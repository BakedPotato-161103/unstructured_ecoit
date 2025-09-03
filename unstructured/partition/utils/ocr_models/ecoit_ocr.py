from __future__ import annotations

import os
from copy import deepcopy
import re
import functools
from typing import TYPE_CHECKING
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import cv2
import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image as PILImage

from unstructured.logger import logger, trace_logger
from unstructured.partition.utils.config import env_config
from unstructured.partition.utils.constants import Source
from unstructured.documents.elements import ElementType

from unstructured.partition.utils.ocr_models.ocr_interface import OCRAgent
from unstructured.utils import requires_dependencies
import unstructured_pytesseract

if TYPE_CHECKING:
    from unstructured_inference.inference.elements import TextRegions
    from unstructured_inference.inference.layoutelement import LayoutElements

# Ecoit import models
from doctr.models.detection.zoo import detection_predictor
from doctr.utils.geometry import detach_scores
from doctr.models.builder import DocumentBuilder

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class Clause: 
    h_quantile = 0.5
    def __init__(self):
        self.words = []
        self.scores = 0.
    
    def parse(self):
        if len(self.words) == 0:
            return []
        words = np.array(self.words)
        words_cc = xyxy2cxcywh(words)
        cy = np.quantile(words_cc[:, 1], Clause.h_quantile)
        h = np.quantile(words_cc[:, 3], 0.8)
        return [[words[0, 0], cy - h/2, words[-1, 2], cy + h/2, self.scores / len(self.words)]]
    
    def append(self, box, score):
        self.words.append(box)
        self.scores += score
    
    def reset(self):
        self.words = []
        self.scores = 0
    
    def length(self) -> float:
        if len(self.words) == 0:
            return 0
        elif len(self.words) == 1:
            x1, _, x2, _ = self.words[0]
            return x2 - x1
        start, end = self.words[0], self.words[-1]
        return float(end[2] - start[0])
    def __len__(self):
        return len(self.words)

# Adding singleton to avoid multiple initialization
def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class OCRAgentECOIT(OCRAgent):
    """ The OCRAgent consists of 2 components: a detector and a recognizer, is sequentially inferred
        Will extend and abtract to any 2-step OCRAgent w/ conditions
    """
    def __init__(self, language="vie"):
        self.doc_builder = DocumentBuilder()
        # Default to Vietnamese since its ermmm for probation
        self.language = "vie"
        self.load()
        
    def load(self):
        # Import Clause height quantile from here
        Clause.h_quantile = env_config.ECOIT_TEXT_HEIGHT_QUANTILE
        self.parse_line = env_config.ECOIT_PARSE_LINE
        # Follows initialization pipeline of Unstructured w/ env_var for config - 
        # kinda risky but it is the best practice we've known
        # Load text detection model
        doctr_det_model = env_config.DOCTR_DET_MODEL
        doctr_device = env_config.DOCTR_DEVICE
        self.det_threshold = env_config.DOCTR_THRESHOLD
        self.det_cluster = env_config.DOCTR_DET_CLUSTER
        logger.info(f"Instantiating text detection model, using {doctr_det_model} on {doctr_device} with {self.det_threshold} thresholding!!!")
        # Batch size is default to 1 bcz unstructured process by page not by doc, gonna change that later. 
        self.detector = detection_predictor(arch=doctr_det_model, pretrained=True, batch_size=1)
        self.detector.model.to(doctr_device)
        
        # Load text recognition model
        logger.info(f"Instantiating VietOCR Predictor using {env_config.VIETOCR_MODEL_NAME} on {env_config.VIETOCR_DEVICE}!!")
        config = Cfg.load_config_from_name(env_config.VIETOCR_MODEL_NAME)
        config['cnn']['pretrained']=True
        config['device'] = env_config.VIETOCR_DEVICE
        config['predictor']['beamsearch'] = env_config.VIETOCR_USE_BEAM_SEARCH
        self.recognizer = Predictor(config)
        self.rec_batch_size = env_config.VIETOCR_BATCH_SIZE
        self.rec_ratio = config['dataset']['image_max_width'] / config['dataset']['image_height'] * env_config.VIETOCR_MAX_TEXT_WIDTH_RATIO  
        self.spacing_ratio = env_config.ECOIT_TEXT_SPACING_RATIO
        
        

    def forward(self, image: PILImage.Image):
        trace_logger.detail("Performing text detection")
        np_image = np.asarray(image)
        det_res = self.detector([np_image])[0]['words']
        # The bounding box takes the form [left, top, right, bottom, prob]
        det_res = det_res[det_res[:, -1] >= self.det_threshold]
        det_boxes, det_scores = det_res[:, :-1], det_res[:, -1]
        # We need to agree on the norm of minimal height. ? By mean height ?? May be takes the 2-nd quantile to ensure unstable symbols.
        det_boxes_cxcywh = xyxy2cxcywh(det_boxes.copy())
        minimal_cell_height = np.quantile(det_boxes_cxcywh[:, -1], 0.5)
        spacing = minimal_cell_height * self.spacing_ratio 
        det_boxes_cxcywh[:, -1] = np.maximum(det_boxes_cxcywh[:, -1], minimal_cell_height)
        det_boxes = cxcywh2xyxy(det_boxes_cxcywh)
        # Get pseudo layout for word clustering.
        if self.det_cluster is True:
            trace_logger.detail("Sorting bounding boxes and merge to patches")
            layout = self.doc_builder(
                                    [np_image],
                                    [det_boxes],
                                    [det_scores],
                                    [[["-", 0.9] for _ in range(det_boxes.shape[0])]],
                                    [np_image.shape[:2]],
                                    [[{"value": 0, "confidence": None} for i in range(len(det_boxes))]],
                                    None,
                                    None,
                                    ).pages[0]
            # Cluster text boxes based on their spacial distance, keep track of line to append \n characters.
            concat_boxes= []
            line_indexes = []
            for block in layout.blocks:
                for line in block.lines:
                    pose, clause = 0, Clause() 
                    for word in line.words:
                        ((x1, y1), (x2, y2)) = word.geometry
                        short_word = (x2 - x1) / minimal_cell_height <= 1
                        clause.append([x1, y1, x2, y2], word.objectness_score)
                        # Hope that this x2 is not reference :) 
                        pose = x2
                        if (clause.length() / minimal_cell_height > self.rec_ratio) or ((x1 - pose > spacing) and (len(clause) > 0)) or not short_word:
                            # print(f"Ratio {clause.length() / minimal_cell_height} too big for {self.rec_ratio} or block is {x1 - pose} far from {spacing}") 
                            if len(clause.words) > 0:
                                concat_boxes.extend(clause.parse())
                                clause.reset()
                        # Add this for comprehension
                    # Directly parse to contents since not lock parsing. 
                    concat_boxes.extend(clause.parse()) 
                    line_indexes.append(len(concat_boxes))
            concat_boxes = np.array(concat_boxes)
            det_boxes, det_scores = concat_boxes[:, :-1], concat_boxes[:, -1]
        # Crop patches
        trace_logger.detail("Preparing for recognition inference")
        # Get concrete boxes instead of proportional ones.
        real_boxes, patch_indexes, patches = self.crop_patches( 
                                                    np_image.copy(), 
                                                    det_boxes.copy(), 
                                                    issorted=env_config.VIETOCR_SORT_PATCHES,
                                                    padding=0
                                                    )
        words, probs = [], []
        with logging_redirect_tqdm():
            for i in tqdm.tqdm(range(0, len(patches), self.rec_batch_size), desc="Recognition processing", leave=False):
                rec_res = self.recognizer.predict_batch(patches[i:min(i+self.rec_batch_size, len(patches))], return_prob=True)
                words.extend([w.strip() for w in rec_res[0]])
                probs.extend(rec_res[1])
        # Reorder the recognition to align with detection ordering
        det_words = list(zip(list(zip(words, probs)), patch_indexes))
        det_words, _ = zip(*sorted(det_words, key= lambda x: x[1]))
        # Since we assume that the page is straight, orientation is unneeded
        # Return a DocTR document object. --> Get the first page as Unstructured Operate page-wise.
        # Arguments list is parsed following DocTR function format.
        output_boxes, output_words = [], []
        if self.parse_line:
            if not self.det_cluster:
                layout = self.doc_builder(
                                            [np_image],
                                            [real_boxes],
                                            [det_scores],
                                            [det_words],
                                            [np_image.shape[:2]],
                                            [[{"value": 0, "confidence": None} for i in range(len(det_boxes))]],
                                            None,
                                            None,
                                            ).pages[0]
                for block in layout.blocks:
                    for line in block.lines:
                        value, words, clause = 0, [], Clause()
                        for word in line.words:
                            ((x1, y1), (x2, y2)) = word.geometry
                            words.append(word.value)
                            value += word.confidence
                            clause.append([x1, y1, x2, y2], word.objectness_score)
                        output_boxes.append(clause.parse()[0][:-1])
                        output_words.append((" ".join(words), value / len(words)) )
            else: 
                st = 0
                for checkpoint in line_indexes:
                    clause = Clause()
                    clause.words = real_boxes[st:checkpoint]
                    clause.scores = det_scores[st:checkpoint].sum()
                    output_boxes.append(clause.parse()[0][:-1])
                    output_words.append((" ".join([det_words[i][0] for i in range(st, checkpoint)]), 
                                         np.mean([det_words[i][1] for i in range(st, checkpoint)])))
                    st = checkpoint
        else:
            output_boxes = real_boxes.tolist()
            output_words = list(det_words)

        return {'boxes': output_boxes, 'preds': output_words}
    
        # DocBuilder formatting 
        # return  self.doc_builder(
        #                         [np_image],
        #                         [det_boxes],
        #                         [det_scores],
        #                         [det_words],
        #                         [np_image.shape[:2]],
        #                         [[{"value": 0, "confidence": None} for i in range(len(det_words))]],
        #                         None,
        #                         None,
        #                         ).pages[0]
    
    # Make sure to use with copy of the bboxes to prevent overwriting ? Or just create a dup :) 
    def crop_patches(self, image: np.ndarray, bboxes: np.ndarray, issorted: bool = True, padding=0):
        
        if isinstance(padding, int):
            padding = (padding, padding)
        bboxes = bboxes.copy()
        xy_shape = list(image.shape[:2][::-1])
        xy_shape.extend(xy_shape)
        bboxes *= xy_shape
        # Recognition pipeline
        # Will direct to temporary storing later, if that is neccessary
        patches = []
        for box in bboxes.astype(int).tolist():
            x_min, y_min, x_max, y_max = box
            x_min, x_max = max(0, x_min - padding[0]), min(xy_shape[0], x_max + padding[0])
            y_min, y_max = max(0, y_min - padding[1]), min(xy_shape[1], y_max + padding[1])
            patches.append(PILImage.fromarray(image[y_min:y_max, x_min:x_max, ::-1].copy()))
        # Sort patches with H/W ratio for efficient recognition inference
        patches = list(enumerate(patches))
        if issorted:
            patches = sorted(patches, key = lambda entry: entry[1].size[0] / entry[1].size[1])
        indexes, patches = zip(*patches)
        return bboxes, indexes, patches

    @requires_dependencies("unstructured_inference")
    def get_layout_elements_from_image(self, image: PILImage.Image) -> LayoutElements:
        from unstructured_inference.inference.layoutelement import LayoutElements

        ocr_regions = self.get_layout_from_image(image)
        # NOTE(christine): For paddle, there is no difference in `ocr_layout` and `ocr_text` in
        # terms of grouping because we get ocr_text from `ocr_layout, so the first two grouping
        # and merging steps are not necessary.
        return LayoutElements(
                                element_coords=ocr_regions.element_coords,
                                texts=ocr_regions.texts,
                                element_class_ids=np.zeros(ocr_regions.texts.shape),
                                element_class_id_map={0: ElementType.UNCATEGORIZED_TEXT},
                            )
    
    def get_layout_from_image(self, image: PILImage.Image) -> TextRegions:
        # Follows paddle_ocr.py format
        ocr_data = self.forward(image)
        ocr_regions = self.parse_data(ocr_data)
        return ocr_regions

    @requires_dependencies("unstructured_inference")
    def parse_data(self, ocr_data):
        """ Parsing  OCR data as dict to Unstructured TextRegions for unified system calling. 
        """
        from unstructured_inference.inference.elements import TextRegions
        from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords
        text_regions: list[TextRegions] = []
        for box, word in zip(ocr_data['boxes'], ocr_data['preds']):
            reg = build_text_region_from_coords(
                        *box, 
                        text=word[0], 
                        source=Source.OCR_ECOIT
                    )
            text_regions.append(reg)
        return TextRegions.from_list(text_regions)

    def get_text_from_image(self, image: PILImage.Image) -> str:
        ocr_regions = self.get_layout_from_image(image)
        return "\n\n".join(ocr_regions.texts)

    def is_text_sorted(self) -> bool:
        False

def xyxy2cxcywh(coords: np.ndarray): 
    cx = coords[:, [0, 2]].mean(axis=1)
    cy = coords[:, [1, 3]].mean(axis=1)
    w = (cx - coords[:, 0]) * 2
    h = (cy - coords[:, 1]) * 2
    return np.stack([cx, cy, w, h], axis=1)

def cxcywh2xyxy(coords: np.ndarray):
    x1y1 = coords[:, :2] - coords[:, 2:] / 2
    x2y2 = coords[:, :2] + coords[:, 2:] / 2
    return np.concatenate([x1y1, x2y2], axis=1)