import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'FastSAM'))

import torch
import numpy as np
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
from .base_segmentor import BaseSegmentor

class FastSAMAdapter(BaseSegmentor):
    def __init__(self, model_path, device='cuda', conf=0.4, iou=0.9):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.model = FastSAM(model_path)
        
    def preprocess(self, image):
        if isinstance(image, str):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        return image
        
    def segment(self, image):
        image = self.preprocess(image)
        everything_results = self.model(image, device=self.device, retina_masks=True, 
                                      imgsz=1024, conf=self.conf, iou=self.iou)
        return everything_results
        
    def postprocess(self, result, image):
        prompt_process = FastSAMPrompt(image, result, device=self.device)
        anns = prompt_process.everything_prompt()
        return anns