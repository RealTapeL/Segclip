import sys
import os

# Add the segmentor directory to the path so we can import fastsam as a package
segmentor_path = os.path.dirname(__file__)
if segmentor_path not in sys.path:
    sys.path.insert(0, segmentor_path)

import torch
import numpy as np
from PIL import Image

# Import FastSAM modules directly from local copy
from fastsam.model import FastSAM
from fastsam.prompt import FastSAMPrompt

# Use absolute import for BaseSegmentor
from src.segmentor.base_segmentor import BaseSegmentor

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