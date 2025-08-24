import numpy as np
from PIL import Image
import torch

class SegCLIPPipeline:
    def __init__(self, segmentor, classifier):
        self.segmentor = segmentor
        self.classifier = classifier
        
    def run(self, image_path, class_names):
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Segment
        seg_result = self.segmentor.segment(image)
        masks = self.segmentor.postprocess(seg_result, np.array(image))
        
        # Extract mask arrays
        mask_arrays = []
        for ann in masks:
            # Instead of directly accessing ann['segmentation'], let's check the structure
            if isinstance(ann, dict) and 'segmentation' in ann:
                mask = ann['segmentation']
            else:
                # If ann itself is the mask
                mask = ann
                
            # Handle different mask formats
            if isinstance(mask, dict):
                # COCO format RLE mask
                try:
                    from pycocotools import mask as mask_utils
                    mask = mask_utils.decode(mask)
                except ImportError:
                    # If pycocotools is not available, skip this mask
                    continue
            elif hasattr(mask, 'cpu'):
                # PyTorch tensor
                mask = mask.cpu().numpy()
            elif torch.is_tensor(mask):
                # PyTorch tensor
                mask = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray) and mask.ndim > 2:
                # If mask has more than 2 dimensions, take the first channel
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
            
            # Only add valid masks
            if isinstance(mask, np.ndarray):
                mask_arrays.append(mask)
            
        # Classify
        results = self.classifier.classify(np.array(image), mask_arrays, class_names)
        
        return results