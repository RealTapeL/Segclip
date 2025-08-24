import numpy as np
from PIL import Image

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
            mask_arrays.append(ann['segmentation'])
            
        # Classify
        results = self.classifier.classify(np.array(image), mask_arrays, class_names)
        
        return results