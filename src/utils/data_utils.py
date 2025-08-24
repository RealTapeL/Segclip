import numpy as np
from PIL import Image

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def mask_to_bbox(mask):
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return None
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]