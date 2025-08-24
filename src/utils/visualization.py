import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def visualize_results(image, results, class_names=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = []
    for i in range(len(results)):
        color = tuple(np.random.randint(0, 255, 3) / 255.0)
        colors.append(color)
        
    for i, result in enumerate(results):
        mask = result['mask']
        class_name = result['class']
        confidence = result['confidence']
        
        ax.imshow(mask, cmap=plt.cm.jet, alpha=0.5)
        # Find a point in the mask to place text
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            y_center = int(np.mean(y_coords))
            x_center = int(np.mean(x_coords))
            ax.text(x_center, y_center, f'{class_name}: {confidence:.2f}', 
                   bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    ax.axis('off')
    return fig

def show_masks(image, masks):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    for ann in masks:
        mask = ann['segmentation']
        ax.imshow(mask, cmap=plt.cm.jet, alpha=0.5)
    ax.axis('off')
    return fig