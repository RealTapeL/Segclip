import sys
import os
import torch
from PIL import Image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.segmentor.fastsam_adapter import FastSAMAdapter
from src.classifier.llava_adapter import LLaVAAdapter
from src.pipeline.segclip_pipeline import SegCLIPPipeline

def save_segmentation_results(image_path, seg_result, masks, fastsam_model, output_root, timestamp):
    """
    Save FastSAM segmentation results
    """
    segmentation_folder = os.path.join(output_root, timestamp, 'segmentation')
    masks_folder = os.path.join(segmentation_folder, 'masks')
    os.makedirs(segmentation_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image)
    
    # Save original image
    original_folder = os.path.join(output_root, timestamp, 'original')
    os.makedirs(original_folder, exist_ok=True)
    original_image.save(os.path.join(original_folder, 'input_image.jpg'))
    
    # Save individual masks
    for i, mask_data in enumerate(masks):
        # Handle different mask formats
        if isinstance(mask_data, dict) and 'segmentation' in mask_data:
            mask = mask_data['segmentation']
        else:
            mask = mask_data
            
        # Convert to numpy if needed
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif torch.is_tensor(mask):
            mask = mask.cpu().numpy()
            
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask[:, :, 0]
            
        # Save mask
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_filename = f"mask_{i+1:03d}.png"
        mask_path = os.path.join(masks_folder, mask_filename)
        mask_image.save(mask_path)
    
    # Create and save full visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(original_array)
    
    # Overlay all masks with different colors
    for i, mask_data in enumerate(masks):
        if isinstance(mask_data, dict) and 'segmentation' in mask_data:
            mask = mask_data['segmentation']
        else:
            mask = mask_data
            
        # Convert to numpy if needed
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif torch.is_tensor(mask):
            mask = mask.cpu().numpy()
            
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask[:, :, 0]
            
        # Show mask
        ax.imshow(mask, cmap=plt.cm.jet, alpha=0.5)
    
    ax.axis('off')
    fig.savefig(os.path.join(segmentation_folder, 'full_visualization.png'), 
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Save full mask
    full_mask = np.zeros_like(original_array[:, :, 0], dtype=np.float32)
    for mask_data in masks:
        if isinstance(mask_data, dict) and 'segmentation' in mask_data:
            mask = mask_data['segmentation']
        else:
            mask = mask_data
            
        # Convert to numpy if needed
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif torch.is_tensor(mask):
            mask = mask.cpu().numpy()
            
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask[:, :, 0]
            
        full_mask = np.maximum(full_mask, mask)
        
    full_mask_image = Image.fromarray((full_mask * 255).astype(np.uint8))
    full_mask_image.save(os.path.join(segmentation_folder, 'full_mask.png'))
    
    print(f"Segmentation results saved to {segmentation_folder}")

def save_classification_results(image_path, results, output_root, timestamp):
    """
    Save masks and images by class names
    """
    classification_folder = os.path.join(output_root, timestamp, 'classification')
    os.makedirs(classification_folder, exist_ok=True)
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image)
    
    # Save results by class
    class_counts = {}  # Track count of each class for naming
    
    for i, result in enumerate(results):
        class_name = result['class']
        
        # Update class count
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
        
        # Create class folder
        class_folder = os.path.join(classification_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Save mask
        mask = result['mask']
        # Ensure mask is in the correct format
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_filename = f"{class_name}_{class_counts[class_name]}_mask.png"
        mask_path = os.path.join(class_folder, mask_filename)
        mask_image.save(mask_path)
        
        # Save masked image
        masked_image_array = original_array * mask[..., np.newaxis]
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        image_filename = f"{class_name}_{class_counts[class_name]}_image.png"
        image_path_file = os.path.join(class_folder, image_filename)
        masked_image.save(image_path_file)
        
        # Save info file
        info_filename = f"{class_name}_{class_counts[class_name]}_info.txt"
        info_path = os.path.join(class_folder, info_filename)
        with open(info_path, 'w') as f:
            f.write(f"Class: {class_name}\n")
            f.write(f"Confidence: {result['confidence']:.3f}\n")
            f.write(f"Scores:\n")
            for class_label, score in result['scores'].items():
                f.write(f"  {class_label}: {score:.3f}\n")
    
    print(f"Classification results saved to {classification_folder}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SegCLIP Inference with LLaVA')
    parser.add_argument('--image_path', type=str, 
                        default='./samples/sample.jpg',
                        help='Path to input image')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=['computer', 'cup', 'pencil'],
                        help='Class names for classification')
    parser.add_argument('--fastsam_model', type=str, 
                        default='../FastSAM/weights/FastSAM-x.pt', 
                        help='FastSAM model path')
    parser.add_argument('--llava_model', type=str, 
                        default='/home/ps/llava-v1.5-7b',
                        help='LLaVA model path')
    parser.add_argument('--output_path', type=str, default='./outputs', help='Output root directory path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for segmentation')
    parser.add_argument('--iou', type=float, default=0.9, help='IoU threshold for segmentation')
    
    args = parser.parse_args()
    
    print(f"Using LLaVA model: {args.llava_model}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize models
    segmentor = FastSAMAdapter(args.fastsam_model, device=args.device, conf=args.conf, iou=args.iou)
    classifier = LLaVAAdapter(args.llava_model, device=args.device)
    
    # Create pipeline
    pipeline = SegCLIPPipeline(segmentor, classifier)
    
    # Run segmentation first to get masks
    image = Image.open(args.image_path).convert('RGB')
    seg_result = segmentor.segment(image)
    masks = segmentor.postprocess(seg_result, np.array(image))
    
    # Save segmentation results
    save_segmentation_results(args.image_path, seg_result, masks, args.fastsam_model, args.output_path, timestamp)
    
    # Run full pipeline for classification
    results = pipeline.run(args.image_path, args.classes)
    
    # Save classification results
    save_classification_results(args.image_path, results, args.output_path, timestamp)
    
    # Print results
    print(f"Results saved to {os.path.join(args.output_path, timestamp)}")
    for i, result in enumerate(results):
        print(f"Object {i+1}: {result['class']} (confidence: {result['confidence']:.3f})")

if __name__ == '__main__':
    main()