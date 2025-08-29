import sys
import os
import torch
from PIL import Image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.segmentor.fastsam_adapter import FastSAMAdapter
from src.classifier.llava_adapter import LLaVAAdapter
from src.pipeline.segclip_pipeline import SegCLIPPipeline

def load_prompts(prompts_file):
    """
    Load prompts from a JSON file
    """
    if not os.path.exists(prompts_file):
        print(f"Warning: Prompts file {prompts_file} not found. Using default prompt.")
        return None
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return prompts
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        return None

def load_custom_prompt(prompt_file):
    """
    Load a custom prompt from a text file
    """
    if not os.path.exists(prompt_file):
        print(f"Warning: Custom prompt file {prompt_file} not found.")
        return None
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    except Exception as e:
        print(f"Error loading custom prompt file: {e}")
        return None

def save_segmentation_results(image_path, seg_result, masks, fastsam_model, output_root, timestamp):
    """
    Save FastSAM segmentation results
    """
    # Ensure output_root is an absolute path
    output_root = os.path.abspath(output_root)
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
    
    # Save FastSAM raw results (the actual FastSAM inference results)
    fastsam_result_folder = os.path.join(segmentation_folder, 'fastsam_inference')
    os.makedirs(fastsam_result_folder, exist_ok=True)
    
    # Save FastSAM visualization using direct FastSAM inference - the correct way
    if seg_result and len(seg_result) > 0:
        # Use the fastsam_inference module to generate the original FastSAM result
        fastsam_output_path = os.path.join(fastsam_result_folder, 'fastsam_segmentation.png')
        
        # Build command to run FastSAM inference
        fastsam_inference_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'segmentor', 'fastsam_inference.py')
        cmd = (f"python {fastsam_inference_path} "
               f"--model_path {fastsam_model} "
               f"--img_path {image_path} "
               f"--output {fastsam_result_folder}/ "
               f"--conf 0.1 "
               f"--iou 0.3")
        
        # Run FastSAM inference
        os.system(cmd)
        
        # Rename the output file to our standard name
        # FastSAM inference saves with the same name as input, so we need to rename it
        input_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(input_filename)
        generated_file = os.path.join(fastsam_result_folder, f"{name}_segmented{ext}")
        if os.path.exists(generated_file):
            target_file = os.path.join(fastsam_result_folder, 'fastsam_segmentation.png')
            if os.path.exists(target_file):
                os.remove(target_file)
            os.rename(generated_file, target_file)
            print(f"FastSAM segmentation result saved to: {target_file}")
        else:
            # Try the original filename if _segmented version doesn't exist
            generated_file = os.path.join(fastsam_result_folder, input_filename)
            if os.path.exists(generated_file):
                target_file = os.path.join(fastsam_result_folder, 'fastsam_segmentation.png')
                if os.path.exists(target_file):
                    os.remove(target_file)
                os.rename(generated_file, target_file)
                print(f"FastSAM segmentation result saved to: {target_file}")
    
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
    # Print the actual files saved
    if os.path.exists(segmentation_folder):
        files = os.listdir(segmentation_folder)
        print(f"Files in segmentation folder: {files}")
    if os.path.exists(masks_folder):
        mask_files = os.listdir(masks_folder)
        print(f"Number of mask files saved: {len(mask_files)}")
    if os.path.exists(fastsam_result_folder):
        fastsam_files = os.listdir(fastsam_result_folder)
        print(f"FastSAM result files: {fastsam_files}")

def save_classification_results(image_path, results, output_root, timestamp):
    """
    Save masks and images by class names
    """
    # Ensure output_root is an absolute path
    output_root = os.path.abspath(output_root)
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
    # Print the actual files saved
    if os.path.exists(classification_folder):
        class_dirs = os.listdir(classification_folder)
        print(f"Classification subdirectories: {class_dirs}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SegCLIP Inference with LLaVA')
    parser.add_argument('--image_path', type=str, 
                        default='./samples/sample.jpg',
                        help='Path to input image')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=None,
                        help='Class names for classification. If not provided, open vocabulary classification is used.')
    parser.add_argument('--open_vocabulary', action='store_true',
                        help='Use open vocabulary classification (equivalent to not providing --classes)')
    parser.add_argument('--prompt_key', type=str, 
                        default=None,
                        help='Key for prompt in prompts.json file')
    parser.add_argument('--custom_prompt', type=str, 
                        default=None,
                        help='Custom prompt for LLaVA classification')
    parser.add_argument('--prompt_file', type=str, 
                        default='/home/ps/few-shot-research/mcxh_img/SegClip/prompt.txt',
                        help='Path to a text file containing a custom prompt')
    parser.add_argument('--prompts_file', type=str, 
                        default='/home/ps/few-shot-research/mcxh_img/SegClip/config/prompts.json',
                        help='Path to prompts JSON file')
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
    parser.add_argument('--max_objects', type=int, default=30, 
                        help='Maximum number of objects to classify (to avoid too many segments)')
    
    args = parser.parse_args()
    
    # Load prompts from JSON file
    prompts = load_prompts(args.prompts_file)
    
    # Load custom prompt from text file if provided
    custom_prompt_from_file = None
    if args.prompt_file:
        custom_prompt_from_file = load_custom_prompt(args.prompt_file)
        if custom_prompt_from_file:
            print(f"Loaded custom prompt from file: {args.prompt_file}")
    
    # Determine classification mode
    use_open_vocabulary = args.open_vocabulary or (args.classes is None and args.custom_prompt is None and args.prompt_key is None and args.prompt_file is None)
    use_custom_prompt = args.custom_prompt is not None
    use_prompt_key = args.prompt_key is not None
    use_prompt_file = args.prompt_file is not None and custom_prompt_from_file is not None
    
    # Determine the prompt to use
    if use_custom_prompt:
        print(f"Using custom prompt: {args.custom_prompt}")
        custom_prompt = args.custom_prompt
        class_names = None
    elif use_prompt_file and custom_prompt_from_file:
        print(f"Using custom prompt from file: {args.prompt_file}")
        custom_prompt = custom_prompt_from_file
        class_names = None
    elif use_prompt_key:
        if prompts and args.prompt_key in prompts:
            custom_prompt = prompts[args.prompt_key]
            print(f"Using prompt key '{args.prompt_key}': {custom_prompt}")
        else:
            custom_prompt = None
            print(f"Prompt key '{args.prompt_key}' not found in prompts file. Using default behavior.")
        class_names = None
    elif use_open_vocabulary:
        print("Using open vocabulary classification - LLaVA will determine object classes automatically")
        custom_prompt = None
        class_names = None
    else:
        print(f"Using specified classes: {args.classes}")
        custom_prompt = None
        class_names = args.classes
    
    print(f"Using LLaVA model: {args.llava_model}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output path is absolute
    output_path = os.path.abspath(args.output_path)
    print(f"Output will be saved to: {output_path}")
    
    # Initialize models
    segmentor = FastSAMAdapter(args.fastsam_model, device=args.device, conf=args.conf, iou=args.iou)
    classifier = LLaVAAdapter(args.llava_model, device=args.device)
    
    # Create pipeline
    pipeline = SegCLIPPipeline(segmentor, classifier)
    
    # Run segmentation first to get masks
    image = Image.open(args.image_path).convert('RGB')
    seg_result = segmentor.segment(image)
    masks = segmentor.postprocess(seg_result, np.array(image))
    
    # Limit number of masks if needed
    if len(masks) > args.max_objects:
        print(f"Limiting number of objects from {len(masks)} to {args.max_objects}")
        masks = masks[:args.max_objects]
    
    # Save segmentation results regardless of what happens next
    try:
        save_segmentation_results(args.image_path, seg_result, masks, args.fastsam_model, output_path, timestamp)
        print("Segmentation results saved successfully")
    except Exception as e:
        print(f"Error saving segmentation results: {e}")
        import traceback
        traceback.print_exc()
    
    # Run full pipeline for classification
    try:
        # Pass custom_prompt to the pipeline - fix the argument passing
        results = pipeline.run(image_path=args.image_path, class_names=class_names, custom_prompt=custom_prompt)
        
        # Limit number of results if needed
        if len(results) > args.max_objects:
            print(f"Limiting number of classification results from {len(results)} to {args.max_objects}")
            results = results[:args.max_objects]
        
        # Save classification results
        save_classification_results(args.image_path, results, output_path, timestamp)
        
        # Print results
        final_output_path = os.path.join(output_path, timestamp)
        print(f"Results saved to {final_output_path}")
        for i, result in enumerate(results):
            print(f"Object {i+1}: {result['class']} (confidence: {result['confidence']:.3f})")
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()
        print("Segmentation completed but classification failed")

if __name__ == '__main__':
    main()