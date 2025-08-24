import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from PIL import Image
from src.segmentor.fastsam_adapter import FastSAMAdapter
from src.classifier.clip_adapter import CLIPAdapter
from src.pipeline.segclip_pipeline import SegCLIPPipeline
from src.utils.visualization import visualize_results

def main():
    parser = argparse.ArgumentParser(description='SegCLIP Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='Class names for classification')
    parser.add_argument('--fastsam_model', type=str, default='FastSAM-x.pt', help='FastSAM model path')
    parser.add_argument('--clip_model', type=str, default='ViT-B/16', help='CLIP model name')
    parser.add_argument('--output_path', type=str, default='output.png', help='Output image path')
    
    args = parser.parse_args()
    
    # Initialize models
    segmentor = FastSAMAdapter(args.fastsam_model)
    classifier = CLIPAdapter(args.clip_model)
    
    # Create pipeline
    pipeline = SegCLIPPipeline(segmentor, classifier)
    
    # Run inference
    results = pipeline.run(args.image_path, args.classes)
    
    # Visualize results
    image = Image.open(args.image_path).convert('RGB')
    fig = visualize_results(image, results)
    fig.savefig(args.output_path)
    print(f"Results saved to {args.output_path}")
    
    # Print results
    for i, result in enumerate(results):
        print(f"Object {i+1}: {result['class']} (confidence: {result['confidence']:.3f})")

if __name__ == '__main__':
    main()