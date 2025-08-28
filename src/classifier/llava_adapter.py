import torch
import numpy as np
from PIL import Image
import os
import sys
from .base_classifier import BaseClassifier
import torch.nn.functional as F
import torchvision.transforms as transforms

class LLaVAAdapter(BaseClassifier):
    def __init__(self, model_path='/home/ps/llava-v1.5-7b', device='cuda'):
        self.device = device
        self.model_path = model_path
        
        try:
            # Get the llava module path
            llava_base_path = os.path.join(os.path.dirname(__file__), 'llava')
            
            # Temporarily modify sys.path to prioritize our llava module
            original_path = sys.path[:]
            sys.path.insert(0, os.path.dirname(llava_base_path))
            
            # Import the modules
            import llava.model.builder
            import llava.mm_utils
            import llava.constants
            
            # Load the model in offline mode
            os.environ['HF_HUB_OFFLINE'] = '1'
            self.tokenizer, self.model, self.image_processor, self.context_len = llava.model.builder.load_pretrained_model(
                model_path, None, 'llava', False, False, device_map="auto", device=device
            )
            
            self.process_images = llava.mm_utils.process_images
            self.tokenizer_image_token = llava.mm_utils.tokenizer_image_token
            self.IMAGE_TOKEN_INDEX = llava.constants.IMAGE_TOKEN_INDEX
            
            # Restore original sys.path
            sys.path[:] = original_path
            
            self.model.eval()
            print(f"Loaded LLaVA model from {model_path}")
        except Exception as e:
            print(f"Failed to load LLaVA model: {e}")
            raise e
            
    def load_model(self, config):
        pass
        
    def preprocess(self, image):
        """Preprocess image for LLaVA"""
        return image
        
    def _prepare_masked_image(self, image, mask):
        """
        Prepare masked region image for LLaVA classification
        """
        # Ensure inputs are numpy arrays
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Handle case where mask is None
        if mask is None:
            return Image.fromarray(image.astype('uint8'))
            
        # Process mask format
        if mask.ndim > 2:
            mask = mask[:, :, 0]
            
        # Ensure mask is binary
        if mask.max() > 1:
            mask = mask / 255.0
            
        # Apply mask to image
        masked_image = image * mask[..., np.newaxis]
        
        # Find mask bounding box
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            # If no mask region, return original image
            return Image.fromarray(image.astype('uint8'))
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop image to mask region with margin
        margin = 20  # Add margin
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        
        cropped_image = masked_image[y_min:y_max, x_min:x_max]
        
        # Create new image with cropped region
        pil_image = Image.fromarray(cropped_image.astype('uint8'))
        return pil_image
        
    def _generate_prompt(self, class_names):
        """
        Generate prompt for LLaVA
        """
        class_list = ", ".join(class_names)
        prompt = f"USER: <image>\nIdentify what is in this image. Select one from the following categories: {class_list}. Answer with just the category name.\nASSISTANT:"
        return prompt
        
    def classify(self, image, masks, class_names):
        """
        Classify masked regions using LLaVA
        """
        results = []
        
        # Handle case where masks is a single mask
        if not isinstance(masks, list):
            masks = [masks]
            
        for i, mask in enumerate(masks):
            # Prepare masked image
            pil_image = self._prepare_masked_image(np.array(image), mask)
            
            # Generate prompt
            prompt = self._generate_prompt(class_names)
            
            # Process image
            image_tensor = self.process_images([pil_image], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                
            # Tokenize prompt
            input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # Generate output
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=256,
                    use_cache=True
                )
                
            # Decode output
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Calculate confidence scores for all classes
            scores = {}
            outputs_lower = outputs.lower()
            for class_name in class_names:
                if class_name.lower() in outputs_lower:
                    scores[class_name] = 1.0
                else:
                    scores[class_name] = 0.0
                    
            # If no exact match, use simple matching
            if all(score == 0.0 for score in scores.values()):
                for class_name in class_names:
                    if class_name.lower() in outputs.lower():
                        scores[class_name] = 1.0
                        break
                        
            # If still no match, assign low confidence to all
            if all(score == 0.0 for score in scores.values()):
                for class_name in class_names:
                    scores[class_name] = 1.0 / len(class_names)
            else:
                # Normalize scores
                total = sum(scores.values())
                scores = {k: v/total for k, v in scores.items()}
                
            # Select class with highest score
            best_class = max(scores, key=scores.get)
            confidence = scores[best_class]
            
            results.append({
                'class': best_class,
                'confidence': confidence,
                'scores': scores,
                'mask': mask
            })
            
        return results