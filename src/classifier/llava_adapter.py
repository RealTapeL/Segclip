import torch
import numpy as np
from PIL import Image
import os
import sys
from .base_classifier import BaseClassifier
import torch.nn.functional as F
import torchvision.transforms as transforms
import re

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
        
    def _generate_prompt_v1(self, class_names):
        """
        Original prompt - simple classification
        """
        class_list = ", ".join(class_names)
        prompt = f"USER: <image>\nIdentify what is in this image. Select one from the following categories: {class_list}. Answer with just the category name.\nASSISTANT:"
        return prompt
        
    def _generate_prompt_v2(self, class_names):
        """
        Improved prompt - request confidence scores
        """
        class_list = ", ".join(class_names)
        prompt = f"USER: <image>\nIdentify what is in this image. Select one from the following categories: {class_list}. Answer with the category name and a confidence score between 0 and 1 in the format 'category_name:confidence_score'. For example, 'computer:0.95'.\nASSISTANT:"
        return prompt
        
    def _generate_prompt_v3(self, class_names):
        """
        Advanced prompt - request probabilities for all classes
        """
        class_list = ", ".join(class_names)
        prompt = f"USER: <image>\nWhat is in this image? Provide confidence scores between 0 and 1 for each of the following categories: {class_list}. Answer in the format 'category1:score1,category2:score2,...'. For example, 'computer:0.95,glass_bottle:0.1'.\nASSISTANT:"
        return prompt
        
    def _parse_v2_response(self, response, class_names):
        """
        Parse response from prompt v2 (single class with confidence)
        """
        # Try to extract class and confidence in format "class:confidence"
        match = re.search(r'([^:]+):(\d*\.?\d+)', response.strip())
        if match:
            predicted_class = match.group(1).strip()
            try:
                confidence = float(match.group(2))
                # Normalize confidence to be between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
                
                # Create scores dictionary
                scores = {}
                for class_name in class_names:
                    if class_name.lower() == predicted_class.lower():
                        scores[class_name] = confidence
                    else:
                        scores[class_name] = (1.0 - confidence) / (len(class_names) - 1) if len(class_names) > 1 else 0.0
                
                return scores
            except ValueError:
                pass
        
        # Fallback to simple matching
        scores = {}
        response_lower = response.lower()
        match_found = False
        for class_name in class_names:
            if class_name.lower() in response_lower:
                scores[class_name] = 1.0
                match_found = True
            else:
                scores[class_name] = 0.0
        
        if not match_found:
            # If no match, assign equal probability
            for class_name in class_names:
                scores[class_name] = 1.0 / len(class_names)
                
        return scores
        
    def _parse_v3_response(self, response, class_names):
        """
        Parse response from prompt v3 (multiple classes with confidences)
        """
        scores = {}
        
        # Try to parse format "class1:score1,class2:score2,..."
        pairs = response.split(',')
        parsed_scores = {}
        for pair in pairs:
            if ':' in pair:
                try:
                    class_part, score_part = pair.split(':', 1)
                    class_name = class_part.strip()
                    score = float(score_part.strip())
                    parsed_scores[class_name] = max(0.0, min(1.0, score))  # Normalize to [0,1]
                except ValueError:
                    continue
        
        # Match parsed classes with our class names
        for class_name in class_names:
            matched = False
            for parsed_class, parsed_score in parsed_scores.items():
                if class_name.lower() == parsed_class.lower():
                    scores[class_name] = parsed_score
                    matched = True
                    break
            if not matched:
                scores[class_name] = 0.0  # Default to 0 if not mentioned
        
        # If no scores were parsed, fallback to simple matching
        if all(score == 0.0 for score in scores.values()):
            response_lower = response.lower()
            match_found = False
            for class_name in class_names:
                if class_name.lower() in response_lower:
                    scores[class_name] = 1.0
                    match_found = True
                else:
                    scores[class_name] = 0.0
            
            if not match_found:
                # If no match, assign equal probability
                for class_name in class_names:
                    scores[class_name] = 1.0 / len(class_names)
        
        return scores
        
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
            
            # Try different prompts in order of complexity
            prompts = [
                self._generate_prompt_v3(class_names),  # Most detailed prompt
                self._generate_prompt_v2(class_names),  # Prompt with confidence
                self._generate_prompt_v1(class_names)   # Simple prompt
            ]
            
            scores = None
            best_response = ""
            
            # Try each prompt until we get a parseable response
            for prompt in prompts:
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
                response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                best_response = response  # Save for potential fallback
                
                # Try to parse the response
                try:
                    if "Provide confidence scores" in prompt:  # v3 prompt
                        scores = self._parse_v3_response(response, class_names)
                    elif "confidence score between 0 and 1" in prompt:  # v2 prompt
                        scores = self._parse_v2_response(response, class_names)
                    else:  # v1 prompt
                        # Simple matching for v1
                        scores = {}
                        response_lower = response.lower()
                        match_found = False
                        for class_name in class_names:
                            if class_name.lower() == response_lower:
                                scores[class_name] = 1.0
                                match_found = True
                            else:
                                scores[class_name] = 0.0
                        
                        if not match_found:
                            # Partial matching
                            for class_name in class_names:
                                if class_name.lower() in response_lower:
                                    scores[class_name] = 0.8
                                else:
                                    scores[class_name] = 0.1
                    
                    # If we successfully got scores, break
                    if scores is not None:
                        break
                except Exception as e:
                    # Continue to next prompt if parsing fails
                    continue
            
            # Fallback if no prompt worked
            if scores is None:
                scores = {}
                response_lower = best_response.lower()
                match_found = False
                for class_name in class_names:
                    if class_name.lower() in response_lower:
                        scores[class_name] = 1.0
                        match_found = True
                    else:
                        scores[class_name] = 0.0
                
                if not match_found:
                    for class_name in class_names:
                        scores[class_name] = 1.0 / len(class_names)
            
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