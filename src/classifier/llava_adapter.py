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
            
            # Load the model
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
        image = np.array(image)
        results = []
        
        with torch.no_grad():
            for i, mask in enumerate(masks):
                # Process mask to correct format
                mask = self._process_mask(mask, image.shape[0], image.shape[1])
                
                # Prepare masked region image
                masked_pil_image = self._prepare_masked_image(image, mask)
                
                # Process image for LLaVA
                image_tensor = self.process_images([masked_pil_image], self.image_processor, {})
                if type(image_tensor) is list:
                    image_tensor = [_image.to(self.model.device, dtype=torch.float16) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                
                # Generate prompt
                prompt = self._generate_prompt(class_names)
                input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
                
                # Generate response
                generated_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    max_new_tokens=128,
                    do_sample=False,
                )
                
                # Decode output
                output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predicted_text = output_text.strip()
                
                # Parse the response to get assistant's answer only
                if "ASSISTANT:" in predicted_text:
                    predicted_text = predicted_text.split("ASSISTANT:")[-1].strip()
                
                # Create scores based on exact match
                scores_dict = {}
                for class_name in class_names:
                    # Simple matching - can be improved with fuzzy matching
                    if class_name.lower() in predicted_text.lower() or predicted_text.lower() in class_name.lower():
                        scores_dict[class_name] = 1.0
                    else:
                        scores_dict[class_name] = 0.0
                
                # If no exact match, assign equal probabilities
                if sum(scores_dict.values()) == 0:
                    for class_name in class_names:
                        scores_dict[class_name] = 1.0 / len(class_names)
                else:
                    # Normalize scores
                    total = sum(scores_dict.values())
                    for class_name in class_names:
                        scores_dict[class_name] /= total
                
                # Get predicted class
                predicted_class = max(scores_dict, key=scores_dict.get)
                confidence = scores_dict[predicted_class]
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': predicted_class,
                    'confidence': confidence,
                    'object_id': i
                })
                
        return results
        
    def _process_mask(self, mask, image_height, image_width):
        """Process mask to ensure correct format"""
        # Handle different mask formats
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask[:, :, 0]
            
        # Ensure mask is the same size as image
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            # Resize mask to match image size
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = transforms.Resize((image_height, image_width), interpolation=Image.NEAREST)(mask_pil)
            mask = np.array(mask_pil) / 255.0
            
        # Ensure mask is binary
        if mask.max() > 1:
            mask = mask / 255.0
            
        return mask