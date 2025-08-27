import torch
import numpy as np
from PIL import Image
import os
from .base_classifier import BaseClassifier
import torch.nn.functional as F

# Import MemoryBank class
from ..utils.memory_bank import MemoryBank

class CLIPAdapter(BaseClassifier):
    def __init__(self, model_name='openai/clip-vit-base-patch16', device='cuda'):
        self.device = device
        self.model_name = model_name
        # Initialize MemoryBank with device parameter
        self.memory_bank = MemoryBank(capacity=1000, device=device)
        
        # Try to load using OpenAI CLIP first
        self.use_transformers = False
        try:
            import clip
            # Check if model_name is a local path
            if os.path.isfile(model_name):
                # For local files, prefer transformers
                raise ImportError("Using transformers for local model file")
            else:
                self.model, self.preprocess = clip.load(model_name, device=device)
                print(f"Loaded CLIP model {model_name} using OpenAI library")
        except Exception as e:
            print(f"Failed to load using OpenAI CLIP: {e}")
            print("Falling back to Hugging Face Transformers")
            # Fallback to Hugging Face Transformers
            self.use_transformers = True
            self._load_with_transformers(model_name, device)
        
        if not self.use_transformers:
            self.model.eval()
            
        # Define prompt templates
        self.prompt_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a clear photo of a {}",
            "a cropped photo of a {}",
        ]
        
    def _load_with_transformers(self, model_name, device):
        from transformers import CLIPProcessor, CLIPModel
        from torchvision import transforms
        
        # Check if model_name is a local path
        if os.path.isfile(model_name) or os.path.isdir(model_name):
            # Local model
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            # Model from Hugging Face Hub
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
        # Define preprocessing manually to match CLIP expectations
        self.transform_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        print(f"Loaded CLIP model {model_name} using Hugging Face Transformers")
        
    def load_model(self, config):
        pass
        
    def preprocess(self, image):
        """Preprocess image"""
        if self.use_transformers:
            return self.transform_preprocess(image)
        else:
            # For OpenAI CLIP, this will be handled in the classify method
            return image
            
    def _get_text_features_with_prompts(self, class_names):
        """
        Get enhanced text features using prompt templates
        """
        if not self.use_transformers:
            import clip
            # Generate multiple prompts for each class
            texts = []
            for template in self.prompt_templates:
                for class_name in class_names:
                    texts.append(template.format(class_name))
            
            # Encode all texts
            tokenized_texts = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokenized_texts)
                
            # Average features for prompts of the same class
            averaged_features = []
            prompts_per_class = len(self.prompt_templates)
            for i in range(len(class_names)):
                start_idx = i * prompts_per_class
                end_idx = (i + 1) * prompts_per_class
                class_features = text_features[start_idx:end_idx]
                averaged_feature = class_features.mean(dim=0, keepdim=True)
                averaged_features.append(averaged_feature)
                
            final_features = torch.cat(averaged_features, dim=0)
            final_features /= final_features.norm(dim=-1, keepdim=True)
            return final_features
        else:
            # For Transformers version, use simple method
            text_inputs = self.processor(
                text=class_names, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features
            
    def _prepare_masked_image(self, image, mask):
        """
        Prepare masked region image for CLIP classification
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
        margin = 10  # Add margin
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        
        cropped_image = masked_image[y_min:y_max, x_min:x_max]
        
        # Create new image with cropped region
        pil_image = Image.fromarray(cropped_image.astype('uint8'))
        return pil_image
        
    def classify(self, image, masks, class_names):
        image = np.array(image)
        results = []
        
        if self.use_transformers:
            return self._classify_with_transformers(image, masks, class_names)
        else:
            return self._classify_with_openai(image, masks, class_names)
                
    def _classify_with_openai(self, image, masks, class_names):
        import clip
        results = []
        
        # Get enhanced text features
        text_features = self._get_text_features_with_prompts(class_names)
        
        with torch.no_grad():
            for i, mask in enumerate(masks):
                # Process mask to correct format
                mask = self._process_mask(mask, image.shape[0], image.shape[1])
                
                # Prepare masked region image
                masked_pil_image = self._prepare_masked_image(image, mask)
                
                # Preprocess image for CLIP
                processed_image = self.preprocess(masked_pil_image).unsqueeze(0).to(self.device)
                
                # Get image features
                image_features = self.model.encode_image(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = 100.0 * image_features @ text_features.T
                scores = similarity[0]
                
                # Average scores for each class (multiple prompts per class)
                prompts_per_class = len(self.prompt_templates)
                class_scores = []
                for j in range(len(class_names)):
                    start_idx = j * prompts_per_class
                    end_idx = (j + 1) * prompts_per_class
                    class_score = scores[start_idx:end_idx].mean()
                    class_scores.append(class_score)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(torch.stack(class_scores), dim=0)
                
                values, indices = probabilities.topk(len(class_names))
                
                # Prepare result
                scores_dict = {class_names[j]: float(values[j]) for j in range(len(class_names))}
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0]),
                    'object_id': i
                })
                
                # Add current features and predicted label to memory bank
                predicted_label = class_names[indices[0].item()]
                self.memory_bank.add(image_features.cpu(), [predicted_label])
                
        return results
        
    def _classify_with_transformers(self, image, masks, class_names):
        results = []
        
        # Get text features
        text_features = self._get_text_features_with_prompts(class_names)
        
        with torch.no_grad():
            for i, mask in enumerate(masks):
                # Process mask to correct format
                mask = self._process_mask(mask, image.shape[0], image.shape[1])
                
                # Prepare masked region image
                masked_pil_image = self._prepare_masked_image(image, mask)
                
                # Preprocess image for CLIP
                processed_image = self.transform_preprocess(masked_pil_image).unsqueeze(0).to(self.device)
                
                # Get image features
                image_features = self.model.get_image_features(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = 100.0 * image_features @ text_features.T
                scores = similarity[0]
                
                # Average scores for each class (multiple prompts per class)
                prompts_per_class = len(self.prompt_templates)
                class_scores = []
                for j in range(len(class_names)):
                    start_idx = j * prompts_per_class
                    end_idx = (j + 1) * prompts_per_class
                    class_score = scores[start_idx:end_idx].mean()
                    class_scores.append(class_score)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(torch.stack(class_scores), dim=0)
                
                values, indices = probabilities.topk(len(class_names))
                
                # Prepare result
                scores_dict = {class_names[j]: float(values[j]) for j in range(len(class_names))}
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0]),
                    'object_id': i
                })
                
                # Add current features and predicted label to memory bank
                predicted_label = class_names[indices[0].item()]
                self.memory_bank.add(image_features.cpu(), [predicted_label])
                
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
            from torchvision import transforms
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = transforms.Resize((image_height, image_width), interpolation=Image.NEAREST)(mask_pil)
            mask = np.array(mask_pil) / 255.0
            
        # Ensure mask is binary
        if mask.max() > 1:
            mask = mask / 255.0
            
        return mask