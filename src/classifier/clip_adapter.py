import torch
import numpy as np
from PIL import Image
import os
from .base_classifier import BaseClassifier

class CLIPAdapter(BaseClassifier):
    def __init__(self, model_name='openai/clip-vit-base-patch16', device='cuda'):
        self.device = device
        self.model_name = model_name
        
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
        """实现preprocess方法"""
        if self.use_transformers:
            return self.transform_preprocess(image)
        else:
            # For OpenAI CLIP, this will be handled in the classify method
            return image
        
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
        
        # Tokenize class names once for all masks
        texts = clip.tokenize(class_names).to(self.device)
        
        with torch.no_grad():
            # Encode text features once for all masks
            text_features = self.model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            for i, mask in enumerate(masks):
                # Process mask to correct format
                mask = self._process_mask(mask, image.shape[0], image.shape[1])
                
                # Apply mask to image
                masked_image = image * mask[..., np.newaxis]
                
                # Convert to PIL Image and preprocess
                pil_image = Image.fromarray(masked_image.astype('uint8'))
                processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Get image features
                image_features = self.model.encode_image(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(class_names))
                
                # Prepare result
                scores_dict = {class_names[j]: float(values[j]) for j in range(len(class_names))}
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0]),
                    'object_id': i
                })
                
        return results
        
    def _classify_with_transformers(self, image, masks, class_names):
        results = []
        
        with torch.no_grad():
            # Encode text features once for all masks
            text_inputs = self.processor(
                text=class_names, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            text_features = self.model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            for i, mask in enumerate(masks):
                # Process mask to correct format
                mask = self._process_mask(mask, image.shape[0], image.shape[1])
                
                # Apply mask to image
                masked_image = image * mask[..., np.newaxis]
                
                # Convert to PIL Image and preprocess
                pil_image = Image.fromarray(masked_image.astype('uint8'))
                processed_image = self.transform_preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Get image features
                image_features = self.model.get_image_features(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(class_names))
                
                # Prepare result
                scores_dict = {class_names[j]: float(values[j]) for j in range(len(class_names))}
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0]),
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
            from torchvision import transforms
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = transforms.Resize((image_height, image_width), interpolation=Image.NEAREST)(mask_pil)
            mask = np.array(mask_pil) / 255.0
            
        # Ensure mask is binary
        if mask.max() > 1:
            mask = mask / 255.0
            
        return mask