import torch
import open_clip
import numpy as np
from PIL import Image
from .base_classifier import BaseClassifier

class CLIPAdapter(BaseClassifier):
    def __init__(self, model_name='ViT-B/16', pretrained='laion2b_s34b_b88k', device='cuda'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
    def load_model(self, config):
        pass
        
    def preprocess_image(self, image, mask):
        masked_image = image * mask[..., None]
        # Convert to PIL Image and preprocess
        pil_image = Image.fromarray(masked_image.astype('uint8'))
        return self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
    def classify(self, image, masks, class_names):
        image = np.array(image)
        results = []
        
        for mask in masks:
            # Process mask to correct format
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                mask = np.expand_dims(mask, axis=-1)
            
            masked_image = image * mask
            
            # Convert to PIL Image and preprocess
            pil_image = Image.fromarray(masked_image.astype('uint8'))
            processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.encode_image(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Encode text
                texts = self.tokenizer(class_names).to(self.device)
                text_features = self.model.encode_text(texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(class_names))
                
                results.append({
                    'mask': mask,
                    'scores': {class_names[i]: float(values[i]) for i in range(len(class_names))},
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0])
                })
                
        return results