import torch
import numpy as np
from PIL import Image
import os
import sys
from .base_classifier import BaseClassifier
import torch.nn.functional as F
import torchvision.transforms as transforms
import re
import hashlib
import json

class LLaVAAdapter(BaseClassifier):
    def __init__(self, model_path='/home/ps/llava-v1.5-7b', device='cuda'):
        self.device = device
        self.model_path = model_path
        self.response_cache = {}  # 缓存字典
        
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
            # 只有在模型文件存在时才设置离线模式
            if os.path.exists(model_path):
                os.environ['HF_HUB_OFFLINE'] = '1'
            else:
                print(f"Model path {model_path} not found, will try to download from HuggingFace")
                
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
        
    def _get_cache_key(self, image_array, mask, class_names, prompt):
        """生成缓存键"""
        # 将图像和mask转换为bytes用于哈希
        image_bytes = image_array.tobytes()
        mask_bytes = mask.tobytes() if mask is not None else b''
        class_names_str = ','.join(sorted(class_names))
        
        # 创建唯一标识符
        key_data = image_bytes + mask_bytes + class_names_str.encode() + prompt.encode()
        cache_key = hashlib.md5(key_data).hexdigest()
        return cache_key
        
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
        
    def _generate_prompt_simple(self, class_names):
        """
        Simple prompt without numbered list to avoid confusion
        """
        class_list = ", ".join(class_names)
        prompt = f"USER: <image>\nCarefully analyze this image and identify what object is shown. Select the single most likely category from the following list: {class_list}. Respond with ONLY the category name. Do not include any other text or explanation.\nASSISTANT:"
        return prompt
        
    def _generate_open_vocabulary_prompt(self):
        """
        Generate prompt for open vocabulary classification
        """
        prompt = "USER: <image>\nCarefully analyze this image and identify what single object is shown. Respond with ONLY the object name in a single word or short phrase. Do not include any other text or explanation.\nASSISTANT:"
        return prompt
        
    def _parse_response(self, response, class_names):
        """
        Parse response with better logic to handle various response formats
        """
        scores = {}
        response_clean = response.strip().lower()
        
        # Remove any trailing punctuation
        response_clean = re.sub(r'[.!?\s]+$', '', response_clean)
        
        # Try to find exact matches
        match_found = False
        for class_name in class_names:
            class_name_clean = class_name.lower()
            # Check for exact match or match with number prefix (e.g., "2. computer" -> "computer")
            if (class_name_clean == response_clean or 
                f"{class_name_clean}" == response_clean or
                class_name_clean in response_clean):
                scores[class_name] = 1.0
                match_found = True
            else:
                scores[class_name] = 0.0
        
        # If no exact match, try partial matching
        if not match_found:
            for class_name in class_names:
                class_name_clean = class_name.lower()
                # Check if response contains class name or vice versa
                if (class_name_clean in response_clean or 
                    response_clean in class_name_clean):
                    # Calculate similarity based on character overlap
                    response_set = set(response_clean)
                    class_set = set(class_name_clean)
                    if len(response_set) > 0 and len(class_set) > 0:
                        overlap = len(response_set & class_set) / len(response_set | class_set)
                        scores[class_name] = min(0.95, max(0.1, overlap))
                    else:
                        scores[class_name] = 0.1
                else:
                    scores[class_name] = 0.0
        
        # If still no matches, assign equal probabilities
        if all(score == 0.0 for score in scores.values()):
            for class_name in class_names:
                scores[class_name] = 1.0 / len(class_names)
                
        return scores
        
    def classify(self, image, masks, class_names=None):
        """
        Classify masked regions using LLaVA with improved prompts and parsing
        """
        # If class_names is None, we perform open vocabulary classification
        open_vocabulary = class_names is None
        
        results = []
        
        # Handle case where masks is a single mask
        if not isinstance(masks, list):
            masks = [masks]
            
        # Convert image to numpy array if it's a tensor
        if isinstance(image, torch.Tensor):
            image_array = image.cpu().numpy()
        else:
            image_array = np.array(image)
            
        # 批量处理所有mask
        pil_images = []
        cache_keys = []
        prompts_list = []
        
        # 使用简单提示避免编号混淆
        for i, mask in enumerate(masks):
            # 生成缓存键
            cache_key = self._get_cache_key(image_array, mask, class_names if class_names else [], 
                                          "open_vocab" if open_vocabulary else "simple_prompt")
            cache_keys.append(cache_key)
            
            # 检查缓存
            if cache_key in self.response_cache:
                results.append(self.response_cache[cache_key])
                continue
                
            # Prepare masked image
            pil_image = self._prepare_masked_image(image_array, mask)
            pil_images.append((i, pil_image, mask, cache_key))  # 保存索引信息
            
            # Use appropriate prompt
            if open_vocabulary:
                prompt = self._generate_open_vocabulary_prompt()
            else:
                prompt = self._generate_prompt_simple(class_names)
            prompts_list.append(prompt)
        
        # 对未缓存的图像进行批量处理
        for idx, (original_index, pil_image, mask, cache_key) in enumerate(pil_images):
            prompt = prompts_list[idx] if idx < len(prompts_list) else (
                self._generate_open_vocabulary_prompt() if open_vocabulary else self._generate_prompt_simple(class_names)
            )
            
            # Process single image
            image_tensor = self.process_images([pil_image], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                
            # Tokenize prompt
            input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # Generate output with optimized parameters
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,  # 改为贪婪搜索而非采样
                    num_beams=1,      # 减少beam search数量
                    max_new_tokens=50,  # 减少最大生成token数
                    use_cache=True
                )
                
            # Decode output
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # 输出LLaVA的原始回复到终端
            print(f"LLaVA Raw Response for object {idx+1}: '{response}'")
            
            if open_vocabulary:
                # For open vocabulary, we just return the response as the class
                result = {
                    'class': response,
                    'confidence': 1.0,  # For open vocabulary, we don't compute confidence scores
                    'scores': {response: 1.0},
                    'mask': mask
                }
            else:
                # Parse response with improved parser
                try:
                    scores = self._parse_response(response, class_names)
                except Exception as e:
                    print(f"Error parsing response: {response}, error: {e}")
                    # Fallback to simple parsing
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
                        for class_name in class_names:
                            scores[class_name] = 1.0 / len(class_names)
                
                # Select class with highest score
                best_class = max(scores, key=scores.get)
                confidence = scores[best_class]
                
                result = {
                    'class': best_class,
                    'confidence': confidence,
                    'scores': scores,
                    'mask': mask
                }
            
            # 缓存结果
            self.response_cache[cache_key] = result
            results.append(result)
            
            # 显存优化：在每次迭代后清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return results