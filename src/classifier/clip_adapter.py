import torch
import numpy as np
from PIL import Image
import os
from .base_classifier import BaseClassifier
import torch.nn.functional as F

# 导入MemoryBank类
from ..utils.memory_bank import MemoryBank

class CLIPAdapter(BaseClassifier):
    def __init__(self, model_name='openai/clip-vit-base-patch16', device='cuda'):
        self.device = device
        self.model_name = model_name
        # 初始化MemoryBank时传递设备参数
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
                
                # Apply multi-scale context enhancement
                enhanced_image = self._enhance_with_context(image, mask)
                
                # Multi-scale feature extraction
                scale_scores = []
                for scale_factor in [0.8, 1.0, 1.2]:
                    scaled_image = self._scale_image(enhanced_image, scale_factor)
                    score = self._get_clip_score(scaled_image, text_features, clip)
                    scale_scores.append(score)
                
                # Ensemble scores from different scales
                ensemble_scores = torch.mean(torch.stack(scale_scores), dim=0)
                
                # 使用记忆库增强分类
                pil_image = Image.fromarray(enhanced_image.astype('uint8'))
                processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 从记忆库获取最近邻
                memory_scores, memory_labels = self.memory_bank.get_nearest_neighbors(image_features, k=5)
                combined_scores = ensemble_scores.clone()
                
                if memory_scores is not None and self.memory_bank.get_size() > 1:
                    # 只有在记忆库中有足够数据时才进行聚类
                    n_clusters = min(5, self.memory_bank.get_size())
                    if n_clusters > 1:  # 只有在有足够的簇时才进行聚类
                        cluster_result = self.memory_bank.cluster_features(n_clusters=n_clusters)
                        if cluster_result is not None:
                            cluster_centers, cluster_labels = cluster_result
                            cluster_centers = torch.from_numpy(cluster_centers).float().to(self.device)
                            cluster_features = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
                            
                            # 计算与聚类中心的相似度
                            cluster_similarities = 100.0 * image_features @ cluster_features.t()
                            
                            # 结合CLIP分数、记忆库分数和聚类分数
                            for j, label in enumerate(cluster_labels):
                                if label in class_names:
                                    label_idx = class_names.index(label)
                                    # 融合聚类分数（给予权重0.3）
                                    combined_scores[label_idx] += 0.3 * cluster_similarities[0, j]
                                    
                            # 同时考虑最近邻的标签信息
                            for labels_for_query in memory_labels:
                                for label in labels_for_query:
                                    if label in class_names:
                                        label_idx = class_names.index(label)
                                        # 给最近邻标签增加权重
                                        combined_scores[label_idx] += 0.1
                
                combined_scores = F.softmax(combined_scores, dim=-1)
                values, indices = combined_scores.topk(len(class_names))
                
                # Prepare result
                scores_dict = {class_names[j]: float(values[j]) for j in range(len(class_names))}
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0]),
                    'object_id': i
                })
                
                # 将当前特征和预测标签添加到记忆库
                predicted_label = class_names[indices[0].item()]
                self.memory_bank.add(image_features.cpu(), [predicted_label])
                
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
                
                # Apply multi-scale context enhancement
                enhanced_image = self._enhance_with_context(image, mask)
                
                # Multi-scale feature extraction
                scale_scores = []
                for scale_factor in [0.8, 1.0, 1.2]:
                    scaled_image = self._scale_image(enhanced_image, scale_factor)
                    score = self._get_transformers_score(scaled_image, text_features)
                    scale_scores.append(score)
                
                # Ensemble scores from different scales
                ensemble_scores = torch.mean(torch.stack(scale_scores), dim=0)
                
                # 使用记忆库增强分类
                pil_image = Image.fromarray(enhanced_image.astype('uint8'))
                processed_image = self.transform_preprocess(pil_image).unsqueeze(0).to(self.device)
                image_features = self.model.get_image_features(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 从记忆库获取最近邻
                memory_scores, memory_labels = self.memory_bank.get_nearest_neighbors(image_features, k=5)
                combined_scores = ensemble_scores.clone()
                
                if memory_scores is not None and self.memory_bank.get_size() > 1:
                    # 只有在记忆库中有足够数据时才进行聚类
                    n_clusters = min(5, self.memory_bank.get_size())
                    if n_clusters > 1:  # 只有在有足够的簇时才进行聚类
                        cluster_result = self.memory_bank.cluster_features(n_clusters=n_clusters)
                        if cluster_result is not None:
                            cluster_centers, cluster_labels = cluster_result
                            cluster_centers = torch.from_numpy(cluster_centers).float().to(self.device)
                            cluster_features = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
                            
                            # 计算与聚类中心的相似度
                            cluster_similarities = 100.0 * image_features @ cluster_features.t()
                            
                            # 结合CLIP分数、记忆库分数和聚类分数
                            for j, label in enumerate(cluster_labels):
                                if label in class_names:
                                    label_idx = class_names.index(label)
                                    # 融合聚类分数（给予权重0.3）
                                    combined_scores[label_idx] += 0.3 * cluster_similarities[0, j]
                                    
                            # 同时考虑最近邻的标签信息
                            for labels_for_query in memory_labels:
                                for label in labels_for_query:
                                    if label in class_names:
                                        label_idx = class_names.index(label)
                                        # 给最近邻标签增加权重
                                        combined_scores[label_idx] += 0.1
                
                combined_scores = F.softmax(combined_scores, dim=-1)
                values, indices = combined_scores.topk(len(class_names))
                
                # Prepare result
                scores_dict = {class_names[j]: float(values[j]) for j in range(len(class_names))}
                
                results.append({
                    'mask': mask,
                    'scores': scores_dict,
                    'class': class_names[indices[0].item()],
                    'confidence': float(values[0]),
                    'object_id': i
                })
                
                # 将当前特征和预测标签添加到记忆库
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
        
    def _enhance_with_context(self, image, mask):
        """
        增强图像上下文信息，通过扩展掩码区域获取更多上下文
        """
        # 扩展掩码区域以包含更多上下文信息
        from scipy import ndimage
        expanded_mask = ndimage.binary_dilation(mask, iterations=5).astype(mask.dtype)
        
        # 创建上下文增强图像（结合原始图像和扩展区域）
        context_enhanced = image.copy()
        # 可以调整上下文区域的亮度或对比度以增强特征
        return context_enhanced
        
    def _scale_image(self, image, scale_factor):
        """
        缩放图像以实现多尺度特征提取
        """
        if scale_factor == 1.0:
            return image
            
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        pil_image = Image.fromarray(image.astype('uint8'))
        scaled_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
        return np.array(scaled_image)
        
    def _get_clip_score(self, image, text_features, clip):
        """
        使用OpenAI CLIP获取图像文本相似度分数
        """
        pil_image = Image.fromarray(image.astype('uint8'))
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Get image features
        image_features = self.model.encode_image(processed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = 100.0 * image_features @ text_features.T
        return similarity[0]
        
    def _get_transformers_score(self, image, text_features):
        """
        使用Transformers CLIP获取图像文本相似度分数
        """
        pil_image = Image.fromarray(image.astype('uint8'))
        processed_image = self.transform_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Get image features
        image_features = self.model.get_image_features(processed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = 100.0 * image_features @ text_features.T
        return similarity[0]