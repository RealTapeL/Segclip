import os
from .paths import MODEL_CONFIGS_DIR

# CLIP模型配置
CLIP_MODEL_CONFIGS = {
    'ViT-B/16': os.path.join(MODEL_CONFIGS_DIR, 'ViT-B-16.json'),
    'ViT-B/32': os.path.join(MODEL_CONFIGS_DIR, 'ViT-B-32.json'),
    'ViT-L/14': os.path.join(MODEL_CONFIGS_DIR, 'ViT-L-14.json'),
    'ViT-H/14': os.path.join(MODEL_CONFIGS_DIR, 'ViT-H-14.json'),
    'ViT-g/14': os.path.join(MODEL_CONFIGS_DIR, 'ViT-g-14.json'),
    'ViT-bigG/14': os.path.join(MODEL_CONFIGS_DIR, 'ViT-bigG-14.json'),
}

# 默认使用的CLIP模型
DEFAULT_CLIP_MODEL = 'ViT-B/16'