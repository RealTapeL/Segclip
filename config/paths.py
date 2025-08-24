import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FastSAM路径
FASTSAM_ROOT = os.path.join(PROJECT_ROOT, '..', 'FastSAM')

# 模型配置路径
MODEL_CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'model_configs')

# 示例图片路径
SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'asserts', 'samples')

# 输出路径
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'asserts', 'outputs')