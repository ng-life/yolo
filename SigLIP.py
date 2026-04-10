import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F

# 1. 检查设备：Mac mini M4 建议使用 mps
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# 2. 加载模型和处理器
# 推荐使用 so400m 版本，兼顾精度与速度
model_id = "google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

def get_image_embedding(image_path):
    # 加载图片并预处理
    image = Image.open(image_path).convert("RGB")
    
    # 转化为模型输入格式
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # 某些 SigLIP 变体会返回 BaseModelOutputWithPooling
        if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
            image_features = image_features.pooler_output
        elif hasattr(image_features, "last_hidden_state") and image_features.last_hidden_state is not None:
            image_features = image_features.last_hidden_state.mean(dim=1)
        # 归一化：确保后续点积可直接作为余弦相似度
        image_embedding = F.normalize(image_features, p=2, dim=-1)
        
    return image_embedding.cpu().numpy()

# 示例：提取两张图片的向量并对比相似度
img1_vec = get_image_embedding("/Users/ng-life/Downloads/20260408_111243.jpg")
img2_vec = get_image_embedding("/Users/ng-life/Downloads/20260408_111243 (1).jpg")

# 计算余弦相似度 (简单点积即可，因为已经归一化了)
similarity = (img1_vec @ img2_vec.T).item()
print(f"图片相似度得分: {similarity:.4f}")