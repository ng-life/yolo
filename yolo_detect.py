from ultralytics import YOLO
import json

# 加载 YOLO11 nano 分类预训练权重
model = YOLO("yolo26m.pt") 

print("加载完成")

# 在 M4 Mac 上直接推理
results = model.predict("./images/sku_images/铁观音.png", device="cpu", conf=0.1) # 使用 MPS 加速
results[0].show()