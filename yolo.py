from ultralytics import YOLOWorld
import json

# 加载 YOLO11 nano 分类预训练权重
model = YOLOWorld("yolov8x-worldv2") 

print("加载完成")

model.set_classes([   
    "packaged product",
    "tea box",
    "product package",
    "retail item", "product",
    "box", "jar", "bag", "packet", "bottle",
    "plastic bag",
    "poly bag",
    "bag",
    "packaging bag"])
# 在 M4 Mac 上直接推理
results = model.predict("./images/sku_images/铁观音.png", device="mps", conf=0.05) # 使用 MPS 加速
results[0].show()