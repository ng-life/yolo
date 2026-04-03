# YOLO × SKU-110K — 从零训练到 CoreML 部署

> 使用 **Ultralytics YOLO** 在 **SKU-110K** 零售货架数据集上训练目标检测模型，
> 并将最优权重导出为 **CoreML (.mlpackage)** 以在 Apple 设备上部署。

---

## 目录

1. [项目结构](#项目结构)
2. [环境要求](#环境要求)
3. [安装](#安装)
4. [数据集说明](#数据集说明)
5. [训练](#训练)
6. [导出 CoreML](#导出-coreml)
7. [推理](#推理)
8. [配置文件说明](#配置文件说明)
9. [常见问题](#常见问题)
10. [引用](#引用)

---

## 项目结构

```
yolo/
├── requirements.txt          # Python 依赖
├── train.py                  # 训练脚本
├── export.py                 # CoreML 导出脚本
├── predict.py                # 推理脚本
├── config/
│   └── train_config.yaml     # 训练超参数 & 数据集配置
└── runs/                     # 训练 / 预测结果（自动生成）
    └── sku110k/
        └── train/
            └── weights/
                ├── best.pt   # 最优检查点
                └── last.pt   # 最后检查点
```

---

## 环境要求

| 项目 | 版本 |
|------|------|
| macOS | 12 Monterey 或更高（CoreML 导出 & 推理） |
| Python | 3.10 + |
| Xcode Command Line Tools | 已安装（`xcode-select --install`） |
| 磁盘空间 | ≥ 20 GB（数据集 13.6 GB + 模型 & 结果） |

> **GPU 加速**：Apple Silicon Mac 自动使用 MPS 后端；英伟达 GPU 需要 CUDA 环境。

---

## 安装

```bash
# 1. 克隆 / 进入项目目录
cd /path/to/yolo

# 2. 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 数据集说明

**SKU-110K** 由 Trax Retail 创建，收录全球零售货架的密集堆叠商品图像。

| 子集 | 图像数 | 用途 |
|------|--------|------|
| train | 8,219 | 训练 |
| val | 588 | 验证 |
| test | 2,936 | 最终评估 |

- **类别数**：1（单类 `object`，即"商品"）
- **原始标注**：CSV 格式（x1, y1, x2, y2 绝对坐标）
- **Ultralytics 自动处理**：首次训练时自动下载（约 13.6 GB）并将标注转换为 YOLO 格式

> 数据集下载脚本内嵌于 Ultralytics 的
> [`SKU-110K.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml)，
> 无需手动下载。

---

## 训练

### 快速开始

```bash
python train.py
```

等价于使用 `config/train_config.yaml` 中的所有默认值：
- 模型：`yolo11n.pt`（最轻量，适合快速验证）
- 数据集：`SKU-110K.yaml`（自动下载）
- 训练轮次：100 epochs
- 图像尺寸：640 × 640

### 使用更大的模型

```bash
# yolo11s → yolo11m → yolo11l → yolo11x（精度逐步提升）
python train.py --model yolo11s.pt --epochs 150 --batch 8
```

### 在 Apple Silicon 上训练（MPS）

```bash
python train.py --device mps --batch 16
```

### 恢复中断的训练

```bash
python train.py --resume runs/sku110k/train/weights/last.pt
```

### 训练结果

训练完成后，结果保存在 `runs/sku110k/train/`：

```
runs/sku110k/train/
├── weights/
│   ├── best.pt          ← 最优模型权重
│   └── last.pt          ← 最后 epoch 权重
├── results.csv          ← 每 epoch 的 mAP / Loss 曲线数据
├── results.png          ← 训练曲线图
├── confusion_matrix.png
└── val_batch*.jpg       ← 验证集预测可视化
```

---

## 导出 CoreML

### 标准导出（FP32）

```bash
python export.py --weights runs/sku110k/train/weights/best.pt
```

生成 `best.mlpackage`（与 `.pt` 文件同目录）。

### FP16 量化（推荐，体积减半，速度翻倍）

```bash
python export.py --weights best.pt --half
```

### INT8 量化（最小体积，适合存储空间受限场景）

```bash
python export.py --weights best.pt --int8
```

### 内嵌 NMS（推荐用于 iOS/macOS App 部署）

将 NMS（非极大值抑制）嵌入模型图，App 代码中无需二次后处理：

```bash
python export.py --weights best.pt --nms --half
```

### 导出参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | — | **必填**，.pt 权重文件路径 |
| `--imgsz` | 640 | 输入图像尺寸（与训练保持一致） |
| `--half` | False | FP16 量化 |
| `--int8` | False | INT8 量化 |
| `--nms` | False | 内嵌 NMS |
| `--batch` | 1 | 导出批大小 |
| `--device` | cpu | 导出设备 |

---

## 推理

### 使用 PyTorch 权重推理

```bash
python predict.py --model best.pt --source /path/to/image.jpg
```

### 使用 CoreML 模型推理（macOS）

```bash
python predict.py --model best.mlpackage --source /path/to/shelf_image.jpg --save
```

### 批量处理目录

```bash
python predict.py --model best.pt --source ./test_images/ --save --conf 0.3
```

### 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | — | **必填**，.pt 或 .mlpackage |
| `--source` | — | **必填**，图像 / 目录 / 视频 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.45 | NMS IoU 阈值 |
| `--max-det` | 1000 | 每图最大检测数（货架图建议 ≥500） |
| `--save` | False | 保存标注结果图 |
| `--show` | False | 实时预览窗口 |

---

## 配置文件说明

`config/train_config.yaml` 涵盖全部 Ultralytics 训练参数，关键项如下：

```yaml
model: yolo11n.pt      # 预训练权重；换成 yolo11s/m/l/x 获得更高精度
data: SKU-110K.yaml    # 数据集（内置，自动下载）
epochs: 100
batch: 16              # -1 = 自动批大小
imgsz: 640
device: ""             # "" = 自动检测 GPU → MPS → CPU
```

所有参数均可通过命令行 `--` 前缀覆盖，例如：

```bash
python train.py --epochs 200 --imgsz 1280 --device 0
```

---

## 常见问题

**Q: 下载数据集速度很慢？**  
A: SKU-110K 原始文件托管在 AWS S3，约 13.6 GB。建议使用稳定网络或配置代理：
```bash
export https_proxy=http://your-proxy:port
python train.py
```

**Q: Apple Silicon 上 MPS 训练报错？**  
A: 确保 PyTorch ≥ 2.3 并将 `batch` 设置为 2 的幂次（2, 4, 8, 16…）。

**Q: CoreML 导出报 `coremltools` 未找到？**  
A: 运行 `pip install coremltools>=8.0`；在 Linux 上需要额外安装 `protobuf`。

**Q: 如何在 iOS App 中使用 .mlpackage？**  
A: 将 `.mlpackage` 拖入 Xcode 项目中，Xcode 会自动生成 Swift 接口。
参考 [Apple 官方文档](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)。

**Q: 验证集 mAP 不高怎么办？**  
A: 尝试：① 使用更大的模型（`yolo11s/m`）；② 增大 `imgsz`（如 1280）；
③ 调整增强参数（关闭不适用于货架场景的 `degrees`）；④ 增加 `epochs`。

---

## 引用

如使用 SKU-110K 数据集，请在论文中引用：

```bibtex
@inproceedings{goldman2019dense,
  author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
  title     = {Precise Detection in Densely Packed Scenes},
  booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
  year      = {2019}
}
```

---

## License

本项目代码使用 **MIT License**。
Ultralytics YOLO 使用 **AGPL-3.0 License**，训练所产生的模型权重同样受此约束。
如需商业用途，请参阅 [Ultralytics 商业授权](https://ultralytics.com/license)。
