# 项目指南 — YOLO × SKU-110K

基于 Ultralytics YOLO 在 SKU-110K 零售货架数据集上训练目标检测模型，并导出为 CoreML 供 Apple 设备部署。

## 架构

| 脚本 | 职责 |
|------|------|
| `train.py` | 训练入口，读取 `config/train_config.yaml`，支持 CLI 覆盖参数 |
| `export.py` | 将 `.pt` 权重导出为 CoreML `.mlpackage` |
| `predict.py` | 推理，支持 `.pt` 和 `.mlpackage` 两种格式 |
| `check_env.py` | 验证环境（Python 版本、MPS/CUDA/CPU 可用性、核心库） |
| `config/train_config.yaml` | 所有训练超参数（模型、优化器、增强、硬件）的单一来源 |

训练结果自动保存至 `runs/sku110k/train/`，最优权重为 `weights/best.pt`，最后检查点为 `weights/last.pt`。

## 构建与运行

```bash
# 安装 / 同步依赖（使用 uv，需 Python ≥ 3.11）
uv sync

# 训练（使用 config/train_config.yaml 中的默认配置）
uv run python train.py

# 覆盖参数
uv run python train.py --model yolo11s.pt --epochs 50 --batch 8 --device mps

# 恢复中断的训练
uv run python train.py --resume runs/sku110k/train/weights/last.pt

# 导出 CoreML（仅 macOS）
uv run python export.py --weights runs/sku110k/train/weights/best.pt

# 推理
uv run python predict.py --model runs/sku110k/train/weights/best.pt --source image.jpg

# 检查环境
uv run python check_env.py
```

## 依赖管理

- **使用 `uv`**，不使用 `pip` 直接安装。
- 依赖声明在 `pyproject.toml`；锁文件为 `uv.lock`，需提交到版本控制。
- 添加依赖：`uv add <package>`；移除依赖：`uv remove <package>`。
- `requirements.txt` 仅作历史参考，**不再维护**。

## 约定

- **超参数只改 `config/train_config.yaml`**，不硬编码在脚本内。
- 设备自动检测（`device: ""`）：优先 GPU → MPS → CPU，无需手动指定。
- 数据增强参数针对货架场景已调优：`flipud: 0.0`（货架不倒置）、`degrees: 0.0`（不旋转）。
- `save_period: 10` — 每 10 轮保存一次检查点，避免训练中断丢失进度。
- CoreML 导出仅在 macOS 上支持；Linux 下只做训练和 ONNX 导出。

## 常见陷阱

- SKU-110K 数据集首次运行时自动下载（**约 13.6 GB**），确保磁盘空间充足。
- `onnxruntime` 最新版不支持 Python 3.10，项目已锁定 `requires-python = ">=3.11"`。
- Apple Silicon 下使用 MPS 加速，显存不足时将 `batch` 调小（如 8）。
- `exist_ok: false` 时重复实验名会报错——用 `--name` 指定不同实验名或改为 `true`。
