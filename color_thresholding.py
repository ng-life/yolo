"""
color_thresholding.py
使用轮廓检测与 Blob 分析，自动识别纯色背景并抠出前景物体。

用法:
    uv run color_thresholding.py --input image.jpg --output result.png
    uv run color_thresholding.py --input image.jpg --output result.png --black-threshold 80
    uv run color_thresholding.py --input image.jpg --output result.png --padding 20 --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 背景颜色自动识别
# ──────────────────────────────────────────────────────────────────────────────

def detect_background_color(image_bgr: np.ndarray, border_width: int = 10) -> np.ndarray:
    """
    通过采样图像四条边缘像素，取中位数作为背景颜色估计值。
    返回形状为 (3,) 的 BGR 数组。
    """
    h, w = image_bgr.shape[:2]
    bw = min(border_width, h // 4, w // 4)

    top    = image_bgr[:bw, :].reshape(-1, 3)
    bottom = image_bgr[h - bw:, :].reshape(-1, 3)
    left   = image_bgr[:, :bw].reshape(-1, 3)
    right  = image_bgr[:, w - bw:].reshape(-1, 3)

    border_pixels = np.concatenate([top, bottom, left, right], axis=0)
    bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
    return bg_color


def is_dark_background(bg_color_bgr: np.ndarray, threshold: int = 80) -> bool:
    """判断背景是否为暗色（亮度低于阈值）。"""
    gray_val = int(bg_color_bgr[0]) * 0.114 + int(bg_color_bgr[1]) * 0.587 + int(bg_color_bgr[2]) * 0.299
    return gray_val < threshold


# ──────────────────────────────────────────────────────────────────────────────
# 前景掩码生成
# ──────────────────────────────────────────────────────────────────────────────

def build_foreground_mask(
    image_bgr: np.ndarray,
    bg_color: np.ndarray,
    color_tolerance: int = 30,
    black_threshold: int = 80,
    min_blob_area: int = 500,
    padding: int = 0,
) -> np.ndarray:
    """
    返回前景掩码（uint8，255=前景，0=背景）。

    策略：
      1. 若背景为暗色（接近纯黑）：用亮度阈值分离前景。
      2. 否则：在 HSV 空间对背景颜色做范围阈值，再取反得前景。
      3. 形态学去噪 + 保留最大 Blob（或面积超过 min_blob_area 的所有 Blob）。
    """
    h, w = image_bgr.shape[:2]
    dark_bg = is_dark_background(bg_color, black_threshold)

    if dark_bg:
        # 背景暗色：用灰度亮度阈值
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)
    else:
        # 亮色背景：HSV 范围阈值
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        bg_hsv = cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]

        lo = np.clip(bg_hsv.astype(int) - [color_tolerance, 60, 60], 0, 255).astype(np.uint8)
        hi = np.clip(bg_hsv.astype(int) + [color_tolerance, 60, 60], 0, 255).astype(np.uint8)

        # 处理色调环绕（红色区域跨越 0/180）
        if lo[0] > hi[0]:
            m1 = cv2.inRange(image_hsv, np.array([0, lo[1], lo[2]]), hi)
            m2 = cv2.inRange(image_hsv, lo, np.array([180, hi[1], hi[2]]))
            bg_mask = cv2.bitwise_or(m1, m2)
        else:
            bg_mask = cv2.inRange(image_hsv, lo, hi)

        mask = cv2.bitwise_not(bg_mask)

    # ── 形态学处理：填洞 + 去噪 ──────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

    # ── Blob 过滤：保留面积 ≥ min_blob_area 的连通区域 ────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_blob_area:
            filtered[labels == i] = 255

    # ── 可选 padding（膨胀）──────────────────────────────────────────────
    if padding > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        filtered = cv2.dilate(filtered, k, iterations=1)

    # 裁回原图尺寸（dilate 不会超界，但保险起见）
    filtered = filtered[:h, :w]
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def remove_background(
    input_path: str | Path,
    output_path: str | Path,
    color_tolerance: int = 30,
    black_threshold: int = 80,
    min_blob_area: int = 500,
    padding: int = 0,
    scale: float = 1.0,
    debug: bool = False,
) -> None:
    import time

    def _tick(label: str, t_prev: float) -> float:
        t_now = time.perf_counter()
        print(f"  [{label}] {(t_now - t_prev) * 1000:.1f}ms")
        return t_now

    t0 = t = time.perf_counter()
    print("[耗时统计]")

    src = Path(input_path)
    dst = Path(output_path)

    image_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image_bgr is None:
        sys.exit(f"[错误] 无法读取图像: {src}")
    t = _tick("读取图像", t)

    # 缩放处理（缩放后用于掩码计算，最终绘框映射回原图）
    orig_h, orig_w = image_bgr.shape[:2]
    if scale != 1.0:
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        proc_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        t = _tick(f"缩放图像 {orig_w}×{orig_h}→{new_w}×{new_h}", t)
    else:
        proc_img = image_bgr

    # 1. 自动检测背景颜色
    bg_color = detect_background_color(proc_img)
    dark = is_dark_background(bg_color, black_threshold)
    t = _tick("检测背景颜色", t)
    print(f"[信息] 原图尺寸: {orig_w}×{orig_h}，处理尺寸: {proc_img.shape[1]}×{proc_img.shape[0]}")
    print(f"[信息] 检测到背景颜色 BGR={bg_color.tolist()}，{'暗色背景' if dark else '亮色背景'}")

    # 2. 生成前景掩码
    mask = build_foreground_mask(
        proc_img,
        bg_color,
        color_tolerance=color_tolerance,
        black_threshold=black_threshold,
        min_blob_area=min_blob_area,
        padding=padding,
    )
    t = _tick("生成前景掩码", t)

    fg_pixels = int(np.sum(mask > 0))
    total_pixels = mask.size
    print(f"[信息] 前景像素占比: {fg_pixels / total_pixels * 100:.1f}%")

    # 3. 在原图上用红色矩形圈出物品区域（坐标映射回原图尺寸）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image_bgr.copy()
    if contours:
        all_points = np.concatenate(contours, axis=0)
        x, y, cw, ch = cv2.boundingRect(all_points)
        if scale != 1.0:
            inv = 1.0 / scale
            x, y, cw, ch = int(x * inv), int(y * inv), int(cw * inv), int(ch * inv)
        cv2.rectangle(result, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
        print(f"[信息] 物品区域: x={x}, y={y}, w={cw}, h={ch}")
    else:
        print("[警告] 未检测到前景物品")
    t = _tick("轮廓检测+绘框", t)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), result)
    t = _tick("保存图像", t)

    total = time.perf_counter() - t0
    print(f"[完成] 已保存至: {dst}（总耗时 {total * 1000:.1f}ms）")

    # 4. Debug 模式：额外保存掩码图
    if debug:
        debug_dir = dst.parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        mask_path = debug_dir / (dst.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)
        print(f"[调试] 掩码已保存: {mask_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="自动识别纯色背景并抠出前景（轮廓检测 + Blob 分析）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,  help="输入图像路径")
    p.add_argument("--output", "-o", required=True,  help="输出图像路径（原图 + 红色物品检测框）")
    p.add_argument("--color-tolerance", type=int, default=30,
                   help="背景颜色容差（HSV 色调偏差，0-90）")
    p.add_argument("--black-threshold",  type=int, default=80,
                   help="判定暗色背景的亮度阈值（0-255），同时用于暗色背景的前景分离")
    p.add_argument("--min-blob-area",  type=int, default=500,
                   help="保留的最小 Blob 面积（像素²）")
    p.add_argument("--padding", type=int, default=0,
                   help="前景掩码向外膨胀的像素数")
    p.add_argument("--scale", type=float, default=1.0,
                   help="处理前缩放比例（0.0-1.0），越小越快，检测精度略降；结果绘在原图上")
    p.add_argument("--debug", action="store_true",
                   help="保存掩码和轮廓可视化图（输出在 <output_dir>/debug/）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    remove_background(
        input_path=args.input,
        output_path=args.output,
        color_tolerance=args.color_tolerance,
        black_threshold=args.black_threshold,
        min_blob_area=args.min_blob_area,
        padding=args.padding,
        scale=args.scale,
        debug=args.debug,
    )
