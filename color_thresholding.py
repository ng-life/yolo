"""
color_thresholding.py
使用四种方法自动识别背景并圈出前景物体。

方法对比：
  threshold  — 全局 HSV 颜色阈值（最快，适合均匀纯色背景）
  floodfill  — 边缘种子 Flood Fill（推荐，仅删除连通背景，不误删同色前景）
  grabcut    — GrabCut 迭代分割（最慢但最精细，适合纹理/渐变背景）
  edges      — Canny 边缘检测 + 膨胀闭运算（适合边缘清晰的物体；可配合 --obb 输出旋转外接框）

用法:
    uv run color_thresholding.py -i image.jpg -o result.png
    uv run color_thresholding.py -i image.jpg --method floodfill
    uv run color_thresholding.py -i image.jpg --method floodfill --flood-tolerance 25
    uv run color_thresholding.py -i image.jpg --method grabcut --debug
    uv run color_thresholding.py -i image.jpg --method edges --obb
    uv run color_thresholding.py -i /path/to/images --method edges --obb
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def order_box_points(points: np.ndarray) -> np.ndarray:
    """将四个顶点排序为左上、右上、右下、左下。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def extract_primary_contour(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """提取最大外轮廓及其填充掩码，忽略周围小杂物。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    primary_contour = max(contours, key=cv2.contourArea)
    primary_mask = np.zeros_like(mask)
    cv2.drawContours(primary_mask, [primary_contour], -1, 255, thickness=cv2.FILLED)
    return primary_contour, primary_mask


def crop_rotated_rect(image_bgr: np.ndarray, rect: tuple) -> np.ndarray | None:
    """按旋转矩形做透视矫正，返回摆正后的 OBB 区域原图内容。结果保持竖向，高大于宽，并额外旋转 180 度。"""
    box = cv2.boxPoints(rect).astype(np.float32)
    src_pts = order_box_points(box)

    width_a = np.linalg.norm(src_pts[2] - src_pts[3])
    width_b = np.linalg.norm(src_pts[1] - src_pts[0])
    height_a = np.linalg.norm(src_pts[1] - src_pts[2])
    height_b = np.linalg.norm(src_pts[0] - src_pts[3])

    dst_w = max(1, int(round(max(width_a, width_b))))
    dst_h = max(1, int(round(max(height_a, height_b))))

    if dst_w <= 1 or dst_h <= 1:
        return None

    dst_pts = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(image_bgr, matrix, (dst_w, dst_h))

    if result.shape[1] > result.shape[0]:
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    result = cv2.rotate(result, cv2.ROTATE_180)

    return result


def iter_input_images(input_path: Path) -> list[Path]:
    """返回待处理图片列表，支持单图或目录。"""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    sys.exit(f"[错误] 输入路径不存在: {input_path}")


def build_output_path(input_file: Path, output_path: str | Path | None, draw_obb: bool, is_batch: bool) -> Path:
    """生成输出路径。批处理时输出到原目录，文件名为 原文件名_obb.jpg。"""
    if is_batch or output_path is None:
        suffix = "_obb.jpg" if draw_obb else "_result" + input_file.suffix
        return input_file.with_name(input_file.stem + suffix)

    output = Path(output_path)
    if output.is_dir():
        suffix = ".jpg" if draw_obb else input_file.suffix
        name = input_file.stem + ("_obb" if draw_obb else "_result") + suffix
        return output / name

    return output


def save_image(output_path: Path, image: np.ndarray) -> None:
    """保存图像。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)


def process_image(
    input_path: str | Path,
    output_path: str | Path | None,
    method: str = "threshold",
    color_tolerance: int = 30,
    black_threshold: int = 80,
    min_blob_area: int = 500,
    padding: int = 0,
    scale: float = 1.0,
    grabcut_margin: float = 0.05,
    grabcut_iter: int = 5,
    grabcut_max_side: int = 400,
    flood_tolerance: int = 20,
    flood_seed_step: int = 8,
    edges_canny_lo: int = 50,
    edges_canny_hi: int = 150,
    edges_dilate_ksize: int = 15,
    edges_dilate_iter: int = 2,
    edges_blur_ksize: int = 9,
    draw_obb: bool = False,
    debug: bool = False,
) -> None:
    src = Path(input_path)
    files = iter_input_images(src)
    if not files:
        sys.exit(f"[错误] 未在目录中找到可处理图片: {src}")

    is_batch = src.is_dir()
    print(f"[信息] 待处理图片数: {len(files)}")
    for file_path in files:
        dst = build_output_path(file_path, output_path, draw_obb, is_batch)
        remove_background(
            input_path=file_path,
            output_path=dst,
            method=method,
            color_tolerance=color_tolerance,
            black_threshold=black_threshold,
            min_blob_area=min_blob_area,
            padding=padding,
            scale=scale,
            grabcut_margin=grabcut_margin,
            grabcut_iter=grabcut_iter,
            grabcut_max_side=grabcut_max_side,
            flood_tolerance=flood_tolerance,
            flood_seed_step=flood_seed_step,
            edges_canny_lo=edges_canny_lo,
            edges_canny_hi=edges_canny_hi,
            edges_dilate_ksize=edges_dilate_ksize,
            edges_dilate_iter=edges_dilate_iter,
            edges_blur_ksize=edges_blur_ksize,
            draw_obb=draw_obb,
            debug=debug,
        )


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


def build_foreground_mask_grabcut(
    image_bgr: np.ndarray,
    border_margin: float = 0.05,
    iterations: int = 5,
    min_blob_area: int = 500,
    padding: int = 0,
    max_side: int = 400,
) -> np.ndarray:
    """
    使用 GrabCut 算法生成前景掩码。
    适用于纹理/网纹等非纯色背景。
    内部将图像缩至 max_side 再运行 GrabCut，掩码 upscale 回原尺寸，
    大幅降低计算量且不影响检测精度。
    """
    h, w = image_bgr.shape[:2]

    # 内部下采样
    gc_scale = min(1.0, max_side / max(h, w))
    if gc_scale < 1.0:
        gc_w = max(1, int(w * gc_scale))
        gc_h = max(1, int(h * gc_scale))
        gc_img = cv2.resize(image_bgr, (gc_w, gc_h), interpolation=cv2.INTER_AREA)
    else:
        gc_img = image_bgr
        gc_h, gc_w = h, w

    mx = max(1, int(gc_w * border_margin))
    my = max(1, int(gc_h * border_margin))
    rect = (mx, my, gc_w - 2 * mx, gc_h - 2 * my)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((gc_h, gc_w), np.uint8)

    cv2.grabCut(gc_img, gc_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # GC_FGD=1（确定前景）或 GC_PR_FGD=3（可能前景）均视为前景
    small_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # upscale 回原尺寸
    if gc_scale < 1.0:
        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = small_mask

    # 形态学填洞 + 去噪
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

    # Blob 过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            filtered[labels == i] = 255

    if padding > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        filtered = cv2.dilate(filtered, k, iterations=1)

    return filtered[:h, :w]


def build_foreground_mask_floodfill(
    image_bgr: np.ndarray,
    tolerance: int = 20,
    min_blob_area: int = 500,
    padding: int = 0,
    seed_step: int = 8,
) -> np.ndarray:
    """
    从图像四条边的种子点进行 Flood Fill，标记与边缘连通的背景，取反得前景掩码。

    对比全局 HSV 阈值的核心优势：
      - 仅删除与图像边缘「物理连通」的背景像素，不会误删颜色与背景
        相同但属于前景的区域（如白背景下的白色商品标签/反光面）。
      - 容差沿连通路径传播，天然处理轻微渐变 / 光照不均的背景。
      - 无需预先检测背景颜色，速度与阈值法相当。
    """
    h, w = image_bgr.shape[:2]

    # OpenCV 的 floodFill 掩码必须比图像大 2px
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    # 从四条边每隔 seed_step 像素取一个种子
    seeds: list[tuple[int, int]] = []
    for x in range(0, w, seed_step):
        seeds.append((0, x))
        seeds.append((h - 1, x))
    for y in range(0, h, seed_step):
        seeds.append((y, 0))
        seeds.append((y, w - 1))

    lo = (tolerance,) * 3
    hi = (tolerance,) * 3
    # FLOODFILL_MASK_ONLY: 只写掩码不改图像；高 8 位指定掩码填充值 255
    flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    work = image_bgr.copy()  # floodFill 会修改 image 本身，用副本保护原图
    for y, x in seeds:
        if ff_mask[y + 1, x + 1] == 0:  # 跳过已被填充的区域，加速
            cv2.floodFill(work, ff_mask, (x, y), (0, 0, 0), loDiff=lo, upDiff=hi, flags=flags)

    # 去掉 1px 边框，还原原始坐标系；被标记的像素 = 背景
    bg_mask = ff_mask[1:-1, 1:-1]
    mask = cv2.bitwise_not(bg_mask)

    # 形态学：填洞 + 去噪
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

    # Blob 过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            filtered[labels == i] = 255

    if padding > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        filtered = cv2.dilate(filtered, k, iterations=1)

    return filtered[:h, :w]


def build_foreground_mask_edges(
    image_bgr: np.ndarray,
    canny_lo: int = 50,
    canny_hi: int = 150,
    dilate_ksize: int = 15,
    dilate_iter: int = 2,
    blur_ksize: int = 9,
    min_blob_area: int = 500,
    padding: int = 0,
) -> np.ndarray:
    """
    基于 Canny 边缘检测 + 形态学操作生成前景掩码。
    适用于前景物体边缘清晰、背景纹理复杂的场景。

    流程:
      1. 转灰度 + 中值滤波消除背景细纹
      2. Canny 边缘检测提取物体轮廓
      3. 膨胀 + 闭运算将边缘"填实"成实心区域
      4. Blob 过滤保留有效连通域
    """
    h, w = image_bgr.shape[:2]

    # 1. 转灰度 + 中值滤波（核必须为奇数）
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.medianBlur(gray, blur_k)

    # 2. Canny 边缘检测
    edge_map = cv2.Canny(blurred, canny_lo, canny_hi)

    # 3. 膨胀 + 闭运算使商品区域"实心化"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
    mask = cv2.dilate(edge_map, kernel, iterations=dilate_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Blob 过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            filtered[labels == i] = 255

    if padding > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        filtered = cv2.dilate(filtered, k, iterations=1)

    return filtered[:h, :w]


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def remove_background(
    input_path: str | Path,
    output_path: str | Path,
    method: str = "threshold",
    color_tolerance: int = 30,
    black_threshold: int = 80,
    min_blob_area: int = 500,
    padding: int = 0,
    scale: float = 1.0,
    grabcut_margin: float = 0.05,
    grabcut_iter: int = 5,
    grabcut_max_side: int = 400,
    flood_tolerance: int = 20,
    flood_seed_step: int = 8,
    edges_canny_lo: int = 50,
    edges_canny_hi: int = 150,
    edges_dilate_ksize: int = 15,
    edges_dilate_iter: int = 2,
    edges_blur_ksize: int = 9,
    draw_obb: bool = False,
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

    print(f"[信息] 原图尺寸: {orig_w}×{orig_h}，处理尺寸: {proc_img.shape[1]}×{proc_img.shape[0]}")
    print(f"[信息] 分割方法: {method}{'（OBB 旋转框）' if draw_obb else ''}")

    # 1 & 2. 生成前景掩码
    if method == "grabcut":
        mask = build_foreground_mask_grabcut(
            proc_img,
            border_margin=grabcut_margin,
            iterations=grabcut_iter,
            min_blob_area=min_blob_area,
            padding=padding,
            max_side=grabcut_max_side,
        )
        t = _tick("GrabCut 分割", t)
    elif method == "floodfill":
        mask = build_foreground_mask_floodfill(
            proc_img,
            tolerance=flood_tolerance,
            min_blob_area=min_blob_area,
            padding=padding,
            seed_step=flood_seed_step,
        )
        t = _tick("FloodFill 分割", t)
    elif method == "edges":
        mask = build_foreground_mask_edges(
            proc_img,
            canny_lo=edges_canny_lo,
            canny_hi=edges_canny_hi,
            dilate_ksize=edges_dilate_ksize,
            dilate_iter=edges_dilate_iter,
            blur_ksize=edges_blur_ksize,
            min_blob_area=min_blob_area,
            padding=padding,
        )
        t = _tick("边缘检测分割", t)
    else:
        bg_color = detect_background_color(proc_img)
        dark = is_dark_background(bg_color, black_threshold)
        t = _tick("检测背景颜色", t)
        print(f"[信息] 检测到背景颜色 BGR={bg_color.tolist()}，{'暗色背景' if dark else '亮色背景'}")
        mask = build_foreground_mask(
            proc_img,
            bg_color,
            color_tolerance=color_tolerance,
            black_threshold=black_threshold,
            min_blob_area=min_blob_area,
            padding=padding,
        )
        t = _tick("阈值分割", t)

    fg_pixels = int(np.sum(mask > 0))
    total_pixels = mask.size
    print(f"[信息] 前景像素占比: {fg_pixels / total_pixels * 100:.1f}%")

    # 3. 根据前景区域输出结果（坐标映射回原图尺寸）
    result = image_bgr.copy()
    primary_contour, primary_mask = extract_primary_contour(mask)
    if primary_contour is not None and primary_mask is not None:
        if draw_obb:
            # 旋转最小外接矩形，坐标先映射回原图尺寸，再输出摆正裁剪结果
            pts = primary_contour.astype(np.float32)
            if scale != 1.0:
                pts = pts / scale
            rect = cv2.minAreaRect(pts)
            cx, cy = rect[0]
            rw, rh = rect[1]
            angle = rect[2]
            print(f"[信息] 物品区域(OBB): 中心=({cx:.0f},{cy:.0f}), 尺寸={rw:.0f}×{rh:.0f}, 角度={angle:.1f}°")

            cropped = crop_rotated_rect(image_bgr, rect)
            if cropped is None:
                print("[警告] OBB 裁剪失败，退回输出原图")
            else:
                result = cropped
        else:
            # 轴对齐外接矩形（红色）
            x, y, cw, ch = cv2.boundingRect(primary_contour)
            if scale != 1.0:
                inv = 1.0 / scale
                x2 = x + cw
                y2 = y + ch
                # 左上角 floor（往外扩），右下角 ceil（往外扩），确保不漏掉边缘像素
                x  = max(0, int(np.floor(x  * inv)))
                y  = max(0, int(np.floor(y  * inv)))
                x2 = min(orig_w, int(np.ceil(x2 * inv)))
                y2 = min(orig_h, int(np.ceil(y2 * inv)))
                cw = x2 - x
                ch = y2 - y
            cv2.rectangle(result, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            print(f"[信息] 物品区域: x={x}, y={y}, w={cw}, h={ch}")
    else:
        print("[警告] 未检测到前景物品")
    t = _tick("轮廓检测+结果生成", t)

    save_image(dst, result)
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
    p.add_argument("--input",  "-i", required=True,  help="输入图像或目录路径")
    p.add_argument("--output", "-o", help="输出图像路径；省略时自动输出到原目录")
    p.add_argument("--method", choices=["threshold", "grabcut", "floodfill", "edges"], default="threshold",
                   help="分割方法：threshold=颜色阈值（纯色背景）；grabcut=GrabCut（纹理背景）；"
                        "floodfill=边缘种子洪泛（推荐）；edges=Canny 边缘检测（适合边缘清晰的物体）")
    p.add_argument("--flood-tolerance", type=int, default=20,
                   help="floodfill 方法的像素颜色容差（0-60），值越大扩散越激进")
    p.add_argument("--flood-seed-step", type=int, default=8,
                   help="floodfill 方法边缘种子点间距（px），越小越密但更慢")
    p.add_argument("--grabcut-margin", type=float, default=0.05,
                   help="GrabCut 边缘背景先验区域比例（0.0-0.2）")
    p.add_argument("--grabcut-iter", type=int, default=5,
                   help="GrabCut 迭代次数，越多越精确但越慢")
    p.add_argument("--grabcut-max-side", type=int, default=400,
                   help="GrabCut 内部处理图像的最大边长（px），越小越快；建议 200-600")
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
    p.add_argument("--obb", action="store_true",
                   help="使用旋转最小外接矩形（OBB）将物体摆正，只保留商品主体并输出透明 PNG")
    p.add_argument("--canny-lo", type=int, default=50,
                   help="edges 方法：Canny 低阈值（0-255）")
    p.add_argument("--canny-hi", type=int, default=150,
                   help="edges 方法：Canny 高阈值（0-255），通常为低阈值的 2-3 倍")
    p.add_argument("--dilate-ksize", type=int, default=15,
                   help="edges 方法：膨胀/闭运算核大小（px），越大越能填实内部空洞")
    p.add_argument("--dilate-iter", type=int, default=2,
                   help="edges 方法：膨胀迭代次数")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_image(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        color_tolerance=args.color_tolerance,
        black_threshold=args.black_threshold,
        min_blob_area=args.min_blob_area,
        padding=args.padding,
        scale=args.scale,
        grabcut_margin=args.grabcut_margin,
        grabcut_iter=args.grabcut_iter,
        grabcut_max_side=args.grabcut_max_side,
        flood_tolerance=args.flood_tolerance,
        flood_seed_step=args.flood_seed_step,
        edges_canny_lo=args.canny_lo,
        edges_canny_hi=args.canny_hi,
        edges_dilate_ksize=args.dilate_ksize,
        edges_dilate_iter=args.dilate_iter,
        draw_obb=args.obb,
        debug=args.debug,
    )
