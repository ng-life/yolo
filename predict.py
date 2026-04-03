#!/usr/bin/env python3
"""
predict.py — Run inference with a trained YOLO or exported CoreML model.

Supports:
  • .pt  — PyTorch weights (any device)
  • .mlpackage — CoreML model (macOS / Apple Silicon only)

Usage
-----
  # PyTorch weights:
  python predict.py --model runs/sku110k/train/weights/best.pt --source image.jpg

  # CoreML model:
  python predict.py --model best.mlpackage --source image.jpg

  # Run on a directory of images and save annotated results:
  python predict.py --model best.pt --source ./test_images/ --save

  # Adjust confidence / IoU thresholds:
  python predict.py --model best.pt --source image.jpg --conf 0.4 --iou 0.5

  # Show live preview (requires a display):
  python predict.py --model best.pt --source image.jpg --show
"""

import argparse
import platform
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "[ERROR] ultralytics not found.\n"
        "  Run:  pip install -r requirements.txt"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on images / video / directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to .pt weights OR .mlpackage CoreML bundle.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Inference source: image path, video path, directory, "
            "webcam index (0), or URL."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0–1).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (0–1).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Maximum detections per image. SKU-110K can have 200+ items per shelf.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            'Inference device: "" (auto), "cpu", "0" (GPU), "mps" (Apple Silicon). '
            "CoreML models ignore this and always run on-device."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save annotated result images/videos.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=False,
        help="Save results as YOLO-format .txt label files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show live preview window (requires display).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="Output directory for saved results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sku110k",
        help="Experiment name (sub-folder of --project).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Bounding-box line width in pixels (1–2 looks best for dense SKU images).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found: {model_path}")

    # CoreML models must run on macOS
    is_coreml = model_path.suffix == ".mlpackage"
    if is_coreml and platform.system() != "Darwin":
        sys.exit(
            "[ERROR] CoreML (.mlpackage) inference is only supported on macOS.\n"
            f"  Detected OS: {platform.system()}"
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model : {model_path}")
    model = YOLO(model_path)

    # ── Inference kwargs ──────────────────────────────────────────────────────
    predict_kwargs: dict = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "save": args.save,
        "save_txt": args.save_txt,
        "show": args.show,
        "project": args.project,
        "name": args.name,
        "line_width": args.line_width,
        "verbose": True,
    }
    if args.device is not None and not is_coreml:
        predict_kwargs["device"] = args.device

    print(f"[INFO] Source        : {args.source}")
    print(f"[INFO] Confidence    : {args.conf}")
    print(f"[INFO] IoU threshold : {args.iou}")
    print(f"[INFO] Max detections: {args.max_det}")
    print()

    # ── Run inference ─────────────────────────────────────────────────────────
    results = model.predict(**predict_kwargs)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_dets = sum(len(r.boxes) for r in results)
    print("\n" + "=" * 60)
    print(f"[DONE] Processed {len(results)} frame(s), {total_dets} total detections.")
    if args.save:
        import os
        out_dir = Path(args.project) / args.name
        print(f"  Saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
