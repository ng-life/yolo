#!/usr/bin/env python3
"""
export.py — Export a trained YOLO model to CoreML (.mlpackage) format.

Ultralytics uses coremltools under the hood.  This script adds:
  • Sensible defaults for SKU-110K (single class 'object')
  • Optional INT8 / FP16 quantisation
  • Basic sanity check after export

Requirements
------------
  macOS (or x86 Linux) is required for CoreML export.
  pip install coremltools>=8.0 onnx onnxruntime

Usage
-----
  # Export best weights from a training run:
  python export.py --weights runs/sku110k/train/weights/best.pt

  # FP16 quantised model (smaller, ~2× faster on Apple Neural Engine):
  python export.py --weights best.pt --half

  # INT8 quantisation (smallest, fastest — slight accuracy trade-off):
  python export.py --weights best.pt --int8

  # Include NMS in the CoreML graph (recommended for on-device inference):
  python export.py --weights best.pt --nms

  # Custom image size:
  python export.py --weights best.pt --imgsz 1280
"""

import argparse
import platform
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------
_OS = platform.system()
if _OS not in ("Darwin", "Linux"):
    sys.exit(
        "[ERROR] CoreML export is only supported on macOS or Linux.\n"
        f"  Detected OS: {_OS}"
    )

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
        description="Export a trained YOLO model to CoreML format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained .pt weights file (e.g. runs/sku110k/train/weights/best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square). Use the same value as during training.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="FP16 quantisation — smaller model, faster on Apple Silicon.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=False,
        help="INT8 quantisation — smallest model size, fastest inference.",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        default=False,
        help=(
            "Embed Non-Maximum Suppression in the exported graph. "
            "Recommended for on-device deployment (skips post-processing in app code)."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for the exported model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Export device: "cpu" or "mps" (Apple Silicon).',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to same directory as --weights. "
            "The .mlpackage bundle is always placed next to the .pt file by Ultralytics."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    weights_path = args.weights.resolve()
    if not weights_path.exists():
        sys.exit(f"[ERROR] Weights file not found: {weights_path}")
    if weights_path.suffix != ".pt":
        sys.exit(f"[ERROR] Expected a .pt file, got: {weights_path}")

    if args.half and args.int8:
        print("[WARN] Both --half and --int8 specified; --int8 takes precedence.")
        args.half = False

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading weights: {weights_path}")
    model = YOLO(weights_path)

    # ── Build export kwargs ───────────────────────────────────────────────────
    export_kwargs: dict = {
        "format": "coreml",
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "nms": args.nms,
        "batch": args.batch,
        "device": args.device,
    }

    print("[INFO] Export settings:")
    for k, v in export_kwargs.items():
        print(f"         {k}: {v}")
    print()

    # ── Export ────────────────────────────────────────────────────────────────
    exported_path = model.export(**export_kwargs)

    # ── Result ────────────────────────────────────────────────────────────────
    exported = Path(str(exported_path))

    print("\n" + "=" * 60)
    if exported.exists():
        size_mb = sum(f.stat().st_size for f in exported.rglob("*") if f.is_file()) / 1e6
        print("[DONE] CoreML export successful.")
        print(f"  Output : {exported}")
        print(f"  Size   : {size_mb:.1f} MB")
        print()
        print("[NEXT] Test the exported model:")
        print(f"  python predict.py --model {exported} --source path/to/image.jpg")
    else:
        print("[WARN] Export finished but .mlpackage not found at expected path.")
        print(f"       Ultralytics reported: {exported_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
