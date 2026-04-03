#!/usr/bin/env python3
"""
train.py — Train a YOLO model on the SKU-110K retail-shelf dataset.

Usage
-----
# Default (reads config/train_config.yaml):
    python train.py

# Override any argument at the command line:
    python train.py --model yolo11s.pt --epochs 50 --batch 8 --device mps

# Resume an interrupted run:
    python train.py --resume runs/sku110k/train/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is importable when run from any working dir.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    import yaml
    from ultralytics import YOLO
except ImportError as exc:
    sys.exit(
        f"[ERROR] Missing dependency: {exc}\n"
        "  Run:  pip install -r requirements.txt"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML training configuration."""
    with open(config_path, encoding="utf-8") as fh:
        # Strip YAML comments — PyYAML handles this natively
        cfg = yaml.safe_load(fh)
    return cfg or {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO on SKU-110K dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "train_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model weights (e.g. yolo11n.pt) or YAML architecture.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML (default: SKU-110K.yaml from Ultralytics built-ins).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (-1 = auto-batch).",
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device: "" (auto), "0" (GPU), "cpu", "mps" (Apple Silicon).',
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt checkpoint to resume training.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Output project directory.",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name.")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Load base config from YAML ────────────────────────────────────────────
    cfg = load_config(args.config) if args.config.exists() else {}

    # ── CLI overrides win over YAML values ────────────────────────────────────
    overrides: dict = {}
    cli_map = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
    }
    for key, val in cli_map.items():
        if val is not None:
            overrides[key] = val

    # Merge: YAML base → CLI overrides
    train_kwargs = {**cfg, **overrides}

    # Pull out the model weight path before passing the rest to model.train()
    model_weights = train_kwargs.pop("model", "yolo11n.pt")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        model = YOLO(args.resume)
        # When resuming, Ultralytics re-uses all original training settings.
        results = model.train(resume=True)
    else:
        print(f"[INFO] Loading model: {model_weights}")
        model = YOLO(model_weights)

        # Ensure dataset defaults to SKU-110K if not specified
        if "data" not in train_kwargs:
            train_kwargs["data"] = "SKU-110K.yaml"

        print("[INFO] Starting training with args:")
        for k, v in train_kwargs.items():
            print(f"         {k}: {v}")
        print()

        results = model.train(**train_kwargs)

    # ── Report results ────────────────────────────────────────────────────────
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("[DONE] Training complete.")
    print(f"  Results saved to : {save_dir}")
    print(f"  Best weights     : {best_weights}")
    if best_weights.exists():
        print("\n[NEXT] Export to CoreML:")
        print(f"  python export.py --weights {best_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
