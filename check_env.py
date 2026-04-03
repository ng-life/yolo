#!/usr/bin/env python3
"""Quick sanity-check for the SKU-110K YOLO project."""
import sys, pathlib, ast, importlib

ROOT = pathlib.Path(__file__).parent.resolve()


def check(label, ok, detail=""):
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


errors = []

# 1. YAML config
import yaml
with open(ROOT / "config" / "train_config.yaml") as f:
    cfg = yaml.safe_load(f)

if not check("YAML model field", isinstance(cfg.get("model"), str), repr(cfg.get("model"))):
    errors.append("YAML model field broken")
check("YAML data field",   cfg.get("data") == "SKU-110K.yaml", cfg.get("data"))
check("YAML epochs field", isinstance(cfg.get("epochs"), int),  str(cfg.get("epochs")))

# 2. Python version
v = sys.version_info
ok = (v.major, v.minor) >= (3, 10)
check(f"Python >= 3.10", ok, f"{v.major}.{v.minor}.{v.micro}")
if not ok:
    errors.append("Python too old")

# 3. Dependencies
deps = {
    "ultralytics": "ultralytics",
    "torch":       "torch",
    "coremltools": "coremltools",
    "onnx":        "onnx",
    "numpy":       "numpy",
    "polars":      "polars",
}
for mod, pkg in deps.items():
    try:
        m = importlib.import_module(mod)
        check(pkg, True, getattr(m, "__version__", "?"))
    except ImportError:
        check(pkg, False, "not installed")
        errors.append(f"missing: {pkg}")

# 4. Hardware
import torch
if torch.backends.mps.is_available():
    check("Apple MPS", True, "Apple Silicon")
elif torch.cuda.is_available():
    check("CUDA", True, f"{torch.version.cuda} / {torch.cuda.get_device_name(0)}")
else:
    check("Accelerator", True, "CPU only (training will be slow)")

# 5. Script syntax
for script in ["train.py", "export.py", "predict.py"]:
    src = (ROOT / script).read_text()
    try:
        ast.parse(src)
        check(f"{script} syntax", True)
    except SyntaxError as e:
        check(f"{script} syntax", False, str(e))
        errors.append(f"syntax error in {script}")

# Summary
print()
if errors:
    print("ISSUES FOUND:")
    for e in errors:
        print(f"  - {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("All checks passed. Ready to train:")
    print("  python train.py")
