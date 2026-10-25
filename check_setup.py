"""
check_setup.py
==============
Run this FIRST, before anything else, to confirm your environment is ready.

    python check_setup.py

It checks the Python version, every required package, that the project files are
present, and (if you have trained already) that the saved model loads correctly.
Nothing is installed or changed by this script -- it only reports.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys

OK = "[ OK ]"
BAD = "[FAIL]"
WARN = "[WARN]"

problems: list[str] = []


def line(status: str, message: str) -> None:
    print(f"{status} {message}")


print("=" * 66)
print("PROJECT SETUP CHECK — AI-Based Arm Deflection Prediction")
print("=" * 66)
print(f"Platform        : {platform.system()} {platform.release()}")
print(f"Python version  : {sys.version.split()[0]}")
print(f"Python location : {sys.executable}")
print()

# ------------------------------------------------------------------ 1. Python
if sys.version_info >= (3, 10):
    line(OK, f"Python {sys.version_info.major}.{sys.version_info.minor} "
             f"(3.10 or newer required)")
else:
    line(BAD, f"Python {sys.version_info.major}.{sys.version_info.minor} is too old. "
              f"Install Python 3.10 or newer.")
    problems.append("Python version too old")

# Detect the classic mistake of using the system Python instead of the venv
in_venv = (hasattr(sys, "real_prefix")
           or sys.prefix != getattr(sys, "base_prefix", sys.prefix))
if in_venv:
    line(OK, "Running inside a virtual environment")
else:
    line(WARN, "Not inside a virtual environment. It will still work, but using a "
               "venv is strongly recommended (see README).")

print()
print("-- Packages " + "-" * 54)

REQUIRED = {
    "numpy": "1.26",
    "pandas": "2.0",
    "sklearn": "1.3",
    "matplotlib": "3.8",
    "streamlit": "1.30",
    "joblib": "1.3",
}
for module, minimum in REQUIRED.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        line(OK, f"{module:<12} {version}")
    except ImportError:
        line(BAD, f"{module:<12} NOT INSTALLED (need >= {minimum})")
        problems.append(f"{module} missing")

print()
print("-- Project files " + "-" * 49)

EXPECTED = {
    "physics.py": "beam equations and validation",
    "model_classes.py": "model classes needed for saving/loading",
    "generate_dataset.py": "creates the dataset",
    "train_model.py": "trains and evaluates the models",
    "app.py": "Streamlit interface",
    "requirements.txt": "dependency list",
    "README.md": "instructions",
}
here = os.path.dirname(os.path.abspath(__file__))
for name, purpose in EXPECTED.items():
    path = os.path.join(here, name)
    if os.path.exists(path):
        line(OK, f"{name:<22} ({purpose})")
    else:
        line(BAD, f"{name:<22} MISSING — {purpose}")
        problems.append(f"{name} missing")

print()
print("-- Generated artefacts " + "-" * 44)

DATA = os.path.join(here, "data", "beam_deflection_dataset.csv")
MODEL = os.path.join(here, "models", "model.joblib")
METRICS = os.path.join(here, "models", "metrics.json")

if os.path.exists(DATA):
    size_mb = os.path.getsize(DATA) / 1e6
    line(OK, f"data/beam_deflection_dataset.csv ({size_mb:.1f} MB)")
else:
    line(WARN, "data/beam_deflection_dataset.csv not found "
               "-> run:  python generate_dataset.py")
    problems.append("dataset not generated yet")

if os.path.exists(MODEL) and os.path.exists(METRICS):
    size_mb = os.path.getsize(MODEL) / 1e6
    line(OK, f"models/model.joblib ({size_mb:.1f} MB)")
    # actually try to load it: this catches version/pickle problems
    try:
        import joblib
        bundle = joblib.load(MODEL)
        assert "models" in bundle and "best_model_display_name" in bundle
        line(OK, f"model loads correctly — best model: "
                 f"{bundle['best_model_display_name']}")
    except Exception as exc:                       # noqa: BLE001
        line(BAD, f"model file exists but FAILED to load: {exc}")
        problems.append("model.joblib cannot be loaded")
else:
    line(WARN, "trained model not found -> run:  python train_model.py")
    problems.append("model not trained yet")

print()
print("=" * 66)
blocking = [p for p in problems
            if not p.startswith(("dataset not generated", "model not trained"))]
if blocking:
    print("RESULT: FIX THESE BEFORE CONTINUING")
    for p in blocking:
        print(f"  - {p}")
elif problems:
    print("RESULT: environment is fine. Next steps still to do:")
    for p in problems:
        print(f"  - {p}")
    print("\n  python generate_dataset.py   (about 2 seconds)")
    print("  python train_model.py        (about 1 minute)")
else:
    print("RESULT: EVERYTHING IS READY.")
    print("Start the app with:   streamlit run app.py")
print("=" * 66)
