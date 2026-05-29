"""Verify all lab dependencies are installed in the active environment."""
import importlib
import sys

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "pytest_cov": "pytest-cov",
    "flake8": "flake8",
    "black": "black",
    "pre_commit": "pre-commit",
    "mypy": "mypy",
    "yaml": "pyyaml",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "psutil": "psutil",
    "tensorboard": "tensorboard",
}

print(f"Python: {sys.executable}\n")

failed = []
for module, label in PACKAGES.items():
    try:
        importlib.import_module(module)
        print(f"[OK] {label}")
    except ImportError as err:
        print(f"[FAIL] {label}: {err}")
        failed.append(label)

if failed:
    print(f"\nMissing packages: {', '.join(failed)}")
    sys.exit(1)

print("\nAll lab dependencies are installed.")
