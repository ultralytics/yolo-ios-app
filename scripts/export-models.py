# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Export all YOLO26 models to CoreML format for the iOS app.

Usage:
    pip install ultralytics
    python scripts/export-models.py

Exports YOLO26 nano models for all 5 tasks (detect, segment, classify, pose, obb)
to CoreML .mlpackage format and copies them into the app's Models/ directories.
"""
import shutil

from pathlib import Path

from ultralytics import YOLO

# Repository root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
APP_MODELS = ROOT / "YOLOiOSApp" / "Models"

# YOLO26 nano models for all 5 tasks
MODELS = {
    "yolo26n.pt": "Detect",
    "yolo26n-cls.pt": "Classify",
    "yolo26n-seg.pt": "Segment",
    "yolo26n-pose.pt": "Pose",
    "yolo26n-obb.pt": "OBB",
}


def main():
    for model_name, task_dir in MODELS.items():
        print(f"\nExporting {model_name} to CoreML...")
        model = YOLO(model_name)
        exported = model.export(format="coreml", int8=True, nms=False)

        # Copy exported .mlpackage to app Models directory
        src = Path(exported)
        dst = APP_MODELS / task_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)
        print(f"  Copied to {dst.relative_to(ROOT)}")

    print("\nAll YOLO26 models exported and copied successfully!")


if __name__ == "__main__":
    main()
