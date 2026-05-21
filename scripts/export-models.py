# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export all YOLO models to Core ML format for the iOS app.

Usage:
    pip install ultralytics
    python scripts/export-models.py

Exports YOLO models (5 sizes x 6 tasks = 30 models) to Core ML .mlpackage
format and copies them into the app's Models/ directories.
"""

import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO

# Repository root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
APP_MODELS = ROOT / "YOLOiOSApp" / "Models"

# All sizes and tasks
SIZES = ["n", "s", "m", "l", "x"]
TASKS = {
    "": "Detect",
    "-seg": "Segment",
    "-sem": "Semantic",
    "-cls": "Classify",
    "-pose": "Pose",
    "-obb": "OBB",
}
# Square exports are best when the same model is used for both portrait and landscape.
# Ultralytics imgsz order is [height, width]; use [640, 384] for portrait-only or [384, 640] for landscape-only.
# Use orientation-only shapes only when inference is locked to that orientation.
IMGSZ = {
    "": 640,
    "-seg": 640,
    "-sem": 640,
    "-cls": 224,
    "-pose": 640,
    "-obb": 1024,
}


def main():
    """Export YOLO26 models to Core ML format and prepare zips for release."""
    for size in SIZES:
        for suffix, task_dir in TASKS.items():
            model_name = f"yolo26{size}{suffix}.pt"
            print(f"\nExporting {model_name} to Core ML...")
            model = YOLO(model_name)
            imgsz = IMGSZ[suffix]
            exported = model.export(format="coreml", int8=True, nms=False, imgsz=[imgsz, imgsz])

            # Copy exported .mlpackage to app Models directory
            src = Path(exported)
            dst = APP_MODELS / task_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(src, dst)
            print(f"  Copied to {dst.relative_to(ROOT)}")

            # Create zip for GitHub release upload
            zip_path = src.with_suffix(".mlpackage.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in src.rglob("*"):
                    zf.write(file, file.relative_to(src.parent))
            print(f"  Zipped to {zip_path.name}")

    print("\nAll models exported, copied, and zipped successfully!")


if __name__ == "__main__":
    main()
