# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export official YOLO26 Core ML assets for the iOS app release.

Usage from the repository root:

    uv venv --python 3.13 .venv
    uv pip install -e "../ultralytics[export]"
    uv run python scripts/export-models.py

The script exports the official YOLO26 task x size matrix to int8 Core ML
`.mlpackage` directories, zips each package as `<model>.mlpackage.zip`, and
optionally uploads the archives to the release used by RemoteModels.swift.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import coremltools as ct
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "coreml"
APP_MODELS_DIR = ROOT / "YOLOiOSApp" / "Models"
DEFAULT_REPO = "ultralytics/yolo-ios-app"
DEFAULT_TAG = "models-v1.0.0"
SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class TaskSpec:
    """Core ML export settings for one prediction task."""

    suffix: str
    model_dir: str
    imgsz: int


TASKS: dict[str, TaskSpec] = {
    "detect": TaskSpec("", "Detect", 640),
    "segment": TaskSpec("-seg", "Segment", 640),
    "semantic": TaskSpec("-sem", "Semantic", 640),
    "depth": TaskSpec("-depth", "Depth", 640),
    "classify": TaskSpec("-cls", "Classify", 224),
    "pose": TaskSpec("-pose", "Pose", 640),
    "obb": TaskSpec("-obb", "OBB", 640),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS))
    parser.add_argument(
        "--copy-to-app",
        action="store_true",
        help="Also copy exported .mlpackage directories into YOLOiOSApp/Models/<Task>/ for local testing.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated .mlpackage.zip files to the GitHub release with gh release upload --clobber.",
    )
    return parser.parse_args()


def zip_mlpackage(package: Path) -> Path:
    """Create a zip archive for a Core ML package directory."""
    zip_path = package.with_suffix(".mlpackage.zip")
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for path in package.rglob("*"):
            archive.write(path, Path(package.name) / path.relative_to(package))
    return zip_path


def verify_mlpackage(package: Path, imgsz: int) -> None:
    """Verify that a Core ML package has the required fixed image input."""
    spec = ct.utils.load_spec(str(package))
    image_inputs = [feature.type.imageType for feature in spec.description.input if feature.type.HasField("imageType")]
    if len(image_inputs) != 1:
        raise ValueError(f"{package.name} has {len(image_inputs)} image inputs; expected 1")
    image_input = image_inputs[0]
    if (image_input.height, image_input.width) != (imgsz, imgsz):
        raise ValueError(f"{package.name} input is {image_input.height}x{image_input.width}; expected {imgsz}x{imgsz}")


def copy_to_app(package: Path, task: TaskSpec) -> None:
    """Copy an exported Core ML package into the app model bundle."""
    destination = APP_MODELS_DIR / task.model_dir / package.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, destination)
    print(f"copied {package.name} -> {destination.relative_to(ROOT)}")


def upload_assets(repo: str, tag: str, assets: list[Path]) -> None:
    """Upload exported Core ML assets to a GitHub release."""
    if not assets:
        return
    command = [
        "gh",
        "release",
        "upload",
        tag,
        "--repo",
        repo,
        *(str(path) for path in assets),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    """Export, package, and optionally upload Core ML assets."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    assets: list[Path] = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        for size in args.sizes:
            model_id = f"yolo26{size}{task.suffix}"
            print(f"\nExporting {model_id} ({task_name}, imgsz={task.imgsz})")
            model = YOLO(f"{model_id}.pt")
            exported = Path(
                model.export(
                    format="coreml",
                    quantize=8,
                    nms=False,
                    end2end=task_name != "depth",
                    imgsz=task.imgsz,
                )
            )
            package = exported.resolve()
            manifest = package / "Manifest.json"
            if not manifest.exists():
                raise FileNotFoundError(f"Export did not create a valid mlpackage: {package}")
            verify_mlpackage(package, task.imgsz)
            if args.copy_to_app:
                copy_to_app(package, task)
            asset = zip_mlpackage(package)
            assets.append(asset)
            print(f"asset {asset.relative_to(ROOT)} input={task.imgsz}x{task.imgsz}")

    if args.upload:
        upload_assets(args.repo, args.tag, assets)

    print(f"\nPrepared {len(assets)} Core ML release assets in {output_dir}")


if __name__ == "__main__":
    main()
