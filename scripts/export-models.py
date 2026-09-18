# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export official YOLO26 Core ML and Core AI assets for the iOS app release.

Usage from the repository root:

    uv venv --python 3.13 .venv
    uv pip install "ultralytics[export-coreml]>=8.4.156" "coreai-torch>=0.4.2"
    uv run python scripts/export-models.py

The script exports the official YOLO26 task x size matrix to int8 Core ML
`.mlpackage` directories and FP16 Core AI `.aimodel` directories (Core AI
export requires macOS 26+ on Apple silicon), zips each as `<model>.<ext>.zip`,
and optionally uploads the archives to the release used by RemoteModels.swift.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import coremltools as ct
from ultralytics import YOLO, __version__
from ultralytics.utils.checks import check_version

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "exports"
APP_MODELS_DIR = ROOT / "YOLOiOSApp" / "Models"
DEFAULT_REPO = "ultralytics/yolo-ios-app"
DEFAULT_TAG = "v8.3.0"
SIZES = ("n", "s", "m", "l", "x")
# Export arguments per format. Core AI has no int8 export, so its assets are FP16.
FORMATS = {"coreml": {"quantize": 8}, "coreai": {"quantize": 16}}


@dataclass(frozen=True)
class TaskSpec:
    """Export settings for one prediction task."""

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
    parser.add_argument("--formats", nargs="+", choices=FORMATS.keys(), default=list(FORMATS))
    parser.add_argument(
        "--copy-to-app",
        action="store_true",
        help="Also copy exported model directories into YOLOiOSApp/Models/<Task>/ for local testing.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated .zip archives to the GitHub release with gh release upload --clobber.",
    )
    return parser.parse_args()


def zip_model(package: Path) -> Path:
    """Create a zip archive for a model directory, keeping the directory as the top-level entry."""
    zip_path = package.with_name(f"{package.name}.zip")
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for path in package.rglob("*"):
            archive.write(path, Path(package.name) / path.relative_to(package))
    return zip_path


def verify_model(package: Path, task_name: str, imgsz: int, quantize: int) -> None:
    """Verify that an exported model has the required mobile export contract."""
    if package.suffix == ".aimodel":  # same keys and string values as the Core ML metadata
        metadata = json.loads((package / "metadata.json").read_text())["creatorDefinedMetadata"]
        if ast.literal_eval(metadata.get("imgsz", "[]")) != [imgsz, imgsz]:
            raise ValueError(f"{package.name} input is {metadata.get('imgsz')}; expected {imgsz}x{imgsz}")
    else:
        spec = ct.utils.load_spec(str(package))
        image_inputs = [f.type.imageType for f in spec.description.input if f.type.HasField("imageType")]
        if len(image_inputs) != 1:
            raise ValueError(f"{package.name} has {len(image_inputs)} image inputs; expected 1")
        image_input = image_inputs[0]
        if (image_input.height, image_input.width) != (imgsz, imgsz):
            raise ValueError(
                f"{package.name} input is {image_input.height}x{image_input.width}; expected {imgsz}x{imgsz}"
            )
        metadata = dict(spec.description.metadata.userDefined)
        if ast.literal_eval(metadata.get("args", "{}")).get("nms") is not False:
            raise ValueError(f"{package.name} metadata does not record nms=False")
    args = ast.literal_eval(metadata.get("args", "{}"))
    expected_end2end = task_name in {"detect", "segment", "pose", "obb"}
    if metadata.get("task") != task_name:
        raise ValueError(f"{package.name} metadata task is {metadata.get('task')}; expected {task_name}")
    if args.get("quantize") != quantize:
        raise ValueError(f"{package.name} metadata does not record quantize={quantize}")
    if metadata.get("end2end") != str(expected_end2end):
        raise ValueError(f"{package.name} end2end metadata is {metadata.get('end2end')}; expected {expected_end2end}")


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def copy_to_app(package: Path, task: TaskSpec) -> None:
    """Copy an exported model directory into the app model bundle."""
    destination = APP_MODELS_DIR / task.model_dir / package.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, destination)
    print(f"copied {package.name} -> {destination.relative_to(ROOT)}")


def upload_assets(repo: str, tag: str, assets: list[Path]) -> None:
    """Upload exported assets to a GitHub release."""
    if not assets:
        return
    command = [
        "gh",
        "release",
        "upload",
        tag,
        "--repo",
        repo,
        "--clobber",
        *(str(path) for path in assets),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    """Export, package, and optionally upload Core ML and Core AI assets."""
    args = parse_args()
    check_version(__version__, ">=8.4.156", name="ultralytics", hard=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    assets: list[Path] = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        for size in args.sizes:
            model_id = f"yolo26{size}{task.suffix}"
            for fmt in args.formats:
                print(f"\nExporting {model_id} ({task_name}, {fmt}, imgsz={task.imgsz})")
                model = YOLO(output_dir / f"{model_id}.pt")
                package = Path(model.export(format=fmt, nms=False, imgsz=task.imgsz, **FORMATS[fmt])).resolve()
                verify_model(package, task_name, task.imgsz, FORMATS[fmt]["quantize"])
                if args.copy_to_app:
                    copy_to_app(package, task)
                asset = zip_model(package)
                assets.append(asset)
                print(f"asset {display_path(asset)} input={task.imgsz}x{task.imgsz}")

    if args.upload:
        upload_assets(args.repo, args.tag, assets)

    print(f"\nPrepared {len(assets)} release assets in {output_dir}")


if __name__ == "__main__":
    main()
