#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download the six nano YOLO26 Core ML models into the iOS app's Models/ folders so the app ships with them and works
# offline straight after a fresh clone. This runs automatically from the YOLOiOSApp "Download YOLO Models" Xcode build
# phase, and is also safe to run by hand from the repository root: $ bash scripts/download-app-models.sh
#
# Model archives come from the GitHub release referenced by RemoteModels.swift. Nothing is committed to git (all
# *.mlpackage assets are gitignored), and the app already falls back to on-demand download at runtime, so a missing
# network connection here only prints a warning instead of failing the build.

BASE_URL="https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Destination Models/ directory: the Xcode build phase passes "${SRCROOT}/Models"; manual runs default to the app's.
MODELS_DIR="${1:-$SCRIPT_DIR/../YOLOiOSApp/Models}"

# Nano model -> task subfolder. These six are the default models bundled with the app, one per supported task.
MODELS=(
  "yolo26n:Detect"
  "yolo26n-seg:Segment"
  "yolo26n-sem:Semantic"
  "yolo26n-cls:Classify"
  "yolo26n-pose:Pose"
  "yolo26n-obb:OBB"
)

download_model() {
  local name="$1" folder="$2"
  local dest_dir="$MODELS_DIR/$folder"
  local dest="$dest_dir/$name.mlpackage"

  # Skip models that are already present and valid (keeps incremental builds fast and works offline).
  if [[ -f "$dest/Manifest.json" ]]; then
    echo "note: $name already present, skipping"
    return 0
  fi

  local work zip extract
  work="$(mktemp -d)"
  zip="$work/$name.mlpackage.zip"
  extract="$work/extract"
  mkdir -p "$extract"

  echo "note: downloading $name..."
  if ! curl -fL "$BASE_URL/$name.mlpackage.zip" -o "$zip" --progress-bar \
    || ! unzip -oq "$zip" -d "$extract"; then
    echo "warning: could not fetch $name; the app will download it on demand at runtime"
    rm -rf "$work"
    return 0
  fi

  # Strip macOS archive cruft, then locate the .mlpackage (the zip may be nested or flat).
  rm -rf "$extract/__MACOSX" 2> /dev/null || true
  find "$extract" -name ".DS_Store" -delete 2> /dev/null || true

  local pkg
  pkg="$(find "$extract" -maxdepth 2 -type d -name "*.mlpackage" | head -n 1)"
  [[ -z "$pkg" && -f "$extract/Manifest.json" ]] && pkg="$extract"
  if [[ -z "$pkg" || ! -f "$pkg/Manifest.json" ]]; then
    echo "warning: $name archive invalid (no Manifest.json); skipping"
    rm -rf "$work"
    return 0
  fi

  mkdir -p "$dest_dir"
  rm -rf "$dest"
  mv "$pkg" "$dest"
  rm -rf "$work"
  echo "note: ✅ $name ready"
}

for entry in "${MODELS[@]}"; do
  download_model "${entry%:*}" "${entry#*:}"
done

echo "note: YOLO nano model setup complete -> $MODELS_DIR"
exit 0
