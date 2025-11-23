#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to download and prepare YOLO model files for testing
# Run from repository root: $ bash scripts/download-models.sh

set -e # Exit immediately if a command fails

BASE_URL="https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../Tests/YOLOTests/Resources"
APP_DIR="$SCRIPT_DIR/../YOLOiOSApp"

# Model to directory mapping - using YOLO26 models (newer, don't need NMS)
MODELS=(
  "yolo26n:DetectModels"
  "yolo26n-seg:SegmentModels"
  "yolo26n-cls:ClassifyModels"
  "yolo26n-pose:PoseModels"
  "yolo26n-obb:OBBModels"
)

# Ensure directories exist
mkdir -p "$OUTPUT_DIR"

process_model() {
  local model_info=$1
  local model_name="${model_info%:*}"
  local app_dir="${model_info#*:}"
  local model_path="$OUTPUT_DIR/$model_name.mlpackage"
  local zip_path="$OUTPUT_DIR/$model_name.mlpackage.zip"
  local app_model_path="$APP_DIR/$app_dir/$model_name.mlpackage"

  # Download and extract if not present
  if [[ ! -d "$model_path" ]]; then
    # Remove existing model directory if present
    rm -rf "$model_path"

    # Create model directory
    mkdir -p "$model_path"

    # Download the model
    echo "Downloading $model_name..."
    curl -L "$BASE_URL/$model_name.mlpackage.zip" -o "$zip_path" --progress-bar

    # Extract the zip file
    echo "Extracting $model_name..."
    unzip -o "$zip_path" -d "$model_path"

    # Remove macOS metadata if present (after extraction)
    rm -rf "$model_path/__MACOSX" 2> /dev/null || true
    find "$model_path" -name "*.DS_Store" -delete 2> /dev/null || true

    # Clean up zip
    rm "$zip_path"

    # Quick verification
    [ -f "$model_path/Manifest.json" ] && echo "✅ Model $model_name ready" || echo "⚠️ Model $model_name may be incomplete"
  fi

  # Copy to app model directory if not present
  if [[ ! -d "$app_model_path" ]]; then
    echo "Copying $model_name to $app_dir..."
    mkdir -p "$APP_DIR/$app_dir"
    cp -r "$model_path" "$app_model_path"
    echo "✅ $model_name copied to app"
  fi
}

# Process each model
for model in "${MODELS[@]}"; do
  process_model "$model"
done

echo "All models prepared successfully!"
