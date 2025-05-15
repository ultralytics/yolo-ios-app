#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to download and prepare YOLO model files for testing
# Run directly from repository root: $ bash Tests/YOLOTests/Resources/download-test-models.sh

set -e # Exit immediately if a command fails

BASE_URL="https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"
MODELS=("yolo11n" "yolo11n-seg" "yolo11n-cls" "yolo11n-pose" "yolo11n-obb")
OUTPUT_DIR="Tests/YOLOTests/Resources"

mkdir -p "$OUTPUT_DIR"

download_model() {
  local model_name=$1
  local model_path="$OUTPUT_DIR/$model_name.mlpackage"
  local zip_path="$OUTPUT_DIR/$model_name.mlpackage.zip"

  # Remove existing model directory if present
  [ -d "$model_path" ] && rm -rf "$model_path"

  # Download the model
  echo "Downloading $model_name..."
  curl -L "$BASE_URL/$model_name.mlpackage.zip" -o "$zip_path" --progress-bar

  # Extract the zip file, excluding macOS metadata
  echo "Extracting $model_name..."
  unzip -o "$zip_path" -d "$OUTPUT_DIR" -x "__MACOSX*" "*.DS_Store"

  # Clean up zip and any stray macOS folders
  rm "$zip_path"
  rm -rf "$OUTPUT_DIR/__MACOSX"

  # Quick verification
  [ -f "$model_path/Manifest.json" ] && echo "✅ Model $model_name ready" || echo "⚠️ Model $model_name may be incomplete"
}

# Download and extract each model
for model in "${MODELS[@]}"; do
  download_model "$model"
done

echo "All models prepared successfully!"
