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
}

copy_to_app() {
  local model_name=$1
  local app_dir=$2
  local model_path="$OUTPUT_DIR/$model_name.mlpackage"
  local app_model_path="../../../YOLOiOSApp/$app_dir/$model_name.mlpackage"
  
  if [[ ! -d "$app_model_path" ]]; then
    echo "Copying $model_name to $app_dir..."
    mkdir -p "../../../YOLOiOSApp/$app_dir"
    cp -r "$model_path" "$app_model_path"
    echo "✅ $model_name copied to app"
  fi
}

# Download and extract each model
for model in "${MODELS[@]}"; do
  download_model "$model"
done

# Copy models to app directories
copy_to_app "yolo11n" "DetectModels"
copy_to_app "yolo11n-seg" "SegmentModels"
copy_to_app "yolo11n-cls" "ClassifyModels"
copy_to_app "yolo11n-pose" "PoseModels"
copy_to_app "yolo11n-obb" "OBBModels"

echo "All models prepared successfully!"
