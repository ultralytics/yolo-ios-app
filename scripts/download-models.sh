#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to download and prepare YOLO model files for the package tests and the app bundle
# Run from repository root: $ bash scripts/download-models.sh

set -e # Exit immediately if a command fails

BASE_URL="https://github.com/ultralytics/yolo-ios-app/releases/download/models-v1.0.0"

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../Tests/YOLOTests/Resources"
APP_DIR="$SCRIPT_DIR/../YOLOiOSApp"

# Model to directory mapping
MODELS=(
  "yolo26n:Models/Detect"
  "yolo26n-seg:Models/Segment"
  "yolo26n-sem:Models/Semantic"
  "yolo26n-depth:Models/Depth"
  "yolo26n-cls:Models/Classify"
  "yolo26n-pose:Models/Pose"
  "yolo26n-obb:Models/OBB"
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

  rm -rf "$model_path"

  # Download the model: -f fails on HTTP errors, --retry covers transient network/server failures,
  # and unzip -t catches truncated or corrupt archives before extraction
  echo "Downloading $model_name..."
  local attempt
  for attempt in 1 2 3; do
    if curl -fL --retry 3 --connect-timeout 15 "$BASE_URL/$model_name.mlpackage.zip" -o "$zip_path" --progress-bar \
      && unzip -tqq "$zip_path" > /dev/null; then
      break
    fi
    rm -f "$zip_path"
    if [[ $attempt -eq 3 ]]; then
      echo "❌ Failed to download a valid $model_name archive after 3 attempts"
      exit 1
    fi
    echo "⚠️ $model_name download failed or archive invalid (attempt $attempt), retrying..."
    sleep 5
  done

  # Extract to temp directory to handle nested structure
  local tmp_dir="$OUTPUT_DIR/_tmp_extract_$$"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  echo "Extracting $model_name..."
  unzip -o "$zip_path" -d "$tmp_dir"

  # Remove macOS metadata if present
  rm -rf "$tmp_dir/__MACOSX" 2> /dev/null || true
  find "$tmp_dir" -name "*.DS_Store" -delete 2> /dev/null || true

  # Handle nested directory: zip may contain model_name.mlpackage/ folder
  if [ -d "$tmp_dir/$model_name.mlpackage" ]; then
    mv "$tmp_dir/$model_name.mlpackage" "$model_path"
  else
    mv "$tmp_dir" "$model_path"
  fi

  # Clean up
  rm -rf "$tmp_dir" "$zip_path"

  # Verify extraction; remove the bad directory so a re-run re-downloads instead of skipping it
  if [[ ! -f "$model_path/Manifest.json" ]]; then
    rm -rf "$model_path"
    echo "❌ Model $model_name is incomplete (missing Manifest.json)"
    exit 1
  fi
  echo "✅ Model $model_name ready"

  echo "Copying $model_name to $app_dir..."
  mkdir -p "$APP_DIR/$app_dir"
  rm -rf "$app_model_path"
  cp -r "$model_path" "$app_model_path"
  echo "✅ $model_name copied to app"
}

# Process each model
for model in "${MODELS[@]}"; do
  process_model "$model"
done

echo "All models prepared successfully!"
