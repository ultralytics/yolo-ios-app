#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to download and prepare YOLO model files for testing
# Run directly from repository root: $ bash Tests/YOLOTests/Resources/download-test-models.sh

set -e # Exit immediately if a command fails

# Define constants
BASE_URL="https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"
MODELS=("yolo11n" "yolo11n-seg" "yolo11n-cls" "yolo11n-pose" "yolo11n-obb")
OUTPUT_DIR="Tests/YOLOTests/Resources"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to download and extract a model
download_model() {
  local model_name=$1
  local model_path="$OUTPUT_DIR/$model_name.mlpackage"
  local zip_path="$OUTPUT_DIR/$model_name.mlpackage.zip"

  # Remove any existing model directory
  if [ -d "$model_path" ]; then
    echo "Removing existing $model_name model"
    rm -rf "$model_path"
  fi

  # Download the model
  echo "Downloading $model_name from $BASE_URL/$model_name.mlpackage.zip"
  curl -L "$BASE_URL/$model_name.mlpackage.zip" -o "$zip_path" --progress-bar

  # Extract the zip file
  echo "Extracting $zip_path"
  unzip -o "$zip_path" -d "$OUTPUT_DIR"

  # Remove the zip file after extraction
  rm "$zip_path"
  echo "Cleaned up $zip_path"

  # Remove macOS metadata folders if they exist
  rm -rf "$OUTPUT_DIR/__MACOSX"

  echo "Successfully prepared $model_name"
  return 0
}

# Download and extract each model
for model in "${MODELS[@]}"; do
  download_model "$model"
done

echo "All models prepared successfully!"
