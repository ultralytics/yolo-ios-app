#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to copy YOLO26 models from yolo26-mobile repo to the iOS app
# Usage: bash scripts/copy-yolo26-models.sh <path-to-yolo26-mobile-repo>

set -e # Exit immediately if a command fails

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/../YOLOiOSApp"

# Check if source directory is provided
if [ -z "$1" ]; then
  echo "Usage: bash scripts/copy-yolo26-models.sh <path-to-yolo26-mobile-repo>"
  echo "Example: bash scripts/copy-yolo26-models.sh ~/Downloads/yolo26-mobile"
  exit 1
fi

SOURCE_DIR="$1/yolo26s_saved_mobile"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "❌ Error: Source directory not found: $SOURCE_DIR"
  echo "Please provide the path to the yolo26-mobile repository"
  exit 1
fi

echo "📦 Copying YOLO26 models from: $SOURCE_DIR"
echo "   To: $APP_DIR/DetectModels/"

# Create DetectModels directory if it doesn't exist
mkdir -p "$APP_DIR/DetectModels"

# Copy all .mlpackage files
MODEL_COUNT=0
for model_file in "$SOURCE_DIR"/*.mlpackage; do
  if [ -d "$model_file" ]; then
    model_name=$(basename "$model_file")
    destination="$APP_DIR/DetectModels/$model_name"
    
    # Remove existing model if present
    if [ -d "$destination" ]; then
      echo "⚠️  Removing existing: $model_name"
      rm -rf "$destination"
    fi
    
    # Copy the model
    echo "📋 Copying: $model_name"
    cp -R "$model_file" "$destination"
    MODEL_COUNT=$((MODEL_COUNT + 1))
  fi
done

if [ $MODEL_COUNT -eq 0 ]; then
  echo "⚠️  No .mlpackage files found in $SOURCE_DIR"
  echo "   Looking for files..."
  ls -la "$SOURCE_DIR" | head -10
else
  echo "✅ Successfully copied $MODEL_COUNT model(s)"
  echo ""
  echo "📝 Next steps:"
  echo "   1. Open Xcode"
  echo "   2. Right-click on 'DetectModels' folder in the project navigator"
  echo "   3. Select 'Add Files to YOLOiOSApp...'"
  echo "   4. Select the copied .mlpackage files"
  echo "   5. Make sure 'Copy items if needed' is checked"
  echo "   6. Make sure 'Create groups' is selected"
  echo "   7. Make sure your app target is checked"
  echo "   8. Click 'Add'"
fi

