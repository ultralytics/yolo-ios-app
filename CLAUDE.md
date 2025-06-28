# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Files
The YOLOiOSApp includes pre-trained YOLO11 CoreML models organized by task type:
- **Location**: `YOLOiOSApp/YOLOiOSApp/Models/`
- **Structure**: Models are organized in folders by task (DetectModels, SegmentModels, ClassifyModels, PoseModels, OBBModels)
- **Format**: `.mlpackage` files (CoreML format)
- **Sizes**: Each task includes models in NANO (n), SMALL (s), MEDIUM (m), LARGE (l), and XLARGE (x) sizes
- **Git**: Model files are excluded from version control via `.gitignore` to keep repository size manageable

## Build and Test Commands

### Swift Package Commands
```bash
# Resolve Swift Package dependencies
xcodebuild -resolvePackageDependencies

# Build the Swift Package
xcodebuild -scheme YOLO -sdk iphonesimulator -destination "platform=iOS Simulator,name=iPhone 14" build

# Run tests with coverage
xcodebuild \
  -scheme YOLO \
  -sdk iphonesimulator \
  -destination "platform=iOS Simulator,name=iPhone 14" \
  -enableCodeCoverage YES \
  clean build test

# Download test models (required for full test suite)
chmod +x Tests/YOLOTests/Resources/download-test-models.sh
Tests/YOLOTests/Resources/download-test-models.sh

# Run a specific test
xcodebuild test -scheme YOLO -destination 'platform=iOS Simulator,name=iPhone 14' -only-testing:YOLOTests/YOLOv11Tests/testObjectDetection

# Generate code coverage report (from CI workflow)
xcrun llvm-cov export -format="lcov" \
  -instr-profile="$PROFILE_PATH" \
  "$EXECUTABLE_PATH" > coverage.lcov
```

### iOS App Commands
```bash
# Build YOLOiOSApp target
xcodebuild -scheme YOLOiOSApp -sdk iphonesimulator build

# Build for device
xcodebuild -scheme YOLOiOSApp -sdk iphoneos build

# Build example apps
xcodebuild -scheme YOLORealTimeSwiftUI -sdk iphonesimulator build
xcodebuild -scheme YOLORealTimeUIKit -sdk iphonesimulator build
xcodebuild -scheme YOLOSingleImageSwiftUI -sdk iphonesimulator build
xcodebuild -scheme YOLOSingleImageUIKit -sdk iphonesimulator build
```

## Architecture Overview

This repository provides a comprehensive YOLO implementation for iOS, consisting of a reusable Swift Package and demonstration applications.

### Repository Structure
- **Sources/YOLO/**: Swift Package containing the core YOLO library
  - `Predictor` protocol: Unified interface for all YOLO tasks
  - Task implementations: ObjectDetector, Classifier, Segmenter, PoseEstimater, ObbDetector
  - UI components: YOLOView (UIKit), YOLOCamera (SwiftUI), BoundingBoxView
  - Utilities: VideoCapture, NonMaxSuppression, ThresholdProvider
  - Model management: Download manager for remote models

- **YOLOiOSApp/**: Main iOS application demonstrating YOLO capabilities
  - Pre-packaged YOLO11 CoreML models organized by task type
  - Modern UI with dark theme and gradient backgrounds
  - Multi-level zoom and real-time performance metrics
  - Model size selector (NANO to XLARGE)
  - Tab-based task switching

- **ExampleApps/**: Four example implementations
  - YOLORealTimeSwiftUI: Real-time detection with SwiftUI
  - YOLORealTimeUIKit: Real-time detection with UIKit
  - YOLOSingleImageSwiftUI: Single image processing with SwiftUI
  - YOLOSingleImageUIKit: Single image processing with UIKit

### Key Design Patterns

1. **Protocol-Based Architecture**: All YOLO tasks conform to the `Predictor` protocol, enabling polymorphic model usage

2. **Callable Syntax**: Swift's callable syntax for intuitive inference:
   ```swift
   let result = model(image)  // Direct call syntax
   ```

3. **Multi-Input Support**: Models accept UIImage, CIImage, CGImage, SwiftUI.Image, file paths, and URLs

4. **Task-Specific Results**: Structured result types for each task (DetectionResult, ClassificationResult, etc.)

5. **Camera Integration**: YOLOCamera SwiftUI view provides complete real-time inference with minimal code

6. **Delegate Pattern**: VideoCapture uses delegates for frame processing callbacks

### Model Architecture
- CoreML format required (.mlpackage)
- Input size: 640x640 (standard YOLO input)
- Supported tasks: Detection, Classification, Segmentation, Pose Estimation, OBB
- Model sizes: NANO (n), SMALL (s), MEDIUM (m), LARGE (l), XLARGE (x)
- INT8 quantization for optimal performance

### Current UI State (feat/uikit-modern-ui branch)
- Modern dark theme with gradient backgrounds
- Yellow bounding boxes for detections
- Real-time FPS and latency metrics
- Zoom controls with 0.5x, 1x, 2x, 3x levels
- Model size selector in navigation bar
- Tab bar for task switching

## Testing Environment
- Test models must be downloaded separately using the provided script
- Tests can run with or without models by setting `SKIP_MODEL_TESTS` flag
- CI/CD uses macOS-15 with iPhone 14 simulator
- Code coverage is tracked via Codecov integration
- For tests requiring environment variables, create a `.env` file with:
  ```
  API_URL=your_api_url
  FIREBASE_API_KEY=your_firebase_key
  ```