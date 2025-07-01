# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Ultralytics YOLO iOS app - a Swift-based computer vision application providing real-time object detection, segmentation, pose estimation, and classification using CoreML. The project includes both a standalone iOS app and a reusable Swift Package for developers.

## Build and Test Commands

### Swift Package Manager
- Run all tests: `swift test`
- Build package: `swift build`
- Clean build: `swift package clean`

### Xcode Commands
- Build and test with simulator: `xcodebuild -scheme YOLO -sdk iphonesimulator -destination "platform=iOS Simulator,name=iPhone 14" clean build test`
- Resolve dependencies: `xcodebuild -resolvePackageDependencies`
- Run tests in Xcode: `Cmd+U`
- Build in Xcode: `Cmd+B`

### Test Setup
Before running tests, download required CoreML models:
```bash
chmod +x Tests/YOLOTests/Resources/download-test-models.sh
Tests/YOLOTests/Resources/download-test-models.sh
```

## High-Level Architecture

### Core Components

1. **Entry Point (`YOLO.swift`)**
   - Main API class with `callAsFunction` for elegant syntax
   - Factory pattern for creating task-specific predictors
   - Supports multiple input formats (UIImage, CIImage, CGImage, paths, URLs)

2. **Task System (`YOLOTask.swift`)**
   - `.detect`: Object detection with bounding boxes
   - `.segment`: Instance segmentation with masks
   - `.pose`: Human pose estimation
   - `.obb`: Oriented bounding box detection
   - `.classify`: Image classification

3. **Prediction Architecture**
   - **`Predictor` Protocol**: Contract for all predictors
   - **`BasePredictor`**: Abstract base implementing common functionality
     - Async CoreML model loading
     - Vision framework integration
     - Performance monitoring
   - **Task-specific predictors**: ObjectDetector, Segmenter, PoseEstimater, ObbDetector, Classifier

4. **UI Layer**
   - **`YOLOCamera`**: SwiftUI camera wrapper
   - **`YOLOView`**: UIKit view with camera management and result visualization
   - **`VideoCapture`**: AVFoundation-based camera handling

5. **Data Flow**
   ```
   Camera → VideoCapture → Predictor → Vision Request → processObservations → YOLOResult → UI
   ```

### Key Design Patterns
- Factory Pattern for model creation
- Protocol-Oriented Design for extensibility
- Delegate Pattern for result callbacks
- Async/Await for background operations

## Development Guidelines

### Requirements
- Swift 5.7+
- Xcode 14.0+
- iOS 16.0+ deployment target
- CoreML models in `.mlpackage` format

### Code Style
- Follow standard Swift conventions
- Use `///` documentation comments for public APIs
- Implement proper error handling with `PredictorError`
- Each component should have corresponding tests

### Adding New Features
To add a new YOLO task:
1. Add case to `YOLOTask` enum
2. Create predictor class inheriting from `BasePredictor`
3. Implement `processObservations()` and `predictOnImage()`
4. Add case to YOLO.init() switch

### Testing
- Models required: yolo11n.mlpackage (and variants for each task)
- Set `SKIP_MODEL_TESTS = true` if models unavailable
- Tests organized by functionality in Tests/YOLOTests/

### Camera Usage
Add to Info.plist: "Privacy - Camera Usage Description"

## Project Structure
```
├── Sources/YOLO/          # Swift Package library
├── YOLOiOSApp/           # Main iOS application
├── ExampleApps/          # Example implementations
├── Tests/YOLOTests/      # Unit tests
└── .github/workflows/    # CI/CD configuration
```

## CI/CD
- Runs on macOS-15 with iPhone simulator
- Automatic code formatting via GitHub Actions
- Code coverage reporting to Codecov
- Test models downloaded automatically in CI