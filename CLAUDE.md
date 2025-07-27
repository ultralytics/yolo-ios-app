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


## External Display Support (Build 467+)

⚠️ **IMPORTANT**: This repository now includes external display support for presentations and demonstrations.

### Current Implementation Status
- ✅ Multi-scene architecture with Scene Delegates
- ✅ iPhone controls + External display visualization
- ✅ Dynamic UI scaling for 5K displays
- ✅ Model synchronization between displays
- ⚠️ Known issues with remote model loading
- ⚠️ UI synchronization timing issues

### Key Features
- **iPhone Controls**: All model selection and parameter adjustment
- **External Display**: Clean visualization with model name, FPS, and detection results
- **Dynamic Scaling**: Automatic UI scaling for large displays (Apple Studio Display 5K tested)
- **UI Toggle**: Eye button (👁️) to show/hide external display UI elements

### Technical Notes
- Uses `UIWindowSceneSessionRoleExternalDisplay` for external window management
- Scene Delegates handle multi-window coordination
- NotificationCenter for inter-scene communication
- YOLOView API methods made public: `switchCameraTapped()`, `sliderChanged()`

### Files Modified for External Display
- `YOLOiOSApp/Info.plist`: Multi-scene configuration
- `YOLOiOSApp/SceneDelegate.swift` & `ExternalSceneDelegate.swift`: Scene management
- `YOLOiOSApp/ExternalViewController.swift`: External display UI controller
- `YOLOiOSApp/ViewController.swift`: Main app with external display coordination
- `Sources/YOLO/YOLOView.swift`: Public API for external access
- `Sources/YOLO/BoundingBoxView.swift`: Dynamic scaling implementation

### Known Issues to Address
1. **Remote Model Loading**: Downloaded models may not sync properly to external display
2. **Initial Sync**: External display may not receive initial model state
3. **Performance**: Dual YOLOView instances need optimization
4. **Error Handling**: Better model loading failure management

### Testing Requirements
- Requires physical iOS device with external display capability
- Apple Studio Display (5K) recommended for testing
- Lightning to HDMI or USB-C display adapters

## Code Organization

### Project Structure Pattern
- Each app target has its own directory with `.xcodeproj` file
- Models organized by task type in dedicated directories (DetectModels/, SegmentModels/, etc.)
- Test files with `.backup` extension contain model-dependent tests
- Each test directory includes a README.md with setup instructions

### Code Style and Formatting
- Swift code follows standard iOS conventions
- No explicit SwiftFormat or SwiftLint configuration files (uses Xcode defaults)
- GitHub Actions uses Ultralytics formatter for consistency
- Test naming convention: `test<FeatureName>_<Scenario>_<ExpectedResult>()`

### Model File Management
- Model files (.mlpackage) contain CoreML models with weights
- Models are included in repository for easy app testing
- Remote model downloading supported via ModelDownloadManager
- Model files organized by task type: Detect, Segment, Classify, Pose, OBB

## License

- Open source: AGPL-3.0 License
- Commercial use: Enterprise License required (ultralytics.com/license)