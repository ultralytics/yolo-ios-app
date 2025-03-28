<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Test Guide

Welcome to the testing guide for the Ultralytics YOLO iOS application. This directory contains comprehensive tests designed to ensure the robustness and correctness of the YOLO framework integration within the iOS environment. To execute these tests successfully, you'll need to download and correctly place the required model files first.

## 🧪 Preparation Before Testing

Follow these steps to set up your testing environment.

### 1. Check the Test Resource Directory

Verify that the following directory exists within your project structure:

```
Tests/YOLOTests/Resources/
```

If this directory is missing, create it using the terminal:

```bash
mkdir -p Tests/YOLOTests/Resources/
```

### 2. Obtain the Required Model Files

The tests require specific [Core ML](https://developer.apple.com/documentation/coreml) model files (`.mlpackage`). Ensure you have the following files ready:

-   `yolo11n.mlpackage`: Standard [object detection](https://docs.ultralytics.com/tasks/detect/) model.
-   `yolo11n-seg.mlpackage`: Model for [instance segmentation](https://docs.ultralytics.com/tasks/segment/).
-   `yolo11n-cls.mlpackage`: Model for [image classification](https://docs.ultralytics.com/tasks/classify/).
-   `yolo11n-pose.mlpackage`: Model for [pose estimation](https://docs.ultralytics.com/tasks/pose/).
-   `yolo11n-obb.mlpackage`: Model for [oriented bounding box](https://docs.ultralytics.com/tasks/obb/) detection.

### 3. Methods to Acquire Model Files

You can obtain the necessary `.mlpackage` files using one of the following methods:

#### Method 1: Download and Convert Official Models

1.  Download the base PyTorch (`.pt`) models from the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) releases or train your own.
2.  Convert these models to the Core ML format using the Ultralytics `export` mode. You'll need a [Python](https://www.python.org/) environment with the `ultralytics` package installed (see the [Ultralytics Quickstart guide](https://docs.ultralytics.com/quickstart/) for installation).

```python
from ultralytics import YOLO

# Ensure you have the base .pt files (e.g., yolo11n.pt, yolo11n-seg.pt, etc.)
# Download them if necessary or use your custom trained models.

# Detection model (YOLO11n)
model_det = YOLO("yolo11n.pt")
model_det.export(format="coreml", nms=True) # Exports yolo11n.mlpackage

# Segmentation model (YOLO11n-Seg)
model_seg = YOLO("yolo11n-seg.pt")
model_seg.export(format="coreml") # Exports yolo11n-seg.mlpackage

# Classification model (YOLO11n-Cls)
model_cls = YOLO("yolo11n-cls.pt")
model_cls.export(format="coreml") # Exports yolo11n-cls.mlpackage

# Pose estimation model (YOLO11n-Pose)
model_pose = YOLO("yolo11n-pose.pt")
model_pose.export(format="coreml") # Exports yolo11n-pose.mlpackage

# OBB (Oriented Bounding Box) model (YOLO11n-OBB)
model_obb = YOLO("yolo11n-obb.pt")
model_obb.export(format="coreml") # Exports yolo11n-obb.mlpackage

# For more details on export options, see: https://docs.ultralytics.com/modes/export/
```

#### Method 2: Use Ultralytics Pre-Exported Models (If Available)

Check the [Ultralytics YOLO11 model page](https://docs.ultralytics.com/models/yolo11/) or the [Ultralytics HUB Models section](https://docs.ultralytics.com/hub/models/) for potentially available pre-exported Core ML models. Note that direct downloads of `.mlpackage` files might not always be provided, making Method 1 the more reliable approach.

### 4. Place the Model Files

After obtaining or exporting the `.mlpackage` files, move or copy them into the designated resource directory:

```
Tests/YOLOTests/Resources/
```

Ensure the filenames match exactly those listed in step 2.

## ▶️ Running the Tests

With the model files correctly placed, you can run the test suite using either [Swift Package Manager (SwiftPM)](https://www.swift.org/package-manager/) or [Xcode](https://developer.apple.com/xcode/).

### Using SwiftPM

Navigate to the root directory of the `yolo-ios-app` package in your terminal and run:

```bash
swift test
```

### Using Xcode

1.  Open the `Package.swift` file located in the root directory of the [yolo-ios-app repository](https://github.com/ultralytics/yolo-ios-app) using Xcode.
2.  Wait for Xcode to resolve package dependencies.
3.  Select **Product** > **Test** from the menu bar, or use the shortcut **⌘U**.

Xcode will build the package and execute all the tests defined in the `YOLOTests` target.

## 🛠️ Troubleshooting

Encountering issues? Here are some common problems and solutions:

### "Test model file not found" Error

If you receive an error message indicating that a model file could not be found:

1.  **Verify Path:** Double-check that all required `.mlpackage` files are present directly inside the `Tests/YOLOTests/Resources/` directory.
2.  **Verify Filenames:** Ensure the filenames exactly match the required names (e.g., `yolo11n.mlpackage`, `yolo11n-seg.mlpackage`, etc.). Check for typos or incorrect extensions.
3.  **Check `Package.swift`:** Confirm that the `Resources` directory is correctly specified as a resource for the `YOLOTests` target in the `Package.swift` file.

### Other Issues

If tests fail or you encounter other problems:

1.  **SwiftPM Version:** Ensure your installed Swift Package Manager version is compatible with the project requirements.
2.  **iOS Target:** The project requires [iOS](https://www.apple.com/ios/ios-18/) 16.0 or later. Make sure your testing environment (simulator or device) meets this requirement.
3.  **Framework Availability:** Confirm that the Core ML and [Vision frameworks](https://developer.apple.com/documentation/vision) are available and correctly linked in your build settings.
4.  **Consult Logs:** Examine the detailed test logs in Xcode or the terminal output for specific error messages that can help pinpoint the issue.
5.  **Check Ultralytics Docs:** Refer to the [Ultralytics documentation](https://docs.ultralytics.com/) or the [FAQ section](https://docs.ultralytics.com/help/FAQ/) for potential solutions.

## Contributing

Contributions to enhance the tests or improve the iOS application are welcome! Please see the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more information on how to get started. Thank you for helping improve Ultralytics YOLO!
