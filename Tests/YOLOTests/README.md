# YOLO Test Guide

This directory contains comprehensive tests for the YOLO framework. To run these tests, you need to download and place the required model files.

## Preparation Before Testing

### 1. Check the Test Resource Directory

Ensure the following directory exists:

```
Tests/YOLOTests/Resources/
```

If it doesn't exist, create it:

```bash
mkdir -p Tests/YOLOTests/Resources/
```

### 2. Obtain the Required Model Files

Prepare the following CoreML model files needed for testing:

- `yolo11n.mlpackage` - Detection model
- `yolo11n-seg.mlpackage` - Segmentation model
- `yolo11n-cls.mlpackage` - Classification model
- `yolo11n-pose.mlpackage` - Pose estimation model
- `yolo11n-obb.mlpackage` - Oriented bounding box model

### 3. Methods to Acquire Model Files

#### Method 1: Download from the Official Source

1. Download models from [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
2. Convert to CoreML format by running the following code in a Python environment:

```python
from ultralytics import YOLO

# Detection model
model = YOLO("yolo11n.pt")
model.export(format="coreml", nms=True)

# Segmentation model
model = YOLO("yolo11n-seg.pt")
model.export(format="coreml")

# Classification model
model = YOLO("yolo11n-cls.pt")
model.export(format="coreml")

# Pose estimation model
model = YOLO("yolo11n-pose.pt")
model.export(format="coreml")

# OBB (Oriented Bounding Box) model
model = YOLO("yolo11n-obb.pt")
model.export(format="coreml")
```

#### Method 2: Use Ultralytics Sample Models

You can also download models from Ultralytics [Model Hub](https://docs.ultralytics.com/models/) and convert them.

### 4. Place the Model Files

Place the converted `.mlpackage` files in the `Tests/YOLOTests/Resources/` directory.

## Running the Tests

Once preparation is complete, you can run the tests using SwiftPM:

```bash
swift test
```

Alternatively, open the project in Xcode and run the tests:

1. Open Package.swift
2. Select Product > Test (⌘U)

## Troubleshooting

### When Model Files Are Not Found

If you see a "Test model file not found" error message:

1. Check that model files are placed in the correct path
2. Verify model filenames and extensions are accurate (e.g., `yolo11n.mlpackage`)
3. Ensure resource settings in Package.swift are correct

### Other Issues

If you encounter problems during test execution, check the following:

1. Compatibility of your Swift Package Manager version
2. Support for the required iOS version (iOS 16.0 or above)
3. Availability of CoreML and Vision frameworks
