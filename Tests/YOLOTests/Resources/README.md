<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Test Resources Directory

This directory is designated for storing model files required to run the Ultralytics YOLO tests within the iOS application environment.

## 📦 Required Model Files

To execute the test suite successfully, ensure the following [Core ML](https://developer.apple.com/documentation/coreml) model files are placed within this directory. These models cover various computer vision tasks supported by Ultralytics YOLO:

- `yolo26n.mlpackage` - [Detection](https://docs.ultralytics.com/tasks/detect/) model
- `yolo26n-seg.mlpackage` - [Segmentation](https://docs.ultralytics.com/tasks/segment/) model
- `yolo26n-cls.mlpackage` - [Classification](https://docs.ultralytics.com/tasks/classify/) model
- `yolo26n-pose.mlpackage` - [Pose estimation](https://docs.ultralytics.com/tasks/pose/) model
- `yolo26n-obb.mlpackage` - [Oriented bounding box](https://docs.ultralytics.com/tasks/obb/) model

**Note**: Due to their significant file sizes, these model files are not included directly in the source code repository. While the necessary directory structure is provided, you must manually add the actual `.mlpackage` files.

## 📁 Directory Structure

The test resources follow this structure:

```
Tests/YOLOTests/Resources/
├── README.md                      # This file
├── yolo26n.mlpackage/             # Detection model package
├── yolo26n-cls.mlpackage/         # Classification model package
├── yolo26n-obb.mlpackage/         # Oriented bounding box model package
├── yolo26n-pose.mlpackage/        # Pose estimation model package
└── yolo26n-seg.mlpackage/         # Segmentation model package
```

Each `.mlpackage` directory should contain a complete Core ML model package with the proper structure including `Manifest.json` and other required files. When downloading or converting models, ensure they maintain this structure.

## 📥 How to Obtain Model Files

### Automated Download (Recommended)

The easiest way to download all required test models is using the automated download script. From the repository root directory, run:

```bash
bash scripts/download-models.sh
```

This script will:

- Download all nano-sized YOLO26 models to this test resources directory
- Copy the models to the appropriate app model directories for use in the iOS app
- Verify model integrity after download

### Manual Setup

For comprehensive instructions on manually acquiring and placing these model files, please consult the main `Tests/YOLOTests/README.md` file located in the parent test directory. This guide provides the necessary steps for exporting models in the required format.

---

We hope this information clarifies the setup process for the test resources. Your contributions towards improving this documentation or the overall testing framework are highly valued! Please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details on how to get involved.
