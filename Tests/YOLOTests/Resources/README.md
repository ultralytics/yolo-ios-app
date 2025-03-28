<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Test Resources Directory

This directory is designated for storing model files required to run the Ultralytics YOLO tests within the iOS application environment.

## 📦 Required Model Files

To execute the test suite successfully, ensure the following [Core ML](https://developer.apple.com/documentation/coreml) model files are placed within this directory. These models cover various computer vision tasks supported by Ultralytics YOLO:

- `yolo11n.mlpackage` - [Detection](https://docs.ultralytics.com/tasks/detect/) model
- `yolo11n-seg.mlpackage` - [Segmentation](https://docs.ultralytics.com/tasks/segment/) model
- `yolo11n-cls.mlpackage` - [Classification](https://docs.ultralytics.com/tasks/classify/) model
- `yolo11n-pose.mlpackage` - [Pose estimation](https://docs.ultralytics.com/tasks/pose/) model
- `yolo11n-obb.mlpackage` - [Oriented bounding box](https://docs.ultralytics.com/tasks/obb/) model

**Note**: Due to their significant file sizes, these model files are not included directly in the source code repository. While the necessary directory structure is provided, you must manually add the actual `.mlpackage` files.

## 📥 How to Obtain Model Files

For comprehensive instructions on acquiring and placing these model files, please consult the main `Tests/YOLOTests/README.md` file located in the parent test directory. This guide provides the necessary steps for exporting models in the required format.

---

We hope this information clarifies the setup process for the test resources. Your contributions towards improving this documentation or the overall testing framework are highly valued! Please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details on how to get involved.
