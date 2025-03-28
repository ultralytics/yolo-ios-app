<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Test Resources Directory

Place model files needed for YOLO tests in this directory.

## 📦 Required Model Files

To run the tests, place the following [Core ML](https://developer.apple.com/documentation/coreml) model files in this directory:

- `yolo11n.mlpackage` - [Detection](https://docs.ultralytics.com/tasks/detect/) model
- `yolo11n-seg.mlpackage` - [Segmentation](https://docs.ultralytics.com/tasks/segment/) model
- `yolo11n-cls.mlpackage` - [Classification](https://docs.ultralytics.com/tasks/classify/) model
- `yolo11n-pose.mlpackage` - [Pose estimation](https://docs.ultralytics.com/tasks/pose/) model
- `yolo11n-obb.mlpackage` - [Oriented bounding box](https://docs.ultralytics.com/tasks/obb/) model

**Note**: These model files are not included in the repository due to their large size. Empty directory structures are provided, but you will need to add the actual model files yourself.

## 📥 How to Obtain Model Files

For detailed instructions on how to acquire these files, please refer to the main `Tests/YOLOTests/README.md`.

---

We hope this guide helps you set up the necessary resources for testing. Contributions to improve this documentation or the testing process are welcome! Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details.
