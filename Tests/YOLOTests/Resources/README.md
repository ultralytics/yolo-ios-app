# Test Resources Directory

Place model files needed for YOLO tests in this directory.

## Required Model Files

To run the tests, place the following CoreML model files in this directory:

- `yolo11n.mlpackage` - Detection model
- `yolo11n-seg.mlpackage` - Segmentation model
- `yolo11n-cls.mlpackage` - Classification model
- `yolo11n-pose.mlpackage` - Pose estimation model
- `yolo11n-obb.mlpackage` - Oriented bounding box model

**Note**: These model files are not included in the repository due to their large size. Empty directory structures are provided, but you will need to add the actual model files yourself.

## How to Obtain Model Files

For detailed instructions on how to acquire these files, please refer to `Tests/YOLOTests/README.md`.
