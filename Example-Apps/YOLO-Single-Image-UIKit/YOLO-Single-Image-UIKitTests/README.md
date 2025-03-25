# YOLO Single Image UIKit Tests

This directory contains unit tests for the YOLO Single Image UIKit example application.

## Running Tests

### Prerequisites

To run these tests, you need to have the following CoreML model file:

- `yolo11x-seg.mlpackage` - Segmentation model

**Note**: This model file is not included in the repository due to its large size.

### Obtaining the Model File

1. Download YOLO11 models from https://github.com/ultralytics/ultralytics
2. Convert it to CoreML format using Ultralytics:

```python
from ultralytics import YOLO

# Segmentation model
model = YOLO("yolo11x-seg.pt")
model.export(format="coreml")
```

3. Place the generated `.mlpackage` file in your Xcode project

### Testing Strategy

These tests verify:
- Model initialization and loading
- Image preprocessing and orientation correction
- Inference results validation
- UI component functionality

Some tests will be skipped if the required model files are not available.