# YOLO Single Image SwiftUI Tests

This directory contains unit tests for the YOLO Single Image SwiftUI example application.

## Running Tests

### Prerequisites

To run these tests, you need to have the following CoreML model file:

- `yolo11n-seg.mlpackage` - Segmentation model

**Note**: This model file is not included in the repository due to its large size.

### Obtaining the Model File

1. Download YOLO11 models from https://github.com/ultralytics/ultralytics
2. Convert it to CoreML format using Ultralytics:

```python
from ultralytics import YOLO

# Segmentation model
model = YOLO("yolo11n-seg.pt")
model.export(format="coreml")
```

3. Place the generated `.mlpackage` file in your Xcode project

### Testing Strategy

These tests verify:
- Model initialization and loading
- SwiftUI view rendering and layout
- Image selection functionality
- Inference results handling

Some tests will be skipped if the required model files are not available.