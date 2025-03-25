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

### Adding Model Files to the Project

**IMPORTANT**: The model file must be added to the **main application target** (YOLO-Single-Image-SwiftUI), not just the test target.

Follow these steps to add the model file correctly:
1. Drag and drop `yolo11n-seg.mlpackage` into your Xcode project
2. In the dialog that appears, ensure the following:
   - Check the "YOLO-Single-Image-SwiftUI" target (main app target)
   - You can also check the "YOLO-Single-Image-SwiftUITests" target, but this alone is not sufficient
   - Select "Create folder references" option (blue folder icon)
3. Click "Finish" to add the model

For best organization, place the model file in a "Models" group in your project.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The YOLO framework looks for models in the main application bundle (Bundle.main), so models must be included in the main target for tests to work properly.

### Testing Strategy

These tests verify:
- Model initialization and loading
- SwiftUI view rendering and layout
- Image selection functionality
- Inference results handling

Some tests will be skipped if the required model files are not available. Set `SKIP_MODEL_TESTS = false` in the test file to run model-dependent tests once you've added the required model files.