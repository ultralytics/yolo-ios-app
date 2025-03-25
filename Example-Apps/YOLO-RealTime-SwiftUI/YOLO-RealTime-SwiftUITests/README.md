# YOLO RealTime SwiftUI Tests

This directory contains unit tests for the YOLO RealTime SwiftUI example application.

## Running Tests

### Prerequisites

To run these tests, you need to have the following CoreML model file:

- `yolo11n-obb.mlpackage` - Oriented Bounding Box (OBB) model

**Note**: This model file is not included in the repository due to its large size.

### Obtaining the Model File

1. Download YOLO11 models from https://github.com/ultralytics/ultralytics
2. Convert it to CoreML format using Ultralytics:

```python
from ultralytics import YOLO

# OBB model
model = YOLO("yolo11n-obb.pt")
model.export(format="coreml")
```

### Adding Model Files to the Project

**IMPORTANT**: The model file must be added to the **main application target** (YOLO-RealTime-SwiftUI), not just the test target.

Follow these steps to add the model file correctly:

1. Drag and drop `yolo11n-obb.mlpackage` into your Xcode project
2. In the dialog that appears, ensure the following:
   - Check the "YOLO-RealTime-SwiftUI" target (main app target)
   - You can also check the "YOLO-RealTime-SwiftUITests" target, but this alone is not sufficient
   - Select "Create folder references" option (blue folder icon)
3. Click "Finish" to add the model

For best organization, place the model file in a "Models" group in your project.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The YOLO framework looks for models in the main application bundle (Bundle.main), so models must be included in the main target for tests to work properly.

### Testing Strategy

These tests verify:

- Model initialization and loading
- Basic camera preview functionality
- UI layout and responsiveness

#### Running Tests Without Models

By default, `SKIP_MODEL_TESTS` is set to `true`, which allows running tests without requiring model files. This skips tests that depend on actual model inference but still verifies basic functionality of the application. This is ideal for CI environments or when you don't have the model files available.

#### Running Tests With Models

If you want to run the full test suite including model-dependent tests:

1. Add the required model files to the main application target as described above
2. Set `SKIP_MODEL_TESTS = false` in the test file
3. Run the tests again

This approach ensures the tests can verify both basic functionality and model integration without requiring large model files for basic testing.
