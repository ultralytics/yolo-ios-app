<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Single Image SwiftUI Tests

This directory contains unit tests for the YOLO Single Image SwiftUI example application, designed to ensure the reliability and correctness of the app's features.

## 🧪 Running Tests

Follow these instructions to set up and run the unit tests for the application.

### Prerequisites

To execute these tests, you will need the following [Core ML](https://developer.apple.com/documentation/coreml) model file:

- `yolo11n-seg.mlpackage` - An Ultralytics YOLO11 [segmentation model](../tasks/segment.md).

**Note**: This model file is **not included** in the repository due to its large size. You must obtain and add it manually.

### Obtaining the Model File

1.  Download pretrained Ultralytics YOLO11 models from the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
2.  Convert the PyTorch model (`.pt`) to Core ML format (`.mlpackage`) using the Ultralytics `export` functionality:

```python
from ultralytics import YOLO

# Load the YOLO11 nano segmentation model
model = YOLO("yolo11n-seg.pt")

# Export the model to Core ML format
model.export(format="coreml")  # Creates yolo11n-seg.mlpackage
```

For more details on exporting models, refer to the [Ultralytics Export documentation](../modes/export.md).

### Adding Model Files to the Project

**IMPORTANT**: The `.mlpackage` model file must be added to the **main application target** (`YOLO-Single-Image-SwiftUI`) in Xcode, not just the test target. The testing framework relies on accessing the model through the main application bundle (`Bundle.main`).

Follow these steps to add the model file correctly using [Xcode](https://developer.apple.com/xcode/):

1.  Drag and drop the `yolo11n-seg.mlpackage` file into your Xcode project navigator.
2.  In the "Choose options for adding these files" dialog:
    - Ensure the checkbox for the **`YOLO-Single-Image-SwiftUI`** target (the main app) is selected.
    - You may optionally select the `YOLO-Single-Image-SwiftUITests` target, but the main target is essential.
    - Select the "Create folder references" option (this usually shows a blue folder icon).
3.  Click "Finish".

For better project organization, consider placing the model file within a "Models" group in your Xcode project structure.

![Adding model to target in Xcode](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

### Testing Strategy

These [unit tests](https://en.wikipedia.org/wiki/Unit_testing) verify several key aspects of the application:

- **Model Initialization**: Checks if the Core ML model loads correctly.
- **SwiftUI Views**: Ensures that the [SwiftUI](https://developer.apple.com/xcode/swiftui/) views render and handle layout as expected.
- **Image Selection**: Validates the functionality for selecting images from the device.
- **Inference Handling**: Tests how the application processes and displays the inference results from the YOLO model.

#### Running Tests Without Models (CI/Basic Checks)

By default, the `SKIP_MODEL_TESTS` flag in the test files is set to `true`. This configuration allows you to run a subset of tests **without requiring the actual model files**. These tests focus on verifying the basic UI functionality, view rendering, and non-inference logic. This setup is ideal for [Continuous Integration (CI)](https://www.atlassian.com/continuous-delivery/continuous-integration) environments or initial setup checks where model files might not be readily available.

#### Running Tests With Models (Full Suite)

To run the complete test suite, including tests that perform actual model inference:

1.  Ensure you have added the required `yolo11n-seg.mlpackage` file to the **main application target** as described in the "Adding Model Files" section.
2.  Locate the `SKIP_MODEL_TESTS` flag within the test source file (e.g., `YOLO_Single_Image_SwiftUITests.swift`) and set it to `false`.
3.  Run the tests again using Xcode's Test navigator (Cmd+U).

This comprehensive approach ensures that tests can validate both the fundamental application structure and the critical model integration and inference pathways, while still offering flexibility for environments without the large model files.

We welcome contributions to improve these tests! Please see the [Ultralytics Contributing Guidelines](https://docs.ultralytics.com/help/contributing/) for more information.
