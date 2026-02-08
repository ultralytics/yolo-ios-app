<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO RealTime UIKit Tests

This directory contains [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for the YOLO RealTime UIKit example application. These tests are designed to ensure the reliability and functionality of the app's various components, contributing to a robust final product. You can find more information about building robust applications in the [Ultralytics documentation](https://docs.ultralytics.com/).

## 🧪 Running Tests

Follow these instructions to set up and run the tests for the application.

### Prerequisites

To execute the full test suite effectively, you will need the following [Apple Core ML](https://developer.apple.com/documentation/coreml) model file:

- `yolo26n.mlpackage`: An [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) model optimized for [object detection](https://docs.ultralytics.com/tasks/detect/) tasks.

**Note**: This model file is not included directly in the repository. Due to its potentially [large size](https://git-lfs.com/), including it could complicate version control management with Git.

### Obtaining the Model File

1.  Download pretrained [Ultralytics YOLO](https://docs.ultralytics.com/models/) models from the official [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
2.  Convert the downloaded model (e.g., `yolo26n.pt`) to the Core ML format using the [Ultralytics `export` mode](https://docs.ultralytics.com/modes/export/). See our [Core ML integration guide](https://docs.ultralytics.com/integrations/coreml/) for detailed instructions.

```python
from ultralytics import YOLO

# Load the YOLO26 nano detection model
model = YOLO("yolo26n.pt")

# Export the model to Core ML format
# See https://docs.ultralytics.com/integrations/coreml/ for more details
model.export(format="coreml")
```

### Adding Model Files to the Project

**IMPORTANT**: The model file (`yolo26n.mlpackage`) **must** be added to the **main application target** (`YOLO-RealTime-UIKit`) within your [Xcode project](https://developer.apple.com/xcode/), not just the test target. This ensures the model is correctly bundled and accessible by the main application.

Follow these steps to add the model file correctly:

1.  Drag and drop `yolo26n.mlpackage` into your Xcode project navigator.
2.  In the "Choose options for adding these files" dialog:
    - Ensure the checkbox for the **"YOLO-RealTime-UIKit" target** (the main app target) is **checked**.
    - Optionally, check the "YOLO-RealTime-UIKitTests" target, but remember the main target is crucial for the tests to access the model.
    - Select the "Create folder references" option (indicated by a blue folder icon). This helps maintain a clean project organization.
3.  Click "Finish" to complete the process.

For optimal project structure, consider placing the model file within a dedicated "Models" group in your Xcode project navigator.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The testing framework relies on accessing models from the main application bundle ([`Bundle.main`](https://developer.apple.com/documentation/foundation/bundle)). Therefore, including the models in the main target is essential for the model-dependent tests to function correctly.

### Testing Strategy

These unit tests are designed to verify several key aspects of the application:

- **Model Initialization and Loading**: Ensures models load correctly and are ready for inference.
- **Camera Session Configuration**: Validates the setup of the [AVCaptureSession](https://developer.apple.com/documentation/avfoundation/avcapturesession) for real-time video input.
- **UI Component Functionality**: Tests the behavior and state of [UIKit components](https://developer.apple.com/documentation/uikit) used in the app.
- **Real-time Inference Processing**: Checks the pipeline for processing frames and performing [real-time model inference](https://www.ultralytics.com/glossary/real-time-inference).

#### Running Tests Without Models

By default, the `SKIP_MODEL_TESTS` flag within the test files is set to `true`. This configuration allows you to run a subset of tests **without** needing the actual `.mlpackage` model files. These tests focus on verifying basic functionality, setup procedures, and UI interactions. This is useful for quick checks or within [Continuous Integration (CI)](https://docs.ultralytics.com/help/CI/) environments where large model files might not be readily available or necessary.

#### Running Tests With Models

To execute the **complete** test suite, including tests that require the model to perform actual inference:

1.  Ensure you have added the required `yolo26n.mlpackage` file to the **main application target** as detailed in the "Adding Model Files to the Project" section above.
2.  Locate the primary test file (e.g., `YOLORealTimeUIKitTests.swift`) and change the `SKIP_MODEL_TESTS` flag to `false`.
3.  Run the tests again using Xcode's testing tools (Shortcut: Cmd+U).

This tiered testing approach provides flexibility, ensuring that both fundamental application logic and the critical model integration can be thoroughly validated, depending on the availability of the model files.

## ✨ Contributing

Contributions to enhance the YOLO RealTime UIKit example application and its tests are highly encouraged! If you have suggestions, identify bugs, or want to propose improvements, please feel free to open an issue or submit a pull request in the main [Ultralytics repository](https://github.com/ultralytics/ultralytics). For more detailed guidance on contributing, please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/). Thank you for helping make Ultralytics better!
