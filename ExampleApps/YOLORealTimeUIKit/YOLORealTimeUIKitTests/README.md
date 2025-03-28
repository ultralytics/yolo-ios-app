<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO RealTime UIKit Tests

This directory contains [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for the YOLO RealTime UIKit example application, designed to ensure the reliability and functionality of the app's components.

## 🧪 Running Tests

Follow these instructions to set up and run the tests for the application.

### Prerequisites

To execute these tests effectively, you will need the following [Apple Core ML](https://developer.apple.com/documentation/coreml) model file:

- `yolo11n.mlpackage` - An [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) [object detection](https://docs.ultralytics.com/tasks/detect/) model.

**Note**: This model file is not included in the repository due to its potentially [large size](https://git-lfs.com/), which can complicate version control.

### Obtaining the Model File

1.  Download [Ultralytics YOLO](https://docs.ultralytics.com/models/) models from the official [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
2.  Convert the downloaded model (e.g., `yolo11n.pt`) to the Core ML format using the [Ultralytics `export` mode](https://docs.ultralytics.com/modes/export/):

```python
from ultralytics import YOLO

# Load the YOLO11 nano detection model
model = YOLO("yolo11n.pt")

# Export the model to Core ML format
# See https://docs.ultralytics.com/integrations/coreml/ for more details
model.export(format="coreml")
```

### Adding Model Files to the Project

**IMPORTANT**: The model file (`yolo11n.mlpackage`) must be added to the **main application target** (`YOLO-RealTime-UIKit`) within your [Xcode project](https://developer.apple.com/xcode/), not just the test target. This ensures the model is accessible within the main application bundle.

Follow these steps to add the model file correctly:

1.  Drag and drop `yolo11n.mlpackage` into your Xcode project navigator.
2.  In the "Choose options for adding these files" dialog:
    - Ensure the checkbox for the "YOLO-RealTime-UIKit" target (the main app target) is checked.
    - Optionally, check the "YOLO-RealTime-UIKitTests" target, but remember the main target is crucial.
    - Select the "Create folder references" option (indicated by a blue folder icon). This helps maintain project organization.
3.  Click "Finish" to complete the process.

For optimal project structure, consider placing the model file within a dedicated "Models" group in your Xcode project.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The testing framework relies on accessing models from the main application bundle ([`Bundle.main`](https://developer.apple.com/documentation/foundation/bundle)), hence the requirement to include models in the main target for tests to function correctly.

### Testing Strategy

These unit tests are designed to verify several key aspects of the application:

- **Model Initialization and Loading**: Ensures models load correctly and are ready for inference.
- **Camera Session Configuration**: Validates the setup of the [camera session](https://developer.apple.com/documentation/avfoundation/avcapturesession) for real-time video input.
- **UI Component Functionality**: Tests the behavior and state of [UI components](https://developer.apple.com/documentation/uikit) used in the app.
- **Real-time Inference Processing**: Checks the pipeline for processing frames and performing [model inference](https://www.ultralytics.com/glossary/real-time-inference).

#### Running Tests Without Models

By default, the `SKIP_MODEL_TESTS` flag in the test files is set to `true`. This configuration allows you to run a subset of tests without needing the actual `.mlpackage` model files. These tests focus on verifying the basic functionality, setup, and UI interactions of the application, making them suitable for quick checks or [Continuous Integration (CI)](https://docs.ultralytics.com/help/CI/) environments where model files might not be readily available.

#### Running Tests With Models

To execute the complete test suite, including tests that depend on the actual model performing inference:

1.  Ensure you have added the required `yolo11n.mlpackage` file to the **main application target** as detailed in the "Adding Model Files to the Project" section.
2.  Locate the test file (e.g., `YOLORealTimeUIKitTests.swift`) and change the `SKIP_MODEL_TESTS` flag to `false`.
3.  Run the tests again using Xcode's testing tools (Cmd+U).

This tiered testing approach ensures that both fundamental application logic and model integration can be thoroughly validated, offering flexibility based on whether the large model files are present.

## ✨ Contributing

Contributions to enhance the YOLO RealTime UIKit example application and its tests are welcome! If you have suggestions, bug fixes, or improvements, please feel free to open an issue or submit a pull request in the [Ultralytics repository](https://github.com/ultralytics/ultralytics). For more detailed guidance, see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/). Thank you for helping improve Ultralytics!
