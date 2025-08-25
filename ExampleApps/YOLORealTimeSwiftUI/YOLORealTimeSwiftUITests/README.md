<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO RealTime SwiftUI Tests

This directory contains [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for the YOLO RealTime SwiftUI example application. These tests are designed to verify the core functionalities of the app using [Xcode's testing framework](https://developer.apple.com/documentation/xctest).

## 🧪 Running Tests

### Prerequisites

To execute the complete test suite, including tests involving model [inference](https://www.ultralytics.com/glossary/real-time-inference), you need the following [Core ML](https://developer.apple.com/documentation/coreml) model file:

- `yolo11n-obb.mlpackage` - An [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) model optimized for Oriented Bounding Box ([OBB](https://docs.ultralytics.com/tasks/obb/)) detection. Learn more about OBB in our [documentation](https://docs.ultralytics.com/tasks/obb/).

**Note**: This model file is not included in the repository due to its size.

### Obtaining the Model File

1.  Download pretrained Ultralytics YOLO models from the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics). You can find various models on our [models page](https://docs.ultralytics.com/models/).
2.  Convert the desired [PyTorch](https://pytorch.org/) model (`.pt`) to Core ML format (`.mlpackage`) using the Ultralytics `export` mode. See the [Export documentation](https://docs.ultralytics.com/modes/export/) and our [Core ML integration guide](https://docs.ultralytics.com/integrations/coreml/) for more details.

```python
from ultralytics import YOLO

# Load a pretrained OBB model (e.g., yolo11n-obb.pt)
model = YOLO("yolo11n-obb.pt")

# Export the model to Core ML format
model.export(format="coreml")  # Creates yolo11n-obb.mlpackage
```

### Adding Model Files to the Project

**IMPORTANT**: The `.mlpackage` file must be added to the **main application target** (`YOLO-RealTime-SwiftUI`), not just the test target. The testing framework loads models from the main application's bundle (`Bundle.main`).

Follow these steps to add the model file correctly within Xcode:

1.  Drag and drop the `yolo11n-obb.mlpackage` file into your Xcode project navigator.
2.  In the "Choose options for adding these files" dialog:
    - Ensure the **"YOLO-RealTime-SwiftUI"** target checkbox is **checked**.
    - Optionally, check the "YOLO-RealTime-SwiftUITests" target, but the main target is essential.
    - Select the **"Create folder references"** option (indicated by a blue folder icon). This helps maintain the project structure.
    - Ensure **"Copy items if needed"** is checked.
3.  Click "Finish".

For better organization, consider placing the model file within a "Models" group in your project structure.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The application framework specifically looks for models within the main application [bundle](https://developer.apple.com/documentation/foundation/bundle), hence the requirement to include them in the main target for tests to access them correctly.

### Testing Strategy

These tests aim to verify several aspects of the application:

- **Model Initialization**: Checks if the Core ML model can be loaded correctly.
- **Camera Functionality**: Ensures the camera preview starts and functions as expected within the [SwiftUI](https://developer.apple.com/xcode/swiftui/) view, using the [Swift programming language](https://developer.apple.com/swift/).
- **UI Layout**: Verifies basic UI elements are present and responsive.
- **Inference (Optional)**: Performs basic checks on the model's inference output, including [bounding box](https://www.ultralytics.com/glossary/bounding-box) predictions, if the model is available.

#### Running Tests Without Models (Default)

By default, the `SKIP_MODEL_TESTS` flag in the test file is set to `true`. This configuration allows you to run the tests **without** needing the `yolo11n-obb.mlpackage` file. Tests that depend on actual model inference will be skipped, but basic application functionality (UI, camera setup) will still be verified. This is useful for quick checks or in [Continuous Integration (CI)](https://en.wikipedia.org/wiki/Continuous_integration) environments where managing large model files might be complex. Check our [Quickstart guide](https://docs.ultralytics.com/quickstart/) for setting up your environment.

#### Running Tests With Models

To run the full test suite, including tests that perform inference:

1.  Ensure you have obtained `yolo11n-obb.mlpackage` and added it to the **main application target** as described above.
2.  Open the relevant test file (e.g., `YOLO_RealTime_SwiftUITests.swift`).
3.  Change the flag `SKIP_MODEL_TESTS` to `false`.
4.  Run the tests using Xcode (Product > Test or `Cmd+U`).

This flexible approach ensures that both core application logic and [model deployment](https://www.ultralytics.com/glossary/model-deployment) integration can be tested effectively, accommodating different development and testing scenarios. Explore more terms in the [Ultralytics Glossary](https://www.ultralytics.com/glossary).

## 🤝 Contributing

Contributions to improve the tests or the application are welcome! If you find issues or have suggestions for improvements, please open an issue or submit a pull request on the main repository. See the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get started.
