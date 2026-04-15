<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Single Image UIKit Tests

This directory contains the [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for the YOLO Single Image UIKit example application, designed to ensure the reliability and correctness of the app's functionality.

## 🧪 Running Tests

Follow these instructions to set up and run the unit tests for the application.

### Prerequisites

To execute the complete test suite, you will need the [Core ML](https://developer.apple.com/documentation/coreml) model the example app loads by default:

- `yolo26n.mlpackage`: An [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) detection model.

**Note**: This model file is not included in the repository due to its significant size. Large files are often excluded from [version control](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_removing_a_file_from_every_commit) to keep repository size manageable.

### Obtaining the Model File

1.  **Download**: Obtain the base PyTorch YOLO26 model (`yolo26n.pt`) from the [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases) or train your own following our [model training tips](https://docs.ultralytics.com/guides/model-training-tips/).
2.  **Convert**: Convert the PyTorch model to the Core ML format using the Ultralytics Python package. Detailed instructions can be found in our [Core ML export documentation](https://docs.ultralytics.com/integrations/coreml/).

```python
from ultralytics import YOLO

# Load the YOLO26 detection model
model = YOLO("yolo26n.pt")

# Export the model to Core ML format
# This will create the yolo26n.mlpackage file
model.export(format="coreml")
```

### Adding Model Files to the Project

**IMPORTANT**: The Core ML model file (`.mlpackage`) must be added to the **main application target** (named `YOLO-Single-Image-UIKit`), not just the test target.

Follow these steps carefully:

1.  Drag and drop the generated `yolo26n.mlpackage` file into your Xcode project navigator.
2.  In the "Choose options for adding these files" dialog:
    - Ensure the checkbox next to the **`YOLO-Single-Image-UIKit`** target is checked. This is crucial.
    - You may optionally check the `YOLO-Single-Image-UIKitTests` target as well, but including it only in the test target is insufficient.
    - Select the "Create folder references" option (indicated by a blue folder icon) for better project organization.
3.  Click "Finish".

Consider placing the model file within a "Models" group in your Xcode project for clarity.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The reason the model must be part of the main application target is that the YOLO framework code within the app loads the model from the main application [bundle](https://developer.apple.com/documentation/foundation/bundle) (`Bundle.main`). Tests run within the context of the app, thus requiring the model to be accessible via this main bundle.

### Testing Strategy

These unit tests are designed to verify several key aspects of the application:

- **Model Handling**: Correct initialization and loading of the Core ML model.
- **Preprocessing**: Accurate image preprocessing steps, including orientation correction.
- **Inference**: Validation of the inference results against expected outputs (when models are present). You can learn more about inference in our [Predict Mode documentation](https://docs.ultralytics.com/modes/predict/).
- **UI**: Basic functionality checks for relevant UI components.

#### Running Tests Without Models

The test file currently sets `SKIP_MODEL_TESTS = false`, so model-dependent checks run by default.

- **Benefits**: If you switch `SKIP_MODEL_TESTS` to `true`, developers can quickly verify the core application logic, UI interactions, and preprocessing steps without needing to download and manage large model files. It's particularly useful for [Continuous Integration (CI)](https://en.wikipedia.org/wiki/Continuous_integration) pipelines where efficiency is key. Check our [CI guide](https://docs.ultralytics.com/help/CI/) for more details.
- **Limitations**: Tests that specifically depend on running inference with the model will be skipped.

#### Running Tests With Models

To run the full test suite, including tests that perform actual model inference:

1.  **Add Models**: Ensure you have obtained and added the required `yolo26n.mlpackage` file to the **main application target** as described in the "Adding Model Files to the Project" section.
2.  **Modify Flag**: Open the relevant test file (e.g., `YOLO_Single_Image_UIKitTests.swift`) and change the flag `SKIP_MODEL_TESTS` to `false`.
3.  **Run Tests**: Execute the tests again through [Xcode](https://developer.apple.com/xcode/) (Product > Test or Command+U).

This comprehensive approach ensures that both the fundamental application structure and the critical model integration points are thoroughly tested, while still offering a lightweight option for basic checks and CI environments. For more information on deploying models, check out our guide on [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options/) and explore platforms like [Ultralytics Platform](https://platform.ultralytics.com).

## 🤝 Contributing

Contributions to improve the tests or the example application are welcome! Please see the main [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for contribution guidelines. Feel free to submit [issues](https://github.com/ultralytics/yolo-ios-app/issues) or [pull requests](https://github.com/ultralytics/yolo-ios-app/pulls).
