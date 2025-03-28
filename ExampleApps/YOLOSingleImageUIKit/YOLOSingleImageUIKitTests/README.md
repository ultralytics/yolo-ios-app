<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Single Image UIKit Tests

This directory contains the [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for the YOLO Single Image UIKit example application, designed to ensure the reliability and correctness of the app's functionality.

## 🧪 Running Tests

Follow these instructions to set up and run the unit tests for the application.

### Prerequisites

To execute the complete test suite, you will need the following [Core ML](https://developer.apple.com/documentation/coreml) model file:

-   `yolo11x-seg.mlpackage`: An [Ultralytics YOLO11](../models/yolo11.md) segmentation model.

**Note**: This model file is not included in the repository due to its significant size. Large files are often excluded from version control to keep repository size manageable.

### Obtaining the Model File

1.  **Download**: Obtain the base PyTorch YOLO11 model (`yolo11x-seg.pt`) from the [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases) or train your own.
2.  **Convert**: Convert the PyTorch model to the Core ML format using the Ultralytics Python package. Detailed instructions can be found in our [Core ML export documentation](https://docs.ultralytics.com/integrations/coreml/).

```python
from ultralytics import YOLO

# Load the YOLO11 segmentation model
model = YOLO("yolo11x-seg.pt")

# Export the model to Core ML format
# This will create the yolo11x-seg.mlpackage file
model.export(format="coreml")
```

### Adding Model Files to the Project

**IMPORTANT**: The Core ML model file (`.mlpackage`) must be added to the **main application target** (named `YOLO-Single-Image-UIKit`), not just the test target.

Follow these steps carefully:

1.  Drag and drop the generated `yolo11x-seg.mlpackage` file into your Xcode project navigator.
2.  In the "Choose options for adding these files" dialog:
    -   Ensure the checkbox next to the **`YOLO-Single-Image-UIKit`** target is checked. This is crucial.
    -   You may optionally check the `YOLO-Single-Image-UIKitTests` target as well, but including it only in the test target is insufficient.
    -   Select the "Create folder references" option (indicated by a blue folder icon) for better project organization.
3.  Click "Finish".

Consider placing the model file within a "Models" group in your Xcode project for clarity.

![Adding model to target](https://docs-assets.developer.apple.com/published/abd9789384/ff4127a0-80a6-4716-b1cd-fc1facce5d8e.png)

The reason the model must be part of the main application target is that the YOLO framework code within the app loads the model from the main application [bundle](https://developer.apple.com/documentation/foundation/bundle) (`Bundle.main`). Tests run within the context of the app, thus requiring the model to be accessible via this main bundle.

### Testing Strategy

These unit tests are designed to verify several key aspects of the application:

-   **Model Handling**: Correct initialization and loading of the Core ML model.
-   **Preprocessing**: Accurate image preprocessing steps, including orientation correction.
-   **Inference**: Validation of the inference results against expected outputs (when models are present).
-   **UI**: Basic functionality checks for relevant UI components.

#### Running Tests Without Models

By default, the test suite is configured to run *without* requiring the actual model files. This is controlled by the `SKIP_MODEL_TESTS` flag within the test code, which is set to `true`.

-   **Benefits**: This allows developers to quickly verify the core application logic, UI interactions, and preprocessing steps without needing to download and manage large model files. It's particularly useful for [Continuous Integration (CI)](https://en.wikipedia.org/wiki/Continuous_integration) pipelines where efficiency is key.
-   **Limitations**: Tests that specifically depend on running inference with the model will be skipped.

#### Running Tests With Models

To run the full test suite, including tests that perform actual model inference:

1.  **Add Models**: Ensure you have obtained and added the required `yolo11x-seg.mlpackage` file to the **main application target** as described in the "Adding Model Files to the Project" section.
2.  **Modify Flag**: Open the relevant test file (e.g., `YOLO_Single_Image_UIKitTests.swift`) and change the flag `SKIP_MODEL_TESTS` to `false`.
3.  **Run Tests**: Execute the tests again through Xcode (Product > Test or Command+U).

This comprehensive approach ensures that both the fundamental application structure and the critical model integration points are thoroughly tested, while still offering a lightweight option for basic checks and CI environments. For more information on deploying models, check out our guide on [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).

## 🤝 Contributing

Contributions to improve the tests or the example application are welcome! Please see the main [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for contribution guidelines. Feel free to submit issues or pull requests.
