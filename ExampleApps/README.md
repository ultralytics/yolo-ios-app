<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Package Examples

Welcome! This directory contains example Xcode projects demonstrating how to integrate the Ultralytics YOLO Package into your iOS applications. Explore how to leverage [Core ML](https://developer.apple.com/documentation/coreml) models for various computer vision tasks directly on Apple devices.

## 🚀 Examples

We provide several sample apps built with both [SwiftUI](https://developer.apple.com/xcode/swiftui/) and [UIKit](https://developer.apple.com/documentation/uikit) to illustrate different use cases:

### YOLO-Single-Image-SwiftUI

- A straightforward SwiftUI application demonstrating inference on a single image selected from the user's photo library using an Ultralytics YOLO Core ML model.

### YOLO-Single-Image-UIKit

- A simple UIKit application showcasing inference on a single image chosen from the photo library, powered by an Ultralytics YOLO Core ML model.

### YOLO-RealTime-SwiftUI

- An example SwiftUI app performing real-time object detection, segmentation, or other tasks using the device's camera feed.

### YOLO-RealTime-UIKit

- An example UIKit app implementing real-time inference from the live camera stream.

## 🛠️ Usage

Follow these steps to get the examples up and running:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ultralytics/yolo-ios-app.git
    cd yolo-ios-app/ExampleApps
    ```

2.  **Open an Example Project:** Navigate to the desired example directory (e.g., `YOLO-RealTime-SwiftUI`) and open the `.xcodeproj` file in [Xcode](https://developer.apple.com/xcode/).

3.  **Add a YOLO Core ML Model:** Drag and drop your chosen `.mlpackage` or `.mlmodel` file into the Xcode project navigator. Ensure it's added to the target membership of the application.

    #### Obtaining YOLO Core ML Models

    You have two primary ways to get Ultralytics YOLO models in Core ML format:

    -   **Download Pre-Exported Models:** Download optimized Core ML INT8 models directly from the [Ultralytics YOLOv8 GitHub releases](https://github.com/ultralytics/ultralytics/releases) (or other YOLO versions). Place the downloaded model file into your Xcode project.
    -   **Export Your Own Models:** Use the `ultralytics` Python package to export models tailored to your needs. This offers flexibility in choosing model types and configurations.
        -   Install the package:
            ```bash
            pip install ultralytics
            ```
        -   Run a Python script to export:
            ```python
            from ultralytics import YOLO

            # Example: Export various YOLOv8 models
            # Replace 'yolov8n.pt' with the specific model you need (e.g., yolov8n-seg.pt)
            # See https://docs.ultralytics.com/models/ for available models

            # Load a model (Detection, Segmentation, Pose, Classification, OBB)
            # model = YOLO("yolov8n.pt") # Detection
            # model = YOLO("yolov8n-seg.pt") # Segmentation
            # model = YOLO("yolov8n-pose.pt") # Pose
            model = YOLO("yolov8n-cls.pt") # Classification
            # model = YOLO("yolov8n-obb.pt") # Oriented Bounding Boxes

            # Export to Core ML INT8 format
            # NMS is typically needed only for Detection models when used standalone.
            # The YOLO package handles NMS internally for other tasks.
            # Refer to Ultralytics export docs: https://docs.ultralytics.com/modes/export/
            model.export(format="coreml", int8=True, nms=False, imgsz=640) # Adjust nms=True for detection if needed

            print(f"Model exported to {model.export_dir}")
            ```
        -   Locate the exported `.mlpackage` file in the specified directory and add it to your Xcode project.

4.  **Configure Signing:** In the project settings under "Signing & Capabilities", select your development team.

5.  **Build and Run:** Select a connected physical iOS device (iPhone or iPad) and click the Run button (▶) in Xcode.

**Important Notes:**

-   Real-time examples require a physical device with a camera; they **cannot** be run on the iOS Simulator.
-   The examples use local Swift Package dependencies. Opening multiple example projects simultaneously might cause Xcode to have trouble resolving these packages. Please open and work with one example project at a time.

## ✅ Testing

Each example app includes unit tests to verify its core functionality. These tests are located within the corresponding `Tests` directory (e.g., `YOLO-Single-Image-SwiftUITests`).

### Running Tests

1.  Open the desired example project in Xcode.
2.  Select the Test navigator (diamond icon).
3.  Choose the test target (e.g., `YOLO-Single-Image-SwiftUITests`).
4.  Run tests using `Cmd+U` or by clicking the play button next to the test target or individual tests.

### Test Configuration

-   **Without Models:** By default, tests are configured to skip model-dependent checks (`SKIP_MODEL_TESTS = true` in the test files). This allows verifying basic UI and logic without needing the potentially large Core ML model files, useful for quick checks and some [CI](https://www.ultralytics.com/glossary/continuous-integration-ci) setups.
-   **With Models:** To perform comprehensive testing including model inference:
    1.  Ensure the required Core ML model(s) are added to the main application target (refer to the specific `README.md` within each test directory for details on required models).
    2.  Set `SKIP_MODEL_TESTS = false` within the relevant test file(s).
    3.  Run the tests again.

Please consult the `README.md` file inside each example's `Tests` directory for detailed instructions on required models, obtaining/exporting them, and the specific functionalities covered by the tests.

## 🤝 Contributing

Contributions to enhance these examples or add new ones are welcome! Please see the main [Ultralytics Contribution Guide](https://docs.ultralytics.com/help/contributing/) for guidelines on how to contribute to our open-source projects. Let's build amazing vision AI applications together!
