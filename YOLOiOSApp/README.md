<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO iOS App

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

The Ultralytics YOLO iOS App makes it easy to experience the power of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) object detection models directly on your Apple device. Explore real-time detection capabilities with various models, bringing state-of-the-art [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) to your fingertips.

<div align="center">
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" target="_blank"><img width="90%" src="https://github.com/ultralytics/ultralytics/assets/26833433/fd3c8a92-fec0-4253-b4ac-ee94f5ced3fb" alt="Ultralytics YOLO iOS App previews"></a>
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Download on the Apple App Store"></a>
</div>

## 🛠️ Quickstart: Setting Up the Ultralytics YOLO iOS App

Getting started with the Ultralytics YOLO iOS App is straightforward. Follow these steps to install and run the app on your iOS device.

### Prerequisites

Ensure you have the following before you begin:

- **Xcode:** The app requires [Xcode](https://developer.apple.com/xcode/) installed on your macOS machine. You can download it from the [Mac App Store](https://apps.apple.com/us/app/xcode/id497799835).
- **iOS Device:** An iPhone or iPad running [iOS 14.0](https://support.apple.com/guide/iphone/iphone-models-compatible-with-ios-18-iphe3fa5df43/ios) or later is needed for testing.
- **Apple Developer Account:** A free [Apple Developer account](https://developer.apple.com/programs/enroll/) is sufficient for testing on your device.

### Installation

1.  **Clone the Repository:**
    Use `git` to clone the repository to your local machine.

    ```sh
    git clone https://github.com/ultralytics/yolo-ios-app.git
    cd yolo-ios-app # Navigate into the cloned directory
    ```

2.  **Open the Project in Xcode:**
    Locate the `YOLO.xcodeproj` file within the cloned directory and open it using Xcode.

    <p align="center">
    <img width="50%" src="https://github.com/ultralytics/ultralytics/assets/26833433/e0053238-4a7c-4d18-8720-6ce24c73dea0" alt="Xcode project structure showing YOLO.xcodeproj">
    </p>

    In Xcode, navigate to the project's target settings. Under the "Signing & Capabilities" tab, select your Apple Developer account to sign the app.

3.  **Add YOLO11 Models:**
    You need models in the [CoreML](https://developer.apple.com/documentation/coreml) format to run inference on iOS. Export INT8 quantized CoreML models using the `ultralytics` Python package (install via `pip install ultralytics` - see our [Quickstart Guide](https://docs.ultralytics.com/quickstart/)) or download pre-exported models from our [GitHub release assets](https://github.com/ultralytics/yolo-ios-app/releases). Place the `.mlpackage` files into the corresponding `YOLO/{TaskName}Models` directory within the Xcode project (e.g., `YOLO/DetectModels`). Refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/) for more details on exporting models for various deployment environments.

    ```python
    from ultralytics import YOLO
    from ultralytics.utils.downloads import zip_directory


    def export_and_zip_yolo_models(
        model_types=("", "-seg", "-cls", "-pose", "-obb"),
        model_sizes=("n", "s", "m", "l", "x"),
    ):
        """Exports YOLO11 models to CoreML format and optionally zips the output packages."""
        for model_type in model_types:
            imgsz = [224, 224] if "cls" in model_type else [640, 384]  # default input image sizes
            nms = True if model_type == "" else False  # only apply NMS to Detect models
            for size in model_sizes:
                model_name = f"yolo11{size}{model_type}"
                model = YOLO(f"{model_name}.pt")
                model.export(format="coreml", int8=True, imgsz=imgsz, nms=nms)
                zip_directory(f"{model_name}.mlpackage").rename(f"{model_name}.mlpackage.zip")


    # Execute with default parameters
    export_and_zip_yolo_models()
    ```

4.  **Run the App:**
    Connect your iOS device via USB. Select your device from the list of run targets in Xcode (next to the stop button). Click the Run button (▶) to build and install the app on your device.

    <p align="center">
    <img width="100%" src="https://github.com/ultralytics/ultralytics/assets/26833433/d2c6a7b7-fa8b-4130-a57f-4241f7a42ff2" alt="Xcode interface showing run target selection and model directories">
    </p>

## 🚀 Usage

The Ultralytics YOLO iOS App offers an intuitive user experience:

- **Real-Time Inference:** Launch the app and point your device's camera at objects. The app will perform real-time [object detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [image classification](https://docs.ultralytics.com/tasks/classify/), or [oriented bounding box detection](https://docs.ultralytics.com/tasks/obb/) depending on the selected task and model.
- **Flexible Task Selection:** Easily switch between different computer vision tasks supported by the loaded models using the app's interface.
- **Multiple AI Models:** Choose from a range of pre-loaded Ultralytics YOLO11 models, from the lightweight YOLO11n ('nano') optimized for edge devices to the powerful YOLO11x ('x-large') for maximum accuracy. You can also deploy and use [custom models](https://docs.ultralytics.com/hub/quickstart/) trained on your own data after exporting them to the CoreML format.

## 🧪 Testing

The YOLO iOS App includes a suite of unit and integration tests to ensure functionality and reliability.

### Model Testing Configuration

The test suite is designed to run with or without the actual CoreML model files:

- **Without Models:** Set `SKIP_MODEL_TESTS = true` in the test target's build settings (under `Build Settings` > `User-Defined`). This allows running tests that don't require model inference, such as UI tests or utility function tests.
- **With Models:** Set `SKIP_MODEL_TESTS = false` and ensure the required model files are added to the project in their respective directories as described below. This enables the full test suite, including tests that perform actual model inference.

### Required Models for Full Testing

To execute the complete test suite (with `SKIP_MODEL_TESTS = false`), include the following **INT8 quantized CoreML models** in your project:

- **Detection:** `yolo11n.mlpackage` (place in `YOLO/DetectModels`)
- **Segmentation:** `yolo11n-seg.mlpackage` (place in `YOLO/SegmentModels`)
- **Pose Estimation:** `yolo11n-pose.mlpackage` (place in `YOLO/PoseModels`)
- **OBB Detection:** `yolo11n-obb.mlpackage` (place in `YOLO/OBBModels`)
- **Classification:** `yolo11n-cls.mlpackage` (place in `YOLO/ClassifyModels`)

You can export these models using the Python script provided in the [Installation section](#add-yolo11-models) or download them directly from the [releases page](https://github.com/ultralytics/yolo-ios-app/releases).

### Running Tests in Xcode

1.  Open the `YOLO.xcodeproj` project in Xcode.
2.  Navigate to the Test Navigator tab (represented by a diamond icon) in the left sidebar.
3.  Select the tests you wish to run (e.g., the entire `YOLOTests` suite or individual test functions).
4.  Click the Run button (play icon) next to your selection to execute the tests on a connected device or simulator.

Review the test files located within the `YOLOTests` directory for specific implementation details and test coverage.

## 💡 Contribute

Contributions power the open-source community! We welcome your involvement in improving Ultralytics projects. Your efforts, whether reporting bugs via [GitHub Issues](https://github.com/ultralytics/yolo-ios-app/issues), suggesting features, or submitting code through Pull Requests, are greatly appreciated.

- Check out our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for detailed instructions on how to get involved.
- Share your feedback and insights on your experience by participating in our brief [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey).
- A big thank you 🙏 to all our contributors who help make our projects better!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics provides two licensing options to accommodate different use cases:

- **AGPL-3.0 License:** Ideal for students, researchers, and enthusiasts who want to experiment, learn, and share their work openly. This [OSI-approved](https://opensource.org/license/agpl-v3/) license promotes collaboration and knowledge sharing within the open-source community. See the [LICENSE](https://github.com/ultralytics/yolo-ios-app/blob/main/LICENSE) file for the full terms and conditions.
- **Enterprise License:** Designed for commercial applications where integrating Ultralytics software into proprietary products or services is necessary. This license allows for commercial use without the open-source requirements of AGPL-3.0. If your project requires an Enterprise License, please contact us through the [Ultralytics Licensing](https://www.ultralytics.com/license) page.

## 🤝 Contact

For bug reports, feature requests, and contributions specifically related to the YOLO iOS App:

- Submit issues on [GitHub Issues](https://github.com/ultralytics/yolo-ios-app/issues). Please provide detailed information for effective troubleshooting.

For general questions, support, and discussions about Ultralytics YOLO models, software, and other projects:

- Join our vibrant community on [Discord](https://discord.com/invite/ultralytics). Connect with other users and the Ultralytics team.

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
