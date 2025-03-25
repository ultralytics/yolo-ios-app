<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 Ultralytics YOLO iOS App

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml) <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

The Yolo-iOS-App is designed to make it easy to try out the YOLO model [YOLO11 object detection models](https://github.com/ultralytics/ultralytics) on your Apple device.

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
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
</div>

## 🛠 Quickstart: Setting Up the Ultralytics YOLO iOS App

Getting started with the Ultralytics YOLO iOS App is straightforward. Follow these steps to install the app on your iOS device.

### Prerequisites

Ensure you have the following before you start:

- **Xcode:** The Ultralytics YOLO iOS App requires Xcode installed on your macOS machine. Download it from the [Mac App Store](https://apps.apple.com/us/app/xcode/id497799835).

- **An iOS Device:** For testing the app, you'll need an iPhone or iPad running [iOS 14.0](https://www.apple.com/ios/ios-18/) or later.

- **An Apple Developer Account:** A free Apple Developer account will suffice for device testing. Sign up [here](https://developer.apple.com/) if you haven't already.

### Installation

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/ultralytics/yolo-ios-app.git
   ```

2. **Open the Project in Xcode:**

   Navigate to the cloned directory and open the `YOLO.xcodeproj` file.

   <p align="center">
   <img width="50%" src="https://github.com/ultralytics/ultralytics/assets/26833433/e0053238-4a7c-4d18-8720-6ce24c73dea0" alt="XCode load project screenshot">
   </p>

   In Xcode, go to the project's target settings and choose your Apple Developer account under the "Signing & Capabilities" tab.

3. **Add YOLO11 Models to the Project:**

   Export CoreML INT8 models using the `ultralytics` Python package (with `pip install ultralytics`), or download them from our [GitHub release assets](https://github.com/ultralytics/yolo-ios-app/releases). You should have 5 YOLO11 models in total. Place these in the `YOLO/{TaskName}Models` directory as seen in the Xcode screenshot below.

   ```python
   from ultralytics import YOLO

   # Loop through all YOLO11 model sizes
   for size in ("n", "s", "m", "l", "x"):
       # Load a YOLO11 PyTorch model
       model = YOLO(f"yolo11{size}.pt")

       # model = YOLO(f"yolo11{size}-seg.pt") # segment model
       # model = YOLO(f"yolo11{size}-cls.pt") # classify model
       # model = YOLO(f"yolo11{size}-pose.pt") # pose model
       # model = YOLO(f"yolo11{size}-obb.pt") # obb model

       # Export the PyTorch model to CoreML INT8 format with NMS layers
       model.export(format="coreml", int8=True, nms=True, imgsz=[640, 384])
   ```

4. **Run the Ultralytics YOLO iOS App:**

   Connect your iOS device and select it as the run target. Press the Run button to install the app on your device.

   <p align="center">
   <img width="100%" src="https://github.com/ultralytics/ultralytics/assets/26833433/d2c6a7b7-fa8b-4130-a57f-4241f7a42ff2" alt="Ultralytics YOLO XCode screenshot">
   </p>

## 🚀 Usage

The Ultralytics YOLO iOS App is designed to be intuitive:

- **Real-Time Detection:** Launch the app and aim your camera at objects to detect them instantly.
- **Frexible Tasks** Select the task you want to use, such as detect, segment, pause, etc.
- **Multiple AI Models:** Select from a range of Ultralytics YOLO11 models, from YOLO11n 'nano' to YOLO11x 'x-large'. You can also deploy and use custom models that you have trained and exported.

## 🧪 Testing

The YOLO iOS App includes comprehensive testing to ensure reliability:

### Model Testing

The app has been designed with testability in mind. Tests can be run with or without CoreML models:

- When `SKIP_MODEL_TESTS = true`, all tests not requiring models will run successfully
- To run full test suite, set `SKIP_MODEL_TESTS = false` and add model files to the project

### Required Models for Testing

To enable full test coverage, you'll need:

- **Detection model**: `yolo11n.mlpackage`
- **Segmentation model**: `yolo11n-seg.mlpackage`
- **Pose estimation model**: `yolo11n-pose.mlpackage` 
- **OBB model**: `yolo11n-obb.mlpackage`
- **Classification model**: `yolo11n-cls.mlpackage`

These files should be placed in the appropriate model directories within the project.

### Running Tests

1. Open the Xcode project
2. Select the test navigator tab
3. Choose which tests to run (individual tests or full test suite)
4. Press the Run button to execute tests

See the test files in each example app directory for specific test implementations.

## 💡 Contribute

We warmly welcome your contributions to Ultralytics' open-source projects! Your support and contributions significantly impact. Get involved by reviewing our [Contributing Guide](https://docs.ultralytics.com/help/contributing/), and share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A massive thank you 🙏 to everyone who contributes!

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## 📄 License

Ultralytics offers two licensing options:

- **AGPL-3.0 License**: An [OSI-approved](https://opensource.org/license) open-source license, perfect for academics, researchers, and enthusiasts. It encourages sharing knowledge and collaboration. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.

- **Enterprise License**: Designed for commercial use, this license permits integrating Ultralytics software into proprietary products and services. For commercial use, please contact us through [Ultralytics Licensing](https://www.ultralytics.com/license).

## 🤝 Contact

- Submit Ultralytics bug reports and feature requests via [GitHub Issues](https://github.com/ultralytics/yolo-ios-app/issues).
- Join our [Discord](https://discord.com/invite/ultralytics) for assistance, questions, and discussions with the community and team!

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
