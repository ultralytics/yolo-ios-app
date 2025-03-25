<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 Ultralytics YOLO for iOS: App and Swift Package

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml) <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

Welcome to the [Ultralytics YOLO iOS App](https://apps.apple.com/us/app/idetection/id1452689527) GitHub repository! 📖 Leveraging Ultralytics' advanced [YOLO11 models](https://github.com/ultralytics/ultralytics), this repository transforms your iOS device into an intelligent detection tool. Explore our guide to get started with the Ultralytics YOLO iOS App and discover the world in a new and exciting way.

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

## 📂 Content

This repository is a comprehensive project that includes:

### [**Ultralytics YOLO iOS App (Main App)**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)

A primary iOS application that allows easy real-time object detection on iOS devices. Simply drag and drop your custom model to use it in the app.

### [**Swift Package (YOLO Library)**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/YOLO)

A lightweight library for iOS, iPadOS, and macOS that simplifies working with YOLO-based models like YOLO11.
Easily integrate YOLO models into your app with a single line of code:

```swift
let result = model(uiImage)
```

```swift
var body: some View {
    YOLOCamera(
        modelPathOrName: "yolo11n-seg",
        task: .segment,
        cameraPosition: .back
    )
    .ignoresSafeArea()
}
```

## 🛠 Quickstart Guide

If you're new to YOLO on mobile or just want to test your own model, we recommend starting with the main YOLOiOSApp.

[**Ultralytics YOLO iOS App (Main App)**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)

If you'd like to integrate YOLO into your own app, check out the Swift Package and example usage.

[**Swift Package (YOLO Library)**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/YOLO)

[**Example Apps**](https://github.com/ultralytics/yolo-ios-app/tree/main/ExampleApps)

## ✨ Highlights

**Real-Time Inference**

Achieve high-speed, high-accuracy object detection on iPhones and iPads using CoreML models.

**Multi OS Support**

The Swift Package supports iOS, iPadOS, and macOS.

**Flexible Tasks**

Supports object detection, with segmentation, classification, pose estimation and oriented bounding box detection in the pipeline.

## 🧪 Testing

The repository includes comprehensive unit tests for both the YOLO Swift Package and the example applications. These tests ensure the reliability and stability of the codebase.

### Running Tests

Tests require CoreML model files which are not included in the repository due to their large size. To run the tests:

1. Set `SKIP_MODEL_TESTS = false` in the test files you want to run with model testing enabled
2. Download the required CoreML models from [Ultralytics](https://github.com/ultralytics/ultralytics)
3. Convert them to CoreML format using the Ultralytics Python library
4. Add the `.mlpackage` files to your Xcode project
5. Run the tests using Xcode's test navigator

If you don't have the model files, you can still run the tests with `SKIP_MODEL_TESTS = true`, which will skip model-dependent tests.

### Test Coverage

- **YOLO Swift Package**: Core functionality tests for object detection, segmentation, pose estimation, etc.
- **Example Apps**: Tests for each example application, verifying UI components, model integration, and real-time inference.

### Test Documentation

Each test directory includes a README.md with specific instructions for testing that component, including:

- Required model files
- How to obtain and convert models
- Testing strategy
- Test case explanations

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