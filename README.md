<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 Ultralytics YOLO for iOS: App and Swift Package

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

Welcome to the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) iOS App GitHub repository! 📖 This project leverages Ultralytics' state-of-the-art [YOLO11 models](https://docs.ultralytics.com/models/yolo11/) to transform your iOS device into a powerful real-time [object detection](https://www.ultralytics.com/glossary/object-detection) tool. Download the app directly from the [App Store](https://apps.apple.com/us/app/idetection/id1452689527) or explore our guide to integrate YOLO capabilities into your own Swift applications.

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

## 📂 Repository Content

This repository provides a comprehensive solution for running YOLO models on Apple platforms:

### [**Ultralytics YOLO iOS App (Main App)**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)

The primary iOS application allows easy real-time object detection using your device's camera or image library. You can easily test your custom [CoreML](https://developer.apple.com/documentation/coreml) models by simply dragging and dropping them into the app.

### [**Swift Package (YOLO Library)**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/YOLO)

A lightweight [Swift](https://developer.apple.com/swift/) package designed for iOS, iPadOS, and macOS. It simplifies the integration and usage of YOLO-based models like YOLO11 within your own applications. Integrate YOLO models effortlessly with minimal code:

```swift
// Perform inference on a UIImage
let result = model(uiImage)
```

```swift
// Use the built-in camera view for real-time detection
var body: some View {
    YOLOCamera(
        modelPathOrName: "yolo11n-seg", // Specify model name or path
        task: .segment,                // Define the task (detect, segment, classify, pose)
        cameraPosition: .back          // Choose camera (back or front)
    )
    .ignoresSafeArea()
}
```

## 🛠️ Quickstart Guide

New to YOLO on mobile or want to quickly test your custom model? Start with the main YOLOiOSApp.

- [**Ultralytics YOLO iOS App (Main App)**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp): The easiest way to experience YOLO detection on iOS.

Ready to integrate YOLO into your own project? Explore the Swift Package and example applications.

- [**Swift Package (YOLO Library)**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/YOLO): Integrate YOLO capabilities into your Swift app.
- [**Example Apps**](https://github.com/ultralytics/yolo-ios-app/tree/main/ExampleApps): See practical implementations using the YOLO Swift Package.

## ✨ Key Highlights

- **Real-Time Inference**: Achieve high-speed, high-accuracy object detection on iPhones and iPads using optimized [CoreML models](https://docs.ultralytics.com/integrations/coreml/).
- **Multi-OS Support**: The Swift Package is compatible with iOS, iPadOS, and macOS, enabling broad application deployment.
- **Flexible Tasks**: Supports [object detection](https://docs.ultralytics.com/tasks/detect/), with [segmentation](https://docs.ultralytics.com/tasks/segment/), [classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented bounding box (OBB)](https://docs.ultralytics.com/tasks/obb/) detection planned for future updates.

## 🧪 Testing Procedures

This repository includes comprehensive unit tests for both the YOLO Swift Package and the example applications, ensuring code reliability and stability.

### Running Tests

Tests require CoreML model files (`.mlpackage`), which are not included in the repository due to their size. To run the tests with model validation:

1.  Set `SKIP_MODEL_TESTS = false` in the relevant test files (e.g., `YOLOv11Tests.swift`).
2.  Download the required models from the [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases) or train your own.
3.  Convert the models to CoreML format using the [Ultralytics Python library's export function](https://docs.ultralytics.com/modes/export/).
4.  Add the exported `.mlpackage` files to your Xcode project, ensuring they are included in the test targets.
5.  Run the tests using Xcode's Test Navigator (Cmd+U).

If you don't have the model files, you can still run tests by keeping `SKIP_MODEL_TESTS = true`. This will skip tests that require loading and running a model.

### Test Coverage

- **YOLO Swift Package**: Includes tests for core functionalities like model loading, preprocessing, inference, and postprocessing across different tasks.
- **Example Apps**: Contains tests verifying UI components, model integration, and real-time inference performance within the sample applications.

### Test Documentation

Each test directory (e.g., `Tests/YOLOTests`) may include a `README.md` with specific instructions for testing that component, covering:

- Required model files and where to obtain them.
- Steps for model conversion and setup.
- Overview of the testing strategy.
- Explanation of key test cases.

## 💡 Contribute

We warmly welcome contributions to our open-source projects! Your support helps us push the boundaries of [AI](https://www.ultralytics.com/glossary/artificial-intelligence-ai). Get involved by reviewing our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) and sharing your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). Thank you 🙏 to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics provides two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: An [OSI-approved](https://opensource.org/license) open-source license ideal for academic research, personal projects, and experimentation. It promotes open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/yolo-ios-app/blob/main/LICENSE) file for details.
- **Enterprise License**: Tailored for commercial applications, this license allows the integration of Ultralytics software and AI models into commercial products and services without the open-source requirements of AGPL-3.0. If your scenario involves commercial use, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 🤝 Contact

- For bug reports and feature requests related to this iOS project, please use [GitHub Issues](https://github.com/ultralytics/yolo-ios-app/issues).
- For questions, discussions, and support regarding Ultralytics technologies, join our active [Discord](https://discord.com/invite/ultralytics) community!

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
