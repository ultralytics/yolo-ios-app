<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 Ultralytics YOLO for iOS: App and Swift Package

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml)
[![CI](https://github.com/ultralytics/yolo-ios-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/ultralytics/yolo-ios-app/branch/main/graph/badge.svg)](https://app.codecov.io/github/ultralytics/yolo-ios-app)
[![CocoaPods](https://img.shields.io/cocoapods/v/UltralyticsYOLO?logo=cocoapods&logoColor=white&label=CocoaPods)](https://cocoapods.org/pods/UltralyticsYOLO)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for iOS provides on-device [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) for [object detection](https://www.ultralytics.com/glossary/object-detection), instance segmentation, semantic segmentation, classification, pose estimation, and oriented bounding box detection. The SDK supports both [YOLO11](https://docs.ultralytics.com/models/yolo11) (with Core ML NMS) and [YOLO26 models](https://platform.ultralytics.com/ultralytics/yolo26) (NMS-free, with Swift-side postprocessing). Download the app from the [App Store](https://apps.apple.com/app/ultralytics-yolo/id1452689527), or integrate the Swift package into your own applications.

<div align="center">
  <br>
  <a href="https://apps.apple.com/app/ultralytics-yolo/id1452689527" target="_blank"><img width="100%" src="https://github.com/user-attachments/assets/d5dab2e7-f473-47ce-bc63-69bef89ba52a" alt="Ultralytics YOLO iOS App previews"></a>
  <br>
  <br>
  <a href="https://apps.apple.com/app/ultralytics-yolo/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
  &nbsp;&nbsp;
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.yolo" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/google-play.svg" width="15%" alt="Get it on Google Play"></a>
</div>

## ✨ Features

- Swift and Core ML throughout, running on the Apple Neural Engine and GPU
- Camera-rate (~30 FPS) real-time inference on recent iPhones — see [docs/performance.md](docs/performance.md) for on-device profiling
- Native UI following Apple interface guidelines
- YOLO26 (NMS-free) and YOLO11 models both supported
- No third-party dependencies — pure Swift on Apple's first-party frameworks

| Feature                               | iOS | Details                                       |
| ------------------------------------- | --- | --------------------------------------------- |
| Object Detection                      | ✅  | Bounding boxes, labels, and confidence scores |
| Instance Segmentation                 | ✅  | Instance masks with boxes and classes         |
| Semantic Segmentation                 | ✅  | Dense per-pixel class maps                    |
| Image Classification                  | ✅  | Top class predictions and scores              |
| Pose Estimation                       | ✅  | Keypoints with boxes and confidence scores    |
| Oriented Bounding Box (OBB) Detection | ✅  | Rotated boxes and polygon corners             |

## 📂 Repository Content

This repository contains two components for running YOLO models on Apple platforms ([Edge AI](https://www.ultralytics.com/glossary/edge-ai)):

### [**Ultralytics YOLO iOS App (Main App)**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)

The primary iOS application allows easy real-time YOLO inference using your device's camera or image library. The shipped app bundles all six official nano Core ML models, larger variants download on demand, and you can also test your custom [Core ML](https://developer.apple.com/documentation/coreml) models by adding them to the app project.

### [**Swift Package (YOLO Library)**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/UltralyticsYOLO)

A lightweight [Swift](https://developer.apple.com/swift/) package designed for iOS and iPadOS. It handles model loading, inference, and postprocessing for YOLO models like YOLO26 in your own applications, with a few lines of [SwiftUI](https://developer.apple.com/xcode/swiftui/):

```swift
// Perform inference on a UIImage
let result = model(uiImage)
```

```swift
// Use the built-in camera view for real-time inference with a model bundled in your app
var body: some View {
    YOLOCamera(
        modelPathOrName: "yolo26n-seg",
        task: .segment,
        cameraPosition: .back
    )
    .ignoresSafeArea()
}
```

## 📦 Official Model Assets

Official models are GitHub release assets, not large files committed to the repositories. The main iOS app downloads the six nano Core ML assets at build time and bundles them into the app; larger app models, the Swift package's `YOLO(url:)` loading, and Flutter package assets download official models on first use and cache them locally.

The main YOLOiOSApp **bundles all six nano models** (one per task: detect, segment, semantic, classify, pose, OBB) into the shipped app, including App Store/archive builds. They are downloaded at build time from the GitHub release assets by a **Download YOLO Models** Xcode build phase that runs [`scripts/download-models.sh`](scripts/download-models.sh) — the `.mlpackage` files are **never committed to the repo** (`*.mlpackage` is gitignored). The step is idempotent and is skipped on GitHub Actions CI, which runs the same script in its own step.

| Runtime asset                 | Used by                                      | Release                                                                                          |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Core ML int8 `.mlpackage.zip` | iOS app, Swift package, Flutter on iOS/macOS | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |
| LiteRT int8 `.tflite`         | Flutter on Android                           | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) |

URL patterns:

- Core ML: `https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip`
- LiteRT: `https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/<model>_int8.tflite`

The iOS app registry is [`RemoteModels.swift`](YOLOiOSApp/YOLOiOSApp/RemoteModels.swift). It enumerates YOLO26 `n/s/m/l/x` assets for detect, segment, semantic, classify, pose, and OBB and points each model ID at the `v8.3.0` Core ML release. The Core ML column below is owned by this repo; the LiteRT column summarizes the Flutter repo's Android export script and release assets.

| Property       | Core ML                             | LiteRT                           |
| -------------- | ----------------------------------- | -------------------------------- |
| Model IDs      | `yolo26{n,s,m,l,x}`                 | `yolo26{n,s,m,l,x}`              |
| Tasks          | detect, seg, sem, cls, pose, obb    | detect, seg, sem, cls, pose, obb |
| Format         | `.mlpackage.zip`                    | `.tflite`                        |
| `quantize`     | `8`                                 | `8`                              |
| `imgsz`        | `224` cls; `1024` OBB; `640` others | `224` cls; `640` others          |
| `nms`          | `False`                             | `False`                          |
| `end2end`      | `True`                              | `False`                          |
| Calibration    | exporter default                    | `TASK2CALIBRATIONDATA` per task  |
| Postprocessing | Swift/Core ML                       | Android native                   |

Core ML assets use `nms=False` and `end2end=True`: `nms=False` suppresses the Core ML NMS pipeline, and `end2end=True` supplies the YOLO26 decoded output contract consumed by the Swift decoders. The LiteRT export script passes both `nms=False` and `end2end=False`; `end2end=False` disables the YOLO26 end-to-end head for the Android LiteRT conversion path.

### Core ML Release Workflow

The authoritative export script is [`scripts/export-models.py`](scripts/export-models.py). It defines the task/size matrix, export image sizes, int8 Core ML settings, `.mlpackage.zip` packaging, optional local app-copy step, and optional GitHub release upload.

```bash
uv venv --python 3.13 .venv
uv pip install -e "../ultralytics[export]"
uv run python scripts/export-models.py
```

Useful variants:

```bash
# Export only nano task models for local validation and copy them into YOLOiOSApp/Models/.
uv run python scripts/export-models.py --sizes n --copy-to-app

# Export all official Core ML assets and upload them to the canonical release.
uv run python scripts/export-models.py --upload --repo ultralytics/yolo-ios-app --tag v8.3.0
```

The script exports from checkpoints named `yolo26<size><suffix>.pt`, for example `yolo26n.pt`, `yolo26s-seg.pt`, `yolo26m-sem.pt`, `yolo26l-pose.pt`, and `yolo26x-obb.pt`. YOLO26 is NMS-free in this SDK, so official Core ML assets are exported with `nms=False` and `end2end=True`; Swift-side postprocessing handles the end2end detect, segment, pose, and OBB outputs (classify and semantic outputs need no NMS decode).

### Android LiteRT Counterparts

The Android assets used by the Flutter package are maintained in the Flutter repo, not this iOS repo. Their canonical export script is `scripts/export-tflite-models.py` in `ultralytics/yolo-flutter-app`; it exports the matching YOLO26 task/size matrix as int8 `.tflite` assets calibrated with the per-task `ultralytics.cfg.TASK2CALIBRATIONDATA` defaults and uploads them to `yolo-flutter-app` `v0.3.5`.

## 🛠️ Quickstart Guide

New to YOLO on mobile or want to quickly test your custom model? Start with the main YOLOiOSApp. The six nano task models are bundled at build time, so the app can run offline after installation; larger model sizes download on demand.

- [**Ultralytics YOLO iOS App (Main App)**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp): The easiest way to experience YOLO inference on iOS.

Ready to integrate YOLO into your own project? Explore the Swift Package and example applications.

- [**Swift Package (YOLO Library)**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/UltralyticsYOLO): Integrate YOLO capabilities into your Swift app.
- [**Example Apps**](https://github.com/ultralytics/yolo-ios-app/tree/main/ExampleApps): See practical implementations using the YOLO Swift Package.

Add the `UltralyticsYOLO` package to your app with Swift Package Manager:

```swift
.package(url: "https://github.com/ultralytics/yolo-ios-app.git", from: "8.9.4")
```

Or with CocoaPods:

```ruby
pod 'UltralyticsYOLO', '~> 8.9'
```

Then `import UltralyticsYOLO` and use the `YOLO` class — see the [Swift Package README](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/UltralyticsYOLO) for full usage. The same `UltralyticsYOLO` package powers both this native iOS app and the [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app), keeping one source of truth across platforms.

## 🧪 Testing Procedures

This repository includes comprehensive [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for both the YOLO Swift Package and the example applications, ensuring code reliability and stability.

### Running Tests

Tests require Core ML model files (`.mlpackage`), which are not committed to the repository due to their size. To run the package tests with model validation, first run the same downloader used by CI and the app build phase:

```bash
bash scripts/download-models.sh
```

This downloads the six nano Core ML packages into `Tests/YOLOTests/Resources/` and copies them into `YOLOiOSApp/Models/<Task>/` for the main app bundle. You can also export or replace these packages with custom Core ML models using the [Ultralytics Python library's export function](https://docs.ultralytics.com/modes/export). If a specific test target supports `SKIP_MODEL_TESTS`, keeping it set to `true` skips tests that require loading and running a model.

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

We warmly welcome contributions to our open-source projects! Your support helps us push the boundaries of [Artificial Intelligence (AI)](https://www.ultralytics.com/glossary/artificial-intelligence-ai). Get involved by reviewing our [Contributing Guide](https://docs.ultralytics.com/help/contributing) and sharing your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). Thank you 🙏 to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics provides two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: An [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license ideal for academic research, personal projects, and experimentation. It promotes open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/yolo-ios-app/blob/main/LICENSE) file for the full license text.
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
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
