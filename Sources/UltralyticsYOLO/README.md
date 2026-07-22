<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Swift Package: Simple, Powerful YOLO Integration in Swift

The YOLO Swift Package provides an easy way to integrate Core ML-exported [Ultralytics YOLO](https://docs.ultralytics.com/) models into your native Swift applications. It supports multiple computer vision tasks, including [Object Detection](https://docs.ultralytics.com/tasks/detect), [Instance Segmentation](https://docs.ultralytics.com/tasks/segment), [Semantic Segmentation](https://docs.ultralytics.com/tasks/semantic), Depth Estimation, [Image Classification](https://docs.ultralytics.com/tasks/classify), [Pose Estimation](https://docs.ultralytics.com/tasks/pose), and [Oriented Bounding Box Detection](https://docs.ultralytics.com/tasks/obb). With minimal code, you can add powerful YOLO-based features to your app and leverage real-time inference with camera streams in both [SwiftUI](https://developer.apple.com/xcode/swiftui/) and [UIKit](https://developer.apple.com/documentation/uikit).

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

- [✨ Features](#-features)
- [📋 Requirements](#-requirements)
- [🚀 Installation](#-installation)
- [💡 Usage](#-usage)
  - [YOLO Class (Inference)](#yolo-class-inference)
  - [YOLOCamera / YOLOView (Real-Time Camera Inference)](#yolocamera--yoloview-real-time-camera-inference)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## ✨ Features

- ✅ **Simple API**: Easily utilize Core ML YOLO models with Python-like code syntax in [Swift](https://developer.apple.com/swift/).
- ✅ **Multiple Task Support**: Handles Object Detection, Instance Segmentation, Semantic Segmentation, Depth Estimation, Classification, Pose Estimation, and Oriented Bounding Box Detection tasks seamlessly. Explore more about these tasks in the [Ultralytics documentation](https://docs.ultralytics.com/tasks).
- ✅ **SwiftUI / UIKit Integration**: Includes pre-built view components for straightforward integration of real-time camera inference.
- ✅ **URL-Based Model Loading**: Load models directly from remote URLs with automatic downloading and caching via the `YOLO` class.
- ✅ **Zero Dependencies**: Pure Swift built only on Apple's first-party frameworks (Foundation, Core ML, Vision, Compression) — **no third-party packages** to vet, license, or keep up to date. Even ZIP extraction for downloaded models is handled by a small, self-contained extractor, so the package installs instantly via [Swift Package Manager](https://www.swift.org/package-manager/) with nothing to resolve.

**Compatibility note:** `YOLOTask.semantic` and `YOLOTask.depth` are public enum cases for semantic segmentation and depth estimation. Apps with exhaustive `switch` statements over `YOLOTask` should add these cases or include a `default`.

## 📋 Requirements

| Platform | Minimum Version | Notes                      |
| -------- | --------------- | -------------------------- |
| iOS      | 13.0+           | Suitable for iPhone / iPad |

- **Swift 5.10+**: Required for modern language features.
- **Xcode 15.3+**: Required for Swift 5.10 (the package's `swift-tools-version`) and the modern [Swift Concurrency](https://developer.apple.com/documentation/swift/concurrency) APIs. Download from the [Apple Developer site](https://developer.apple.com/xcode/). Xcode 17 is recommended on recent macOS versions.

## 🚀 Installation

### Swift Package Manager

In Xcode, navigate to `File > Add Packages...` and enter the repository URL:

```
https://github.com/ultralytics/yolo-ios-app.git
```

Select the repository when it appears, then choose the `main` branch or the latest version tag.

In the "Choose Package Products for yolo-ios-app" dialog, add the `UltralyticsYOLO` product to your app target and click **Add Package**.

Alternatively, declare the dependency in your own `Package.swift`:

```swift
// In your Package.swift dependencies array
dependencies: [
    .package(url: "https://github.com/ultralytics/yolo-ios-app.git", from: "8.9.11")
]

// In your target's dependencies
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "UltralyticsYOLO", package: "yolo-ios-app") // Use the package name defined above
    ]
)
```

Once added, the package is integrated automatically.

### CocoaPods

`UltralyticsYOLO` is also published to [CocoaPods](https://cocoapods.org/) trunk. Add it to your `Podfile`:

```ruby
pod 'UltralyticsYOLO', '~> 8.9'
```

Then run `pod install`.

## 💡 Usage

The YOLO Swift Package primarily provides two main components: the **`YOLO` class** for inference and **`YOLOCamera` / `YOLOView`** for real-time camera integration.

### Import

Start by importing the package in your Swift files:

```swift
import UltralyticsYOLO
```

### YOLO Class (Inference)

Use the `YOLO` class for performing inference on static images ([`UIImage`](https://developer.apple.com/documentation/uikit/uiimage), `CIImage`, `CGImage`), image file paths, or URLs. It supports Object Detection, Instance Segmentation, Semantic Segmentation, Depth Estimation, Classification, Pose Estimation, and Oriented Bounding Box Detection.

Initialize the `YOLO` class with a valid Ultralytics YOLO model exported to Core ML format. You can load an official model from a remote URL, point to your own local `.mlpackage` or `.mlmodelc`, or reference a model already included in your app [bundle](https://developer.apple.com/documentation/foundation/bundle).

```swift
import UltralyticsYOLO
import UIKit

// --- Initialization ---
// Start with an official Ultralytics model URL.
// The model is downloaded once, cached, and then loaded from disk on later runs.
var model: YOLO?
let officialModelURL = URL(
    string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo26n.mlpackage.zip"
)!
model = YOLO(url: officialModelURL, task: .detect) { result in
    switch result {
    case .success:
        print("Model ready")
    case .failure(let error):
        print("Failed to load model: \(error)")
    }
}

// Or load your own fine-tuned Core ML export from disk.
model = YOLO("/path/to/your-custom-model.mlpackage", task: .detect) { result in
    // handle result
    _ = result
}

// Or load a model you've bundled into your app by resource name.
model = YOLO("yolo26n", task: .detect) { result in
    // handle result
    _ = result
}

// --- Inference ---
// Once the load completion has fired, call the model with any supported input type.
guard let model = model, let image = UIImage(named: "your_image_name") else { return }

let output: YOLOResult = model(image)   // UIImage overload; also accepts CIImage, CGImage,
                                        // a bundle resource name, or a local file path.

// Detection boxes (applies to .detect, .segment, .pose, and similar tasks).
for box in output.boxes {
    print("\(box.cls) — conf \(box.conf) at \(box.xywh)")
}

// Task-specific fields on YOLOResult:
//   .masks           — Segmenter  (combined mask CGImage + per-instance mask arrays)
//   .semanticMask    — SemanticSegmenter (dense class map + overlay CGImage)
//   .depthMap        — DepthEstimator (metric values + colorized CGImage)
//   .probs           — Classifier (top1 / top5 labels and scores)
//   .keypointsList   — PoseEstimator
//   .obb             — ObbDetector (oriented bounding boxes)
```

### YOLOCamera / YOLOView (Real-Time Camera Inference)

The package provides convenient SwiftUI (`YOLOCamera`) and UIKit (`YOLOView`) components for real-time inference using the device's camera stream. Add these views to your layout, and they handle the camera input and on-device model inference automatically.

**Note:** If you use the real-time camera feature, be sure to add "Privacy - Camera Usage Description" to your app's Info.plist.

#### SwiftUI Example

```swift
import UltralyticsYOLO
import SwiftUI

struct CameraView: View {
    var body: some View {
        // Real-time inference using a model bundled in the app (e.g. yolo26n-seg.mlpackage
        // added to the app target). To use an official hosted asset, download and unzip it
        // first (e.g. with `YOLOModelDownloader`) and pass its local path or file URL.
        YOLOCamera(
            modelPathOrName: "yolo26n-seg",
            task: .segment,
            cameraPosition: .back
        ) { result in
            // Called once per processed frame.
            print("Detected \(result.boxes.count) objects")
        }
        .edgesIgnoringSafeArea(.all)
    }
}

// Alternative: use your own custom model already included in the app bundle.
struct CameraViewWithBundledModel: View {
    var body: some View {
        YOLOCamera(
            modelPathOrName: "my-custom-yolo-seg",
            task: .segment,
            cameraPosition: .back
        )
        .edgesIgnoringSafeArea(.all)
    }
}
```

#### UIKit Example

```swift
import AVFoundation
import UIKit
import UltralyticsYOLO

class CameraViewController: UIViewController {
    var yoloView: YOLOView?

    override func viewDidLoad() {
        super.viewDidLoad()

        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self, granted else {
                    print("Camera permission denied")
                    return
                }
                let view = YOLOView(
                    frame: self.view.bounds,
                    modelPathOrName: "my-custom-yolo-seg", // Bundle resource name
                    task: .segment
                )
                view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
                view.onDetection = { result in
                    print("\(result.boxes.count) detections")
                }
                self.view.addSubview(view)
                self.yoloView = view
            }
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        yoloView?.resume()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        yoloView?.stop()
    }
}
```

With just a few lines of code, you can integrate real-time, YOLO-based inference into your application's camera feed. For more advanced use cases, explore the customization options available for these components.

## ⚙️ How to Obtain YOLO Core ML Models

You can use the official hosted assets, export the official matrix yourself, or load your own fine-tuned Core ML model.

### Official Hosted Assets

Official Core ML assets are hosted in [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0). They are int8 `.mlpackage.zip` archives named by model ID, for example `yolo26n.mlpackage.zip`, `yolo26n-seg.mlpackage.zip`, and `yolo26x-obb.mlpackage.zip`.

| Runtime asset                 | Used by                                      | Release                                                                                          |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Core ML int8 `.mlpackage.zip` | iOS app, Swift package, Flutter on iOS/macOS | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |
| LiteRT w8a32 `.tflite`        | Flutter on Android                           | [yolo-flutter-app `v0.6.6`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6) |

URL patterns:

- Core ML: `https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip`
- LiteRT: `https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/<model>_w8a32.tflite`

The `YOLO` class can load a Core ML release URL directly; it downloads once and caches the compiled model locally. If you download manually, unzip the `.mlpackage.zip` asset and add the `.mlpackage` to your app target's "Copy Bundle Resources" build phase.

The [repository root README](../../README.md#-official-model-assets) is the authoritative reference for official model properties, including `imgsz`, `quantize`, `nms`, `end2end`, calibration, postprocessing, and release hosting.

### Reproduce The Official Core ML Assets

The published `v8.3.0` binary properties are recorded in the
[repository root README](../../README.md#-official-model-assets). The export workflow in
[`scripts/export-models.py`](../../scripts/export-models.py) defines future exports, Core ML int8 settings,
`.mlpackage.zip` packaging, the optional local app-copy step, and optional GitHub release upload.

```bash
uv venv --python 3.13 .venv
uv pip install -e "../ultralytics[export]"
uv run python scripts/export-models.py
```

Use `--copy-to-app` to copy exported packages into `YOLOiOSApp/Models/<Task>/` for local app testing. After creating a
new release, use `--upload --repo ultralytics/yolo-ios-app --tag vX.Y.Z` to publish the generated archives; never reuse
`v8.3.0` or another tag already consumed by released apps.

YOLO26 is NMS-free in this SDK. The shipped Core ML assets use `nms=False`; detect, segment, pose, and OBB use
`end2end=True`, while classification, semantic, and depth use `end2end=False`. The Swift package applies task-specific
postprocessing internally. Legacy YOLO11 models exported with Core ML NMS (`nms=True`) remain supported; the predictor
detects the model's `nms` metadata flag at load time and dispatches to the correct path.

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ultralytics/yolo-ios-app). Check the [contributing guide](https://docs.ultralytics.com/help/contributing) for more details on how to get involved. We appreciate your help in making this package better! You can also join the conversation on [Discord](https://discord.com/invite/ultralytics).

## 📜 License

This project is licensed under the [AGPL-3.0 License](https://opensource.org/license/agpl-3.0). See the [LICENSE](https://github.com/ultralytics/yolo-ios-app/blob/main/LICENSE) file for details. For alternative licensing options, please visit [Ultralytics Licensing](https://www.ultralytics.com/license).
