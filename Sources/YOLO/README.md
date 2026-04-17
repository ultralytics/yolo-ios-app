<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# YOLO Swift Package: Simple, Powerful YOLO Integration in Swift

The YOLO Swift Package provides an easy way to integrate Core ML-exported [Ultralytics YOLO](https://docs.ultralytics.com/) models into your native Swift applications. It supports multiple computer vision tasks, including [Object Detection](https://docs.ultralytics.com/tasks/detect/), [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/), [Image Classification](https://docs.ultralytics.com/tasks/classify/), [Pose Estimation](https://docs.ultralytics.com/tasks/pose/), and [Oriented Bounding Box Detection](https://docs.ultralytics.com/tasks/obb/). With minimal code, you can add powerful YOLO-based features to your app and leverage real-time inference with camera streams in both [SwiftUI](https://developer.apple.com/xcode/swiftui/) and [UIKit](https://developer.apple.com/documentation/uikit).

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
- ✅ **Multiple Task Support**: Handles Object Detection, Segmentation, Classification, Pose Estimation, and Oriented Bounding Box Detection tasks seamlessly. Explore more about these tasks in the [Ultralytics documentation](https://docs.ultralytics.com/tasks/).
- ✅ **SwiftUI / UIKit Integration**: Includes pre-built view components for straightforward integration of real-time camera inference.
- ✅ **URL-Based Model Loading**: Load models directly from remote URLs with automatic downloading and caching functionality.
- ✅ **Lightweight & Extensible**: Installs quickly via [Swift Package Manager](https://www.swift.org/package-manager/) with no external dependencies beyond Apple's frameworks.

## 📋 Requirements

| Platform | Minimum Version | Notes                      |
| -------- | --------------- | -------------------------- |
| iOS      | 16.0+           | Suitable for iPhone / iPad |

- **Swift 5.10+**: Required for modern language features.
- **Xcode 15.3+**: Required to build against the iOS 16 SDK and use the modern [Swift Concurrency](https://developer.apple.com/documentation/swift/concurrency) APIs. Download from the [Apple Developer site](https://developer.apple.com/xcode/). Xcode 17 is recommended on recent macOS versions.

## 🚀 Installation

### Swift Package Manager

In Xcode, navigate to `File > Add Packages...` and enter the repository URL:

```
https://github.com/ultralytics/yolo-ios-app.git
```

Select the repository when it appears, then choose the `main` branch or the latest version tag.

In the "Choose Package Products for yolo-ios-app" dialog, add the `YOLO` product to your app target and click **Add Package**.

Alternatively, declare the dependency in your own `Package.swift`:

```swift
// In your Package.swift dependencies array
dependencies: [
    .package(url: "https://github.com/ultralytics/yolo-ios-app.git", branch: "main") // Or specify a version tag
]

// In your target's dependencies
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "YOLO", package: "yolo-ios-app") // Use the package name defined above
    ]
)
```

Once added, the YOLO Swift Package will be automatically integrated into your project.

## 💡 Usage

The YOLO Swift Package primarily provides two main components: the **`YOLO` class** for inference and **`YOLOCamera` / `YOLOView`** for real-time camera integration.

### Import

Start by importing the package in your Swift files:

```swift
import YOLO
```

### YOLO Class (Inference)

Use the `YOLO` class for performing inference on static images ([`UIImage`](https://developer.apple.com/documentation/uikit/uiimage), `CIImage`, `CGImage`), image file paths, or URLs. It supports various tasks like Object Detection, Segmentation, Classification, Pose Estimation, and Oriented Bounding Box Detection.

Initialize the `YOLO` class with a valid Ultralytics YOLO model exported to Core ML format. You can load an official model from a remote URL, point to your own local `.mlpackage` or `.mlmodelc`, or reference a model already included in your app [bundle](https://developer.apple.com/documentation/foundation/bundle).

```swift
import YOLO
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
//   .probs           — Classifier (top1 / top5 labels and scores)
//   .keypointsList   — PoseEstimator
//   .obb             — ObbDetector (oriented bounding boxes)
```

### YOLOCamera / YOLOView (Real-Time Camera Inference)

The package provides convenient SwiftUI (`YOLOCamera`) and UIKit (`YOLOView`) components for real-time inference using the device's camera stream. Add these views to your layout, and they handle the camera input and on-device model inference automatically.

**Note:** If you use the real-time camera feature, be sure to add "Privacy - Camera Usage Description" to your app's Info.plist.

#### SwiftUI Example

```swift
import YOLO
import SwiftUI

struct CameraView: View {
    private static let modelURL = URL(
        string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo26n-seg.mlpackage.zip"
    )!

    var body: some View {
        // Real-time inference using an official downloadable model.
        YOLOCamera(
            url: Self.modelURL,
            task: .segment,
            cameraPosition: .back
        ) { result in
            // Called once per processed frame.
            print("Detected \(result.boxes.count) objects")
        }
        .edgesIgnoringSafeArea(.all)
    }
}

// Alternative: use your own model already included in the app bundle.
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
import YOLO

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

You can get [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models compatible with this package in Core ML format using two methods:

### 1. Download Pre-Exported Models

You can download pre-exported Core ML models (compiled `.mlmodelc` directories or `.mlmodel` files) directly from the Assets section of the [Ultralytics YOLO releases page](https://github.com/ultralytics/ultralytics/releases). Look for files ending in `.mlpackage` or `.mlmodel`. We recommend using models quantized to [INT8](https://www.ultralytics.com/glossary/model-quantization) for better performance on mobile devices.

[Download YOLO Core ML Models (GitHub Releases)](https://github.com/ultralytics/ultralytics/releases)

After downloading, add the `.mlmodel` file or the `.mlmodelc` directory (often within an `.mlpackage`) to your Xcode project. Ensure it's included in your app target's "Copy Bundle Resources" build phase.

### 2. Export Using the Ultralytics Python Package

You can export models to the Core ML format yourself using the `ultralytics` Python package. This gives you more control over the export process, such as choosing different model sizes or export settings.

First, install the `ultralytics` package using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install ultralytics
```

Then, use the following Python script to export your desired [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) models. The example below exports YOLO26 detection models in various sizes to Core ML INT8 format. YOLO26 is NMS-free, so [NMS](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) is not needed during export.

```python
from ultralytics import YOLO

# Example: Export YOLO26 detection models (NMS-free)
for size in ("n", "s", "m", "l", "x"):
    # Load a YOLO26 PyTorch model
    model = YOLO(f"yolo26{size}.pt")  # Assumes you have the .pt file locally or downloads it

    # Export the PyTorch model to Core ML INT8 format (YOLO26 is NMS-free)
    # imgsz can be adjusted based on expected input size
    model.export(format="coreml", int8=True, nms=False, imgsz=[640, 384])
    print(f"Exported yolo26{size}.mlmodel (NMS-free)")

# Example: Export a YOLO26 segmentation model (without Core ML NMS)
seg_model = YOLO("yolo26n-seg.pt")
seg_model.export(format="coreml", int8=True, imgsz=[640, 384])  # NMS=False (or omitted) for non-detection tasks
print("Exported yolo26n-seg.mlmodel without NMS")

# Similarly for other tasks:
# cls_model = YOLO("yolo26n-cls.pt")
# cls_model.export(format="coreml", int8=True, imgsz=[224, 224]) # Classification often uses smaller imgsz

# pose_model = YOLO("yolo26n-pose.pt")
# pose_model.export(format="coreml", int8=True, imgsz=[640, 384])

# obb_model = YOLO("yolo26n-obb.pt")
# obb_model.export(format="coreml", int8=True, imgsz=[640, 384])
```

This script assumes you have the base [PyTorch](https://pytorch.org/) (`.pt`) models available. For detailed export options, refer to the [Ultralytics Core ML export documentation](https://docs.ultralytics.com/integrations/coreml/).

**Important Note on NMS:** The SDK supports both architectures. YOLO26 is NMS-free — export with `nms=False` (or omit the argument) and the Swift package applies postprocessing internally. Legacy YOLO11 models exported with Core ML NMS (`nms=True`) are also supported; the predictor detects the model's `nms` metadata flag at load time and dispatches to the correct path.

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ultralytics/yolo-ios-app). Check the [contributing guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get involved. We appreciate your help in making this package better! You can also join the conversation on [Discord](https://discord.com/invite/ultralytics).

## 📜 License

This project is licensed under the [AGPL-3.0 License](https://opensource.org/license/agpl-3.0). See the [LICENSE](https://github.com/ultralytics/yolo-ios-app/blob/main/LICENSE) file for details. For alternative licensing options, please visit [Ultralytics Licensing](https://www.ultralytics.com/license).
