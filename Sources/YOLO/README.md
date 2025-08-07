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
- [🤝 Contributing](contributing)
- [📜 License](license)

## ✨ Features

- ✅ **Simple API**: Easily utilize Core ML YOLO models with Python-like code syntax in [Swift](https://developer.apple.com/swift/).
- ✅ **Multiple Task Support**: Handles Object Detection, Segmentation, Classification, Pose Estimation, and Oriented Bounding Box Detection tasks seamlessly. Explore more about these tasks in the [Ultralytics documentation](https://docs.ultralytics.com/tasks/).
- ✅ **SwiftUI / UIKit Integration**: Includes pre-built view components for straightforward integration of real-time camera inference.
- ✅ **URL-Based Model Loading**: Load models directly from remote URLs with automatic downloading and caching functionality.
- ✅ **Lightweight & Extensible**: Installs quickly via [Swift Package Manager](https://www.swift.org/package-manager/) with no external dependencies beyond Apple's frameworks.

## 📋 Requirements

| Platform | Minimum Version | Notes                                                                                                    |
| -------- | --------------- | -------------------------------------------------------------------------------------------------------- |
| iOS      | 13.0+           | Suitable for iPhone / iPad                                                                               |
| macOS    | 10.15+          | Camera functionality may depend on hardware availability                                                 |
| tvOS     | 13.0+           | Consider performance implications of [Core ML](https://developer.apple.com/documentation/coreml) on tvOS |
| watchOS  | 6.0+            | Limited use cases due to hardware constraints                                                            |

- **Swift 5.7+**: Required for modern language features.
- **Xcode 14.0+**: Needed to leverage Core ML and the latest [Swift Concurrency](https://developer.apple.com/documentation/swift/concurrency) features. Download from the [Apple Developer site](https://developer.apple.com/xcode/).

## 🚀 Installation

### Swift Package Manager

In Xcode, navigate to `File > Add Packages...` and enter the repository URL:

```
https://github.com/ultralytics/yolo-ios-app.git
```

Select the repository when it appears. Choose the `main` branch or the latest version tag.

Next, in the "Choose Package Products for yolo-ios-app.git" popup, specify your app project in Add to Target and click Add package.

If the package has been added to your project, you’re successful.

(Optional)

Or specify the target in your `Package.swift` file:

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

Initialize the `YOLO` class with a valid Ultralytics YOLO model exported to Core ML format (either a compiled `.mlmodelc` directory included in your app [bundle](https://developer.apple.com/documentation/foundation/bundle) or a path to an uncompiled `.mlmodel` file).

```swift
import YOLO
import UIKit // Or AppKit for macOS

// --- Initialization ---
// Initialize with a model file name included in the app bundle (automatically finds .mlmodelc)
guard let model = try? YOLO(modelFileName: "yolo11n", task: .detect) else {
    fatalError("Failed to load YOLO model.")
}

// Or initialize with a specific path to a .mlmodel file
let modelPath = Bundle.main.path(forResource: "yolo11n", ofType: "mlmodel")!
guard let model = try? YOLO(modelPath: modelPath, task: .detect) else {
    fatalError("Failed to load YOLO model.")
}

// Or initialize with a remote URL (model will be downloaded and cached automatically)
let modelURL = URL(string: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.mlpackage.zip")!
guard let model = try? YOLO(url: modelURL, task: .detect) else {
    fatalError("Failed to load YOLO model from URL.")
}

// --- Inference ---
// Load an image (replace with your image loading logic)
guard let image = UIImage(named: "your_image_name") else { // Or load CGImage, CIImage
    fatalError("Failed to load image.")
}

// Perform inference
do {
    let results = try model.predict(source: image) // Can also accept CGImage, CIImage, file path String, or URL

    // Process results based on the task
    switch model.task {
    case .detect:
        // Access detection results (bounding boxes, confidences, classes)
        for result in results {
            print("Detected object: \(result.label) with confidence \(result.confidence) at \(result.rect)")
        }
    case .segment:
        // Access segmentation results (masks, bounding boxes, etc.)
        // Note: Mask processing might require additional steps depending on your needs.
        for result in results {
            print("Segmented object: \(result.label) with mask area...") // Access result.mask
        }
    // Add cases for .classify, .pose, .obb as needed
    default:
        print("Processing results for task: \(model.task)")
    }

} catch {
    print("Error performing inference: \(error)")
}
```

### YOLOCamera / YOLOView (Real-Time Camera Inference)

The package provides convenient SwiftUI (`YOLOCamera`) and UIKit (`YOLOView`) components for real-time inference using the device's camera stream. Add these views to your layout, and they handle the camera input and on-device model inference automatically.

**\*If you use the real-time camera feature, be sure to add "Privacy - Camera Usage Description" to your app's Info.Plist.**

#### SwiftUI Example

```swift
import YOLO // Ensure YOLO is imported
import SwiftUI

struct CameraView: View {
    var body: some View {
        // Use YOLOCamera for real-time inference in SwiftUI
        YOLOCamera(
            modelPathOrName: "yolo11n-seg", // Model file name in bundle
            task: .segment,             // Specify the task
            cameraPosition: .back       // Use the back camera
            // Optional confidenceThreshold parameter can be added here
        )
        .edgesIgnoringSafeArea(.all) // Optional: make the view full-screen
        // Add error handling if needed
        .onAppear {
            // Request camera permissions if not already granted
        }
    }
}

// Alternative: Initialize with remote URL
struct CameraViewWithURL: View {
    var body: some View {
        YOLOCamera(
            url: URL(string: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.mlpackage.zip")!,
            task: .segment,
            cameraPosition: .back
        )
        .edgesIgnoringSafeArea(.all)
    }
}
```

#### UIKit Example

```swift
import YOLO // Ensure YOLO is imported
import UIKit
import AVFoundation // Needed for camera permission check

class CameraViewController: UIViewController {
    var yoloView: YOLOView?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCameraView()
    }

    func setupCameraView() {
        // Check for camera permissions using AVFoundation
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if granted {
                    // Initialize YOLOView on the main thread after permission check
                    self.yoloView = YOLOView(
                        frame: self.view.bounds,
                        modelFileName: "yolo11n-seg", // Model file name in bundle
                        task: .segment,             // Specify the task
                        cameraPosition: .back       // Use the back camera
                        // Optional confidenceThreshold parameter can be added here
                    )
                    // Handle potential initialization errors
                    if let yoloView = self.yoloView {
                        yoloView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
                        self.view.addSubview(yoloView)
                        // Start the camera session if needed (often handled internally by YOLOView)
                    } else {
                        print("Error: Failed to initialize YOLOView.")
                        // Show an error message to the user
                    }
                } else {
                    print("Error: Camera permission denied.")
                    // Show an error message or guide the user to settings
                }
            }
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // Stop the camera session when the view disappears (often handled internally)
        yoloView?.stopSession()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // Restart the camera session if needed (often handled internally)
        yoloView?.startSession()
    }
}
```

With just a few lines of code, you can integrate real-time, YOLO-based inference into your application’s camera feed. For more advanced use cases, explore the customization options available for these components.

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

Then, use the following Python script to export your desired [YOLO11](https://docs.ultralytics.com/models/yolo11/) models (or other YOLO versions like [YOLOv8](https://docs.ultralytics.com/models/yolov8/)). The example below exports YOLO11 detection models in various sizes to Core ML INT8 format, including [NMS](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) for detection models.

```python
from ultralytics import YOLO

# Example: Export YOLO11 detection models
for size in ("n", "s", "m", "l", "x"):
    # Load a YOLO11 PyTorch model
    model = YOLO(f"yolo11{size}.pt")  # Assumes you have the .pt file locally or downloads it

    # Export the PyTorch model to CoreML INT8 format (with NMS for detection)
    # imgsz can be adjusted based on expected input size
    model.export(format="coreml", int8=True, nms=True, imgsz=[640, 384])
    print(f"Exported yolo11{size}.mlmodel with NMS")

# Example: Export a YOLO11 segmentation model (without CoreML NMS)
seg_model = YOLO("yolo11n-seg.pt")
seg_model.export(format="coreml", int8=True, imgsz=[640, 384])  # NMS=False (or omitted) for non-detection tasks
print("Exported yolo11n-seg.mlmodel without NMS")

# Similarly for other tasks:
# cls_model = YOLO("yolo11n-cls.pt")
# cls_model.export(format="coreml", int8=True, imgsz=[224, 224]) # Classification often uses smaller imgsz

# pose_model = YOLO("yolo11n-pose.pt")
# pose_model.export(format="coreml", int8=True, imgsz=[640, 384])

# obb_model = YOLO("yolo11n-obb.pt")
# obb_model.export(format="coreml", int8=True, imgsz=[640, 384])
```

This script assumes you have the base [PyTorch](https://pytorch.org/) (`.pt`) models available. For detailed export options, refer to the [Ultralytics Core ML export documentation](https://docs.ultralytics.com/integrations/coreml/).

**Important Note on NMS:** The `nms=True` flag during export adds Core ML's built-in Non-Maximum Suppression layers, which is **only applicable to detection models**. For segmentation, pose estimation, and OBB tasks, export with `nms=False` (or omit the argument). This YOLO Swift Package includes optimized Swift implementations of NMS for these other tasks, which are applied automatically after inference. Using `nms=True` for non-detection models may lead to export errors or incorrect behavior.

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ultralytics/yolo-ios-app). Check the [`contributing`](contributing) guide for more details on how to get involved. We appreciate your help in making this package better! You can also join the conversation on [Discord](https://discord.com/invite/ultralytics).

## 📜 License

This project is licensed under the [AGPL-3.0 License](https://opensource.org/license/agpl-v3). See the [`license`](license) file for details. For alternative licensing options, please visit [Ultralytics Licensing](https://www.ultralytics.com/license).
