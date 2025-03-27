# YOLO Package: Simple, Powerful YOLO Integration in Swift

YOLO Package is a Swift package that makes it easy to integrate Core ML-exported YOLO models into your app. It supports multiple tasks such as Object Detection, Segmentation, Classification, Pose Estimation, and Oriented Bounding Box Detection. With minimal code, you can add YOLO-based features to your app and even use real-time inference with camera streams in both SwiftUI and UIKit

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [YOLO Class (Inference)](#yolo-class)
  - [YOLOCamera / YOLOView (Real-Time Camera Inference)](#yolocamera--yoloview)
- [Contributing](contributing)
- [License](license)

## Features

- ✅ **Simple API**: Easily utilize Core ML YOLO models with Python-like code in Swift.
- ✅ **Multiple Task Support**: Object Detection, Segmentation, Classification, Pose Estimation, and Oriented Bounding Box Detection.
- ✅ **SwiftUI / UIKit Integration**: Pre-built view components for real-time camera inference.
- ✅ **Lightweight & Extensible**: Installs quickly via Swift Package Manager with no extra dependencies.

## Requirements

| Platform | Minimum Version | Notes                                         |
| -------- | --------------- | --------------------------------------------- |
| iOS      | 13.0+           | Suitable for iPhone / iPad                    |
| macOS    | 10.15+          | Camera functionality may be unavailable       |
| tvOS     | 13.0+           | Consider performance with Core ML on tvOS     |
| watchOS  | 6.0+            | Limited use cases due to hardware constraints |

** Swift 5.7+**

Built and managed via Swift Package Manager.

** Xcode 14.0+**

Required to leverage Core ML and the latest Swift Concurrency features.

## Installation

### Swift Package Manager

In Xcode, go to File > Add Packages... and enter the URL of this repository:

```swift
dependencies: [
    .package(url: "https://github.com/ultralytics/yolo-ios-app.git")
]
```

Then, specify it in your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "YOLO", package: "YOLO")
    ]
)
```

Once added, YOLOSwift will be automatically integrated into your project

## Usage

YOLO Package primarily provides two main components: the **YOLO class** and **YOLOCamera / YOLOView**.

### Import

```swift
import YOLO
```

### YOLO Class

**(Inference)**

Use the YOLO class for inference on static images, image files, or other UIImage inputs. It supports tasks like Object Detection, Segmentation, Classification, Pose Estimation, and Oriented Bounding Box Detection. Simply provide a valid YOLO model (either .mlmodelc or a local path/string).

```swift
let model = YOLO("yolo11n", task: .detect) # bundle file name, local path
let result = model(image) # SwifUIImage, UIImage, CIImage, CGImage, bundle name, local path, remote URL
```

### YOLOCamera / YOLOView

**(Real-Time Camera Inference)**

YOLO Package also provides SwiftUI and UIKit components for real-time inference on camera streams. Simply add these views to your layout, and the camera input + on-device model inference will be handled automatically.

SwiftUI Example

```swift
import YOLO
import SwiftUI

struct CameraView: View {
    var body: some View {
        YOLOCamera(
            modelPathOrName: "yolo11n-seg",
            task: .segment,
            cameraPosition: .back
        )
        .edgesIgnoringSafeArea(.all)
    }
}
```

UIKit Example

```swift
import YOLO
import UIKit

class CameraViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()

        let yoloView = YOLOView(
            frame: view.bounds,
            modelPathOrName: "yolo11n-seg",
            task: .segment
        )
        view.addSubview(yoloView)
    }
}
```

In just a few lines of code, you can bring YOLO-based, real-time inference to your application’s camera feed.

## How to Obtain YOLO CoreML Models

You can obtain YOLO CoreML models using either of the following methods:

### Download from GitHub Release Assets

You can download the CoreML INT8 models directly from the official YOLO GitHub release page.

Download YOLO CoreML Models (GitHub)

Place the downloaded models into your Xcode project directory.

### Export using Python

You can also export CoreML INT8 models yourself using the ultralytics Python package.

First, install the required package:

pip install ultralytics

Then, run the following Python script to export the desired models:

```python
from ultralytics import YOLO

# Export for all YOLO11 model sizes
for size in ("n", "s", "m", "l", "x"):
    # Load a YOLO11 PyTorch model
    model = YOLO(f"yolo11{size}.pt")

    # Export the PyTorch model to CoreML INT8 format (with NMS layers)
    model.export(format="coreml", int8=True, nms=True, imgsz=[640, 384])

    # You can specify different task models as follows:
    # model = YOLO(f"yolo11{size}-seg.pt")   # segmentation
    # model = YOLO(f"yolo11{size}-cls.pt")   # classification
    # model = YOLO(f"yolo11{size}-pose.pt")  # pose estimation
    # model = YOLO(f"yolo11{size}-obb.pt")   # oriented bounding box

    # Export the PyTorch model to CoreML INT8 format (without NMS layers)
    model.export(
        format="coreml", int8=True, imgsz=[640, 384]
    )  # For use with the package, do not add NMS to any models other than detection.
```

Note: CoreMLTools' NMS is only applicable to detection models, so models for segment, pose, and obb tasks need to write NMS in Swift. This library includes NMS for these tasks.
