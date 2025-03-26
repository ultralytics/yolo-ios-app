//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing a UIKit example for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController demonstrates how to implement real-time object detection using YOLO models in UIKit.
//  It provides a camera interface that continuously detects objects in the camera feed using the YOLO framework.
//  The example shows how to initialize the YOLO model for detection, set up a camera preview, and display
//  detection results in real-time with bounding boxes and labels.

import UIKit
import YOLO

/// A view controller that demonstrates real-time object detection, segmentation, or other YOLO tasks using the UIKit framework.
///
/// This view controller sets up a `YOLOView` which handles camera input, model inference, and visualization
/// of the detection results in real-time. It uses a segmentation model by default but can be modified
/// to use other YOLO model types.
///
/// - Note: This example requires camera permissions to be added to the Info.plist file.
/// - Important: The app requires at least iOS 16.0 or higher to run.
class ViewController: UIViewController {

  /// The YOLO view that handles camera capture, model inference, and visualization.
  var yoloView: YOLOView!

  override func viewDidLoad() {
    super.viewDidLoad()
    // Initialize YOLOView with a segmentation model
    // You can change the model or task type to use detection, classification, etc.
    yoloView = YOLOView(frame: view.bounds, modelPathOrName: "yolo11n", task: .detect)
    view.addSubview(yoloView)
  }
}
