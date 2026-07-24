// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing a UIKit example for real-time object
//  detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController shows how to run real-time object detection with YOLO models in UIKit. It provides a camera
//  interface that continuously detects objects in the camera feed using the YOLO framework, including how to initialize
//  the YOLO model, set up a camera preview, and display detection results in real-time with bounding boxes and labels.

import UIKit
import UltralyticsYOLO

/// A view controller that demonstrates real-time object detection, segmentation, or other YOLO tasks using the UIKit
/// framework.
///
/// Sets up a `YOLOView` which handles camera input, model inference, and visualization of the results in real-time.
/// Uses a detection model by default but can be modified to use other YOLO task types.
///
/// - Note: This example requires camera permissions to be added to the Info.plist file.
/// - Important: The app requires at least iOS 16.0 to run.
class ViewController: UIViewController {

  /// The YOLO view that handles camera capture, model inference, and visualization.
  var yoloView: YOLOView!

  override func viewDidLoad() {
    super.viewDidLoad()
    // Initialize YOLOView with a detection model.
    // Change the model or task to use segmentation, semantic segmentation, classification, pose, or OBB.
    yoloView = YOLOView(frame: view.bounds, modelPathOrName: "yolo26n", task: .detect)
    view.addSubview(yoloView)
  }
}
