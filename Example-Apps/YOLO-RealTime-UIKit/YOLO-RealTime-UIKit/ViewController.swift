//
//  ViewController.swift
//  YOLO-RealTime-UIKit
//
//  Created by Ultralytics
//  License: MIT
//

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
    yoloView = YOLOView(frame: view.bounds, modelPathOrName: "yolo11m-seg", task: .segment)
    view.addSubview(yoloView)
  }
}
