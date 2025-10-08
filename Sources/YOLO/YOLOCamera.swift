// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing real-time camera-based object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOCamera component provides a SwiftUI view for real-time object detection using device cameras.
//  It wraps the underlying YOLOView component to provide a clean SwiftUI interface for camera feed processing,
//  model inference, and result display. The component automatically handles camera setup, frame capture,
//  model loading, and inference processing, making it simple to add real-time object detection capabilities
//  to SwiftUI applications with minimal code. Results are exposed through a callback for custom handling
//  of detection results.

import AVFoundation
import SwiftUI

/// A SwiftUI view that provides real-time camera-based object detection using YOLO models.
public struct YOLOCamera: View {
  @State private var yoloResult: YOLOResult?

  public let modelPathOrName: String?
  public let modelURL: URL?
  public let task: YOLOTask
  public let cameraPosition: AVCaptureDevice.Position
  public let onDetection: ((YOLOResult) -> Void)? 

  public init(
    modelPathOrName: String,
    task: YOLOTask = .detect,
    cameraPosition: AVCaptureDevice.Position = .back,
    onDetection: ((YOLOResult) -> Void)? = nil 
  ) {
    self.modelPathOrName = modelPathOrName
    self.modelURL = nil
    self.task = task
    self.cameraPosition = cameraPosition
    self.onDetection = onDetection  
  }

  public init(
    url: URL,
    task: YOLOTask = .detect,
    cameraPosition: AVCaptureDevice.Position = .back,
    onDetection: ((YOLOResult) -> Void)? = nil
  ) {
    self.modelPathOrName = nil
    self.modelURL = url
    self.task = task
    self.cameraPosition = cameraPosition
    self.onDetection = onDetection 
  }

  public var body: some View {
    YOLOViewRepresentable(
      modelPathOrName: modelPathOrName,
      modelURL: modelURL,
      task: task,
      cameraPosition: cameraPosition
    ) { result in
      self.yoloResult = result
      self.onDetection?(result) 
    }
  }
}

struct YOLOViewRepresentable: UIViewRepresentable {
  let modelPathOrName: String?
  let modelURL: URL?
  let task: YOLOTask
  let cameraPosition: AVCaptureDevice.Position
  let onDetection: ((YOLOResult) -> Void)?

  func makeUIView(context: Context) -> YOLOView {
    let modelPath = modelURL?.path ?? modelPathOrName ?? ""
    assert(!modelPath.isEmpty, "Either modelPathOrName or modelURL must be provided")
    return YOLOView(frame: .zero, modelPathOrName: modelPath, task: task)
  }

  func updateUIView(_ uiView: YOLOView, context: Context) {
    uiView.onDetection = onDetection
  }
}
