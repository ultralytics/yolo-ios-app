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
  public let modelPathOrName: String?
  public let modelURL: URL?
  public let task: YOLOTask
  public let cameraPosition: AVCaptureDevice.Position
  public let onDetection: ((YOLOResult) -> Void)?

  /// Creates a camera view that loads a model from the app bundle or a local file path.
  ///
  /// - Parameters:
  ///   - modelPathOrName: A resource name to look up in the main bundle (e.g. `"yolo11n"` — the
  ///     initializer searches for `.mlmodelc` then `.mlpackage`) or an absolute filesystem path
  ///     to a `.mlmodel`/`.mlpackage` file.
  ///   - task: The YOLO task to run (detect/segment/classify/pose/obb). Defaults to `.detect`.
  ///   - cameraPosition: Which camera to use. Defaults to `.back`.
  ///   - onDetection: Optional callback fired with each frame's inference result.
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

  /// Creates a camera view that downloads (and caches) a model from a remote URL.
  ///
  /// - Parameters:
  ///   - url: Remote URL pointing at a zipped `.mlpackage` or `.mlmodel`. Cached locally by
  ///     `YOLOModelCache` so subsequent launches skip the download.
  ///   - task: The YOLO task to run (detect/segment/classify/pose/obb). Defaults to `.detect`.
  ///   - cameraPosition: Which camera to use. Defaults to `.back`.
  ///   - onDetection: Optional callback fired with each frame's inference result.
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
    let view = YOLOView(frame: .zero, modelPathOrName: modelPath, task: task)
    if cameraPosition == .front {
      view.pendingCameraPosition = .front
    }
    return view
  }

  func updateUIView(_ uiView: YOLOView, context: Context) {
    uiView.onDetection = onDetection
  }
}
