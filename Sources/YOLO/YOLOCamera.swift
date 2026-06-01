// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing real-time camera-based object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLOCamera is a SwiftUI wrapper around YOLOView that runs real-time inference on the device camera. It manages
//  camera setup, frame capture, model loading, and inference, and forwards results through an optional callback.

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
  ///   - task: The YOLO task to run (detect/segment/semantic/classify/pose/obb). Defaults to `.detect`.
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
  ///   - task: The YOLO task to run (detect/segment/semantic/classify/pose/obb). Defaults to `.detect`.
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
