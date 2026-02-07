// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import SwiftUI
import YOLOCore

/// SwiftUI camera view with real-time YOLO inference and task-appropriate overlays.
///
/// ```swift
/// // Minimal — uses built-in overlays
/// YOLOCamera(model: "yolo26n", task: .detect)
///
/// // Custom result handling
/// YOLOCamera(model: "yolo26n", task: .detect) { result in
///     // build your own UI from result
/// }
/// ```
public struct YOLOCamera: View {
  let modelPathOrName: String
  let task: YOLOTask
  let onResult: ((YOLOResult) -> AnyView)?

  @State private var session: YOLOSession?
  @State private var latestResult: YOLOResult?
  @State private var error: Error?
  @State private var isLoading = true

  /// Creates a YOLOCamera with built-in overlays for the given task.
  public init(model: String, task: YOLOTask) {
    self.modelPathOrName = model
    self.task = task
    self.onResult = nil
  }

  /// Creates a YOLOCamera with a custom result view builder.
  public init<Content: View>(
    model: String, task: YOLOTask,
    @ViewBuilder content: @escaping (YOLOResult) -> Content
  ) {
    self.modelPathOrName = model
    self.task = task
    self.onResult = { result in AnyView(content(result)) }
  }

  public var body: some View {
    GeometryReader { geometry in
      ZStack {
        if let error {
          Text(error.localizedDescription)
            .foregroundStyle(.white)
            .padding()
        } else if isLoading {
          ProgressView("Loading model...")
            .foregroundStyle(.white)
        } else if let session {
          // Camera preview
          CameraPreview(session: session.cameraProvider.captureSession)
            .ignoresSafeArea()

          // Overlay — must ignore safe area to match CameraPreview's full-screen bounds
          if let result = latestResult {
            if let onResult {
              onResult(result)
            } else {
              overlayForTask(result: result, size: geometry.size)
                .ignoresSafeArea()
            }
          }
        }
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .background(Color.black)
    }
    .task {
      await startSession()
    }
    .onDisappear {
      session?.stop()
    }
  }

  @MainActor
  private func startSession() async {
    do {
      let model = try await YOLO(modelPathOrName, task: task)
      let newSession = try await YOLOSession(model: model, camera: .back)
      self.session = newSession
      self.isLoading = false

      // Consume results stream
      for await result in newSession.results {
        self.latestResult = result
      }
    } catch {
      self.error = error
      self.isLoading = false
    }
  }

  @ViewBuilder
  private func overlayForTask(result: YOLOResult, size: CGSize) -> some View {
    let frameSize = result.orig_shape
    switch task {
    case .detect:
      DetectionOverlay(boxes: result.boxes, frameSize: frameSize, viewSize: size)
    case .segment:
      SegmentationOverlay(
        boxes: result.boxes, masks: result.masks, frameSize: frameSize, viewSize: size)
    case .pose:
      PoseOverlay(
        boxes: result.boxes, keypointsList: result.keypointsList, frameSize: frameSize,
        viewSize: size)
    case .obb:
      OBBOverlay(obbResults: result.obb, frameSize: frameSize, viewSize: size)
    case .classify:
      ClassificationBanner(probs: result.probs)
    }
  }
}
