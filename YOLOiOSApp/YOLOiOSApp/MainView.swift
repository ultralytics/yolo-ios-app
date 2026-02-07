// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOUI

/// Primary view for the YOLO iOS App with task selection, model selection, and camera inference.
struct MainView: View {
  @State private var selectedTaskIndex = 2  // Default: Detect
  @State private var selectedModelSize: ModelSize = .n
  @State private var session: YOLOSession?
  @State private var latestResult: YOLOResult?
  @State private var isLoading = false
  @State private var loadingMessage = ""
  @State private var errorMessage: String?

  private let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
    ("Classify", "Models/Classify", .classify),
    ("Segment", "Models/Segment", .segment),
    ("Detect", "Models/Detect", .detect),
    ("Pose", "Models/Pose", .pose),
    ("OBB", "Models/OBB", .obb),
  ]

  var body: some View {
    GeometryReader { geometry in
      ZStack {
        Color.black.ignoresSafeArea()

        // Camera preview + overlay
        if let session {
          CameraPreview(session: session.cameraProvider.captureSession)
            .ignoresSafeArea()

          if let result = latestResult {
            overlayForTask(result: result, size: geometry.size)
          }
        }

        // Loading overlay
        if isLoading {
          Color.black.opacity(0.5)
            .ignoresSafeArea()
          VStack(spacing: 12) {
            ProgressView()
              .tint(.white)
            Text(loadingMessage)
              .font(.callout)
              .foregroundStyle(.white)
          }
        }

        // Controls overlay
        VStack {
          controlsHeader
          Spacer()
          statusFooter
        }
      }
    }
    .task {
      await loadModel()
    }
  }

  // MARK: - Controls

  private var controlsHeader: some View {
    VStack(spacing: 8) {
      // Task picker
      Picker("Task", selection: $selectedTaskIndex) {
        ForEach(tasks.indices, id: \.self) { index in
          Text(tasks[index].name).tag(index)
        }
      }
      .pickerStyle(.segmented)
      .onChange(of: selectedTaskIndex) { _, _ in
        selectedModelSize = .n
        Task { await loadModel() }
      }

      // Model size picker
      Picker("Size", selection: $selectedModelSize) {
        ForEach(ModelSize.allCases, id: \.self) { size in
          Text(size.rawValue.uppercased()).tag(size)
        }
      }
      .pickerStyle(.segmented)
      .onChange(of: selectedModelSize) { _, _ in
        Task { await loadModel() }
      }
    }
    .padding(.horizontal)
    .padding(.top, 8)
  }

  private var statusFooter: some View {
    HStack {
      if let result = latestResult {
        Text(String(format: "%.1f FPS", result.fps ?? 0))
          .font(.caption.monospacedDigit())
          .foregroundStyle(.white)

        Spacer()

        Text(currentModelDisplayName)
          .font(.caption)
          .foregroundStyle(.white)
      }

      if let errorMessage {
        Text(errorMessage)
          .font(.caption)
          .foregroundStyle(.red)
      }
    }
    .padding(.horizontal)
    .padding(.bottom, 8)
  }

  // MARK: - Model Loading

  private var currentTask: (name: String, folder: String, yoloTask: YOLOTask) {
    tasks[selectedTaskIndex]
  }

  private var currentModelDisplayName: String {
    let taskSuffix: String
    switch currentTask.yoloTask {
    case .detect: taskSuffix = ""
    case .segment: taskSuffix = "-seg"
    case .classify: taskSuffix = "-cls"
    case .pose: taskSuffix = "-pose"
    case .obb: taskSuffix = "-obb"
    }
    return "yolo26\(selectedModelSize.rawValue)\(taskSuffix)"
  }

  @MainActor
  private func loadModel() async {
    isLoading = true
    loadingMessage = "Loading \(currentModelDisplayName)..."
    errorMessage = nil

    // Stop existing session
    session?.stop()
    session = nil
    latestResult = nil

    do {
      // Try to find model in bundle folder first
      let modelName = resolveModelPath()
      let model = try await YOLO(modelName, task: currentTask.yoloTask)
      let newSession = try await YOLOSession(model: model, camera: .back)
      self.session = newSession
      self.isLoading = false

      // Consume results
      for await result in newSession.results {
        self.latestResult = result
      }
    } catch {
      self.errorMessage = error.localizedDescription
      self.isLoading = false
    }
  }

  private func resolveModelPath() -> String {
    let taskSuffix: String
    switch currentTask.yoloTask {
    case .detect: taskSuffix = ""
    case .segment: taskSuffix = "-seg"
    case .classify: taskSuffix = "-cls"
    case .pose: taskSuffix = "-pose"
    case .obb: taskSuffix = "-obb"
    }

    // Try to find model in task-specific folder
    let folder = currentTask.folder
    if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
      let fileManager = FileManager.default
      if let contents = try? fileManager.contentsOfDirectory(
        at: folderURL, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
      {
        // Find matching model by size
        let sizeChar = selectedModelSize.rawValue
        for url in contents {
          let name = url.deletingPathExtension().lastPathComponent.lowercased()
          if name.contains(sizeChar) && name.hasPrefix("yolo") {
            return url.path
          }
        }
        // Fall back to first available model
        if let first = contents.first(where: {
          ["mlmodelc", "mlpackage", "mlmodel"].contains($0.pathExtension)
        }) {
          return first.path
        }
      }
    }

    // Fall back to bundle root with standard naming
    return "yolo26\(selectedModelSize.rawValue)\(taskSuffix)"
  }

  // MARK: - Overlay

  @ViewBuilder
  private func overlayForTask(result: YOLOResult, size: CGSize) -> some View {
    switch currentTask.yoloTask {
    case .detect:
      DetectionOverlay(boxes: result.boxes, viewSize: size)
    case .segment:
      SegmentationOverlay(boxes: result.boxes, masks: result.masks, viewSize: size)
    case .pose:
      PoseOverlay(boxes: result.boxes, keypointsList: result.keypointsList, viewSize: size)
    case .obb:
      OBBOverlay(obbResults: result.obb, viewSize: size)
    case .classify:
      ClassificationBanner(probs: result.probs)
    }
  }
}

/// Model size enum for the App Store app.
enum ModelSize: String, CaseIterable {
  case n, s, m, l, x
}
