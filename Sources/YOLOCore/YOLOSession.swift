// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import Foundation

/// Real-time camera inference session that streams YOLO results.
///
/// ```swift
/// let model = try await YOLO("yolo26n", task: .detect)
/// let session = try await YOLOSession(model: model, camera: .back)
///
/// for await result in session.results {
///     print(result.boxes)
/// }
/// ```
@MainActor
public final class YOLOSession {
  /// The YOLO model used for inference.
  public let model: YOLO

  /// The camera provider.
  public let cameraProvider: CameraProvider

  /// Mutable configuration for thresholds. Changes take effect on the next frame.
  public var configuration: YOLOConfiguration {
    get { model.configuration }
    set { model.configuration = newValue }
  }

  /// The most recent result, suitable for SwiftUI @Observable binding.
  public private(set) var latestResult: YOLOResult?

  /// The results stream.
  public let results: AsyncStream<YOLOResult>
  private let resultsContinuation: AsyncStream<YOLOResult>.Continuation

  private var inferenceTask: Task<Void, Never>?

  /// Creates a new session with the given model and camera position.
  public init(model: YOLO, camera: CameraPosition = .back) async throws {
    self.model = model
    self.cameraProvider = CameraProvider()

    let (stream, continuation) = AsyncStream<YOLOResult>.makeStream()
    self.results = stream
    self.resultsContinuation = continuation

    let frameStream = try await cameraProvider.start(position: camera)

    // Start inference loop
    inferenceTask = Task.detached { [weak self] in
      for await pixelBuffer in frameStream {
        guard let self = await self else { break }
        let result = await self.runInference(pixelBuffer: pixelBuffer)
        await self.updateLatestResult(result)
        await self.yieldResult(result)
      }
    }
  }

  /// Pauses the camera and inference.
  public func pause() {
    cameraProvider.pause()
  }

  /// Resumes the camera and inference.
  public func resume() {
    cameraProvider.resume()
  }

  /// Stops the session permanently.
  public func stop() {
    inferenceTask?.cancel()
    inferenceTask = nil
    cameraProvider.stop()
    resultsContinuation.finish()
  }

  deinit {
    inferenceTask?.cancel()
    cameraProvider.stop()
    resultsContinuation.finish()
  }

  // MARK: - Private

  private nonisolated func runInference(pixelBuffer: CVPixelBuffer) -> YOLOResult {
    model(pixelBuffer)
  }

  private func updateLatestResult(_ result: YOLOResult) {
    latestResult = result
  }

  private nonisolated func yieldResult(_ result: YOLOResult) {
    resultsContinuation.yield(result)
  }
}
