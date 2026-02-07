// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import CoreVideo

/// Camera position for capture.
public enum CameraPosition: Sendable {
  case front
  case back

  var avPosition: AVCaptureDevice.Position {
    switch self {
    case .front: return .front
    case .back: return .back
    }
  }
}

/// Provides camera frames as an AsyncStream of CVPixelBuffer.
public final class CameraProvider: NSObject, @unchecked Sendable {
  /// The capture session for external preview layer access.
  public let captureSession = AVCaptureSession()

  private let videoOutput = AVCaptureVideoDataOutput()
  private let cameraQueue = DispatchQueue(label: "com.ultralytics.camera-queue")
  private var continuation: AsyncStream<CVPixelBuffer>.Continuation?
  private var captureDevice: AVCaptureDevice?

  /// Creates a camera provider for the specified position.
  public func start(position: CameraPosition = .back) async throws -> AsyncStream<CVPixelBuffer> {
    // Check camera permission
    let status = AVCaptureDevice.authorizationStatus(for: .video)
    switch status {
    case .authorized:
      break
    case .notDetermined:
      let granted = await AVCaptureDevice.requestAccess(for: .video)
      guard granted else { throw PredictorError.cameraPermissionDenied }
    default:
      throw PredictorError.cameraPermissionDenied
    }

    let stream = AsyncStream<CVPixelBuffer> { continuation in
      self.continuation = continuation
      continuation.onTermination = { @Sendable _ in
        // Cleanup when stream ends
      }
    }

    try setupCamera(position: position)
    cameraQueue.async { [weak self] in
      self?.captureSession.startRunning()
    }

    return stream
  }

  /// Pauses frame capture.
  public func pause() {
    cameraQueue.async { [weak self] in
      self?.captureSession.stopRunning()
    }
  }

  /// Resumes frame capture.
  public func resume() {
    cameraQueue.async { [weak self] in
      guard let self, !self.captureSession.isRunning else { return }
      self.captureSession.startRunning()
    }
  }

  /// Stops capture and ends the stream.
  public func stop() {
    cameraQueue.async { [weak self] in
      guard let self else { return }
      self.captureSession.stopRunning()
      self.continuation?.finish()
      self.continuation = nil
    }
  }

  private func setupCamera(position: CameraPosition) throws {
    captureSession.beginConfiguration()
    captureSession.sessionPreset = .hd1280x720

    guard let device = bestCaptureDevice(position: position.avPosition) else {
      captureSession.commitConfiguration()
      throw PredictorError.cameraSetupFailed
    }
    captureDevice = device

    let videoInput = try AVCaptureDeviceInput(device: device)
    guard captureSession.canAddInput(videoInput) else {
      captureSession.commitConfiguration()
      throw PredictorError.cameraSetupFailed
    }
    captureSession.addInput(videoInput)

    // Configure IOSurface-backed pixel buffers for zero-copy ANE access
    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA),
      kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)

    guard captureSession.canAddOutput(videoOutput) else {
      captureSession.commitConfiguration()
      throw PredictorError.cameraSetupFailed
    }
    captureSession.addOutput(videoOutput)

    // Set orientation after adding output
    if let connection = videoOutput.connection(with: .video) {
      connection.videoOrientation = .portrait
      if position == .front {
        connection.isVideoMirrored = true
      }
    }

    // Configure auto-focus
    do {
      try device.lockForConfiguration()
      if device.isFocusModeSupported(.continuousAutoFocus), device.isFocusPointOfInterestSupported {
        device.focusMode = .continuousAutoFocus
        device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
      }
      device.exposureMode = .continuousAutoExposure
      device.unlockForConfiguration()
    } catch {
      // Non-fatal: continue without auto-focus
    }

    captureSession.commitConfiguration()
  }

  private func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
    if let device = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: position) {
      return device
    } else if let device = AVCaptureDevice.default(
      .builtInWideAngleCamera, for: .video, position: position)
    {
      return device
    }
    return AVCaptureDevice.default(for: .video)
  }
}

extension CameraProvider: AVCaptureVideoDataOutputSampleBufferDelegate {
  public func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    let sendableBuffer = UnsafeSendable(pixelBuffer)
    continuation?.yield(sendableBuffer.value)
  }
}
/// Wrapper to suppress Sendable checking for types that are manually managed for thread safety
private struct UnsafeSendable<T>: @unchecked Sendable {
  let value: T
  init(_ value: T) {
    self.value = value
  }
}
