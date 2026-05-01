// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, managing camera capture for real-time inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The VideoCapture component manages the camera and video processing pipeline for real-time
//  object detection. It handles setting up the AVCaptureSession, managing camera devices,
//  configuring camera properties like focus and exposure, and processing video frames for
//  model inference. The class delivers capture frames to the predictor component for real-time
//  analysis and returns results through delegate callbacks. It also supports camera controls
//  such as switching between front and back cameras, zooming, and capturing still photos.

import AVFoundation
import CoreVideo
import UIKit
import Vision

let physicalLensTypes: [AVCaptureDevice.DeviceType] = [
  .builtInUltraWideCamera,
  .builtInWideAngleCamera,
  .builtInTelephotoCamera,
]

/// Protocol for receiving video capture frame processing results.
@MainActor
public protocol VideoCaptureDelegate: AnyObject {
  func onPredict(result: YOLOResult)
  func onInferenceTime(speed: Double, fps: Double)
}

func captureDevices(position: AVCaptureDevice.Position) -> [AVCaptureDevice] {
  if position == .front {
    return bestCaptureDevice(position: position).map { [$0] } ?? []
  }

  let discoverySession = AVCaptureDevice.DiscoverySession(
    deviceTypes: physicalLensTypes,
    mediaType: .video,
    position: position
  )

  return discoverySession.devices
    .sorted { $0.deviceType.lensSortOrder < $1.deviceType.lensSortOrder }
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
  let preferredTypes: [AVCaptureDevice.DeviceType] =
    position == .back
    ? [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera]
    : [.builtInTrueDepthCamera, .builtInWideAngleCamera]

  for deviceType in preferredTypes {
    if let device = AVCaptureDevice.default(deviceType, for: .video, position: position) {
      return device
    }
  }

  return nil
}

func zoomFactor(for lensDevice: AVCaptureDevice, on virtualDevice: AVCaptureDevice) -> CGFloat? {
  guard lensDevice.position == virtualDevice.position else { return nil }
  let constituentDevices = virtualDevice.constituentDevices
    .filter { physicalLensTypes.contains($0.deviceType) }
    .sorted { $0.deviceType.lensSortOrder < $1.deviceType.lensSortOrder }
  guard constituentDevices.count > 1 else { return nil }

  let lensIndex =
    constituentDevices.firstIndex { $0.uniqueID == lensDevice.uniqueID }
    ?? constituentDevices.firstIndex { $0.deviceType == lensDevice.deviceType }
  guard let lensIndex else { return nil }

  let switchOverZoomFactors = virtualDevice.virtualDeviceSwitchOverVideoZoomFactors.map {
    CGFloat(truncating: $0)
  }
  let zoomFactors = [virtualDevice.minAvailableVideoZoomFactor] + switchOverZoomFactors
  guard lensIndex < zoomFactors.count else { return nil }

  return min(
    max(zoomFactors[lensIndex], virtualDevice.minAvailableVideoZoomFactor),
    virtualDevice.maxAvailableVideoZoomFactor
  )
}

func displayZoomFactor(_ zoomFactor: CGFloat, for device: AVCaptureDevice) -> CGFloat {
  if #available(iOS 18.0, *) {
    return zoomFactor * device.displayVideoZoomFactorMultiplier
  }
  return zoomFactor
}

func displayZoomFactor(
  for lensDevice: AVCaptureDevice, activeDevice: AVCaptureDevice?
) -> CGFloat? {
  let candidates = [activeDevice, bestCaptureDevice(position: lensDevice.position)].compactMap {
    $0
  }
  for candidate in candidates {
    if let zoomFactor = zoomFactor(for: lensDevice, on: candidate) {
      return displayZoomFactor(zoomFactor, for: candidate)
    }
  }
  return nil
}

extension AVCaptureDevice.DeviceType {
  fileprivate var lensSortOrder: Int {
    switch self {
    case .builtInUltraWideCamera: return 0
    case .builtInWideAngleCamera: return 1
    case .builtInTelephotoCamera: return 2
    default: return 3
    }
  }
}

extension AVCaptureVideoOrientation {
  init?(_ interfaceOrientation: UIInterfaceOrientation) {
    switch interfaceOrientation {
    case .portrait: self = .portrait
    case .portraitUpsideDown: self = .portraitUpsideDown
    case .landscapeLeft: self = .landscapeLeft
    case .landscapeRight: self = .landscapeRight
    default: return nil
    }
  }

  /// Maps a `UIDeviceOrientation` to the matching video orientation. Unknown device
  /// orientations (face-up/down) return `nil` so callers can preserve the existing setting.
  init?(_ deviceOrientation: UIDeviceOrientation) {
    switch deviceOrientation {
    case .portrait: self = .portrait
    case .portraitUpsideDown: self = .portraitUpsideDown
    case .landscapeLeft: self = .landscapeRight
    case .landscapeRight: self = .landscapeLeft
    default: return nil
    }
  }
}

public final class VideoCapture: NSObject, @unchecked Sendable {
  public var predictor: Predictor?
  public var previewLayer: AVCaptureVideoPreviewLayer?
  public weak var delegate: VideoCaptureDelegate?
  var captureDevice: AVCaptureDevice?

  let captureSession = AVCaptureSession()
  var videoInput: AVCaptureDeviceInput? = nil
  let videoOutput = AVCaptureVideoDataOutput()
  let cameraQueue = DispatchQueue(label: "camera-queue")
  var inferenceOK = true
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false

  private var currentBuffer: CVPixelBuffer?
  private lazy var imageContext = CIContext()
  private var frameCaptureCompletion: ((UIImage?) -> Void)?

  deinit {
    captureSession.stopRunning()
    videoOutput.setSampleBufferDelegate(nil, queue: nil)
  }

  public func setUp(
    sessionPreset: AVCaptureSession.Preset = .hd1280x720,
    position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation,
    completion: @escaping @Sendable (Bool) -> Void
  ) {
    cameraQueue.async { [weak self] in
      guard let self = self else {
        DispatchQueue.main.async {
          completion(false)
        }
        return
      }
      let success = self.setUpCamera(
        sessionPreset: sessionPreset, position: position, orientation: orientation)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }

  func setUpCamera(
    sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation
  ) -> Bool {
    captureSession.beginConfiguration()
    defer {
      captureSession.commitConfiguration()
    }
    captureSession.sessionPreset = sessionPreset

    guard let device = bestCaptureDevice(position: position) else {
      return false
    }
    captureDevice = device

    do {
      videoInput = try AVCaptureDeviceInput(device: device)
    } catch {
      YOLOLog.error("Failed to create video input: \(error)")
      return false
    }

    guard let input = videoInput else { return false }
    if captureSession.canAddInput(input) {
      captureSession.addInput(input)
    } else {
      YOLOLog.warning("Cannot add video input to session")
    }
    let videoOrientation = AVCaptureVideoOrientation(orientation) ?? .portrait
    let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
    previewLayer.connection?.videoOrientation = videoOrientation
    configureVideoMirroring(previewLayer.connection, isMirrored: position == .front)
    self.previewLayer = previewLayer

    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }
    // We want the buffers to be in portrait orientation otherwise they are
    // rotated by 90 degrees. Need to set this _after_ addOutput()!
    let connection = videoOutput.connection(with: AVMediaType.video)
    connection?.videoOrientation = videoOrientation
    configureVideoMirroring(connection, isMirrored: position == .front)

    guard configureCameraDevice(device) else {
      return false
    }

    return true
  }

  func selectCaptureDevice(
    _ device: AVCaptureDevice,
    videoOrientation: AVCaptureVideoOrientation,
    completion: @escaping (Bool) -> Void
  ) {
    cameraQueue.async { [weak self] in
      guard let self else {
        DispatchQueue.main.async { completion(false) }
        return
      }

      let success = self.selectCaptureDeviceSync(device, videoOrientation: videoOrientation)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }

  private func selectCaptureDeviceSync(
    _ device: AVCaptureDevice, videoOrientation: AVCaptureVideoOrientation
  ) -> Bool {
    if device.position == .back,
      selectVirtualRearLens(device, videoOrientation: videoOrientation)
    {
      return true
    }

    guard captureDevice?.uniqueID != device.uniqueID else { return true }

    return switchCaptureInput(to: device, videoOrientation: videoOrientation)
  }

  private func selectVirtualRearLens(
    _ lensDevice: AVCaptureDevice, videoOrientation: AVCaptureVideoOrientation
  ) -> Bool {
    let candidates = [captureDevice, bestCaptureDevice(position: .back)].compactMap { $0 }
    for virtualDevice in candidates {
      guard let zoomFactor = zoomFactor(for: lensDevice, on: virtualDevice) else { continue }
      if captureDevice?.uniqueID != virtualDevice.uniqueID,
        !switchCaptureInput(to: virtualDevice, videoOrientation: videoOrientation)
      {
        return false
      }
      return rampZoom(to: zoomFactor, on: virtualDevice)
    }
    return false
  }

  private func switchCaptureInput(
    to device: AVCaptureDevice, videoOrientation: AVCaptureVideoOrientation
  ) -> Bool {
    let newInput: AVCaptureDeviceInput
    do {
      newInput = try AVCaptureDeviceInput(device: device)
    } catch {
      YOLOLog.error("Failed to create video input: \(error)")
      return false
    }

    guard configureCameraDevice(device) else {
      return false
    }

    captureSession.beginConfiguration()
    defer {
      captureSession.commitConfiguration()
    }

    let currentInput = videoInput ?? captureSession.inputs.first as? AVCaptureDeviceInput
    if let currentInput {
      captureSession.removeInput(currentInput)
    }

    guard captureSession.canAddInput(newInput) else {
      if let currentInput, captureSession.canAddInput(currentInput) {
        captureSession.addInput(currentInput)
      }
      return false
    }

    captureSession.addInput(newInput)
    videoInput = newInput
    captureDevice = device

    let videoConnection = videoOutput.connection(with: .video)
    videoConnection?.videoOrientation = videoOrientation
    configureVideoMirroring(videoConnection, isMirrored: device.position == .front)
    previewLayer?.connection?.videoOrientation = videoOrientation
    configureVideoMirroring(previewLayer?.connection, isMirrored: device.position == .front)

    return true
  }

  private func rampZoom(to zoomFactor: CGFloat, on device: AVCaptureDevice) -> Bool {
    do {
      try device.lockForConfiguration()
      defer { device.unlockForConfiguration() }

      let clampedZoomFactor = min(
        max(zoomFactor, device.minAvailableVideoZoomFactor),
        device.maxAvailableVideoZoomFactor
      )
      if device.isRampingVideoZoom {
        device.cancelVideoZoomRamp()
      }
      device.ramp(toVideoZoomFactor: clampedZoomFactor, withRate: 20)
      return true
    } catch {
      YOLOLog.error("Zoom configuration failed: \(error.localizedDescription)")
      return false
    }
  }

  func start() {
    cameraQueue.async { [weak self] in
      guard let self, !self.captureSession.isRunning else { return }
      self.captureSession.startRunning()
    }
  }

  func stop() {
    cameraQueue.async { [weak self] in
      guard let self, self.captureSession.isRunning else { return }
      self.captureSession.stopRunning()
    }
  }

  func captureNextFrame(completion: @escaping (UIImage?) -> Void) {
    cameraQueue.async { [weak self] in
      guard let self, self.captureSession.isRunning else {
        DispatchQueue.main.async { completion(nil) }
        return
      }
      guard self.frameCaptureCompletion == nil else {
        DispatchQueue.main.async { completion(nil) }
        return
      }
      self.frameCaptureCompletion = completion
    }
  }

  private func predictOnFrame(sampleBuffer: CMSampleBuffer) {
    guard let predictor = predictor else {
      return
    }
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
      }

      predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
      currentBuffer = nil
    }
  }

  func updateVideoOrientation(orientation: AVCaptureVideoOrientation) {
    cameraQueue.async { [weak self] in
      guard let self, let connection = self.videoOutput.connection(with: .video) else { return }

      connection.videoOrientation = orientation
      let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
      self.configureVideoMirroring(connection, isMirrored: currentInput?.device.position == .front)
      self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
      self.configureVideoMirroring(
        self.previewLayer?.connection, isMirrored: connection.isVideoMirrored)
    }
  }

  private func configureVideoMirroring(_ connection: AVCaptureConnection?, isMirrored: Bool) {
    guard let connection else { return }
    guard connection.isVideoMirroringSupported else { return }
    connection.automaticallyAdjustsVideoMirroring = false
    connection.isVideoMirrored = isMirrored
  }

  private func configureCameraDevice(_ device: AVCaptureDevice) -> Bool {
    do {
      try device.lockForConfiguration()
      if device.isFocusModeSupported(.continuousAutoFocus),
        device.isFocusPointOfInterestSupported
      {
        device.focusMode = .continuousAutoFocus
        device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
      }
      if device.isExposureModeSupported(.continuousAutoExposure) {
        device.exposureMode = .continuousAutoExposure
      }
      if #available(iOS 18.0, *), device.position == .back {
        device.videoZoomFactor = min(
          max(1 / device.displayVideoZoomFactorMultiplier, device.minAvailableVideoZoomFactor),
          device.maxAvailableVideoZoomFactor
        )
      }
      device.unlockForConfiguration()
      return true
    } catch {
      YOLOLog.error("Camera configuration failed: \(error.localizedDescription)")
      return false
    }
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  public func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    let requestedFrameCompletion = frameCaptureCompletion
    if requestedFrameCompletion != nil {
      frameCaptureCompletion = nil
    }
    defer {
      if let requestedFrameCompletion {
        let capturedImage = CMSampleBufferGetImageBuffer(sampleBuffer).flatMap { pixelBuffer in
          let image = CIImage(cvPixelBuffer: pixelBuffer)
          return imageContext.createCGImage(image, from: image.extent).map {
            UIImage(cgImage: $0)
          }
        }
        DispatchQueue.main.async {
          requestedFrameCompletion(capturedImage)
        }
      }
    }
    guard inferenceOK else { return }
    predictOnFrame(sampleBuffer: sampleBuffer)
  }
}

extension VideoCapture: ResultsListener, InferenceTimeListener {
  public func on(inferenceTime: Double, fpsRate: Double) {
    DispatchQueue.main.async { [weak self] in
      self?.delegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
    }
  }

  public func on(result: YOLOResult) {
    DispatchQueue.main.async { [weak self] in
      self?.delegate?.onPredict(result: result)
    }
  }
}
