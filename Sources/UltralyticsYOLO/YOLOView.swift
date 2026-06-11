// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing the core UI component for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLOView is the primary UIView for real-time YOLO inference. It owns the camera session, loads the model, runs
//  the video frame pipeline, and renders detection overlays — bounding boxes, segmentation masks, pose skeletons,
//  or oriented boxes depending on the active task. It also exposes UI controls for confidence/IoU/max-detections
//  and supports pinch-to-zoom and photo capture with overlays burned in.

import AVFoundation
import UIKit
import Vision

func aspectFillDisplayRect(for normalizedRect: CGRect, imageSize: CGSize, viewSize: CGSize)
  -> CGRect
{
  guard imageSize.width > 0, imageSize.height > 0, viewSize.width > 0, viewSize.height > 0 else {
    return .zero
  }
  let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
  let scaledImageSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
  let offset = CGPoint(
    x: (scaledImageSize.width - viewSize.width) / 2,
    y: (scaledImageSize.height - viewSize.height) / 2
  )
  return CGRect(
    x: normalizedRect.minX * imageSize.width * scale - offset.x,
    y: normalizedRect.minY * imageSize.height * scale - offset.y,
    width: normalizedRect.width * imageSize.width * scale,
    height: normalizedRect.height * imageSize.height * scale
  )
}

/// Delegate that receives per-frame performance metrics and YOLO inference results from a `YOLOView`.
public protocol YOLOViewDelegate: AnyObject {
  /// Called when performance metrics (FPS and inference time) are updated.
  func yoloView(_: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double)

  /// Called when a new inference result is available.
  func yoloView(_: YOLOView, didReceiveResult result: YOLOResult)
}

extension YOLOViewDelegate {
  /// Default no-op so conformers only implement the callbacks they need.
  public func yoloView(_: YOLOView, didReceiveResult: YOLOResult) {}
}

private let defaultMaxDetectionItems = 100

/// A UIView that runs real-time YOLO inference and renders detection, segmentation, pose, or OBB overlays.
@MainActor
public final class YOLOView: UIView, VideoCaptureDelegate {

  /// Delegate receiving performance metrics and YOLO inference results for each frame.
  public weak var delegate: YOLOViewDelegate?

  public func onInferenceTime(speed: Double, fps: Double) {
    self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // speed arrives already in ms
    self.delegate?.yoloView(self, didUpdatePerformance: fps, inferenceTime: speed)
  }

  public func onPredict(result: YOLOResult) {
    // Notify consumers of detection results regardless of overlay visibility.
    delegate?.yoloView(self, didReceiveResult: result)
    onDetection?(result)

    let breakdownTotal = result.preMs + result.inferenceMs + result.postMs
    labelBreakdown.text =
      breakdownTotal > 0
      ? String(
        format: "%.1f pre · %.1f inference · %.1f post", result.preMs, result.inferenceMs,
        result.postMs)
      : ""

    // Consumers drawing their own overlays disable the built-in rendering below via `showOverlays`;
    // its `didSet` already cleared anything previously drawn, so just skip.
    guard showOverlays else { return }

    task == .obb ? showOBBs(predictions: result) : showBoxes(predictions: result)

    if task == .segment || task == .semantic {
      if let maskImage = task == .segment
        ? result.masks?.combinedMask : result.semanticMask?.maskImage
      {
        guard let maskLayer = self.maskLayer else {
          return
        }
        maskLayer.isHidden = false
        maskLayer.frame = imageFrameInOverlay(for: result.orig_shape)
        maskLayer.contents = maskImage
      }
    } else if task == .classify {
      self.overlayYOLOClassificationsCALayer(on: self, result: result)
    } else if task == .pose {
      self.removeAllSubLayers(parentLayer: poseLayer)
      var keypointList = [[(x: Float, y: Float)]]()
      var confsList = [[Float]]()

      for keypoint in result.keypointsList {
        keypointList.append(keypoint.xyn)
        confsList.append(keypoint.conf)
      }
      guard let poseLayer = poseLayer else { return }
      let imageFrame = imageFrameInOverlay(for: result.orig_shape)
      poseLayer.frame = imageFrame
      drawKeypoints(
        keypointsList: keypointList, confsList: confsList, boundingBoxes: result.boxes,
        on: poseLayer, imageViewSize: imageFrame.size)
    }
  }

  public var onDetection: ((YOLOResult) -> Void)?

  /// Controls whether the built-in prediction overlays (boxes, masks, pose, classification) are drawn.
  /// When `false`, inference and the result callbacks (`delegate`, `onDetection`) keep firing, but nothing is
  /// rendered — letting consumers draw fully custom overlays. Defaults to `true`.
  public var showOverlays: Bool = true {
    didSet {
      if !showOverlays { clearPredictionOverlays() }
    }
  }

  private var videoCapture: VideoCapture
  private var busy = false
  var task = YOLOTask.detect
  var modelName: String = ""
  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  public var sliderNumItems = UISlider()
  public var labelSliderNumItems = UILabel()
  public var sliderConf = UISlider()
  public var labelSliderConf = UILabel()
  public var sliderIoU = UISlider()
  public var labelSliderIoU = UILabel()
  public var labelName = UILabel()
  public var labelFPS = UILabel()
  /// Smaller secondary line under `labelFPS` showing the pre/inference/post breakdown in ms.
  public var labelBreakdown = UILabel()
  public var labelZoom = UILabel()
  public var activityIndicator = UIActivityIndicatorView()
  public var playButton = UIButton()
  public var pauseButton = UIButton()
  public var switchCameraButton = UIButton()
  public var shareButton = UIButton()
  public var infoButton = UIButton()
  public var toolbar = UIView()
  let selection = UISelectionFeedbackGenerator()
  private let lensControl = UISegmentedControl()
  private let lensCaptionLabel = UILabel()
  private let torchButton = UIButton()
  private let torchCaptionLabel = UILabel()
  private var isTorchOn = false
  private var cameraTransitionView: UIView?
  private var lensDevices = [AVCaptureDevice]()
  private var selectedLensDeviceID: String?
  private var cameraSwitchInProgress = false
  private var overlayLayer = CALayer()
  private var maskLayer: CALayer?
  private var poseLayer: CALayer?

  private let minimumZoom: CGFloat = 1.0
  private let maximumZoom: CGFloat = 10.0
  private var lastZoomFactor: CGFloat = 1.0
  private var pausedShareImage: UIImage?

  /// Pending camera position to apply after async camera setup completes.
  public var pendingCameraPosition: AVCaptureDevice.Position?

  deinit {
    NotificationCenter.default.removeObserver(self)
    videoCapture.stop()
  }

  public init(
    frame: CGRect,
    modelPathOrName: String,
    task: YOLOTask
  ) {
    self.videoCapture = VideoCapture()
    super.init(frame: frame)
    setModel(modelPathOrName: modelPathOrName, task: task)
    setUpOrientationChangeNotification()
    self.setUpBoundingBoxViews()
    self.setupUI()
    self.videoCapture.delegate = self
    start(position: .back)
    overlayLayer.frame = bounds
  }

  /// Creates a YOLOView with no model attached; starts the camera preview only.
  public override init(frame: CGRect) {
    self.videoCapture = VideoCapture()
    self.task = .detect  // Default task
    super.init(frame: frame)
    setUpOrientationChangeNotification()
    self.setUpBoundingBoxViews()
    self.setupUI()
    self.videoCapture.delegate = self
    start(position: .back)
    overlayLayer.frame = bounds
  }

  required init?(coder: NSCoder) {
    self.videoCapture = VideoCapture()
    super.init(coder: coder)
  }

  public override func awakeFromNib() {
    super.awakeFromNib()
    setUpOrientationChangeNotification()
    setUpBoundingBoxViews()
    setupUI()
    videoCapture.delegate = self
    start(position: .back)
    overlayLayer.frame = bounds
  }

  public func setModel(
    modelPathOrName: String,
    task: YOLOTask,
    completion: (@Sendable (Result<Void, Error>) -> Void)? = nil
  ) {
    // If modelPathOrName is empty, it means no model was provided yet
    if modelPathOrName.isEmpty {
      self.activityIndicator.stopAnimating()
      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }

    activityIndicator.startAnimating()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    removeClassificationLayers()

    self.task = task
    setupSublayers()

    guard let modelURL = ModelPathResolver.resolve(modelPathOrName) else {
      activityIndicator.stopAnimating()
      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }

    modelName = modelURL.deletingPathExtension().lastPathComponent

    BasePredictor.create(for: task, modelURL: modelURL, isRealTime: true) {
      @Sendable [weak self] result in
      Task { @MainActor in
        guard let self = self else { return }
        self.activityIndicator.stopAnimating()
        switch result {
        case .success(let predictor):
          self.videoCapture.predictor = predictor
          predictor.setNumItemsThreshold(numItems: self.getNumItemsThreshold())
          self.labelName.text = processString(self.modelName)
          completion?(.success(()))
        case .failure(let error):
          YOLOLog.error("Failed to load model: \(error)")
          completion?(.failure(error))
        }
      }
    }
  }

  /// Capture session preset for the camera feed (default `.hd1280x720`).
  ///
  /// The default captures at 720p rather than `.photo`: full-sensor frames must be downscaled to the model input
  /// every frame — the dominant preprocessing cost — and 720p roughly doubles sustained throughput with no accuracy
  /// change for standard 640px models. Models with larger inputs picking out small objects can request a higher
  /// preset (e.g. `.hd1920x1080`); unsupported presets fall back to `.high` then `.photo`. Changing this while the
  /// camera is running restarts the session with the new preset.
  public var captureSessionPreset: AVCaptureSession.Preset = .hd1280x720 {
    didSet {
      guard oldValue != captureSessionPreset, videoCapture.previewLayer != nil else { return }
      let position = videoCapture.captureDevice?.position ?? .back
      videoCapture.stop()
      start(position: position)
    }
  }

  private func start(position: AVCaptureDevice.Position) {
    guard !busy else { return }
    busy = true
    videoCapture.setUp(
      sessionPreset: captureSessionPreset, position: position, videoOrientation: currentVideoOrientation()
    ) {
      [weak self] success in
      Task { @MainActor in
        guard let self = self else { return }
        defer { self.busy = false }
        guard success else { return }
        if let previewLayer = self.videoCapture.previewLayer {
          self.layer.insertSublayer(previewLayer, at: 0)
          previewLayer.frame = self.bounds
          for box in self.boundingBoxViews {
            box.addToLayer(previewLayer)
          }
          previewLayer.addSublayer(self.overlayLayer)
        }
        self.videoCapture.start()

        if let pending = self.pendingCameraPosition, pending != .back {
          self.pendingCameraPosition = nil
          self.switchCameraTapped()
        }
        if let device = self.videoCapture.captureDevice {
          self.lastZoomFactor = device.videoZoomFactor
          self.labelZoom.text = self.zoomLabelText(
            rawZoomFactor: self.lastZoomFactor, device: device)
        }
        self.updateLensControl()
      }
    }
  }

  public func stop() {
    // Stopping the capture session turns the hardware torch off; keep the chip truthful.
    setTorchUI(on: false)
    videoCapture.stop()
  }

  public func resume() {
    pausedShareImage = nil
    videoCapture.start()
  }

  // MARK: - Threshold Configuration Methods

  /// Sets the maximum number of detection items to include in results.
  /// - Parameter numItems: The maximum number of items to include (default is 100).
  public func setNumItemsThreshold(_ numItems: Int) {
    sliderNumItems.value = Float(numItems)
    (videoCapture.predictor as? BasePredictor)?.setNumItemsThreshold(numItems: numItems)
  }

  /// Gets the current maximum number of detection items.
  /// - Returns: The current threshold value.
  public func getNumItemsThreshold() -> Int { Int(sliderNumItems.value) }

  /// Sets the confidence threshold for filtering results.
  /// - Parameter confidence: The confidence threshold value (0.0 to 1.0, default is 0.25).
  public func setConfidenceThreshold(_ confidence: Double) {
    guard validateUnitRange(confidence, name: "Confidence threshold") else { return }
    sliderConf.value = Float(confidence)
    labelSliderConf.text = String(format: "%.2f Confidence Threshold", confidence)
    (videoCapture.predictor as? BasePredictor)?.setConfidenceThreshold(confidence: confidence)
  }

  /// Gets the current confidence threshold.
  /// - Returns: The current confidence threshold value.
  public func getConfidenceThreshold() -> Double { Double(sliderConf.value) }

  /// Sets the IoU (Intersection over Union) threshold for non-maximum suppression.
  /// - Parameter iou: The IoU threshold value (0.0 to 1.0, default is 0.7).
  public func setIouThreshold(_ iou: Double) {
    guard validateUnitRange(iou, name: "IoU threshold") else { return }
    sliderIoU.value = Float(iou)
    labelSliderIoU.text = String(format: "%.2f IoU Threshold", iou)
    (videoCapture.predictor as? BasePredictor)?.setIouThreshold(iou: iou)
  }

  /// Gets the current IoU threshold.
  /// - Returns: The current IoU threshold value.
  public func getIouThreshold() -> Double { Double(sliderIoU.value) }

  /// Sets all thresholds at once.
  /// - Parameters:
  ///   - numItems: The maximum number of items to include.
  ///   - confidence: The confidence threshold value (0.0 to 1.0).
  ///   - iou: The IoU threshold value (0.0 to 1.0).
  public func setThresholds(numItems: Int? = nil, confidence: Double? = nil, iou: Double? = nil) {
    numItems.map { setNumItemsThreshold($0) }
    confidence.map { setConfidenceThreshold($0) }
    iou.map { setIouThreshold($0) }
  }

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      let boxView = BoundingBoxView()

      // Check if this is likely an external display based on view size
      let viewBounds = self.bounds
      let maxDimension = max(viewBounds.width, viewBounds.height)

      // External displays are typically much larger than iPhone screens
      // iPhone screens are typically < 1000 points in their largest dimension
      if maxDimension > 1000 {
        // Scale font size and line width for external display
        // Use a proportional size similar to what's used elsewhere (height * 0.025-0.03)
        let scaledFontSize = max(24, viewBounds.height * 0.03)
        let scaledLineWidth = max(6, viewBounds.height * 0.005)
        boxView.setFontSize(scaledFontSize)
        boxView.setLineWidth(scaledLineWidth)
      }

      boundingBoxViews.append(boxView)
    }
  }

  private func imageFrameInOverlay(for imageSize: CGSize) -> CGRect {
    aspectFillDisplayRect(
      for: CGRect(x: 0, y: 0, width: 1, height: 1),
      imageSize: imageSize,
      viewSize: bounds.size
    )
  }

  func setupMaskLayerIfNeeded() {
    if maskLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      layer.name = "maskLayer"
      layer.magnificationFilter = .linear
      layer.minificationFilter = .linear
      self.overlayLayer.addSublayer(layer)
      self.maskLayer = layer
    }
  }

  func setupPoseLayerIfNeeded() {
    if poseLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      self.overlayLayer.addSublayer(layer)
      self.poseLayer = layer
    }
  }

  public func resetLayers() {
    removeAllSubLayers(parentLayer: maskLayer)
    removeAllSubLayers(parentLayer: poseLayer)
    removeAllSubLayers(parentLayer: overlayLayer)

    maskLayer = nil
    poseLayer = nil
  }

  func setupSublayers() {
    resetLayers()

    switch task {
    case .segment, .semantic:
      setupMaskLayerIfNeeded()
    case .pose:
      setupPoseLayerIfNeeded()
    default: break
    }
  }

  func removeAllSubLayers(parentLayer: CALayer?) {
    guard let parentLayer = parentLayer else { return }
    parentLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
    parentLayer.contents = nil
  }

  func showBoxes(predictions: YOLOResult) {
    let maxVisible = min(predictions.boxes.count, 50, boundingBoxViews.count)

    let viewSize = bounds.size
    for i in 0..<maxVisible {
      let prediction = predictions.boxes[i]
      let rect = aspectFillDisplayRect(
        for: prediction.xywhn,
        imageSize: predictions.orig_shape,
        viewSize: viewSize
      )
      showBox(at: i, prediction: prediction, frame: rect)
    }

    for i in maxVisible..<boundingBoxViews.count {
      boundingBoxViews[i].hide()
    }
  }

  func showOBBs(predictions: YOLOResult) {
    let maxVisible = min(predictions.obb.count, 50, boundingBoxViews.count)

    let viewSize = bounds.size
    for i in 0..<maxVisible {
      let detection = predictions.obb[i]
      let box = detection.box
      let rect = CGRect(
        x: CGFloat(box.cx - box.w / 2),
        y: CGFloat(box.cy - box.h / 2),
        width: CGFloat(box.w),
        height: CGFloat(box.h)
      )
      let frame = aspectFillDisplayRect(
        for: rect,
        imageSize: predictions.orig_shape,
        viewSize: viewSize
      )
      showBox(
        at: i,
        className: detection.cls,
        confidence: CGFloat(detection.confidence),
        classIndex: detection.index,
        frame: frame,
        angle: CGFloat(box.angle)
      )
    }

    for i in maxVisible..<boundingBoxViews.count {
      boundingBoxViews[i].hide()
    }
  }

  /// Configures the ith bounding box view with the prediction's color/label/alpha.
  @inline(__always)
  private func showBox(at index: Int, prediction: Box, frame: CGRect) {
    showBox(
      at: index,
      className: prediction.cls,
      confidence: CGFloat(prediction.conf),
      classIndex: prediction.index,
      frame: frame
    )
  }

  @inline(__always)
  private func showBox(
    at index: Int,
    className: String,
    confidence: CGFloat,
    classIndex: Int,
    frame: CGRect,
    angle: CGFloat? = nil
  ) {
    let color = ultralyticsColors[classIndex % ultralyticsColors.count]
    let label = DetectionLabelStyle.text(className: className, confidence: confidence)
    let alpha = DetectionLabelStyle.alpha(confidence: confidence)
    boundingBoxViews[index].show(
      frame: frame, label: label, color: color, alpha: alpha, angle: angle)
  }

  func removeClassificationLayers() {
    if let sublayers = self.layer.sublayers {
      for layer in sublayers where layer.name == "YOLOOverlayLayer" {
        layer.removeFromSuperlayer()
      }
    }
  }

  /// Hides and clears all built-in prediction overlays (boxes, classification, mask, pose).
  private func clearPredictionOverlays() {
    boundingBoxViews.forEach { $0.hide() }
    removeClassificationLayers()
    maskLayer?.isHidden = true
    maskLayer?.contents = nil
    removeAllSubLayers(parentLayer: poseLayer)
  }

  func overlayYOLOClassificationsCALayer(on view: UIView, result: YOLOResult) {

    removeClassificationLayers()

    let overlayLayer = CALayer()
    overlayLayer.frame = view.bounds
    overlayLayer.name = "YOLOOverlayLayer"

    guard let top1 = result.probs?.top1,
      let top1Conf = result.probs?.top1Conf
    else {
      return
    }

    var colorIndex = 0
    if let index = result.names.firstIndex(of: top1) {
      colorIndex = index % ultralyticsColors.count
    }
    let color = ultralyticsColors[colorIndex]

    let confidence = CGFloat(top1Conf)
    let labelText = DetectionLabelStyle.text(className: top1, confidence: confidence)

    let textLayer = CATextLayer()

    // Check if this is likely an external display and scale font accordingly
    let viewBounds = self.bounds
    let maxDimension = max(viewBounds.width, viewBounds.height)
    let fontSize: CGFloat

    if maxDimension > 1000 {
      fontSize = max(24, viewBounds.height * 0.03)
    } else {
      fontSize = 18
    }

    DetectionLabelStyle.configure(textLayer, fontSize: fontSize)
    textLayer.string = labelText
    let alpha = DetectionLabelStyle.alpha(confidence: confidence)
    textLayer.foregroundColor = UIColor.white.withAlphaComponent(alpha).cgColor
    textLayer.backgroundColor = color.withAlphaComponent(alpha).cgColor
    let textSize = DetectionLabelStyle.size(for: labelText, fontSize: fontSize)
    textLayer.frame = CGRect(
      x: (viewBounds.width - textSize.width) / 2,
      y: (viewBounds.height - textSize.height) / 2,
      width: textSize.width,
      height: textSize.height
    )

    overlayLayer.addSublayer(textLayer)

    view.layer.addSublayer(overlayLayer)
  }

  private func setupUI() {
    labelName.text = processString(modelName)
    labelName.textAlignment = .center
    labelName.font = UIFont.systemFont(ofSize: 24, weight: .medium)
    labelName.textColor = .white
    labelName.font = UIFont.preferredFont(forTextStyle: .title1)
    self.addSubview(labelName)

    labelFPS.text = String(format: "%.1f FPS - %.1f ms", 0.0, 0.0)
    labelFPS.textAlignment = .center
    labelFPS.textColor = .white
    labelFPS.font = UIFont.preferredFont(forTextStyle: .body)
    self.addSubview(labelFPS)

    labelBreakdown.text = ""
    labelBreakdown.textAlignment = .center
    labelBreakdown.textColor = UIColor.white.withAlphaComponent(0.7)
    labelBreakdown.font = UIFont.preferredFont(forTextStyle: .caption1)
    self.addSubview(labelBreakdown)

    labelSliderNumItems.isHidden = true
    configureSlider(sliderNumItems, min: 1, max: 100, value: Float(defaultMaxDetectionItems))
    sliderNumItems.isHidden = true

    labelSliderConf.text = "0.25 Confidence Threshold"
    labelSliderConf.textAlignment = .left
    labelSliderConf.textColor = .white
    labelSliderConf.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderConf)

    configureSlider(sliderConf, min: 0, max: 1, value: 0.25)
    self.addSubview(sliderConf)

    labelSliderIoU.text = "0.70 IoU Threshold"
    labelSliderIoU.textAlignment = .left
    labelSliderIoU.textColor = .white
    labelSliderIoU.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderIoU)

    configureSlider(sliderIoU, min: 0, max: 1, value: 0.7)
    self.addSubview(sliderIoU)

    self.labelSliderConf.text = "0.25 Confidence Threshold"
    self.labelSliderIoU.text = "0.70 IoU Threshold"

    labelZoom.text = "1.00x"
    labelZoom.textColor = .white
    labelZoom.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
    labelZoom.textAlignment = .center
    self.addSubview(labelZoom)

    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)

    playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
    pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
    switchCameraButton = UIButton()
    switchCameraButton.setImage(
      UIImage(systemName: "camera.rotate", withConfiguration: config), for: .normal)
    shareButton.setImage(
      UIImage(systemName: "square.and.arrow.up", withConfiguration: config), for: .normal)
    infoButton.setImage(UIImage(systemName: "info.circle", withConfiguration: config), for: .normal)
    infoButton.accessibilityLabel = "Ultralytics"
    playButton.isEnabled = false
    pauseButton.isEnabled = true
    playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
    pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
    switchCameraButton.addTarget(self, action: #selector(switchCameraTapped), for: .touchUpInside)
    infoButton.addTarget(self, action: #selector(infoTapped), for: .touchUpInside)

    setupToolbar()
    setupLensControl()

    self.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
  }

  /// Applies the shared slider styling and wires up `sliderChanged` as the value-changed handler.
  private func configureSlider(_ slider: UISlider, min: Float, max: Float, value: Float) {
    slider.minimumValue = min
    slider.maximumValue = max
    slider.value = value
    slider.minimumTrackTintColor = .white
    slider.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    slider.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
  }

  /// Configures the toolbar background and adds the play/pause/camera/share/info buttons as subviews.
  private func setupToolbar() {
    toolbar.backgroundColor = .black.withAlphaComponent(0.7)
    [playButton, pauseButton, switchCameraButton, shareButton, infoButton].forEach { button in
      button.tintColor = .white
      toolbar.addSubview(button)
    }
    self.addSubview(toolbar)
  }

  private func setupLensControl() {
    lensControl.isHidden = true
    lensControl.backgroundColor = UIColor.black.withAlphaComponent(0.38)
    lensControl.selectedSegmentTintColor = UIColor.white.withAlphaComponent(0.18)
    lensControl.setTitleTextAttributes(
      [
        .foregroundColor: UIColor.white,
        .font: UIFont.systemFont(ofSize: 13, weight: .semibold),
      ], for: .normal)
    lensControl.setTitleTextAttributes(
      [
        .foregroundColor: UIColor.systemYellow,
        .font: UIFont.systemFont(ofSize: 13, weight: .bold),
      ], for: .selected)
    lensControl.addTarget(self, action: #selector(lensChanged(_:)), for: .valueChanged)
    self.addSubview(lensControl)

    lensCaptionLabel.isHidden = true
    lensCaptionLabel.textAlignment = .center
    lensCaptionLabel.textColor = UIColor.white.withAlphaComponent(0.78)
    lensCaptionLabel.font = UIFont.systemFont(ofSize: 11, weight: .medium)
    self.addSubview(lensCaptionLabel)

    torchButton.isHidden = true
    torchButton.backgroundColor = UIColor.black.withAlphaComponent(0.38)
    // Capsule, matching the lens pill next to it (UISegmentedControl renders as a capsule).
    torchButton.layer.cornerRadius = 17
    torchButton.addTarget(self, action: #selector(torchTapped), for: .touchUpInside)
    self.addSubview(torchButton)

    torchCaptionLabel.isHidden = true
    torchCaptionLabel.text = "Torch on"
    torchCaptionLabel.textColor = .systemYellow
    torchCaptionLabel.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
    // Shrink rather than truncate when the clamped frame is a few points short (375pt-wide triple-lens phones),
    // mirroring the Flutter row's scale-down behavior.
    torchCaptionLabel.adjustsFontSizeToFitWidth = true
    torchCaptionLabel.minimumScaleFactor = 0.7
    self.addSubview(torchCaptionLabel)
    setTorchUI(on: false)
  }

  public override func layoutSubviews() {
    super.layoutSubviews()
    overlayLayer.frame = bounds
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)

    updateLensControlVisibility()

    if isLandscape {
      layoutLandscape()
    } else {
      layoutPortrait()
    }

    self.videoCapture.previewLayer?.frame = self.bounds
    cameraTransitionView?.frame = self.bounds
  }

  public override func didMoveToWindow() {
    super.didMoveToWindow()
    guard window != nil else { return }
    videoCapture.updateVideoOrientation(orientation: currentVideoOrientation())
    videoCapture.frameSizeCaptured = false
  }

  /// Lays out the controls and overlays for landscape orientation.
  private func layoutLandscape() {
    let width = bounds.width
    let height = bounds.height
    let topMargin: CGFloat = 0
    let titleLabelHeight: CGFloat = height * 0.1

    labelName.frame = CGRect(x: 0, y: topMargin, width: width, height: titleLabelHeight)

    let subLabelHeight: CGFloat = height * 0.04
    labelFPS.frame = CGRect(
      x: 0, y: center.y - height * 0.24 - subLabelHeight,
      width: width, height: subLabelHeight
    )
    labelBreakdown.frame = CGRect(
      x: 0, y: labelFPS.frame.maxY, width: width, height: subLabelHeight * 0.8
    )

    let sliderWidth: CGFloat = width * 0.2
    let sliderHeight: CGFloat = height * 0.06

    let bottomMargin: CGFloat = lensControl.isHidden ? 80 : 144
    let totalSliderHeight = (sliderHeight + 3) * 6
    let startY = height - bottomMargin - totalSliderHeight

    let thresholdStartY = startY + (sliderHeight + 3) * 2 + 10

    labelSliderConf.frame = CGRect(
      x: width * 0.1, y: thresholdStartY,
      width: sliderWidth * 1.5, height: sliderHeight
    )

    sliderConf.frame = CGRect(
      x: width * 0.1, y: labelSliderConf.frame.maxY + 3,
      width: sliderWidth, height: sliderHeight
    )

    labelSliderIoU.frame = CGRect(
      x: width * 0.1, y: sliderConf.frame.maxY + 3,
      width: sliderWidth * 1.5, height: sliderHeight
    )

    sliderIoU.frame = CGRect(
      x: width * 0.1, y: labelSliderIoU.frame.maxY + 3,
      width: sliderWidth, height: sliderHeight
    )

    let zoomLabelWidth: CGFloat = width * 0.2
    labelZoom.frame = CGRect(
      x: center.x - zoomLabelWidth / 2, y: self.bounds.maxY - 120,
      width: zoomLabelWidth, height: height * 0.03
    )

    layoutToolbarButtons(width: width, height: height)
    layoutLensControl(width: width, height: height)
  }

  /// Lays out the controls and overlays for portrait orientation.
  private func layoutPortrait() {
    let width = bounds.width
    let height = bounds.height
    let topMargin: CGFloat = 0
    let titleLabelHeight: CGFloat = height * 0.1

    labelName.frame = CGRect(x: 0, y: topMargin, width: width, height: titleLabelHeight)

    let subLabelHeight: CGFloat = height * 0.04
    labelFPS.frame = CGRect(
      x: 0, y: labelName.frame.maxY + 15,
      width: width, height: subLabelHeight
    )
    labelBreakdown.frame = CGRect(
      x: 0, y: labelFPS.frame.maxY, width: width, height: subLabelHeight * 0.8
    )

    let sliderWidth: CGFloat = width * 0.46
    let sliderHeight: CGFloat = height * 0.02
    let leftPadding: CGFloat = 20

    let cameraControlsOffset: CGFloat = lensControl.isHidden ? 0 : 64
    let thresholdStartY =
      center.y + height * 0.16 + sliderHeight * 2 + 40 - cameraControlsOffset

    labelSliderConf.frame = CGRect(
      x: leftPadding, y: thresholdStartY,
      width: sliderWidth * 1.5, height: sliderHeight
    )

    sliderConf.frame = CGRect(
      x: leftPadding, y: labelSliderConf.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )

    labelSliderIoU.frame = CGRect(
      x: leftPadding, y: sliderConf.frame.maxY + 10,
      width: sliderWidth * 1.5, height: sliderHeight
    )

    sliderIoU.frame = CGRect(
      x: leftPadding, y: labelSliderIoU.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )

    let zoomLabelWidth: CGFloat = width * 0.2
    labelZoom.frame = CGRect(
      x: center.x - zoomLabelWidth / 2, y: self.bounds.maxY - 120,
      width: zoomLabelWidth, height: height * 0.03
    )

    layoutToolbarButtons(width: width, height: height)
    layoutLensControl(width: width, height: height)
  }

  /// Lays out toolbar buttons; shared between portrait and landscape layouts.
  private func layoutToolbarButtons(width: CGFloat, height: CGFloat) {
    let toolBarHeight: CGFloat = 66
    let buttonHeight: CGFloat = toolBarHeight * 0.75
    let horizontalInset = buttonHeight * 0.25

    toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
    playButton.frame = CGRect(x: horizontalInset, y: 0, width: buttonHeight, height: buttonHeight)
    pauseButton.frame = CGRect(
      x: playButton.frame.maxX, y: 0, width: buttonHeight, height: buttonHeight
    )
    switchCameraButton.frame = CGRect(
      x: pauseButton.frame.maxX, y: 0, width: buttonHeight, height: buttonHeight
    )
    shareButton.frame = CGRect(
      x: switchCameraButton.frame.maxX, y: 0, width: buttonHeight, height: buttonHeight
    )
    infoButton.frame = CGRect(
      x: width - buttonHeight - horizontalInset, y: 0, width: buttonHeight, height: buttonHeight
    )
  }

  private func layoutLensControl(width: CGFloat, height: CGFloat) {
    guard !lensControl.isHidden else { return }
    let toolBarHeight: CGFloat = 66
    let controlHeight: CGFloat = 34
    let captionHeight: CGFloat = 14
    let zoomHeight: CGFloat = 18
    let controlWidth = min(CGFloat(lensDevices.count) * 60, width - 40)
    lensControl.frame = CGRect(
      x: (width - controlWidth) / 2,
      y: height - toolBarHeight - controlHeight - 14,
      width: controlWidth,
      height: controlHeight
    )
    // Torch chip sits directly right of the centered lens pill, with its "Torch on" note beside it —
    // mirrors the Flutter showcase layout (`yolo-flutter-app/lib/widgets/lens_picker.dart`).
    torchButton.frame = CGRect(
      x: lensControl.frame.maxX + 6,
      y: lensControl.frame.minY,
      width: 41,
      height: controlHeight
    )
    let captionX = torchButton.frame.maxX + 6
    torchCaptionLabel.frame = CGRect(
      x: captionX,
      y: lensControl.frame.minY,
      width: min(60, max(0, width - captionX)),
      height: controlHeight
    )
    lensCaptionLabel.frame = CGRect(
      x: 20,
      y: lensControl.frame.minY - captionHeight - 4,
      width: width - 40,
      height: captionHeight
    )
    labelZoom.frame = CGRect(
      x: center.x - 50,
      y: lensCaptionLabel.frame.minY - zoomHeight - 2,
      width: 100,
      height: zoomHeight
    )
  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation(orientation: currentVideoOrientation())
    videoCapture.frameSizeCaptured = false
  }

  private func currentVideoOrientation() -> AVCaptureVideoOrientation {
    if window?.screen != UIScreen.main {
      return AVCaptureVideoOrientation(UIDevice.current.orientation)
        ?? videoCapture.previewLayer?.connection?.videoOrientation
        ?? .portrait
    }

    if let interfaceOrientation = window?.windowScene?.interfaceOrientation,
      let videoOrientation = AVCaptureVideoOrientation(interfaceOrientation)
    {
      return videoOrientation
    }

    if let videoOrientation = videoCapture.previewLayer?.connection?.videoOrientation {
      return videoOrientation
    }

    return AVCaptureVideoOrientation(UIDevice.current.orientation) ?? .portrait
  }

  @objc public func sliderChanged(_ sender: Any) {
    guard let slider = sender as? UISlider else { return }
    let predictor = videoCapture.predictor as? BasePredictor

    if slider === sliderNumItems {
      let numItems = Int(sliderNumItems.value)
      predictor?.setNumItemsThreshold(numItems: numItems)
    } else if slider === sliderConf {
      let conf = Double(round(100 * sliderConf.value)) / 100
      self.labelSliderConf.text = String(format: "%.2f Confidence Threshold", conf)
      predictor?.setConfidenceThreshold(confidence: conf)
    } else if slider === sliderIoU {
      let iou = Double(round(100 * sliderIoU.value)) / 100
      self.labelSliderIoU.text = String(format: "%.2f IoU Threshold", iou)
      predictor?.setIouThreshold(iou: iou)
    }
  }

  /// Update thresholds programmatically (for external display sync).
  public func updateThresholds(conf: Double, iou: Double, numItems: Int) {
    sliderConf.value = Float(conf)
    sliderIoU.value = Float(iou)
    sliderNumItems.value = Float(numItems)
    self.labelSliderConf.text = String(format: "%.2f Confidence Threshold", conf)
    self.labelSliderIoU.text = String(format: "%.2f IoU Threshold", iou)

    if let predictor = videoCapture.predictor as? BasePredictor {
      predictor.setConfidenceThreshold(confidence: conf)
      predictor.setIouThreshold(iou: iou)
      predictor.setNumItemsThreshold(numItems: numItems)
    }
  }

  @objc func pinch(_ pinch: UIPinchGestureRecognizer) {
    guard let device = videoCapture.captureDevice else { return }

    // Return zoom value between the minimum and maximum zoom values
    func minMaxZoom(_ factor: CGFloat) -> CGFloat {
      return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
    }

    func update(scale factor: CGFloat) {
      do {
        try device.lockForConfiguration()
        defer {
          device.unlockForConfiguration()
        }
        device.videoZoomFactor = factor
      } catch {
        YOLOLog.error("Zoom configuration failed: \(error.localizedDescription)")
      }
    }

    let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
    switch pinch.state {
    case .began, .changed:
      update(scale: newScaleFactor)
      self.labelZoom.text = zoomLabelText(rawZoomFactor: newScaleFactor, device: device)
      updateSelectedLens(rawZoomFactor: newScaleFactor, device: device)
      self.labelZoom.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
    case .ended:
      lastZoomFactor = minMaxZoom(newScaleFactor)
      update(scale: lastZoomFactor)
      updateSelectedLens(rawZoomFactor: lastZoomFactor, device: device)
      self.labelZoom.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
    default: break
    }
  }

  /// Turns the torch on/off and returns the actual resulting state (`false` when the active device has no torch or
  /// configuration fails), keeping the torch chip in sync with the hardware.
  @discardableResult
  public func setTorchMode(_ enabled: Bool) -> Bool {
    var on = false
    defer { setTorchUI(on: on) }
    guard let device = videoCapture.captureDevice, device.hasTorch else { return false }

    do {
      try device.lockForConfiguration()
      defer {
        device.unlockForConfiguration()
      }

      if enabled {
        try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
      } else {
        device.torchMode = .off
      }
      on = device.torchMode == .on
      return on
    } catch {
      YOLOLog.error("Torch configuration failed: \(error.localizedDescription)")
      return false
    }
  }

  @objc private func torchTapped() {
    selection.selectionChanged()
    setTorchMode(!isTorchOn)
  }

  /// Syncs the torch chip and its "Torch on" note to the given state, matching the Flutter showcase torch chip.
  /// 13pt is the SF Symbols font size whose rendered bolt matches Flutter's 17pt icon box.
  private func setTorchUI(on: Bool) {
    isTorchOn = on
    let config = UIImage.SymbolConfiguration(pointSize: 13, weight: .regular, scale: .default)
    torchButton.setImage(
      UIImage(systemName: on ? "bolt.fill" : "bolt.slash.fill", withConfiguration: config),
      for: .normal)
    torchButton.tintColor = on ? .systemYellow : .white
    torchButton.accessibilityLabel = on ? "Turn torch off" : "Turn torch on"
    torchCaptionLabel.isHidden = torchButton.isHidden || !on
  }

  @objc func playTapped() {
    selection.selectionChanged()
    pausedShareImage = nil
    self.videoCapture.start()
    playButton.isEnabled = false
    pauseButton.isEnabled = true
  }

  @objc func pauseTapped() {
    selection.selectionChanged()
    playButton.isEnabled = true
    pauseButton.isEnabled = false
    // Stopping the capture session turns the hardware torch off; keep the chip truthful.
    setTorchUI(on: false)
    videoCapture.captureNextFrame { [weak self] image in
      self?.pausedShareImage = image
      self?.videoCapture.stop()
    }
  }

  @objc func switchCameraTapped() {
    selection.selectionChanged()
    pausedShareImage = nil

    let currentPosition = videoCapture.captureDevice?.position ?? .back
    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back
    let newCameraDevice =
      nextCameraPosition == .back
      ? captureDevices(position: .back).first { $0.deviceType == .builtInWideAngleCamera }
      : bestCaptureDevice(position: nextCameraPosition)
    guard let newCameraDevice else { return }
    switchToCamera(newCameraDevice)
  }

  @objc private func lensChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()
    guard sender.selectedSegmentIndex >= 0,
      sender.selectedSegmentIndex < lensDevices.count
    else {
      return
    }
    switchToCamera(lensDevices[sender.selectedSegmentIndex])
  }

  @objc private func infoTapped() {
    selection.selectionChanged()
    guard let viewController = nearestViewController() else { return }

    let infoViewController = YOLOInfoViewController()
    let navigationController = UINavigationController(rootViewController: infoViewController)
    navigationController.modalPresentationStyle = .pageSheet
    if #available(iOS 15.0, *), let sheet = navigationController.sheetPresentationController {
      sheet.detents = [.medium(), .large()]
      sheet.prefersGrabberVisible = true
      sheet.prefersScrollingExpandsWhenScrolledToEdge = true
    }
    viewController.present(navigationController, animated: true)
  }

  private func nearestViewController() -> UIViewController? {
    var responder: UIResponder? = self
    while let currentResponder = responder {
      if let viewController = currentResponder as? UIViewController {
        return viewController
      }
      responder = currentResponder.next
    }
    return nil
  }

  private func switchToCamera(_ device: AVCaptureDevice) {
    guard !cameraSwitchInProgress else { return }
    let changesPosition = videoCapture.captureDevice?.position != device.position
    cameraSwitchInProgress = true
    setCameraControlsEnabled(false)
    if changesPosition {
      showCameraTransition()
    }

    videoCapture.selectCaptureDevice(device, videoOrientation: currentVideoOrientation()) {
      [weak self] success in
      guard let self else { return }

      self.cameraSwitchInProgress = false
      self.setCameraControlsEnabled(true)
      if changesPosition {
        self.hideCameraTransition()
      }

      guard success else {
        self.updateLensControl()
        return
      }

      let activeDevice = self.videoCapture.captureDevice ?? device
      self.selectedLensDeviceID = device.position == .back ? device.uniqueID : nil
      let rawZoomFactor =
        zoomFactor(for: device, on: activeDevice)
        ?? min(max(activeDevice.videoZoomFactor, self.minimumZoom), self.maximumZoom)
      self.lastZoomFactor = rawZoomFactor
      self.labelZoom.text = self.zoomLabelText(rawZoomFactor: rawZoomFactor, device: activeDevice)
      // Switching the camera input drops the torch (the new device may not have one); sync the chip
      // with the actual hardware state so it never reads stale.
      self.setTorchUI(on: activeDevice.torchMode == .on)
      self.updateLensControl()
    }
  }

  private func showCameraTransition() {
    cameraTransitionView?.removeFromSuperview()

    let transitionView = UIView(frame: bounds)
    transitionView.isUserInteractionEnabled = false
    transitionView.backgroundColor = .black

    if let snapshot = snapshotView(afterScreenUpdates: false) {
      snapshot.frame = transitionView.bounds
      transitionView.addSubview(snapshot)
    }

    let blurView = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterialDark))
    blurView.frame = transitionView.bounds
    blurView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    transitionView.addSubview(blurView)

    addSubview(transitionView)  // top of the hierarchy: cover the whole HUD (labels, sliders, toolbar) uniformly
    cameraTransitionView = transitionView
  }

  private func hideCameraTransition() {
    guard let transitionView = cameraTransitionView else { return }
    cameraTransitionView = nil
    UIView.animate(
      withDuration: 0.18,
      delay: 0.06,
      options: [.beginFromCurrentState, .curveEaseOut]
    ) {
      transitionView.alpha = 0
    } completion: { _ in
      transitionView.removeFromSuperview()
    }
  }

  private func setCameraControlsEnabled(_ isEnabled: Bool) {
    switchCameraButton.isEnabled = isEnabled
    lensControl.isEnabled = isEnabled
    UIView.animate(withDuration: 0.12) {
      self.switchCameraButton.alpha = isEnabled ? 1 : 0.45
      self.lensControl.alpha = isEnabled ? 1 : 0.55
    }
  }

  private func updateLensControl() {
    let currentDevice = videoCapture.captureDevice
    let position = currentDevice?.position ?? .back
    lensDevices =
      position == .front ? currentDevice.map { [$0] } ?? [] : captureDevices(position: position)
    if let selectedLensDeviceID,
      !lensDevices.contains(where: { $0.uniqueID == selectedLensDeviceID })
    {
      self.selectedLensDeviceID = nil
    }
    if position == .back, let currentDevice, selectedLensDeviceID == nil {
      selectedLensDeviceID =
        lensDevice(
          rawZoomFactor: currentDevice.videoZoomFactor, device: currentDevice)?.uniqueID
    }
    lensControl.removeAllSegments()

    for (index, device) in lensDevices.enumerated() {
      lensControl.insertSegment(withTitle: lensTitle(for: device), at: index, animated: false)
    }

    let selectedDeviceID = selectedLensDeviceID ?? currentDevice?.uniqueID
    lensControl.selectedSegmentIndex =
      lensDevices.firstIndex { $0.uniqueID == selectedDeviceID }
      ?? UISegmentedControl.noSegment
    lensCaptionLabel.text = selectedLensDevice().map { lensCaption(for: $0) }
    updateLensControlVisibility()
    setNeedsLayout()
  }

  private func updateLensControlVisibility() {
    lensControl.isHidden = switchCameraButton.isHidden || lensDevices.isEmpty
    lensCaptionLabel.isHidden = lensControl.isHidden
    // Hide the torch chip when the active device has no torch (e.g. front camera) — a visible chip
    // that can't do anything reads as broken.
    torchButton.isHidden = lensControl.isHidden || videoCapture.captureDevice?.hasTorch != true
    torchCaptionLabel.isHidden = torchButton.isHidden || !isTorchOn
  }

  private func lensTitle(for device: AVCaptureDevice) -> String {
    if device.position == .front {
      return "1"
    }

    switch device.deviceType {
    case .builtInUltraWideCamera, .builtInWideAngleCamera, .builtInTelephotoCamera:
      if let displayZoomFactor = displayZoomFactor(
        for: device, activeDevice: videoCapture.captureDevice)
      {
        return zoomTitle(displayZoomFactor)
      }
      return fallbackLensTitle(for: device)
    default: return device.localizedName
    }
  }

  private func fallbackLensTitle(for device: AVCaptureDevice) -> String {
    switch device.deviceType {
    case .builtInUltraWideCamera: return "0.5"
    case .builtInWideAngleCamera: return "1"
    case .builtInTelephotoCamera: return "2"
    default: return device.localizedName
    }
  }

  private func zoomTitle(_ zoomFactor: CGFloat) -> String {
    let roundedZoom = (zoomFactor * 10).rounded() / 10
    return roundedZoom.truncatingRemainder(dividingBy: 1) == 0
      ? String(format: "%.0f", roundedZoom)
      : String(format: "%.1f", roundedZoom)
  }

  private func zoomLabelText(rawZoomFactor: CGFloat, device: AVCaptureDevice) -> String {
    String(format: "%.2fx", displayZoomFactor(rawZoomFactor, for: device))
  }

  private func selectedLensDevice() -> AVCaptureDevice? {
    guard let selectedLensDeviceID else { return videoCapture.captureDevice }
    return lensDevices.first { $0.uniqueID == selectedLensDeviceID } ?? videoCapture.captureDevice
  }

  private func updateSelectedLens(rawZoomFactor: CGFloat, device: AVCaptureDevice) {
    guard device.position == .back else {
      selectedLensDeviceID = nil
      return
    }

    guard let selectedLens = lensDevice(rawZoomFactor: rawZoomFactor, device: device) else {
      return
    }

    selectedLensDeviceID = selectedLens.uniqueID
    lensControl.selectedSegmentIndex =
      lensDevices.firstIndex { $0.uniqueID == selectedLens.uniqueID }
      ?? UISegmentedControl.noSegment
    lensCaptionLabel.text = lensCaption(for: selectedLens)
  }

  private func lensDevice(rawZoomFactor: CGFloat, device: AVCaptureDevice) -> AVCaptureDevice? {
    let lensZooms = lensDevices.compactMap { lens -> (device: AVCaptureDevice, zoom: CGFloat)? in
      guard lens.position == .back, physicalLensTypes.contains(lens.deviceType),
        let zoom = zoomFactor(for: lens, on: device)
      else {
        return nil
      }
      return (lens, zoom)
    }.sorted { $0.zoom < $1.zoom }

    return lensZooms.last(where: { rawZoomFactor >= $0.zoom - 0.01 })?.device
      ?? lensZooms.first?.device
  }

  private func lensCaption(for device: AVCaptureDevice) -> String {
    if device.position == .front {
      return "Front camera"
    }

    switch device.deviceType {
    case .builtInUltraWideCamera: return "Ultra wide camera"
    case .builtInWideAngleCamera: return "Wide camera"
    case .builtInTelephotoCamera: return "Telephoto camera"
    default: return device.localizedName
    }
  }

  public func capturePhoto(completion: @escaping (UIImage?) -> Void) {
    if let pausedShareImage, !videoCapture.captureSession.isRunning {
      completion(renderShareImage(pausedShareImage))
      return
    }

    videoCapture.captureNextFrame { [weak self] image in
      guard let self, let image else {
        completion(nil)
        return
      }
      completion(self.renderShareImage(image))
    }
  }

  private func renderShareImage(_ image: UIImage) -> UIImage? {
    let imageView = UIImageView(image: image)
    imageView.contentMode = .scaleAspectFill
    imageView.frame = bounds
    let imageLayer = imageView.layer
    layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

    var tempViews = [UIView]()
    let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
    for info in boundingBoxInfos where !info.isHidden {
      let boxView = createBoxView(from: info)
      boxView.frame = info.rect
      addSubview(boxView)
      tempViews.append(boxView)
    }

    UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
    drawHierarchy(in: bounds, afterScreenUpdates: true)
    let snapshot = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    imageLayer.removeFromSuperlayer()
    tempViews.forEach { $0.removeFromSuperview() }
    return snapshot
  }

  public func setInferenceFlag(ok: Bool) {
    videoCapture.inferenceOK = ok
  }
}

public func processString(_ input: String) -> String {
  var output = input.replacingOccurrences(
    of: "yolo",
    with: "YOLO",
    options: .caseInsensitive,
    range: nil
  )

  output = output.replacingOccurrences(
    of: "obb",
    with: "OBB",
    options: .caseInsensitive,
    range: nil
  )

  guard !output.isEmpty else {
    return output
  }

  let first = output[output.startIndex]
  let firstUppercased = String(first).uppercased()

  if String(first) != firstUppercased {
    output = firstUppercased + output.dropFirst()
  }

  return output
}
