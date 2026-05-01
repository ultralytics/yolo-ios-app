// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing the core UI component for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOView class is the primary UI component for displaying real-time YOLO model results.
//  It handles camera setup, model loading, video frame processing, rendering of detection results,
//  and user interactions such as pinch-to-zoom. The view can display bounding boxes, masks for segmentation,
//  pose estimation keypoints, and oriented bounding boxes depending on the active task. It includes
//  UI elements for controlling inference settings such as confidence threshold and IoU threshold,
//  and provides functionality for capturing photos with detection results overlaid.

import AVFoundation
import UIKit
import Vision

/// YOLOView Delegate Protocol - Provides performance metrics and YOLO results for each frame
public protocol YOLOViewDelegate: AnyObject {
  /// Called when performance metrics (FPS and inference time) are updated
  func yoloView(_: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double)

  /// Called when detection results are available
  func yoloView(_: YOLOView, didReceiveResult result: YOLOResult)

}

private let defaultMaxDetectionItems = 100

/// A UIView component that provides real-time object detection, segmentation, and pose estimation capabilities.
@MainActor
public final class YOLOView: UIView, VideoCaptureDelegate {

  /// Delegate object - Receives performance metrics and YOLO detection results
  public weak var delegate: YOLOViewDelegate?

  public func onInferenceTime(speed: Double, fps: Double) {
    self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
    self.delegate?.yoloView(self, didUpdatePerformance: fps, inferenceTime: speed)
  }

  public func onPredict(result: YOLOResult) {
    // Notify delegate of detection results
    delegate?.yoloView(self, didReceiveResult: result)

    showBoxes(predictions: result)
    onDetection?(result)

    if task == .segment {
      if let maskImage = result.masks?.combinedMask {
        guard let maskLayer = self.maskLayer else {
          self.videoCapture.predictor?.isUpdating = false
          return
        }
        maskLayer.isHidden = false
        maskLayer.frame = self.overlayLayer.bounds
        maskLayer.contents = maskImage
      }
      self.videoCapture.predictor?.isUpdating = false
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
      drawKeypoints(
        keypointsList: keypointList, confsList: confsList, boundingBoxes: result.boxes,
        on: poseLayer, imageViewSize: overlayLayer.frame.size)
    } else if task == .obb {
      guard let obbLayer = self.obbLayer else { return }
      let obbDetections = result.obb
      self.obbRenderer.drawObbDetectionsWithReuse(
        obbDetections: obbDetections,
        on: obbLayer,
        imageViewSize: self.overlayLayer.frame.size
      )
    }
  }

  public var onDetection: ((YOLOResult) -> Void)?
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
  private var cameraTransitionView: UIView?
  private var lensDevices = [AVCaptureDevice]()
  private var selectedLensDeviceID: String?
  private var cameraSwitchInProgress = false
  private var overlayLayer = CALayer()
  private var maskLayer: CALayer?
  private var poseLayer: CALayer?
  private var obbLayer: CALayer?

  let obbRenderer = OBBRenderer()

  private let minimumZoom: CGFloat = 1.0
  private let maximumZoom: CGFloat = 10.0
  private var lastZoomFactor: CGFloat = 1.0

  private var photoCaptureCompletion: ((UIImage?) -> Void)?

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
    setupOverlayLayer()
  }

  /// Initialize YOLOView without a model (camera only)
  public override init(frame: CGRect) {
    self.videoCapture = VideoCapture()
    self.task = .detect  // Default task
    super.init(frame: frame)
    setUpOrientationChangeNotification()
    self.setUpBoundingBoxViews()
    self.setupUI()
    self.videoCapture.delegate = self
    start(position: .back)
    setupOverlayLayer()
  }

  required init?(coder: NSCoder) {
    self.videoCapture = VideoCapture()
    super.init(coder: coder)
  }

  public override func awakeFromNib() {
    super.awakeFromNib()
    Task { @MainActor in
      setUpOrientationChangeNotification()
      setUpBoundingBoxViews()
      setupUI()
      videoCapture.delegate = self
      start(position: .back)
      setupOverlayLayer()
    }
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
          if task == .obb { self.obbLayer?.isHidden = false }
          completion?(.success(()))
        case .failure(let error):
          YOLOLog.error("Failed to load model: \(error)")
          completion?(.failure(error))
        }
      }
    }
  }

  private func start(position: AVCaptureDevice.Position) {
    if !busy {
      busy = true
      let orientation = UIDevice.current.orientation
      videoCapture.setUp(sessionPreset: .photo, position: position, orientation: orientation) {
        [weak self] success in
        Task { @MainActor in
          guard let self = self else { return }
          if success {
            // Add the video preview into the UI.
            if let previewLayer = self.videoCapture.previewLayer {
              self.layer.insertSublayer(previewLayer, at: 0)
              self.videoCapture.previewLayer?.frame = self.bounds  // resize preview layer
              for box in self.boundingBoxViews {
                box.addToLayer(previewLayer)
              }
            }
            self.videoCapture.previewLayer?.addSublayer(self.overlayLayer)
            // Once everything is set up, we can start capturing live video.
            self.videoCapture.start()

            // Apply deferred camera position if set (e.g., front camera from SwiftUI)
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

            self.busy = false
          }
        }
      }
    }
  }

  public func stop() {
    videoCapture.stop()
  }

  public func resume() {
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

  func setupOverlayLayer() {
    let width = self.bounds.width
    let height = self.bounds.height

    var ratio: CGFloat = 1.0
    if videoCapture.captureSession.sessionPreset == .photo {
      ratio = (4.0 / 3.0)
    } else {
      ratio = (16.0 / 9.0)
    }
    var offSet = CGFloat.zero
    var margin = CGFloat.zero
    if self.bounds.width < self.bounds.height {
      offSet = height / ratio
      margin = (offSet - self.bounds.width) / 2
      self.overlayLayer.frame = CGRect(
        x: -margin, y: 0, width: offSet, height: self.bounds.height)
    } else {
      offSet = width / ratio
      margin = (offSet - self.bounds.height) / 2
      self.overlayLayer.frame = CGRect(
        x: 0, y: -margin, width: self.bounds.width, height: offSet)
    }
  }

  func setupMaskLayerIfNeeded() {
    if maskLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      layer.name = "maskLayer"
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

  func setupObbLayerIfNeeded() {
    if obbLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      self.overlayLayer.addSublayer(layer)
      self.obbLayer = layer
    }
  }

  public func resetLayers() {
    removeAllSubLayers(parentLayer: maskLayer)
    removeAllSubLayers(parentLayer: poseLayer)
    removeAllSubLayers(parentLayer: obbLayer)
    removeAllSubLayers(parentLayer: overlayLayer)

    maskLayer = nil
    poseLayer = nil
    obbLayer = nil
  }

  func setupSublayers() {
    resetLayers()

    switch task {
    case .segment:
      setupMaskLayerIfNeeded()
    case .pose:
      setupPoseLayerIfNeeded()
    case .obb:
      setupObbLayerIfNeeded()
      if let obbLayer = obbLayer, obbLayer.superlayer !== overlayLayer {
        overlayLayer.addSublayer(obbLayer)
      }
      obbLayer?.isHidden = false
    default: break
    }
  }

  func removeAllSubLayers(parentLayer: CALayer?) {
    guard let parentLayer = parentLayer else { return }
    parentLayer.sublayers?.forEach { layer in
      layer.removeFromSuperlayer()
    }
    parentLayer.sublayers = nil
    parentLayer.contents = nil
  }

  func showBoxes(predictions: YOLOResult) {
    let width = self.bounds.width
    let height = self.bounds.height
    let maxVisible = min(predictions.boxes.count, 50, boundingBoxViews.count)

    if UIDevice.current.orientation.isLandscape {
      let frameAspect = videoCapture.longSide / videoCapture.shortSide
      let viewAspect = width / height
      let scale: CGFloat
      let offsetX: CGFloat
      let offsetY: CGFloat
      if frameAspect > viewAspect {
        scale = height / videoCapture.shortSide
        offsetX = (videoCapture.longSide * scale - width) / 2
        offsetY = 0
      } else {
        scale = width / videoCapture.longSide
        offsetX = 0
        offsetY = (videoCapture.shortSide * scale - height) / 2
      }
      for i in 0..<maxVisible {
        let prediction = predictions.boxes[i]
        var rect = flippedNormalizedRect(prediction.xywhn)
        rect.origin.x = rect.origin.x * videoCapture.longSide * scale - offsetX
        rect.origin.y =
          height
          - (rect.origin.y * videoCapture.shortSide * scale
            - offsetY
            + rect.size.height * videoCapture.shortSide * scale)
        rect.size.width *= videoCapture.longSide * scale
        rect.size.height *= videoCapture.shortSide * scale
        showBox(at: i, prediction: prediction, frame: rect)
      }
    } else {
      let aspect: CGFloat =
        videoCapture.captureSession.sessionPreset == .photo ? (4.0 / 3.0) : (16.0 / 9.0)
      var ratio = (height / width) / aspect
      for i in 0..<maxVisible {
        let prediction = predictions.boxes[i]
        var displayRect = flippedNormalizedRect(prediction.xywhn)
        if UIDevice.current.orientation == .portraitUpsideDown {
          displayRect.origin.x = 1.0 - displayRect.origin.x - displayRect.width
        } else if UIDevice.current.orientation == .unknown {
          YOLOLog.warning("Device orientation is unknown; predictions may be affected")
        }
        if ratio >= 1 {
          let offset = (1 - ratio) * (0.5 - displayRect.minX)
          let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
          displayRect = displayRect.applying(transform)
          displayRect.size.width *= ratio
        } else {
          let offset = (ratio - 1) * (0.5 - displayRect.maxY)
          let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
          displayRect = displayRect.applying(transform)
          ratio = (height / width) / (3.0 / 4.0)
          displayRect.size.height /= ratio
        }
        displayRect = VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))
        showBox(at: i, prediction: prediction, frame: displayRect)
      }
    }

    for i in maxVisible..<boundingBoxViews.count {
      boundingBoxViews[i].hide()
    }
  }

  /// Flips a normalized Vision rect from bottom-origin to top-origin for display.
  @inline(__always)
  private func flippedNormalizedRect(_ r: CGRect) -> CGRect {
    CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
  }

  /// Configures the ith bounding box view with the prediction's color/label/alpha.
  @inline(__always)
  private func showBox(at index: Int, prediction: Box, frame: CGRect) {
    let confidence = CGFloat(prediction.conf)
    let color = ultralyticsColors[prediction.index % ultralyticsColors.count]
    let label = String(format: "%@ %.1f", prediction.cls, confidence * 100)
    let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
    boundingBoxViews[index].show(frame: frame, label: label, color: color, alpha: alpha)
  }

  func removeClassificationLayers() {
    if let sublayers = self.layer.sublayers {
      for layer in sublayers where layer.name == "YOLOOverlayLayer" {
        layer.removeFromSuperlayer()
      }
    }
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

    let confidencePercent = round(top1Conf * 1000) / 10
    let labelText = " \(top1) \(confidencePercent)% "

    let textLayer = CATextLayer()
    textLayer.contentsScale = UIScreen.main.scale  // Retina display support
    textLayer.alignmentMode = .left

    // Check if this is likely an external display and scale font accordingly
    let viewBounds = self.bounds
    let maxDimension = max(viewBounds.width, viewBounds.height)
    let fontSize: CGFloat

    if maxDimension > 1000 {
      fontSize = max(36, viewBounds.height * 0.04)
    } else {
      fontSize = viewBounds.height * 0.035
    }

    textLayer.font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    textLayer.fontSize = fontSize
    textLayer.foregroundColor = UIColor.white.cgColor
    textLayer.backgroundColor = color.cgColor
    textLayer.cornerRadius = 4
    textLayer.masksToBounds = true

    textLayer.string = labelText
    let textAttributes: [NSAttributedString.Key: Any] = [
      .font: UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    ]
    let textSize = (labelText as NSString).size(withAttributes: textAttributes)
    let width: CGFloat = textSize.width + 10
    let x: CGFloat = self.center.x - (width / 2)
    let y: CGFloat = self.center.y - textSize.height
    let height: CGFloat = textSize.height + 4

    textLayer.frame = CGRect(x: x, y: y, width: width, height: height)

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

  /// Configure a slider with common settings
  private func configureSlider(_ slider: UISlider, min: Float, max: Float, value: Float) {
    slider.minimumValue = min
    slider.maximumValue = max
    slider.value = value
    slider.minimumTrackTintColor = .white
    slider.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    slider.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
  }

  /// Setup toolbar with consistent styling
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
  }

  public override func layoutSubviews() {
    super.layoutSubviews()
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)

    // Apply consistent toolbar styling
    applyToolbarStyling(isLandscape: isLandscape)
    updateLensControlVisibility()

    if isLandscape {
      layoutLandscape()
    } else {
      layoutPortrait()
    }

    self.videoCapture.previewLayer?.frame = self.bounds
    cameraTransitionView?.frame = self.bounds
  }

  /// Apply consistent toolbar and button styling
  private func applyToolbarStyling(isLandscape: Bool) {
    toolbar.backgroundColor = .black.withAlphaComponent(0.7)
    let buttonColor: UIColor = isLandscape ? .white : .white
    [playButton, pauseButton, switchCameraButton, shareButton, infoButton].forEach { button in
      button.tintColor = buttonColor
    }
  }

  /// Layout views for landscape orientation
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

  /// Layout views for portrait orientation
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

  /// Layout toolbar buttons (shared between orientations)
  private func layoutToolbarButtons(width: CGFloat, height: CGFloat) {
    let toolBarHeight: CGFloat = 66
    let buttonHeight: CGFloat = toolBarHeight * 0.75

    toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
    playButton.frame = CGRect(x: 0, y: 0, width: buttonHeight, height: buttonHeight)
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
      x: width - buttonHeight, y: 0, width: buttonHeight, height: buttonHeight
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
    if let interfaceOrientation = window?.windowScene?.interfaceOrientation,
      let videoOrientation = AVCaptureVideoOrientation(interfaceOrientation)
    {
      return videoOrientation
    }

    if let videoOrientation = AVCaptureVideoOrientation(UIDevice.current.orientation) {
      return videoOrientation
    }

    return videoCapture.previewLayer?.connection?.videoOrientation ?? .portrait
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

  @objc func playTapped() {
    selection.selectionChanged()
    self.videoCapture.start()
    playButton.isEnabled = false
    pauseButton.isEnabled = true
  }

  @objc func pauseTapped() {
    selection.selectionChanged()
    self.videoCapture.stop()
    playButton.isEnabled = true
    pauseButton.isEnabled = false
  }

  @objc func switchCameraTapped() {
    selection.selectionChanged()

    let currentPosition = videoCapture.captureDevice?.position ?? .back
    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back
    guard let newCameraDevice = bestCaptureDevice(position: nextCameraPosition) else { return }
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
    if let url = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(url)
    }
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

      self.selectedLensDeviceID = device.position == .back ? device.uniqueID : nil
      let activeDevice = self.videoCapture.captureDevice ?? device
      let rawZoomFactor =
        zoomFactor(for: device, on: activeDevice)
        ?? min(max(activeDevice.videoZoomFactor, self.minimumZoom), self.maximumZoom)
      self.lastZoomFactor = rawZoomFactor
      self.labelZoom.text = self.zoomLabelText(rawZoomFactor: rawZoomFactor, device: activeDevice)
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

    insertSubview(transitionView, belowSubview: labelName)
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
    guard photoCaptureCompletion == nil else {
      completion(nil)  // Previous capture still in progress
      return
    }
    self.photoCaptureCompletion = completion
    let settings = AVCapturePhotoSettings()
    self.videoCapture.photoOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate
    )
  }

  public func setInferenceFlag(ok: Bool) {
    videoCapture.inferenceOK = ok
  }
}

// MARK: - Helper Methods for Layer Copying

extension YOLOView {
  /// Copies layer properties from source to destination
  private func copyLayerProperties(from source: CALayer, to destination: CALayer) {
    destination.opacity = source.opacity
    destination.transform = source.transform
    destination.masksToBounds = source.masksToBounds
    destination.contentsGravity = source.contentsGravity
    destination.contentsRect = source.contentsRect
    destination.contentsCenter = source.contentsCenter
    destination.compositingFilter = source.compositingFilter
    destination.backgroundColor = source.backgroundColor
    destination.cornerRadius = source.cornerRadius
  }

  /// Copies CAShapeLayer properties
  private func copyShapeLayer(_ shapeLayer: CAShapeLayer) -> CAShapeLayer {
    let copy = CAShapeLayer()
    copy.frame = shapeLayer.frame
    copy.path = shapeLayer.path
    copy.strokeColor = shapeLayer.strokeColor
    copy.lineWidth = shapeLayer.lineWidth
    copy.fillColor = shapeLayer.fillColor
    copy.opacity = shapeLayer.opacity
    return copy
  }

  /// Copies CATextLayer properties
  private func copyTextLayer(_ textLayer: CATextLayer) -> CATextLayer {
    let copy = CATextLayer()
    copy.frame = textLayer.frame
    copy.string = textLayer.string
    copy.font = textLayer.font
    copy.fontSize = textLayer.fontSize
    copy.foregroundColor = textLayer.foregroundColor
    copy.backgroundColor = textLayer.backgroundColor
    copy.alignmentMode = textLayer.alignmentMode
    copy.opacity = textLayer.opacity
    return copy
  }

  /// Creates a copy of a visualization layer for capture
  private func copyVisualizationLayer(_ layer: CALayer, isFullFrame: Bool = false) -> CALayer? {
    let tempLayer = CALayer()
    let overlayFrame = self.overlayLayer.frame

    if isFullFrame {
      tempLayer.frame = CGRect(
        x: overlayFrame.origin.x,
        y: overlayFrame.origin.y,
        width: overlayFrame.width,
        height: overlayFrame.height
      )
    } else {
      // For mask layer - adjust frame to be relative to main view
      let layerFrame = layer.frame
      tempLayer.frame = CGRect(
        x: overlayFrame.origin.x + layerFrame.origin.x,
        y: overlayFrame.origin.y + layerFrame.origin.y,
        width: layerFrame.width,
        height: layerFrame.height
      )
      tempLayer.contents = layer.contents
    }

    tempLayer.opacity = layer.opacity
    copyLayerProperties(from: layer, to: tempLayer)

    // Copy sublayers if present
    if let sublayers = layer.sublayers {
      for sublayer in sublayers {
        if let shapeLayer = sublayer as? CAShapeLayer {
          tempLayer.addSublayer(copyShapeLayer(shapeLayer))
        } else if let textLayer = sublayer as? CATextLayer {
          tempLayer.addSublayer(copyTextLayer(textLayer))
        } else {
          let copyLayer = CALayer()
          copyLayer.frame = sublayer.frame
          copyLayerProperties(from: sublayer, to: copyLayer)
          tempLayer.addSublayer(copyLayer)
        }
      }
    }
    return tempLayer
  }
}

extension YOLOView: AVCapturePhotoCaptureDelegate {
  nonisolated public func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      YOLOLog.error("Photo capture error: \(error.localizedDescription)")
    }
    if let dataImage = photo.fileDataRepresentation() {
      guard let dataProvider = CGDataProvider(data: dataImage as CFData),
        let cgImageRef = CGImage(
          jpegDataProviderSource: dataProvider, decode: nil, shouldInterpolate: true,
          intent: .defaultIntent)
      else {
        Task { @MainActor [weak self] in
          self?.photoCaptureCompletion?(nil)
          self?.photoCaptureCompletion = nil
        }
        return
      }

      Task { @MainActor [weak self] in
        guard let self = self else { return }

        var isCameraFront = false
        if let currentInput = self.videoCapture.captureSession.inputs.first
          as? AVCaptureDeviceInput,
          currentInput.device.position == .front
        {
          isCameraFront = true
        }
        var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
        switch UIDevice.current.orientation {
        case .landscapeLeft:
          orientation = isCameraFront ? .downMirrored : .up
        case .landscapeRight:
          orientation = isCameraFront ? .upMirrored : .down
        default:
          break
        }
        var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
        if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
          let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
        {
          image = UIImage(cgImage: cgImage)
        }
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFill
        imageView.frame = self.frame
        let imageLayer = imageView.layer
        self.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

        // Add visualization layers
        var tempLayers = [CALayer]()

        // Add mask layer if present (for segmentation task)
        if let maskLayer = self.maskLayer, !maskLayer.isHidden {
          if let tempLayer = copyVisualizationLayer(maskLayer, isFullFrame: false) {
            self.layer.addSublayer(tempLayer)
            tempLayers.append(tempLayer)
          }
        }

        // Add pose layer if present (for pose task)
        if let poseLayer = self.poseLayer {
          if let tempLayer = copyVisualizationLayer(poseLayer, isFullFrame: true) {
            self.layer.addSublayer(tempLayer)
            tempLayers.append(tempLayer)
          }
        }

        // Add OBB layer if present (for OBB task)
        if let obbLayer = self.obbLayer, !obbLayer.isHidden {
          if let tempLayer = copyVisualizationLayer(obbLayer, isFullFrame: true) {
            self.layer.addSublayer(tempLayer)
            tempLayers.append(tempLayer)
          }
        }

        var tempViews = [UIView]()
        let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
        for info in boundingBoxInfos where !info.isHidden {
          let boxView = createBoxView(from: info)
          boxView.frame = info.rect

          self.addSubview(boxView)
          tempViews.append(boxView)
        }
        let captureBounds = self.bounds
        UIGraphicsBeginImageContextWithOptions(captureBounds.size, true, 0.0)
        self.drawHierarchy(in: captureBounds, afterScreenUpdates: true)
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        imageLayer.removeFromSuperlayer()
        for layer in tempLayers {
          layer.removeFromSuperlayer()
        }
        for v in tempViews {
          v.removeFromSuperview()
        }
        photoCaptureCompletion?(img)
        photoCaptureCompletion = nil
      }
    } else {
      YOLOLog.error("AVCapturePhotoCaptureDelegate: photo has no file data")
    }
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
