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
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double)

  /// Called when detection results are available
  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult)

}

/// A UIView component that provides real-time object detection, segmentation, and pose estimation capabilities.
@MainActor
public class YOLOView: UIView, VideoCaptureDelegate {

  /// Delegate object - Receives performance metrics and YOLO detection results
  public weak var delegate: YOLOViewDelegate?

  func onInferenceTime(speed: Double, fps: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
      // Notify delegate of performance metrics

      self.delegate?.yoloView(self, didUpdatePerformance: fps, inferenceTime: speed)
    }
  }

  func onPredict(result: YOLOResult) {
    // Notify delegate of detection results
    delegate?.yoloView(self, didReceiveResult: result)

    showBoxes(predictions: result)
    onDetection?(result)

    if task == .segment {
      DispatchQueue.main.async {
        if let maskImage = result.masks?.combinedMask {

          guard let maskLayer = self.maskLayer else { return }

          maskLayer.isHidden = false
          maskLayer.frame = self.overlayLayer.bounds
          maskLayer.contents = maskImage

          self.videoCapture.predictor.isUpdating = false
        } else {
          self.videoCapture.predictor.isUpdating = false
        }
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
      drawKeypoints(
        keypointsList: keypointList, confsList: confsList, boundingBoxes: result.boxes,
        on: poseLayer, imageViewSize: overlayLayer.frame.size, originalImageSize: result.orig_shape)
    } else if task == .obb {
      //            self.setupObbLayerIfNeeded()
      guard let obbLayer = self.obbLayer else { return }
      let obbDetections = result.obb
      self.obbRenderer.drawObbDetectionsWithReuse(
        obbDetections: obbDetections,
        on: obbLayer,
        imageViewSize: self.overlayLayer.frame.size,
        originalImageSize: result.orig_shape,  // Example
        lineWidth: 3
      )
    }
  }

  var onDetection: ((YOLOResult) -> Void)?
  private var videoCapture: VideoCapture
  private var busy = false
  private var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  var task = YOLOTask.detect
  var colors: [String: UIColor] = [:]
  var modelName: String = ""
  var classes: [String] = []
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
  public var toolbar = UIView()
  let selection = UISelectionFeedbackGenerator()
  private var overlayLayer = CALayer()
  private var maskLayer: CALayer?
  private var poseLayer: CALayer?
  private var obbLayer: CALayer?

  let obbRenderer = OBBRenderer()

  private let minimumZoom: CGFloat = 1.0
  private let maximumZoom: CGFloat = 10.0
  private var lastZoomFactor: CGFloat = 1.0

  public var capturedImage: UIImage?
  private var photoCaptureCompletion: ((UIImage?) -> Void)?

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
    completion: ((Result<Void, Error>) -> Void)? = nil
  ) {
    activityIndicator.startAnimating()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    removeClassificationLayers()

    self.task = task
    setupSublayers()

    var modelURL: URL?
    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    // Determine model URL
    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage")
      || lowercasedPath.hasSuffix(".mlmodelc")
    {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      if fileManager.fileExists(atPath: possibleURL.path) {
        modelURL = possibleURL
      }
    } else {
      if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      {
        modelURL = compiledURL
      } else if let packageURL = Bundle.main.url(
        forResource: modelPathOrName, withExtension: "mlpackage")
      {
        modelURL = packageURL
      }
    }

    guard let unwrappedModelURL = modelURL else {
      let error = PredictorError.modelFileNotFound
      fatalError(error.localizedDescription)
    }

    modelName = unwrappedModelURL.deletingPathExtension().lastPathComponent

    // Common success handling for all tasks
    func handleSuccess(predictor: Predictor) {
      self.videoCapture.predictor = predictor
      self.activityIndicator.stopAnimating()
      self.labelName.text = processString(modelName)
      completion?(.success(()))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      print("Failed to load model with error: \(error)")
      self.activityIndicator.stopAnimating()
      completion?(.failure(error))
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .segment:
      Segmenter.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .pose:
      PoseEstimator.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .obb:
      ObbDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          self?.obbLayer?.isHidden = false

          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
      ObjectDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }
    }
  }

  private func start(position: AVCaptureDevice.Position) {
    if !busy {
      busy = true
      let orientation = UIDevice.current.orientation
      videoCapture.setUp(sessionPreset: .photo, position: position, orientation: orientation) {
        success in
        // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
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

          self.busy = false
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

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
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
      // Specify contentsGravity or backgroundColor as needed
      // layer.contentsGravity = .resizeAspectFill
      // layer.backgroundColor = UIColor.clear.cgColor

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
    removeAllSubLayers(parentLayer: overlayLayer)

    maskLayer = nil
    poseLayer = nil
    obbLayer?.isHidden = true
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
      overlayLayer.addSublayer(obbLayer!)
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

  func addMaskSubLayers() {
    guard let maskLayer = maskLayer else { return }
    self.overlayLayer.addSublayer(maskLayer)
  }

  func showBoxes(predictions: YOLOResult) {

    let width = self.bounds.width
    let height = self.bounds.height
    var resultCount = 0

    resultCount = predictions.boxes.count

    if UIDevice.current.orientation == .portrait {

      var ratio: CGFloat = 1.0

      if videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)
      } else {
        ratio = (height / width) / (16.0 / 9.0)
      }

      self.labelSliderNumItems.text =
        String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
      for i in 0..<boundingBoxViews.count {
        if i < (resultCount) && i < 50 {
          var rect = CGRect.zero
          var label = ""
          var boxColor: UIColor = .white
          var confidence: CGFloat = 0
          var alpha: CGFloat = 0.9
          var bestClass = ""

          let prediction = predictions.boxes[i]
          rect = CGRect(
            x: prediction.xywhn.minX, y: 1 - prediction.xywhn.maxY, width: prediction.xywhn.width,
            height: prediction.xywhn.height)
          bestClass = prediction.cls
          confidence = CGFloat(prediction.conf)
          let colorIndex = prediction.index % ultralyticsColors.count
          boxColor = ultralyticsColors[colorIndex]
          label = String(format: "%@ %.1f", bestClass, confidence * 100)
          alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          var displayRect = rect
          switch UIDevice.current.orientation {
          case .portraitUpsideDown:
            displayRect = CGRect(
              x: 1.0 - rect.origin.x - rect.width,
              y: 1.0 - rect.origin.y - rect.height,
              width: rect.width,
              height: rect.height)
          case .landscapeLeft:
            displayRect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .landscapeRight:
            displayRect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .unknown:
            print("The device orientation is unknown, the predictions may be affected")
            fallthrough
          default: break
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

          boundingBoxViews[i].show(
            frame: displayRect, label: label, color: boxColor, alpha: alpha)

        } else {
          boundingBoxViews[i].hide()
        }
      }
    } else {
      resultCount = predictions.boxes.count
      self.labelSliderNumItems.text =
        String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"

      let frameAspectRatio = videoCapture.longSide / videoCapture.shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / videoCapture.shortSide
        scaleX = scaleY
        offsetX = (videoCapture.longSide * scaleX - width) / 2
      } else {
        scaleX = width / videoCapture.longSide
        scaleY = scaleX
        offsetY = (videoCapture.shortSide * scaleY - height) / 2
      }

      for i in 0..<boundingBoxViews.count {
        if i < resultCount && i < 50 {
          var rect = CGRect.zero
          var label = ""
          var boxColor: UIColor = .white
          var confidence: CGFloat = 0
          var alpha: CGFloat = 0.9
          var bestClass = ""

          let prediction = predictions.boxes[i]
          rect = CGRect(
            x: prediction.xywhn.minX,
            y: 1 - prediction.xywhn.maxY,
            width: prediction.xywhn.width,
            height: prediction.xywhn.height
          )
          bestClass = prediction.cls
          confidence = CGFloat(prediction.conf)

          let colorIndex = predictions.boxes[i].index % ultralyticsColors.count
          boxColor = ultralyticsColors[colorIndex]
          label = String(format: "%@ %.1f", bestClass, confidence * 100)
          alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)

          rect.origin.x = rect.origin.x * videoCapture.longSide * scaleX - offsetX
          rect.origin.y =
            height
            - (rect.origin.y * videoCapture.shortSide * scaleY
              - offsetY
              + rect.size.height * videoCapture.shortSide * scaleY)
          rect.size.width *= videoCapture.longSide * scaleX
          rect.size.height *= videoCapture.shortSide * scaleY

          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: boxColor,
            alpha: alpha
          )
        } else {
          boundingBoxViews[i].hide()
        }
      }
    }
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
    let fontSize = self.bounds.height * 0.02
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

    labelSliderNumItems.text = "0 items (max 30)"
    labelSliderNumItems.textAlignment = .left
    labelSliderNumItems.textColor = .white
    labelSliderNumItems.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderNumItems)

    configureSlider(sliderNumItems, min: 1, max: 100, value: 30)
    self.addSubview(sliderNumItems)

    labelSliderConf.text = "0.25 Confidence Threshold"
    labelSliderConf.textAlignment = .left
    labelSliderConf.textColor = .white
    labelSliderConf.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderConf)

    configureSlider(sliderConf, min: 0, max: 1, value: 0.25)
    self.addSubview(sliderConf)

    labelSliderIoU.text = "0.45 IoU Threshold"
    labelSliderIoU.textAlignment = .left
    labelSliderIoU.textColor = .white
    labelSliderIoU.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderIoU)

    configureSlider(sliderIoU, min: 0, max: 1, value: 0.45)
    self.addSubview(sliderIoU)

    self.labelSliderNumItems.text = "0 items (max " + String(Int(sliderNumItems.value)) + ")"
    self.labelSliderConf.text = "0.25 Confidence Threshold"
    self.labelSliderIoU.text = "0.45 IoU Threshold"

    labelZoom.text = "1.00x"
    labelZoom.textColor = .white
    labelZoom.font = UIFont.systemFont(ofSize: 14)
    labelZoom.textAlignment = .center
    labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    self.addSubview(labelZoom)

    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)

    playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
    pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
    switchCameraButton = UIButton()
    switchCameraButton.setImage(
      UIImage(systemName: "camera.rotate", withConfiguration: config), for: .normal)
    
    playButton.isEnabled = false
    pauseButton.isEnabled = true
    playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
    pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
    switchCameraButton.addTarget(self, action: #selector(switchCameraTapped), for: .touchUpInside)
    
    setupToolbar()
    
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
    [playButton, pauseButton, switchCameraButton].forEach { button in
      button.tintColor = .white
      toolbar.addSubview(button)
    }
    self.addSubview(toolbar)
  }

  public override func layoutSubviews() {
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
    
    // Apply consistent toolbar styling
    applyToolbarStyling(isLandscape: isLandscape)
    
    if isLandscape {
      layoutLandscape()
    } else {
      layoutPortrait()
    }

    self.videoCapture.previewLayer?.frame = self.bounds
  }
  
  /// Apply consistent toolbar and button styling
  private func applyToolbarStyling(isLandscape: Bool) {
    toolbar.backgroundColor = .black.withAlphaComponent(0.7)
    let buttonColor: UIColor = isLandscape ? .white : .white
    [playButton, pauseButton, switchCameraButton].forEach { button in
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
    let sliderHeight: CGFloat = height * 0.1
    
    labelSliderNumItems.frame = CGRect(
      x: width * 0.1, y: labelFPS.frame.minY - sliderHeight,
      width: sliderWidth, height: sliderHeight
    )
    
    sliderNumItems.frame = CGRect(
      x: width * 0.1, y: labelSliderNumItems.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )
    
    labelSliderConf.frame = CGRect(
      x: width * 0.1, y: sliderNumItems.frame.maxY + 10,
      width: sliderWidth * 1.5, height: sliderHeight
    )
    
    sliderConf.frame = CGRect(
      x: width * 0.1, y: labelSliderConf.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )
    
    labelSliderIoU.frame = CGRect(
      x: width * 0.1, y: sliderConf.frame.maxY + 10,
      width: sliderWidth * 1.5, height: sliderHeight
    )
    
    sliderIoU.frame = CGRect(
      x: width * 0.1, y: labelSliderIoU.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )
    
    let zoomLabelWidth: CGFloat = width * 0.2
    labelZoom.frame = CGRect(
      x: center.x - zoomLabelWidth / 2, y: self.bounds.maxY - 120,
      width: zoomLabelWidth, height: height * 0.03
    )
    
    layoutToolbarButtons(width: width, height: height)
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
    
    sliderNumItems.frame = CGRect(
      x: width * 0.01, y: center.y - sliderHeight - height * 0.24,
      width: sliderWidth, height: sliderHeight
    )
    
    labelSliderNumItems.frame = CGRect(
      x: width * 0.01, y: sliderNumItems.frame.minY - sliderHeight - 10,
      width: sliderWidth, height: sliderHeight
    )
    
    labelSliderConf.frame = CGRect(
      x: width * 0.01, y: center.y + height * 0.24,
      width: sliderWidth * 1.5, height: sliderHeight
    )
    
    sliderConf.frame = CGRect(
      x: width * 0.01, y: labelSliderConf.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )
    
    labelSliderIoU.frame = CGRect(
      x: width * 0.01, y: sliderConf.frame.maxY + 10,
      width: sliderWidth * 1.5, height: sliderHeight
    )
    
    sliderIoU.frame = CGRect(
      x: width * 0.01, y: labelSliderIoU.frame.maxY + 10,
      width: sliderWidth, height: sliderHeight
    )
    
    let zoomLabelWidth: CGFloat = width * 0.2
    labelZoom.frame = CGRect(
      x: center.x - zoomLabelWidth / 2, y: self.bounds.maxY - 120,
      width: zoomLabelWidth, height: height * 0.03
    )
    
    layoutToolbarButtons(width: width, height: height)
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
  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    var orientation: AVCaptureVideoOrientation = .portrait
    switch UIDevice.current.orientation {
    case .portrait:
      orientation = .portrait
    case .portraitUpsideDown:
      orientation = .portraitUpsideDown
    case .landscapeRight:
      orientation = .landscapeLeft
    case .landscapeLeft:
      orientation = .landscapeRight
    default:
      return
    }
    videoCapture.updateVideoOrientation(orientation: orientation)

    //      frameSizeCaptured = false
  }

  @objc func sliderChanged(_ sender: Any) {

    if sender as? UISlider === sliderNumItems {
      if let predictor = videoCapture.predictor as? BasePredictor {
        let numItems = Int(sliderNumItems.value)
        predictor.setNumItemsThreshold(numItems: numItems)
      }
    }
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    self.labelSliderConf.text = String(conf) + " Confidence Threshold"
    self.labelSliderIoU.text = String(iou) + " IoU Threshold"
    if let predictor = videoCapture.predictor as? BasePredictor {
      predictor.setIouThreshold(iou: iou)
      predictor.setConfidenceThreshold(confidence: conf)

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
        print("\(error.localizedDescription)")
      }
    }

    let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
    switch pinch.state {
    case .began, .changed:
      update(scale: newScaleFactor)
      self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
    case .ended:
      lastZoomFactor = minMaxZoom(newScaleFactor)
      update(scale: lastZoomFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
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

    self.videoCapture.captureSession.beginConfiguration()
    let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
    self.videoCapture.captureSession.removeInput(currentInput!)
    guard let currentPosition = currentInput?.device.position else { return }

    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back

    let newCameraDevice = bestCaptureDevice(position: nextCameraPosition)

    guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else {
      return
    }

    self.videoCapture.captureSession.addInput(videoInput1)
    var orientation: AVCaptureVideoOrientation = .portrait
    switch UIDevice.current.orientation {
    case .portrait:
      orientation = .portrait
    case .portraitUpsideDown:
      orientation = .portraitUpsideDown
    case .landscapeRight:
      orientation = .landscapeLeft
    case .landscapeLeft:
      orientation = .landscapeRight
    default:
      return
    }
    self.videoCapture.updateVideoOrientation(orientation: orientation)

    self.videoCapture.captureSession.commitConfiguration()
  }

  public func capturePhoto(completion: @escaping (UIImage?) -> Void) {
    self.photoCaptureCompletion = completion
    let settings = AVCapturePhotoSettings()
    usleep(20_000)  // short 10 ms delay to allow camera to focus
    self.videoCapture.photoOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate
    )
  }

  public func setInferenceFlag(ok: Bool) {
    videoCapture.inferenceOK = ok
  }
}

extension YOLOView: AVCapturePhotoCaptureDelegate {
  public func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      print("error occurred : \(error.localizedDescription)")
    }
    if let dataImage = photo.fileDataRepresentation() {
      let dataProvider = CGDataProvider(data: dataImage as CFData)
      let cgImageRef: CGImage! = CGImage(
        jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
        intent: .defaultIntent)
      var isCameraFront = false
      if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
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
      
      // Create a temporary container view for off-screen compositing
      let containerView = UIView(frame: self.bounds)
      containerView.backgroundColor = .black
      
      // Add the captured image as the base layer
      let imageView = UIImageView(image: image)
      imageView.contentMode = .scaleAspectFill
      imageView.frame = containerView.bounds
      containerView.addSubview(imageView)

      // Add mask layer if present (for segmentation task)
      if let maskLayer = self.maskLayer, !maskLayer.isHidden {
        // Create a temporary copy of the mask layer for capture
        let tempLayer = CALayer()
        // Calculate the correct frame relative to the main view
        let overlayFrame = self.overlayLayer.frame
        let maskFrame = maskLayer.frame

        // Adjust mask frame to be relative to the main view, not overlayLayer
        tempLayer.frame = CGRect(
          x: overlayFrame.origin.x + maskFrame.origin.x,
          y: overlayFrame.origin.y + maskFrame.origin.y,
          width: maskFrame.width,
          height: maskFrame.height
        )
        tempLayer.contents = maskLayer.contents
        tempLayer.contentsGravity = maskLayer.contentsGravity
        tempLayer.contentsRect = maskLayer.contentsRect
        tempLayer.contentsCenter = maskLayer.contentsCenter
        tempLayer.opacity = maskLayer.opacity
        tempLayer.compositingFilter = maskLayer.compositingFilter
        tempLayer.transform = maskLayer.transform
        tempLayer.masksToBounds = maskLayer.masksToBounds
        containerView.layer.addSublayer(tempLayer)
      }

      // Add pose layer if present (for pose task)
      if let poseLayer = self.poseLayer {
        // Create a temporary copy of the pose layer including all sublayers
        let tempLayer = CALayer()
        let overlayFrame = self.overlayLayer.frame

        // Set frame relative to main view
        tempLayer.frame = CGRect(
          x: overlayFrame.origin.x,
          y: overlayFrame.origin.y,
          width: overlayFrame.width,
          height: overlayFrame.height
        )
        tempLayer.opacity = poseLayer.opacity

        // Copy all sublayers (keypoints and skeleton lines)
        if let sublayers = poseLayer.sublayers {
          for sublayer in sublayers {
            let copyLayer = CALayer()
            copyLayer.frame = sublayer.frame
            copyLayer.backgroundColor = sublayer.backgroundColor
            copyLayer.cornerRadius = sublayer.cornerRadius
            copyLayer.opacity = sublayer.opacity

            // If it's a shape layer (for lines), copy the path
            if let shapeLayer = sublayer as? CAShapeLayer {
              let copyShapeLayer = CAShapeLayer()
              copyShapeLayer.frame = shapeLayer.frame
              copyShapeLayer.path = shapeLayer.path
              copyShapeLayer.strokeColor = shapeLayer.strokeColor
              copyShapeLayer.lineWidth = shapeLayer.lineWidth
              copyShapeLayer.fillColor = shapeLayer.fillColor
              copyShapeLayer.opacity = shapeLayer.opacity
              tempLayer.addSublayer(copyShapeLayer)
            } else {
              tempLayer.addSublayer(copyLayer)
            }
          }
        }

        containerView.layer.addSublayer(tempLayer)
      }

      // Add OBB layer if present (for OBB task)
      if let obbLayer = self.obbLayer, !obbLayer.isHidden {
        // Create a temporary copy of the OBB layer including all sublayers
        let tempLayer = CALayer()
        let overlayFrame = self.overlayLayer.frame

        tempLayer.frame = CGRect(
          x: overlayFrame.origin.x,
          y: overlayFrame.origin.y,
          width: overlayFrame.width,
          height: overlayFrame.height
        )
        tempLayer.opacity = obbLayer.opacity

        // Copy all sublayers
        if let sublayers = obbLayer.sublayers {
          for sublayer in sublayers {
            if let shapeLayer = sublayer as? CAShapeLayer {
              let copyShapeLayer = CAShapeLayer()
              copyShapeLayer.frame = shapeLayer.frame
              copyShapeLayer.path = shapeLayer.path
              copyShapeLayer.strokeColor = shapeLayer.strokeColor
              copyShapeLayer.lineWidth = shapeLayer.lineWidth
              copyShapeLayer.fillColor = shapeLayer.fillColor
              copyShapeLayer.opacity = shapeLayer.opacity
              tempLayer.addSublayer(copyShapeLayer)
            } else if let textLayer = sublayer as? CATextLayer {
              let copyTextLayer = CATextLayer()
              copyTextLayer.frame = textLayer.frame
              copyTextLayer.string = textLayer.string
              copyTextLayer.font = textLayer.font
              copyTextLayer.fontSize = textLayer.fontSize
              copyTextLayer.foregroundColor = textLayer.foregroundColor
              copyTextLayer.backgroundColor = textLayer.backgroundColor
              copyTextLayer.alignmentMode = textLayer.alignmentMode
              copyTextLayer.opacity = textLayer.opacity
              tempLayer.addSublayer(copyTextLayer)
            }
          }
        }

        containerView.layer.addSublayer(tempLayer)
      }

      // Add bounding boxes
      let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
      for info in boundingBoxInfos where !info.isHidden {
        let boxView = createBoxView(from: info)
        boxView.frame = info.rect
        containerView.addSubview(boxView)
      }
      
      // Render the container view to image
      UIGraphicsBeginImageContextWithOptions(containerView.bounds.size, true, 0.0)
      guard let context = UIGraphicsGetCurrentContext() else {
        photoCaptureCompletion?(nil)
        photoCaptureCompletion = nil
        return
      }
      
      // Draw the container view's layer hierarchy
      containerView.layer.render(in: context)
      
      // Add Ultralytics logo overlay
      if let logoImage = UIImage(named: "ultralytics_yolo_logotype.png") {
        let logoWidth: CGFloat = containerView.bounds.width * 0.45 // 45% of view width (much larger)
        let logoHeight = logoWidth * (logoImage.size.height / logoImage.size.width)
        let logoX = containerView.bounds.width - logoWidth - 30 // 30pt padding from right
        let logoY = containerView.bounds.height - logoHeight - 30 // 30pt padding from bottom
        let logoRect = CGRect(x: logoX, y: logoY, width: logoWidth, height: logoHeight)
        
        // Draw logo without background
        logoImage.draw(in: logoRect)
      }
      
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      
      // No cleanup needed since we used a temporary container view
      photoCaptureCompletion?(img)
      photoCaptureCompletion = nil
    } else {
      print("AVCapturePhotoCaptureDelegate Error")
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
