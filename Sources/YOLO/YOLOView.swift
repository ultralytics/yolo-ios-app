import UIKit
import Vision
import AVFoundation

@MainActor
public class YOLOView: UIView, VideoCaptureDelegate{
    func onInferenceTime(speed: Double, fps: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
        }
    }
    
    func onPredict(result: YOLOResult) {
        
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
            var keypointList = [[(x:Float, y:Float)]]()
            var confsList = [[Float]]()
            
            for keypoint in result.keypointsList {
                keypointList.append(keypoint.xyn)
                confsList.append(keypoint.conf)
            }
            guard let poseLayer = poseLayer else { return }
            drawKeypoints(keypointsList: keypointList, confsList: confsList, boundingBoxes: result.boxes,  on: poseLayer, imageViewSize: overlayLayer.frame.size, originalImageSize: result.orig_shape)
        } else if task == .obb {
            //            self.setupObbLayerIfNeeded()
            guard let obbLayer = self.obbLayer else { return }
            let obbDetections = result.obb
            self.obbRenderer.drawObbDetectionsWithReuse(
                obbDetections: obbDetections,
                on: obbLayer,
                imageViewSize: self.overlayLayer.frame.size,
                originalImageSize: result.orig_shape, // 例
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
        task: YOLOTask) {
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
        if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage") || lowercasedPath.hasSuffix(".mlmodelc") {
            let possibleURL = URL(fileURLWithPath: modelPathOrName)
            if fileManager.fileExists(atPath: possibleURL.path) {
                modelURL = possibleURL
            }
        } else {
            if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc") {
                modelURL = compiledURL
            } else if let packageURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlpackage") {
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
            self.labelName.text = modelName
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
            Classifier.create(unwrappedModelURL: unwrappedModelURL,isRealTime: true) { [weak self] result in
                switch result {
                case .success(let predictor):
                    handleSuccess(predictor: predictor)
                case .failure(let error):
                    handleFailure(error)
                }
            }
            
        case .segment:
            Segmenter.create(unwrappedModelURL: unwrappedModelURL,isRealTime: true) { [weak self] result in
                switch result {
                case .success(let predictor):
                    handleSuccess(predictor: predictor)
                case .failure(let error):
                    handleFailure(error)
                }
            }
            
        case .pose:
            PoseEstimater.create(unwrappedModelURL: unwrappedModelURL,isRealTime: true) { [weak self] result in
                switch result {
                case .success(let predictor):
                    handleSuccess(predictor: predictor)
                case .failure(let error):
                    handleFailure(error)
                }
            }
            
        case .obb:
            ObbDetector.create(unwrappedModelURL: unwrappedModelURL,isRealTime: true) { [weak self] result in
                switch result {
                case .success(let predictor):
                    self?.obbLayer?.isHidden = false
                    
                    handleSuccess(predictor: predictor)
                case .failure(let error):
                    handleFailure(error)
                }
            }
            
        default:
            ObjectDetector.create(unwrappedModelURL: unwrappedModelURL,isRealTime: true) { [weak self] result in
                switch result {
                case .success(let predictor):
                    handleSuccess(predictor: predictor)
                case .failure(let error):
                    handleFailure(error)
                }
            }
        }
    }
    
    private func start(position: AVCaptureDevice.Position){
        if !busy {
            busy = true
            let orientation = UIDevice.current.orientation
            videoCapture.setUp(sessionPreset: .photo, position: position, orientation: orientation) { success in
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
    
    public func stop(){
        videoCapture.stop()
    }
    
    public func resume(){
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
            // 必要に応じて contentsGravity や backgroundColor 等を指定
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
        default:break
        }
    }
    
    func removeAllSubLayers(parentLayer:CALayer?) {
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
            
            self.labelSliderNumItems.text = String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
            for i in 0..<boundingBoxViews.count {
                if i < (resultCount) && i < 50 {
                    var rect = CGRect.zero
                    var label = ""
                    var boxColor: UIColor = .white
                    var confidence: CGFloat = 0
                    var alpha: CGFloat = 0.9
                    var bestClass = ""
                    
                    switch task {
                    case .detect:
                        let prediction = predictions.boxes[i]
                        rect = CGRect(x: prediction.xywhn.minX, y: 1-prediction.xywhn.maxY, width: prediction.xywhn.width, height: prediction.xywhn.height)
                        bestClass = prediction.cls
                        confidence = CGFloat(prediction.conf)
                        let colorIndex = prediction.index % ultralyticsColors.count
                        boxColor = ultralyticsColors[colorIndex]
                        label = String(format: "%@ %.1f", bestClass, confidence * 100)
                        alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                    default:
                        let prediction = predictions.boxes[i]
                        let clsIndex = prediction.index
                        rect = prediction.xywhn
                        bestClass = prediction.cls
                        confidence = CGFloat(prediction.conf)
                        label = String(format: "%@ %.1f", bestClass, confidence * 100)
                        let colorIndex = prediction.index % ultralyticsColors.count
                        boxColor = ultralyticsColors[colorIndex]
                        alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                        
                    }
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
                        if task == .detect {
                            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                            displayRect = displayRect.applying(transform)
                        } else {
                            let transform = CGAffineTransform(translationX: offset, y: 0)
                            displayRect = displayRect.applying(transform)
                        }
                        displayRect.size.width *= ratio
                    } else {
                        if task == .detect {
                            let offset = (ratio - 1) * (0.5 - displayRect.maxY)
                            
                            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                            displayRect = displayRect.applying(transform)
                        } else {
                            let offset = (ratio - 1) * (0.5 - displayRect.minY)
                            let transform = CGAffineTransform(translationX: 0, y: offset)
                            displayRect = displayRect.applying(transform)
                        }
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
            self.labelSliderNumItems.text = String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
            
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
                    
                    switch task {
                    case .detect:
                        let prediction = predictions.boxes[i]
                        // detectタスクの場合は、いままで通り「y を 1 - maxY」で反転
                        rect = CGRect(
                            x: prediction.xywhn.minX,
                            y: 1 - prediction.xywhn.maxY,
                            width: prediction.xywhn.width,
                            height: prediction.xywhn.height
                        )
                        bestClass = prediction.cls
                        confidence = CGFloat(prediction.conf)
                        
                    default:
                        let prediction = predictions.boxes[i]
                        // ここを detect と同じように y を反転する
                        rect = CGRect(
                            x: prediction.xywhn.minX,
                            y: 1 - prediction.xywhn.maxY,
                            width: prediction.xywhn.width,
                            height: prediction.xywhn.height
                        )
                        bestClass = prediction.cls
                        confidence = CGFloat(prediction.conf)
                    }
                    
                    // ラベルや色の設定は共通でOK
                    let colorIndex = predictions.boxes[i].index % ultralyticsColors.count
                    boxColor = ultralyticsColors[colorIndex]
                    label = String(format: "%@ %.1f", bestClass, confidence * 100)
                    alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                    
                    // 以下はスケーリング・オフセット処理 (もともとのままでOK)
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
            }        }
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
              let top1Conf = result.probs?.top1Conf else {
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
        textLayer.contentsScale = UIScreen.main.scale  // Retina対応
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
            .font : UIFont.systemFont(ofSize: fontSize, weight: .semibold)
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
        labelName.text = modelName
        labelName.textAlignment = .center
        labelName.font = UIFont.systemFont(ofSize: 24, weight: .medium)
        labelName.textColor = .black
        labelName.font = UIFont.preferredFont(forTextStyle: .title1)
        self.addSubview(labelName)
        
        labelFPS.text = String(format: "%.1f FPS - %.1f ms", 0.0, 0.0)
        labelFPS.textAlignment = .center
        labelFPS.textColor = .black
        labelFPS.font = UIFont.preferredFont(forTextStyle: .body)
        self.addSubview(labelFPS)
        
        labelSliderNumItems.text = "0 items (max 30)"
        labelSliderNumItems.textAlignment = .left
        labelSliderNumItems.textColor = .black
        labelSliderNumItems.font = UIFont.preferredFont(forTextStyle: .subheadline)
        self.addSubview(labelSliderNumItems)
        
        sliderNumItems.minimumValue = 0
        sliderNumItems.maximumValue = 100
        sliderNumItems.value = 30
        sliderNumItems.minimumTrackTintColor = .darkGray
        sliderNumItems.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
        sliderNumItems.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
        self.addSubview(sliderNumItems)
        
        labelSliderConf.text = "0.25 Confidence Threshold"
        labelSliderConf.textAlignment = .left
        labelSliderConf.textColor = .black
        labelSliderConf.font = UIFont.preferredFont(forTextStyle: .subheadline)
        self.addSubview(labelSliderConf)
        
        sliderConf.minimumValue = 0
        sliderConf.maximumValue = 1
        sliderConf.value = 0.25
        sliderConf.minimumTrackTintColor = .darkGray
        sliderConf.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
        sliderConf.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
        self.addSubview(sliderConf)
        
        labelSliderIoU.text = "0.45 IoU Threshold"
        labelSliderIoU.textAlignment = .left
        labelSliderIoU.textColor = .black
        labelSliderIoU.font = UIFont.preferredFont(forTextStyle: .subheadline)
        self.addSubview(labelSliderIoU)
        
        sliderIoU.minimumValue = 0
        sliderIoU.maximumValue = 1
        sliderIoU.value = 0.45
        sliderIoU.minimumTrackTintColor = .darkGray
        sliderIoU.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
        sliderIoU.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
        self.addSubview(sliderIoU)
        
        self.labelSliderNumItems.text = "0 items (max " + String(Int(sliderNumItems.value)) + ")"
        self.labelSliderConf.text = "0.25 Confidence Threshold"
        self.labelSliderIoU.text = "0.45 IoU Threshold"
        
        labelZoom.text = "1.00x"
        labelZoom.textColor = .black
        labelZoom.font = UIFont.systemFont(ofSize: 14)
        labelZoom.textAlignment = .center
        labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
        self.addSubview(labelZoom)
        
        let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
        
        
        playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
        playButton.tintColor = .systemGray
        pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
        pauseButton.tintColor = .systemGray
        switchCameraButton = UIButton()
        switchCameraButton.setImage(UIImage(systemName: "camera.rotate", withConfiguration: config), for: .normal)
        switchCameraButton.tintColor = .systemGray
        playButton.isEnabled = false
        pauseButton.isEnabled = true
        playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
        pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
        switchCameraButton.addTarget(self, action: #selector(switchCameraTapped), for: .touchUpInside)
        toolbar.backgroundColor = .darkGray.withAlphaComponent(0.7)
        self.addSubview(toolbar)
        toolbar.addSubview(playButton)
        toolbar.addSubview(pauseButton)
        toolbar.addSubview(switchCameraButton)
        
        self.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
    }
    
    public override func layoutSubviews() {
        setupOverlayLayer()
        let isLandscape = bounds.width > bounds.height
        activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
        if isLandscape {
            toolbar.backgroundColor = .clear
            playButton.tintColor = .darkGray
            pauseButton.tintColor = .darkGray
            switchCameraButton.tintColor = .darkGray
            
            let width = bounds.width
            let height = bounds.height
            
            let topMargin: CGFloat = 0
            
            let titleLabelHeight: CGFloat = height * 0.1
            labelName.frame = CGRect(
                x: 0,
                y: topMargin,
                width: width,
                height: titleLabelHeight
            )
            
            let subLabelHeight: CGFloat = height * 0.04
            labelFPS.frame = CGRect(
                x: 0,
                y: center.y - height * 0.24 - subLabelHeight,
                width: width,
                height: subLabelHeight
            )
            
            let sliderWidth: CGFloat = width * 0.2
            let sliderHeight: CGFloat = height * 0.1
            
            labelSliderNumItems.frame = CGRect(
                x: width * 0.1,
                y: labelFPS.frame.minY - sliderHeight,
                width: sliderWidth,
                height: sliderHeight
            )
            
            sliderNumItems.frame = CGRect(
                x: width * 0.1,
                y: labelSliderNumItems.frame.maxY + 10,
                width: sliderWidth,
                height: sliderHeight
            )
            
            labelSliderConf.frame = CGRect(
                x: width * 0.1,
                y: sliderNumItems.frame.maxY + 10,
                width: sliderWidth * 1.5,
                height: sliderHeight
            )
            
            sliderConf.frame = CGRect(
                x: width * 0.1,
                y: labelSliderConf.frame.maxY + 10,
                width: sliderWidth,
                height: sliderHeight
            )
            
            labelSliderIoU.frame = CGRect(
                x: width * 0.1,
                y: sliderConf.frame.maxY + 10,
                width: sliderWidth * 1.5,
                height: sliderHeight
            )
            
            sliderIoU.frame = CGRect(
                x: width * 0.1,
                y: labelSliderIoU.frame.maxY + 10,
                width: sliderWidth,
                height: sliderHeight
            )
            
            
            let zoomLabelWidth: CGFloat = width * 0.2
            labelZoom.frame = CGRect(
                x: center.x - zoomLabelWidth / 2,
                y: self.bounds.maxY - 120,
                width: zoomLabelWidth,
                height: height * 0.03
            )
            
            let toolBarHeight: CGFloat = 66
            let buttonHeihgt: CGFloat = toolBarHeight * 0.75
            toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
            playButton.frame = CGRect(x: 0, y: 0, width: buttonHeihgt, height: buttonHeihgt)
            pauseButton.frame = CGRect(x: playButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
            switchCameraButton.frame = CGRect(x: pauseButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
        } else {
            toolbar.backgroundColor = .darkGray.withAlphaComponent(0.7)
            playButton.tintColor = .systemGray
            pauseButton.tintColor = .systemGray
            switchCameraButton.tintColor = .systemGray
            
            let width = bounds.width
            let height = bounds.height
            
            let topMargin: CGFloat = height * 0.02
            
            let titleLabelHeight: CGFloat = height * 0.1
            labelName.frame = CGRect(
                x: 0,
                y: topMargin,
                width: width,
                height: titleLabelHeight
            )
            
            let subLabelHeight: CGFloat = height * 0.04
            labelFPS.frame = CGRect(
                x: 0,
                y: labelName.frame.maxY + 15,
                width: width,
                height: subLabelHeight
            )
            
            let sliderWidth: CGFloat = width * 0.46
            let sliderHeight: CGFloat = height * 0.02
            
            sliderNumItems.frame = CGRect(
                x: width * 0.01,
                y: center.y - sliderHeight - height * 0.24,
                width: sliderWidth,
                height: sliderHeight
            )
            
            labelSliderNumItems.frame = CGRect(
                x: width * 0.01,
                y: sliderNumItems.frame.minY - sliderHeight - 10,
                width: sliderWidth,
                height: sliderHeight
            )
            
            labelSliderConf.frame = CGRect(
                x: width * 0.01,
                y: center.y + height * 0.24,
                width: sliderWidth * 1.5,
                height: sliderHeight
            )
            
            sliderConf.frame = CGRect(
                x: width * 0.01,
                y: labelSliderConf.frame.maxY + 10,
                width: sliderWidth,
                height: sliderHeight
            )
            
            labelSliderIoU.frame = CGRect(
                x: width * 0.01,
                y: sliderConf.frame.maxY + 10,
                width: sliderWidth * 1.5,
                height: sliderHeight
            )
            
            sliderIoU.frame = CGRect(
                x: width * 0.01,
                y: labelSliderIoU.frame.maxY + 10,
                width: sliderWidth,
                height: sliderHeight
            )
            
            
            let zoomLabelWidth: CGFloat = width * 0.2
            labelZoom.frame = CGRect(
                x: center.x - zoomLabelWidth / 2,
                y: self.bounds.maxY - 120,
                width: zoomLabelWidth,
                height: height * 0.03
            )
            
            let toolBarHeight: CGFloat = 66
            let buttonHeihgt: CGFloat = toolBarHeight * 0.75
            toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
            playButton.frame = CGRect(x: 0, y: 0, width: buttonHeihgt, height: buttonHeihgt)
            pauseButton.frame = CGRect(x: playButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
            switchCameraButton.frame = CGRect(x: pauseButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
        }
        
        self.videoCapture.previewLayer?.frame = self.bounds
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
        videoCapture.updateVideoOrientation(orientation:orientation)
        
        //      frameSizeCaptured = false
    }
    
    @objc func sliderChanged(_ sender: Any) {
        
        if sender as? UISlider === sliderNumItems {
            if let detector = videoCapture.predictor as? ObjectDetector {
                let numItems = Int(sliderNumItems.value)
                detector.setNumItemsThreshold(numItems: numItems)
            }
        }
        let conf = Double(round(100 * sliderConf.value)) / 100
        let iou = Double(round(100 * sliderIoU.value)) / 100
        self.labelSliderConf.text = String(conf) + " Confidence Threshold"
        self.labelSliderIoU.text = String(iou) + " IoU Threshold"
        if let detector = videoCapture.predictor as? ObjectDetector {
            detector.setIouThreshold(iou: iou)
            detector.setConfidenceThreshold(confidence: conf)
            
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

extension YOLOView: @preconcurrency AVCapturePhotoCaptureDelegate {
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
            let imageView = UIImageView(image: image)
            imageView.contentMode = .scaleAspectFill
            imageView.frame = self.frame
            let imageLayer = imageView.layer
            self.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)
            
            var tempViews = [UIView]()
            let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
            for info in boundingBoxInfos where !info.isHidden {
                let boxView = createBoxView(from: info)
                boxView.frame = info.rect
                
                self.addSubview(boxView)
                tempViews.append(boxView)
            }
            let bounds = UIScreen.main.bounds
            UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
            self.drawHierarchy(in: bounds, afterScreenUpdates: true)
            let img = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            imageLayer.removeFromSuperlayer()
            for v in tempViews {
                v.removeFromSuperview()
            }
            photoCaptureCompletion?(img)
            photoCaptureCompletion = nil
        } else {
            print("AVCapturePhotoCaptureDelegate Error")
        }
    }
}
