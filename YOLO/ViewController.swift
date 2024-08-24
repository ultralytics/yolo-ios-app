//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLOv8 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision

var mlModel = try! yolov8m(configuration: .init()).model

@available(iOS 15.0, *)
class ViewController: UIViewController {
  @IBOutlet var videoPreview: UIView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet var playButtonOutlet: UIBarButtonItem!
  @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
  @IBOutlet var slider: UISlider!
  @IBOutlet var sliderConf: UISlider!
  @IBOutlet weak var sliderConfLandScape: UISlider!
  @IBOutlet var sliderIoU: UISlider!
  @IBOutlet weak var sliderIoULandScape: UISlider!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelZoom: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var labelSlider: UILabel!
  @IBOutlet weak var labelSliderConf: UILabel!
  @IBOutlet weak var labelSliderConfLandScape: UILabel!
  @IBOutlet weak var labelSliderIoU: UILabel!
  @IBOutlet weak var labelSliderIoULandScape: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var toolBar: UIToolbar!
  var maskLayer: CALayer = CALayer()

  let selection = UISelectionFeedbackGenerator()
  var detector = try! VNCoreMLModel(for: mlModel)
  var session: AVCaptureSession!
  var videoCapture: VideoCapture!
  var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  // var cameraOutput: AVCapturePhotoOutput!

  // Developer mode
  let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")  // developer mode selected in settings
  let save_detections = false  // write every detection to detections.txt
  let save_frames = false  // write every frame to frames.txt

  lazy var visionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: {
        [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    return request
  }()

  enum Task {
    case detect
    case pose
  }
  
  var task: Task = .detect
  var confidenceThreshold: Float = 0.25
  var iouThreshold: Float = 0.4

  override func viewDidLoad() {
    super.viewDidLoad()
    slider.value = 30
    setLabels()
    setUpBoundingBoxViews()
    setUpOrientationChangeNotification()
    startVideo()
    // setModel()
  }

  override func viewWillTransition(
    to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
  ) {
    super.viewWillTransition(to: size, with: coordinator)

    if size.width > size.height {
      labelSliderConf.isHidden = true
      sliderConf.isHidden = true
      labelSliderIoU.isHidden = true
      sliderIoU.isHidden = true
      toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = false
      sliderConfLandScape.isHidden = false
      labelSliderIoULandScape.isHidden = false
      sliderIoULandScape.isHidden = false

    } else {
      labelSliderConf.isHidden = false
      sliderConf.isHidden = false
      labelSliderIoU.isHidden = false
      sliderIoU.isHidden = false
      toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(nil, forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = true
      sliderConfLandScape.isHidden = true
      labelSliderIoULandScape.isHidden = true
      sliderIoULandScape.isHidden = true
    }
    self.videoCapture.previewLayer?.frame = CGRect(
      x: 0, y: 0, width: size.width, height: size.height)
      coordinator.animate(
        alongsideTransition: { context in
        },
        completion: { context in
          self.setupMaskLayer()
        })
  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation()
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  @IBAction func indexChanged(_ sender: Any) {
    selection.selectionChanged()
    activityIndicator.startAnimating()
    setModel()
    setUpBoundingBoxViews()
    activityIndicator.stopAnimating()
  }

  func setModel() {
      /// Switch model
    switch task {
    case .detect:
      switch segmentedControl.selectedSegmentIndex {
      case 0:
        self.labelName.text = "YOLOv8n"
        mlModel = try! yolov8n(configuration: .init()).model
      case 1:
        self.labelName.text = "YOLOv8s"
        mlModel = try! yolov8s(configuration: .init()).model
      case 2:
        self.labelName.text = "YOLOv8m"
        mlModel = try! yolov8m(configuration: .init()).model
      case 3:
        self.labelName.text = "YOLOv8l"
        mlModel = try! yolov8l(configuration: .init()).model
      case 4:
        self.labelName.text = "YOLOv8x"
        mlModel = try! yolov8x(configuration: .init()).model
      default:
        break
      }
      
    case .pose:
      switch segmentedControl.selectedSegmentIndex {
      case 0:
        self.labelName.text = "YOLOv8n"
        mlModel = try! yolov8n_pose(configuration: .init()).model
      case 1:
        self.labelName.text = "YOLOv8s"
        mlModel = try! yolov8s_pose(configuration: .init()).model
        
      case 2:
        self.labelName.text = "YOLOv8m"
        mlModel = try! yolov8m_pose(configuration: .init()).model
      case 3:
        self.labelName.text = "YOLOv8l"
        mlModel = try! yolov8l_pose(configuration: .init()).model
      case 4:
        self.labelName.text = "YOLOv8x"
        mlModel = try! yolov8x_pose(configuration: .init()).model
      default: break
      }

    }

    DispatchQueue.global(qos: .userInitiated).async { [self] in

      /// VNCoreMLModel
      detector = try! VNCoreMLModel(for: mlModel)
      detector.featureProvider = ThresholdProvider()

      /// VNCoreMLRequest
      let request = VNCoreMLRequest(
        model: detector,
        completionHandler: { [weak self] request, error in
          self?.processObservations(for: request, error: error)
      })
      request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
      visionRequest = request
      t2 = 0.0  // inference dt smoothed
      t3 = CACurrentMediaTime()  // FPS start
      t4 = 0.0  // FPS dt smoothed
    }
  }

  /// Update thresholds from slider values
  @IBAction func sliderChanged(_ sender: Any) {
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    self.labelSliderConf.text = String(conf) + " Confidence Threshold"
    self.labelSliderIoU.text = String(iou) + " IoU Threshold"
    detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
  }

    @IBAction func taskSegmentControlChanged(_ sender: UISegmentedControl) {
      self.removeAllMaskSubLayers()

      switch sender.selectedSegmentIndex {
      case 0:
        if self.task != .detect {
          self.task = .detect
          self.setModel()
        }
      case 1:
        if self.task != .pose {
          self.task = .pose
          for i in 0..<self.boundingBoxViews.count {
            self.boundingBoxViews[i].hide()
          }
          self.setModel()
        }
      default:
        break
      }
        
    }
    
    @IBAction func takePhoto(_ sender: Any?) {
    let t0 = DispatchTime.now().uptimeNanoseconds

    // 1. captureSession and cameraOutput
    // session = videoCapture.captureSession  // session = AVCaptureSession()
    // session.sessionPreset = AVCaptureSession.Preset.photo
    // cameraOutput = AVCapturePhotoOutput()
    // cameraOutput.isHighResolutionCaptureEnabled = true
    // cameraOutput.isDualCameraDualPhotoDeliveryEnabled = true
    // print("1 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)

    // 2. Settings
    let settings = AVCapturePhotoSettings()
    // settings.flashMode = .off
    // settings.isHighResolutionPhotoEnabled = cameraOutput.isHighResolutionCaptureEnabled
    // settings.isDualCameraDualPhotoDeliveryEnabled = self.videoCapture.cameraOutput.isDualCameraDualPhotoDeliveryEnabled

    // 3. Capture Photo
    usleep(20_000)  // short 10 ms delay to allow camera to focus
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    print("3 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
  }

  @IBAction func logoButton(_ sender: Any) {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  func setLabels() {
    self.labelName.text = "YOLOv8m"
    self.labelVersion.text = "Version " + UserDefaults.standard.string(forKey: "app_version")!
  }

  @IBAction func playButton(_ sender: Any) {
    selection.selectionChanged()
    self.videoCapture.start()
    playButtonOutlet.isEnabled = false
    pauseButtonOutlet.isEnabled = true
  }

  @IBAction func pauseButton(_ sender: Any?) {
    selection.selectionChanged()
    self.videoCapture.stop()
    playButtonOutlet.isEnabled = true
    pauseButtonOutlet.isEnabled = false
  }

  @IBAction func switchCameraTapped(_ sender: Any) {
    self.videoCapture.captureSession.beginConfiguration()
    let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
    self.videoCapture.captureSession.removeInput(currentInput!)
    // let newCameraDevice = currentInput?.device == .builtInWideAngleCamera ? getCamera(with: .front) : getCamera(with: .back)

    let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
    guard let videoInput1 = try? AVCaptureDeviceInput(device: device) else {
      return
    }

    self.videoCapture.captureSession.addInput(videoInput1)
    self.videoCapture.captureSession.commitConfiguration()
  }

  // share image
  @IBAction func shareButton(_ sender: Any) {
    selection.selectionChanged()
    let settings = AVCapturePhotoSettings()
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
  }

  // share screenshot
  @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
    // let layer = UIApplication.shared.keyWindow!.layer
    // let scale = UIScreen.main.scale
    // UIGraphicsBeginImageContextWithOptions(layer.frame.size, false, scale);
    // layer.render(in: UIGraphicsGetCurrentContext()!)
    // let screenshot = UIGraphicsGetImageFromCurrentImageContext()
    // UIGraphicsEndImageContext()

    // let screenshot = UIApplication.shared.screenShot
    // UIImageWriteToSavedPhotosAlbum(screenshot!, nil, nil, nil)
  }

  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  var colors: [String: UIColor] = [:]
  var classes: [String] = []

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

    // Retrieve class labels directly from the CoreML model's class labels, if available.
    if task == .detect {
      guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
          fatalError("Class labels are missing from the model description")
      }
      classes = classLabels
        // Assign random colors to the classes.
      var count = 0
      for label in classLabels {
        let color = ultralyticsColorsolors[count]
        count += 1
        if count > 19 {
          count = 0
        }
        if colors[label] == nil {  // if key not in dict
          colors[label] = color
        }
      }

    }
  }

  func startVideo() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    videoCapture.setUp(sessionPreset: .photo) { success in
      // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
        }
        
        self.setupMaskLayer()
        self.videoPreview.layer.addSublayer(self.maskLayer)

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxViews {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  func predict(sampleBuffer: CMSampleBuffer) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      let imageOrientation: CGImagePropertyOrientation
      switch UIDevice.current.orientation {
      case .portrait:
        imageOrientation = .up
      case .portraitUpsideDown:
        imageOrientation = .down
      case .landscapeLeft:
        imageOrientation = .up
      case .landscapeRight:
        imageOrientation = .up
      case .unknown:
        imageOrientation = .up
      default:
        imageOrientation = .up
      }

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
        t0 = CACurrentMediaTime()  // inference start
        do {
          try handler.perform([visionRequest])
        } catch {
          print(error)
        }
        t1 = CACurrentMediaTime() - t0  // inference dt
      }

      currentBuffer = nil
    }
  }

  func processObservations(for request: VNRequest, error: Error?) {
    switch task {
    case .detect:

      DispatchQueue.main.async {
        if let results = request.results as? [VNRecognizedObjectObservation] {
          self.show(predictions: results, predsPose: [])
        } else {
          self.show(predictions: [], predsPose: [])
        }

        // Measure FPS
        if self.t1 < 10.0 {  // valid dt
          self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
        }
        self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
        self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
        self.t3 = CACurrentMediaTime()
      }
        
    case .pose:
      if let results = request.results as? [VNCoreMLFeatureValueObservation] {
        DispatchQueue.main.async { [self] in

          if let prediction = results.first?.featureValue.multiArrayValue {

            let preds = PostProcessPose(
              prediction: prediction, confidenceThreshold: self.confidenceThreshold,
              iouThreshold: self.iouThreshold)
            var boxes = [(CGRect, Float)]()
            var kpts = [[Float]]()

            for pred in preds {
              boxes.append((pred.0, pred.1))
              kpts.append(pred.2)
            }
            self.show(predictions: [], predsPose: preds)
            self.maskLayer.sublayers?.forEach { $0.removeFromSuperlayer() }

            self.drawKeypoints(
              keypointsList: kpts, boundingBoxes: boxes, on: maskLayer,
              imageViewSize: maskLayer.bounds.size, originalImageSize: maskLayer.bounds.size)

          } else {
            self.show(predictions: [], predsPose: [])
          }
          if self.t1 < 10.0 {  // valid dt
            self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
          }
          self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
          self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
          self.t3 = CACurrentMediaTime()
        }
      }
    }

  }

  // Save text file
  func saveText(text: String, file: String = "saved.txt") {
    if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
      let fileURL = dir.appendingPathComponent(file)

      // Writing
      do {  // Append to file if it exists
        let fileHandle = try FileHandle(forWritingTo: fileURL)
        fileHandle.seekToEndOfFile()
        fileHandle.write(text.data(using: .utf8)!)
        fileHandle.closeFile()
      } catch {  // Create new file and write
        do {
          try text.write(to: fileURL, atomically: false, encoding: .utf8)
        } catch {
          print("no file written")
        }
      }

      // Reading
      // do {let text2 = try String(contentsOf: fileURL, encoding: .utf8)} catch {/* error handling here */}
    }
  }

  // Save image file
  func saveImage() {
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    let fileURL = dir!.appendingPathComponent("saved.jpg")
    let image = UIImage(named: "ultralytics_yolo_logotype.png")
    FileManager.default.createFile(
      atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
  }

  // Return hard drive space (GB)
  func freeSpace() -> Double {
    let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
    do {
      let values = try fileURL.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey
      ])
      return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9  // Bytes to GB
    } catch {
      print("Error retrieving storage capacity: \(error.localizedDescription)")
    }
    return 0
  }

  // Return RAM usage (GB)
  func memoryUsage() -> Double {
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    if kerr == KERN_SUCCESS {
      return Double(taskInfo.resident_size) / 1E9  // Bytes to GB
    } else {
      return 0
    }
  }

  func show(predictions: [VNRecognizedObjectObservation], predsPose: [(CGRect, Float, [Float])]) {
    let width = videoPreview.bounds.width  // 375 pix
    let height = videoPreview.bounds.height  // 812 pix
    var str = ""

    // ratio = videoPreview AR divided by sessionPreset AR
    var ratio: CGFloat = 1.0
    if videoCapture.captureSession.sessionPreset == .photo {
      ratio = (height / width) / (4.0 / 3.0)  // .photo
    } else {
      ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
    }

    // date
    let date = Date()
    let calendar = Calendar.current
    let hour = calendar.component(.hour, from: date)
    let minutes = calendar.component(.minute, from: date)
    let seconds = calendar.component(.second, from: date)
    let nanoseconds = calendar.component(.nanosecond, from: date)
    let sec_day =
      Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day

    var resultCount = 0

    switch task {
      case .detect:
        resultCount = predictions.count
      case .pose:
        resultCount = predsPose.count
    }

    self.labelSlider.text =
      String(predictions.count) + " items (max " + String(Int(slider.value)) + ")"

    for i in 0..<boundingBoxViews.count {
      if i < (resultCount) && i < Int(slider.value) {
        var rect = CGRect.zero
        var label = ""
        var boxColor: UIColor = .white
        var confidence: CGFloat = 0
        var alpha: CGFloat = 0.9
        var bestClass = ""
        switch task {
        case .detect:
          let prediction = predictions[i]
          rect = prediction.boundingBox
          bestClass = prediction.labels[0].identifier
          confidence = CGFloat(prediction.labels[0].confidence)
          label = String(format: "%@ %.1f", bestClass, confidence * 100)
          boxColor = colors[bestClass] ?? UIColor.white
          alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
        case .pose:
          let predPose = predsPose[i]
          let box = predPose.0
          let conf = predPose.1
          rect = CGRect(
              x: box.minX / 640, y: box.minY / 640, width: box.width / 640, height: box.height / 640)
          bestClass = "person"
          confidence = CGFloat(conf)
          label = String(format: "%@ %.1f", bestClass, confidence * 100)
          boxColor = ultralyticsColorsolors[0]
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

        if developerMode {
          if save_detections {
            str += String(
                format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
          }
        }

        } else {
          boundingBoxViews[i].hide()
      }
    }
  

    // Write
    if developerMode {
      if save_detections {
        saveText(text: str, file: "detections.txt")  // Write stats for each detection
      }
      if save_frames {
        str = String(
          format: "%.3f %.3f %.3f %.3f %.1f %.1f %.1f\n",
          sec_day, freeSpace(), memoryUsage(), UIDevice.current.batteryLevel,
          self.t1 * 1000, self.t2 * 1000, 1 / self.t4)
        saveText(text: str, file: "frames.txt")  // Write stats for each image
      }
    }

    // Debug
    // print(str)
    // print(UIDevice.current.identifierForVendor!)
    // saveImage()
  }

  // Pinch to Zoom Start ---------------------------------------------------------------------------------------------
  let minimumZoom: CGFloat = 1.0
  let maximumZoom: CGFloat = 10.0
  var lastZoomFactor: CGFloat = 1.0

  @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
    let device = videoCapture.captureDevice

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
  }  // Pinch to Zoom End --------------------------------------------------------------------------------------------
}  // ViewController class End

@available(iOS 15.0, *)
extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
    predict(sampleBuffer: sampleBuffer)
  }
}

@available(iOS 15.0, *)
// Programmatically save image
extension ViewController: AVCapturePhotoCaptureDelegate {
  func photoOutput(
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
      var orientation = CGImagePropertyOrientation.right
      switch UIDevice.current.orientation {
      case .landscapeLeft:
        orientation = .up
      case .landscapeRight:
        orientation = .down
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
      imageView.frame = videoPreview.frame
      let imageLayer = imageView.layer
      videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

      let bounds = UIScreen.main.bounds
      UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
      self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      imageLayer.removeFromSuperlayer()
      let activityViewController = UIActivityViewController(
        activityItems: [img!], applicationActivities: nil)
      activityViewController.popoverPresentationController?.sourceView = self.View0
      self.present(activityViewController, animated: true, completion: nil)
      //
      //            // Save to camera roll
      //            UIImageWriteToSavedPhotosAlbum(img!, nil, nil, nil);
    } else {
      print("AVCapturePhotoCaptureDelegate Error")
    }
  }
}
