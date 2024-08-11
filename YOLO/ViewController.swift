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
import CoreMedia
import CoreML
import UIKit
import Vision

var mlModel = try! yolov8m(configuration: .init()).model
enum Task {
    case detect
    case human
}

class ViewController: UIViewController {
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var segmentedControl: UISegmentedControl!
    @IBOutlet weak var taskSegmentControl: UISegmentedControl!
    @IBOutlet weak var trackingLabel: UILabel!
    @IBOutlet weak var trackingSwitch: UISwitch!
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
    
    @IBOutlet weak var saveDataButton: UIBarButtonItem!
    
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
    var developerMode = UserDefaults.standard.bool(forKey: "developer_mode")   // developer mode selected in settings
    var save_detections = false  // write every detection to detections.txt
    let save_frames = false  // write every frame to frames.txt
    var save_strings:[String] = []
    let saveQueue = DispatchQueue(label: "com.ultralytics.saveQueue")
    
    lazy var visionRequest: VNCoreMLRequest = {
        let request = VNCoreMLRequest(model: detector, completionHandler: {
            [weak self] request, error in
            self?.processObservations(for: request, error: error)
        })
        // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
        request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
        return request
    }()
    
    var task: Task = .detect
    var confidenceThreshold:Float = 0.25
    var iouThreshold:Float = 0.4
    var tracking = false
    var tracker = TrackingModel()

    override func viewDidLoad() {
        super.viewDidLoad()
        NotificationCenter.default.addObserver(self, selector: #selector(updateDeveloperMode), name: .settingsChanged, object: nil)
        slider.value = 30
        taskSegmentControl.selectedSegmentIndex = 0
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        startVideo()
        // setModel()
    }
    
    @objc func updateDeveloperMode() {
        let userDefaults = UserDefaults.standard
        developerMode = userDefaults.bool(forKey: "developer_mode")
        if !developerMode {
            save_detections = false
            saveDataButton.isEnabled = false
            saveDataButton.tintColor = UIColor.clear
        } else {
            saveDataButton.isEnabled = true
            saveDataButton.tintColor = nil
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if developerMode {
            save_strings = []
            save_detections = false
            saveDataButton.isEnabled = true
            saveDataButton.tintColor = nil
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self, name: .settingsChanged, object: nil)
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator) {
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
        self.videoCapture.previewLayer?.frame = CGRect(x: 0, y: 0, width: size.width, height: size.height)

    }
    
    private func setUpOrientationChangeNotification() {
         NotificationCenter.default.addObserver(self, selector: #selector(orientationDidChange), name: UIDevice.orientationDidChangeNotification, object: nil)
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
        case .human:
            switch segmentedControl.selectedSegmentIndex {
            case 0:
                self.labelName.text = "YOLOv8n"
                if #available(iOS 15.0, *) {
                    mlModel = try! yolov8n_human(configuration: .init()).model
                } else {
                    // Fallback on earlier versions
                }
            case 1:
                self.labelName.text = "YOLOv8s"
                if #available(iOS 15.0, *) {
                    mlModel = try! yolov8s_human(configuration: .init()).model
                } else {
                    // Fallback on earlier versions
                }
            case 2:
                self.labelName.text = "YOLOv8m"
                if #available(iOS 15.0, *) {
                    mlModel = try! yolov8m_human(configuration: .init()).model
                } else {
                    // Fallback on earlier versions
                }
            case 3:
                self.labelName.text = "YOLOv8l"
                if #available(iOS 15.0, *) {
                    mlModel = try! yolov8l_human(configuration: .init()).model
                } else {
                    // Fallback on earlier versions
                }
            case 4:
                self.labelName.text = "YOLOv8x"
                if #available(iOS 15.0, *) {
                    mlModel = try! yolov8x_human(configuration: .init()).model
                } else {
                    // Fallback on earlier versions
                }
            default:
                break
            }
            
        }
        DispatchQueue.global(qos: .userInitiated).async { [self] in

        /// VNCoreMLModel
        detector = try! VNCoreMLModel(for: mlModel)
        detector.featureProvider = ThresholdProvider()
        
        /// VNCoreMLRequest
        let request = VNCoreMLRequest(model: detector, completionHandler: { [weak self] request, error in
            self?.processObservations(for: request, error: error)
        })
        request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
        visionRequest = request
        t2 = 0.0 // inference dt smoothed
        t3 = CACurrentMediaTime()  // FPS start
        t4 = 0.0  // FPS dt smoothed
        }
    }
    
    /// Update thresholds from slider values
    @IBAction func sliderChanged(_ sender: Any) {
        self.confidenceThreshold = sliderConf.value
        self.iouThreshold = sliderIoU.value
        let conf = Double(round(100 * sliderConf.value)) / 100
        let iou = Double(round(100 * sliderIoU.value)) / 100
        self.labelSliderConf.text = String(conf) + " Confidence Threshold"
        self.labelSliderIoU.text = String(iou) + " IoU Threshold"
        detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
    }
    
    @IBAction func taskSegmentControlChanged(_ sender: UISegmentedControl) {
        save_strings.removeAll()
        saveDataButton.tintColor = nil
        save_detections = false

        switch sender.selectedSegmentIndex {
        case 0:
            if self.task != .detect {
                self.trackingLabel.isHidden = true
                self.trackingSwitch.isHidden = true
                self.task = .detect
                self.setModel()
            }
        case 1:
            if self.task != .human {
                self.task = .human
                for i in 0..<self.boundingBoxViews.count {
                    self.boundingBoxViews[i].hide()
                }
                self.trackingLabel.isHidden = false
                self.trackingSwitch.isHidden = false
                self.setModel()
            }
        default:
            break
        }
    }
     
    @IBAction func TrackingSwitch(_ sender: UISwitch) {
        tracking.toggle()
        if tracking {
            sender.isOn = true
        } else {
            sender.isOn = false
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
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
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
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }
  
    @IBAction func saveData(_ sender: Any) {
        save_detections.toggle()
        if !save_detections {
            saveDataButton.tintColor = nil

            DispatchQueue.global(qos: .userInitiated).async { [self] in
                
                if let url = saveDetectionResultsToCSV(detectionResults: save_strings, task: task) {
                    DispatchQueue.main.async {
                        
                        let activityViewController = UIActivityViewController(activityItems: [url], applicationActivities: nil)
                        activityViewController.popoverPresentationController?.sourceView = self.view
                        activityViewController.completionWithItemsHandler = { activity, success, items, error in
                            if success {
                                do {
                                    try FileManager.default.removeItem(at: url)
                                    print("CSV file deleted successfully")
                                } catch {
                                    print("Failed to delete CSV file: \(error)")
                                }
                            } else if let error = error {
                                print("Activity failed: \(error)")
                            } else {
                                print("Activity was cancelled.")
                            }
                        }

                        self.present(activityViewController, animated: true, completion: nil)
                    }
                } else {
                    print("Failed to create CSV file.")
                }
                save_strings.removeAll()
            }
        } else {
            saveDataButton.tintColor = UIColor.red
        }
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
            
            // Assign random colors to the classes.
            for label in classLabels {
                if colors[label] == nil {  // if key not in dict
                    colors[label] = UIColor(red: CGFloat.random(in: 0...1),
                                            green: CGFloat.random(in: 0...1),
                                            blue: CGFloat.random(in: 0...1),
                                            alpha: 0.6)
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
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
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
                    self.show(predictions: results, persons: [])
                } else {
                    self.show(predictions: [], persons: [])
                }
                
                // Measure FPS
                if self.t1 < 10.0 {  // valid dt
                    self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
                }
                self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
                self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
                self.t3 = CACurrentMediaTime()
            }
        case .human:
            if let results = request.results as? [VNCoreMLFeatureValueObservation] {
                DispatchQueue.main.async {
                    
                    if let prediction = results.first?.featureValue.multiArrayValue {

                        let pred = PostProcessHuman(prediction:prediction, confidenceThreshold: self.confidenceThreshold, iouThreshold: self.iouThreshold)
                        var persons:[Person] = []
                        if !self.tracking {
                            persons = toPerson(boxesAndScoresAndFeatures: pred)
                        } else {
                            persons = self.tracker.track(boxesAndScoresAndFeatures: pred)
                        }
                        self.show(predictions: [], persons: persons)
                    } else {
                        self.show(predictions: [], persons: [])
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
    
    func measureFPS() {
        
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
        FileManager.default.createFile(atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
    }
    
    // Return hard drive space (GB)
    func freeSpace() -> Double {
        let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
        do {
            let values = try fileURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9   // Bytes to GB
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
            return Double(taskInfo.resident_size) / 1E9   // Bytes to GB
        } else {
            return 0
        }
    }
    
    func show(predictions: [VNRecognizedObjectObservation], persons: [Person]) {
        let width = videoPreview.bounds.width
        let height = videoPreview.bounds.height
        var str = ""
        
        var ratio: CGFloat = 1.0
        
        if videoCapture.captureSession.sessionPreset == .photo {
            ratio = (height / width) / (4.0 / 3.0)
        } else {
            ratio = (height / width) / (16.0 / 9.0)
        }
        
        let date = Date()
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: date)
        let minutes = calendar.component(.minute, from: date)
        let seconds = calendar.component(.second, from: date)
        let nanoseconds = calendar.component(.nanosecond, from: date)
        let sec_day = Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9
        
        var resultCount = 0
        
        switch task {
        case .detect:
            resultCount = predictions.count
        case .human:
            resultCount = persons.count
        }
        self.labelSlider.text = String(resultCount) + " items (max " + String(Int(slider.value)) + ")"
        for i in 0..<boundingBoxViews.count {
            if i < (resultCount) && i < Int(slider.value) {
                var rect = CGRect.zero
                var label = ""
                var boxColor:UIColor = .white
                var confidence:CGFloat = 0
                var alpha:CGFloat = 0.9
                var innerTexts = ""
                var bestClass = ""
                switch task  {
                case .detect:
                    let prediction = predictions[i]
                    rect = prediction.boundingBox
                    bestClass = prediction.labels[0].identifier
                    confidence = CGFloat(prediction.labels[0].confidence)
                    label = String(format: "%@ %.1f", bestClass, confidence * 100)
                    boxColor = colors[bestClass] ?? UIColor.white
                    alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                case .human:
                    let person = persons[i]
                    let box = person.box
                    rect = CGRect(x: box.minX/640, y: box.minY/640, width: box.width/640, height: box.height/640)
                    confidence = CGFloat(person.score)
                    if person.index == -1 {
                        label = "person"
                    } else {
                        label = String(format: "%@ %.1f", "ID: \(person.index)", confidence * 100)
                    }
                    innerTexts = "weight: " + String(format: "%.2f", person.weight) + "\n" + "height: " + String(format: "%.2f", person.height) + "\n" + "age: " + String(person.age) + "\n" + person.gender + ": " + String(format: "%.2f", person.genderConfidence) + "\n" + person.race + ": " + String(format: "%.2f", person.raceConfidence)
                    boxColor = person.color
                    
                }
                var displayRect = rect
                switch UIDevice.current.orientation {
                case .portraitUpsideDown:
                    displayRect = CGRect(x: 1.0 - rect.origin.x - rect.width,
                                         y: 1.0 - rect.origin.y - rect.height,
                                         width: rect.width,
                                         height: rect.height)
                case .landscapeLeft:
                    displayRect = CGRect(x: rect.origin.x,
                                         y: rect.origin.y,
                                         width: rect.width,
                                         height: rect.height)
                case .landscapeRight:
                    displayRect = CGRect(x: rect.origin.x,
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
                
                boundingBoxViews[i].show(frame: displayRect, label: label, color: boxColor, alpha: alpha, innerTexts: innerTexts)
                
                if developerMode {
                    if save_detections {
                        saveQueue.async { [self] in
                            guard save_detections else {return}
                            var str = ""
                            switch task {
                            case .detect:
                                str += String(format: "%.3f,%.3f,%.3f,%@, %.2f,%.3f,%.3f,%.3f,%.3f\n",
                                                       sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                                                       rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
                            case .human:
                                let person = persons[i]
                                var id = ""
                                if person.index == -1 {
                                    id = "-"
                                } else {
                                    id = String(person.index)
                                }
                                str += String(format: "%.3f,%.3f,%.3f,%@, %.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%d,%@,%.2f,%@,%.2f\n",
                                              sec_day, freeSpace(), UIDevice.current.batteryLevel, id, confidence,
                                              rect.origin.x, rect.origin.y, rect.size.width, rect.size.height, person.weight, person.height, person.age, person.gender, person.genderConfidence, person.race, person.raceConfidence)
                            }
                            
                            save_strings.append(str)
                        }
                    }
                }
                
            } else {
                boundingBoxViews[i].hide()
            }
        }
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
        case .began: fallthrough
        case .changed:
            update(scale: newScaleFactor)
            self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
            self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
        case .ended:
            lastZoomFactor = minMaxZoom(newScaleFactor)
            update(scale: lastZoomFactor)
            self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
        default: break
        }
    }  // Pinch to Zoom Start


    // ------------------------------------------------------------------------------------------
}  // ViewController class End

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        predict(sampleBuffer: sampleBuffer)
    }
}

// Programmatically save image
extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("error occurred : \(error.localizedDescription)")
        }
        if let dataImage = photo.fileDataRepresentation() {
            let dataProvider = CGDataProvider(data: dataImage as CFData)
            let cgImageRef: CGImage! = CGImage(jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
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
               let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent) {
               image = UIImage(cgImage: cgImage)
                
            }
            let imageView = UIImageView(image: image)
            imageView.contentMode = .scaleAspectFill
            imageView.frame = videoPreview.frame
            let imageLayer = imageView.layer
            var sublayers = videoPreview.layer.sublayers ?? []
            let insertIndex = max(sublayers.count - 1, 0)
            videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)
            
            let bounds = UIScreen.main.bounds
            UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
            self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
            let img = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            imageLayer.removeFromSuperlayer()
            let activityViewController = UIActivityViewController(activityItems: [img!], applicationActivities: nil)
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



