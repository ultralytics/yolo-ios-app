//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLO11 models on iOS devices.
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

var mlModel = try! yolo11m(configuration: .init()).model

class ViewController: UIViewController {
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var modelSegmentedControl: UISegmentedControl!
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
    @IBOutlet weak var taskSegmentControl: UISegmentedControl!
    
    // views for tasks
    var classifyOverlay: UILabel!
    var segmentPoseOverlay: CALayer = CALayer()
    
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
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    var frameSizeCaptured = false
    
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
        case classify
        case segment
        case pose
        case obb
    }
    
    var task: Task = .detect
    var confidenceThreshold: Float = 0.25
    var iouThreshold: Float = 0.4

    var classifyLabels = [String]()
    var colorsForMasks: [(red: UInt8, green: UInt8, blue: UInt8)] = []
    var classes: [String] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setUpOrientationChangeNotification()
        startVideo()
        setupUI()
        // setModel()
    }
    
    func setupUI() {
        slider.value = 30
        setLabels()
        setUpBoundingBoxViews()
        setupColors()
        setupClassifyOverlay()
        setupSegmentPoseOverlay()
    }
    
    func switchUIForTask() {
        switch task {
        case .detect:
            classifyOverlay.isHidden = true
            segmentPoseOverlay.isHidden = true
            updateSlider(show: true)
        case .classify:
            classifyOverlay.isHidden = false
            hideBoundingBoxes()
            segmentPoseOverlay.isHidden = true
            updateSlider(show: false)
        case .segment:
            segmentPoseOverlay.isHidden = false
            hideBoundingBoxes()
            classifyOverlay.isHidden = true
            updateSlider(show: true)
        case .pose:
            segmentPoseOverlay.isHidden = false
            hideBoundingBoxes()
            classifyOverlay.isHidden = true
            updateSlider(show: true)
        default:
            break
        }
    }
    
    func updateSlider(show:Bool) {
        if show {
            labelSlider.isHidden = false
            labelSliderConf.isHidden = false
            labelSliderIoU.isHidden = false
            labelSliderConfLandScape.isHidden = false
            labelSliderIoULandScape.isHidden = false
            slider.isHidden = false
            sliderConf.isHidden = false
            sliderIoU.isHidden = false
            sliderConfLandScape.isHidden = false
            sliderIoULandScape.isHidden = false

        } else {
            labelSlider.isHidden = true
            labelSliderConf.isHidden = true
            labelSliderIoU.isHidden = true
            labelSliderConfLandScape.isHidden = true
            labelSliderIoULandScape.isHidden = true
            sliderConfLandScape.isHidden = true
            sliderIoULandScape.isHidden = true

        }
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
                self.setupSegmentPoseOverlay()
                self.updateClassifyOverlay()

            }
        )
    }
    
    private func setUpOrientationChangeNotification() {
        NotificationCenter.default.addObserver(
            self, selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification, object: nil)
    }
    
    @objc func orientationDidChange() {
        videoCapture.updateVideoOrientation()
        //      frameSizeCaptured = false
    }
    
    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }
    
    @IBAction func taskSegmentChanged(_ sender: UISegmentedControl) {
        switch sender.selectedSegmentIndex {
        case 0:
            task = .detect
            modelSegmentedControl.setEnabled(true, forSegmentAt: 1)
            modelSegmentedControl.setEnabled(true, forSegmentAt: 2)
            modelSegmentedControl.setEnabled(true, forSegmentAt: 3)
            modelSegmentedControl.setEnabled(true, forSegmentAt: 4)
        case 1:
            task = .classify
            modelSegmentedControl.selectedSegmentIndex = 0
            updateModelSegmentControl(enableModelIndex: [0,1,2,3,4], unableModelIndex: [])
            showClassifyUI()
        case 2:
            task = .segment
            modelSegmentedControl.selectedSegmentIndex = 0
            updateModelSegmentControl(enableModelIndex: [0], unableModelIndex: [1,2,3,4])
        case 3:
            task = .pose
            modelSegmentedControl.selectedSegmentIndex = 0
            updateModelSegmentControl(enableModelIndex: [0], unableModelIndex: [1,2,3,4])
        default:
            updateModelSegmentControl(enableModelIndex: [], unableModelIndex: [0,1,2,3,4])
        }
        switchUIForTask()
        setModel()
        if task == .classify {
            setupClassifyLabels()
        } else {
            setupColors()
        }
    }
    
    func updateModelSegmentControl(enableModelIndex:[Int], unableModelIndex:[Int]) {
        for index in enableModelIndex {
            modelSegmentedControl.setEnabled(true, forSegmentAt: index)
        }
        
        for index in unableModelIndex {
            modelSegmentedControl.setEnabled(false, forSegmentAt: index)
        }
    }
    
    @IBAction func indexChanged(_ sender: Any) {
        selection.selectionChanged()
        activityIndicator.startAnimating()
        setModel()
        setUpBoundingBoxViews()
        setupColors()
        activityIndicator.stopAnimating()
    }
    
    /// Update thresholds from slider values
    @IBAction func sliderChanged(_ sender: Any) {
        let conf = Double(round(100 * sliderConf.value)) / 100
        let iou = Double(round(100 * sliderIoU.value)) / 100
        self.labelSliderConf.text = String(conf) + " Confidence Threshold"
        self.labelSliderIoU.text = String(iou) + " IoU Threshold"
        detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
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
        self.labelName.text = "YOLO11m"
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
    var colorsForMask: [(red: UInt8, green: UInt8, blue: UInt8)] = []
    let ultralyticsColorsolors: [UIColor] = [
        UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),  // #042AFF
        UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),  // #0BDBEB
        UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),  // #F3F3F3
        UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),  // #00DFB7
        UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),  // #111F68
        UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),  // #FF6FDD
        UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),  // #FF444F
        UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),  // #CCED00
        UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),  // #00F344
        UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),  // #BD00FF
        UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),  // #00B4FF
        UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),  // #DD00BA
        UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),  // #00FFFF
        UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),  // #26C000
        UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),  // #01FFB3
        UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),  // #7D24FF
        UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),  // #7B0068
        UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),  // #FF1B6C
        UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),  // #FC6D2F
        UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),  // #A2FF0B
    ]
    
    func setUpBoundingBoxViews() {
        // Ensure all bounding box views are initialized up to the maximum allowed.
        while boundingBoxViews.count < maxBoundingBoxViews {
            boundingBoxViews.append(BoundingBoxView())
        }
        
        // Retrieve class labels directly from the CoreML model's class labels, if available.
    }
    
    func setupColors() {
        guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
            print("Class labels are missing from the model description")
            return
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
        
        count = 0
        for (key, color) in colors {
            let color = ultralyticsColorsolors[count]
            count += 1
            if count > 19 {
                count = 0
            }
            guard let colorForMask = color.toRGBComponents() else { fatalError() }
            colorsForMask.append(colorForMask)
        }
    }
    
    func setupClassifyLabels() {
        guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
            print("Class labels are missing from the model description")
            return
        }
        classifyLabels = classLabels
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
                self.videoPreview.layer.addSublayer(self.segmentPoseOverlay)

                // Add the bounding box layers to the UI, on top of the video preview.
                for box in self.boundingBoxViews {
                    box.addToLayer(self.videoPreview.layer)
                }
                
                // Once everything is set up, we can start capturing live video.
                self.videoCapture.start()
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

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        predict(sampleBuffer: sampleBuffer)
    }
}

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
