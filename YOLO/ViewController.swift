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

class ViewController: UIViewController {
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var playButtonOutlet: UIBarButtonItem!
    @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelZoom: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    
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
    let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")   // developer mode selected in settings
    let save_detections = false  // write every detection to detections.txt
    let save_frames = false  // write every frame to frames.txt
    // Global ORTSession initialized in the viewDidLoad
    var ortSession: ORTSession?
    var poseUtil: OnnxPoseUtils?
    
    func getOnnxModelPath() -> String{
        guard let modelPath = Bundle.main.path(forResource: "yolov8n-pose-pre", ofType: "onnx") else { fatalError("Error in finding model") }
        return modelPath
    }
    
    func setModelOnnx() {
        do {
            guard let modelPath = Bundle.main.path(forResource: "yolov8n-pose-pre", ofType: "onnx") else { fatalError("Error in finding model") }
            let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.info)
            let ortSessionOptions = try ORTSessionOptions()
            try ortSessionOptions.registerCustomOps(functionPointer: RegisterCustomOps) // Register the bridging-header in Build settings
            ortSession = try ORTSession(
                env: ortEnv, modelPath: modelPath, sessionOptions: ortSessionOptions)
        } catch {
            print(error)
            fatalError("Error in instantiating the ONNX model")
        }
        t2 = 0.0 // inference dt smoothed
        t3 = CACurrentMediaTime()  // FPS start
        t4 = 0.0  // FPS dt smoothed
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        //load the ONNX model
        setLabels()
        setUpBoundingBoxViews()
        startVideo()
        //setModel()
        poseUtil = OnnxPoseUtils()
        setModelOnnx()
    }
    
    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
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
        self.labelName.text = "YOLOv8n Pose"
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
        
        // let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)!
        guard let videoInput1 = try? AVCaptureDeviceInput(device: device) else {
            return
        }
        
        self.videoCapture.captureSession.addInput(videoInput1)
        self.videoCapture.captureSession.commitConfiguration()
    }
    
    // share image
    @IBAction func shareButton(_ sender: Any) {
        selection.selectionChanged()
        let bounds = UIScreen.main.bounds
        //let bounds = self.View0.bounds
        
        UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
        self.View0.drawHierarchy(in: bounds, afterScreenUpdates: false)
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        let activityViewController = UIActivityViewController(activityItems: [img!], applicationActivities: nil)
        activityViewController.popoverPresentationController?.sourceView = self.View0
        self.present(activityViewController, animated: true, completion: nil)
        // playButton("")
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
    
    let maxBoundingBoxViews = 1
    var boundingBoxViews = [BoundingBoxView]()
    var colors: [String: UIColor] = [:]
    
    func setUpBoundingBoxViews() {
        // Ensure all bounding box views are initialized up to the maximum allowed.
        while boundingBoxViews.count < maxBoundingBoxViews {
            boundingBoxViews.append(BoundingBoxView())
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
            let onnxHandler = VNOnnxHandler(cvImageBufffer: pixelBuffer, session: ortSession!)
            DispatchQueue.main.async {
                if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
                    self.t0 = CACurrentMediaTime()  // inference start
                    do {
                        self.videoPreview.layer.sublayers = nil // Remove all previous layers to avoid an OOM problem
                        //let outputTensor = try onnxHandler.perform()
                        
                        let outputImage = try onnxHandler.performImage(poseUtil: self.poseUtil!)
                        if self.t1 < 10.0 {  // valid dt
                            self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
                        }
                        self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
                        self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
                        self.t3 = CACurrentMediaTime()
                        let l = CALayer()
                        l.contents = outputImage.cgImage
                        l.contentsGravity = .resizeAspect
                        l.isHidden = true
                        l.frame = self.videoPreview.bounds
                        self.videoPreview.layer.addSublayer(l)
                        l.isHidden = false
                        //self.processOnnxObservations(for: outputTensor, inputImage: UIImage(cgImage: CGImage.create(from: pixelBuffer)!))
                    } catch {
                        print("Error in model execution \(error)")
                    }
                    self.t1 = CACurrentMediaTime() - self.t0  // inference dt
                }
                self.currentBuffer = nil
            }
        }
    }
    
    /// Convert the outputTensor values into a layer for super imposing on the videoPreview layer
    ///     - Params
    ///        - request: The output tensor after procesing the image
    ///        - inputImage: The original image that was proccessed by the onnx model
    func processOnnxObservations(for request: ORTValue, inputImage: UIImage) {
        DispatchQueue.main.async {
            self.showOnnx(opTensor: request, inputImage: inputImage)
            // Measure FPS
            if self.t1 < 10.0 {  // valid dt
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
            self.t3 = CACurrentMediaTime()
        }
    }
    
    func showOnnx(opTensor: ORTValue, inputImage: UIImage) {
        let targetWidth = videoPreview.bounds.width  // 375 pix
        let targetHeight = videoPreview.bounds.height  // 812 pix
        var str = ""
        
        // ratio = videoPreview AR divided by sessionPreset AR
        var ratio: CGFloat = 1.0
        if videoCapture.captureSession.sessionPreset == .photo {
            ratio = (targetHeight / targetWidth) / (4.0 / 3.0)  // .photo
        } else {
            ratio = (targetHeight / targetWidth) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
        }
        
        let widthRatio = targetWidth / inputImage.size.width
        let heightRatio = targetHeight / inputImage.size.height
        let scaleFactor = min(widthRatio, heightRatio)
        
        // date for developer mode
        let date = Date()
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: date)
        let minutes = calendar.component(.minute, from: date)
        let seconds = calendar.component(.second, from: date)
        let nanoseconds = calendar.component(.nanosecond, from: date)
        let sec_day = Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day
        
        
        // pose datapoints
        var keypoints:[Float32] = Array()
        do {
            let output = try opTensor.tensorData()
            var arr2 = Array<Float32>(repeating: 0, count: output.count/MemoryLayout<Float32>.stride)   // Do not change the datatype Float32
            _ = arr2.withUnsafeMutableBytes { output.copyBytes(to: $0) }
            // 57 is hardcoded based on the keypoints returned from the model. Refer to the Netron visualisation for the output shape
            if (!arr2.isEmpty) {
                for i in stride(from: arr2.count-57, to: arr2.count, by: 1) {
                    keypoints.append(arr2[i])
                }
            }
        } catch {
            print(error)
            fatalError("Output tensor processing failed")
        }
        if (keypoints.count > 0) {
            let box = keypoints[0..<4] // The first 4 points are the bounding box co-ords.
            // Refer yolov8_pose_e2e.py run_inference method under the https://onnxruntime.ai/docs/tutorials/mobile/pose-detection.html
            
            let half_w = Double(box[2] / 2 )
            let half_h = Double(box[3] / 2 )
            let x = (Double(box[0]) - Double(half_w)) * widthRatio
            let y = (Double(box[1]) - Double(half_h)) * heightRatio
            //let rect = CGRect(x: -x, y: y, width: Double(half_w * 2), height: Double(half_h * 2))
            
            var keypointsWithoutBoxes = Array(keypoints[6..<keypoints.count]) // Based on 17 key
            //keypointsWithoutBoxes = keypointsWithoutBoxes.map { Float($0) * Float(scaleFactor) }
            
            for i in 0..<boundingBoxViews.count {
                
                //var rect = prediction.boundingBox  // normalized xywh, origin lower left
                let rect = CGRect(x: x, y: y, width: Double(half_w), height: Double(half_h))
                
                // This part is commented because I am unable to figure out the scaling part
                /*
                 switch UIDevice.current.orientation {
                 case .portraitUpsideDown:
                 rect = CGRect(x: 1.0 - rect.origin.x - rect.width,
                 y: 1.0 - rect.origin.y - rect.height,
                 width: rect.width,
                 height: rect.height)
                 case .landscapeLeft:
                 rect = CGRect(x: rect.origin.y,
                 y: 1.0 - rect.origin.x - rect.width,
                 width: rect.height,
                 height: rect.width)
                 case .landscapeRight:
                 rect = CGRect(x: 1.0 - rect.origin.y - rect.height,
                 y: rect.origin.x,
                 width: rect.height,
                 height: rect.width)
                 case .unknown:
                 print("The device orientation is unknown, the predictions may be affected")
                 fallthrough
                 default: break
                 }
                 
                 if ratio >= 1 { // iPhone ratio = 1.218
                 let offset = (1 - ratio) * (0.5 - rect.minX)
                 let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                 rect = rect.applying(transform)
                 rect.size.width *= ratio
                 } else { // iPad ratio = 0.75
                 let offset = (ratio - 1) * (0.5 - rect.maxY)
                 let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                 rect = rect.applying(transform)
                 rect.size.height /= ratio
                 }
                 */
                
                NSLog("Rect origin \(rect.origin.debugDescription)")
                NSLog("Rect size \(rect.size.debugDescription)")
                NSLog("Input image : \(inputImage.size.debugDescription)")
                NSLog("Video frame \(videoPreview.frame)")
                
                // Scale normalized to pixels [375, 812] [width, height]
                //rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))
                
                // The labels array is a list of VNClassificationObservation objects,
                // with the highest scoring class first in the list.
                let bestClass = "class"
                let confidence = 0.1
                // print(confidence, rect)  // debug (confidence, xywh) with xywh origin top left (pixels)
                
                // Show the bounding box.
                
                boundingBoxViews[i].showOnnx(frame: rect,
                                             keypoints: keypointsWithoutBoxes, widthRatio: Float(widthRatio), heightRatio: Float(heightRatio)) // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
                
                if developerMode {
                    if save_detections {
                        str += String(format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                                      sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                                      rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
                    }
                }
            }
            
        }
        // Write
        if developerMode {
            if save_detections {
                saveText(text: str, file: "detections.txt")  // Write stats for each detection
            }
            if save_frames {
                str = String(format: "%.3f %.3f %.3f %.3f %.1f %.1f %.1f\n",
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
    }  // Pinch to Zoom Start ------------------------------------------------------------------------------------------
}  // ViewController class End

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        if let buffer = sampleBuffer.imageBuffer {
            predict(sampleBuffer: sampleBuffer)
        }
        
    }
}

// Programmatically save image
extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("error occurred : \(error.localizedDescription)")
        }
        if let dataImage = photo.fileDataRepresentation() {
            print(UIImage(data: dataImage)?.size as Any)
            let dataProvider = CGDataProvider(data: dataImage as CFData)
            let cgImageRef: CGImage! = CGImage(jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
            let image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: UIImage.Orientation.right)
            
            // Save to camera roll
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil);
        } else {
            print("AVCapturePhotoCaptureDelegate Error")
        }
    }
}



