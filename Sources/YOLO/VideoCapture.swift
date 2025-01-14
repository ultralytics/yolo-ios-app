import AVFoundation
import CoreVideo
import UIKit
import Vision

@MainActor
protocol VideoCaptureDelegate: AnyObject {
    func onPredict(result: YOLOResult)
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice {
    // print("USE TELEPHOTO: ")
    // print(UserDefaults.standard.bool(forKey: "use_telephoto"))

    if UserDefaults.standard.bool(forKey: "use_telephoto"), let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: position) {
        return device
    } else if let device = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: position) {
        return device
    } else if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
        return device
    } else {
        fatalError("Missing expected back camera device.")
    }
}

class VideoCapture: NSObject,@unchecked Sendable {
    var predictor:Predictor!
    var previewLayer: AVCaptureVideoPreviewLayer?
    weak var delegate: VideoCaptureDelegate?
    var captureDevice: AVCaptureDevice?
    let captureSession = AVCaptureSession()
    var videoInput: AVCaptureDeviceInput? = nil
    let videoOutput = AVCaptureVideoDataOutput()
    var photoOutput = AVCapturePhotoOutput()
    let cameraQueue = DispatchQueue(label: "camera-queue")
    var lastCapturedPhoto: UIImage? = nil
    private var currentBuffer: CVPixelBuffer?

    func setUp(sessionPreset: AVCaptureSession.Preset = .hd1280x720,
                      position: AVCaptureDevice.Position,
                      completion: @escaping (Bool) -> Void) {
        cameraQueue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset, position: position)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }

    func setUpCamera(sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset

        captureDevice = bestCaptureDevice(position: position)
        videoInput = try! AVCaptureDeviceInput(device: captureDevice!)

        if captureSession.canAddInput(videoInput!) {
            captureSession.addInput(videoInput!)
        }

        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
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
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
            photoOutput.isHighResolutionCaptureEnabled = true
            photoOutput.isLivePhotoCaptureEnabled = photoOutput.isLivePhotoCaptureSupported
        }

        // We want the buffers to be in portrait orientation otherwise they are
        // rotated by 90 degrees. Need to set this _after_ addOutput()!
        // let curDeviceOrientation = UIDevice.current.orientation
        let connection = videoOutput.connection(with: AVMediaType.video)
        connection?.videoOrientation = .portrait
        if position == .front{
            connection?.isVideoMirrored = true
        }

        // Configure captureDevice
        do {
            try captureDevice!.lockForConfiguration()
        } catch {
            print("device configuration not working")
        }
        // captureDevice.setFocusModeLocked(lensPosition: 1.0, completionHandler: { (time) -> Void in })
        if captureDevice!.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus), captureDevice!.isFocusPointOfInterestSupported {
            captureDevice!.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
            captureDevice!.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
        }
        captureDevice!.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
        captureDevice!.unlockForConfiguration()

        captureSession.commitConfiguration()
        return true
    }

    func start() {
        if !captureSession.isRunning {
            DispatchQueue.global().async {
                self.captureSession.startRunning()
            }
        }
    }

    func stop() {
        if captureSession.isRunning {
            DispatchQueue.global().async {
                self.captureSession.stopRunning()
            }
        }
    }
    
    func setZoomRatio(ratio: CGFloat){
        do {
            try captureDevice!.lockForConfiguration()
            defer {
                captureDevice!.unlockForConfiguration()
            }
            captureDevice!.videoZoomFactor = ratio
        } catch { }
    }
    
    private func predictOnFrame(sampleBuffer: CMSampleBuffer) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            
            /// - Tag: MappingOrientation
            // The frame is always oriented based on the camera sensor,
            // so in most cases Vision needs to rotate it for the model to work as expected.
            var imageOrientation: CGImagePropertyOrientation = .up
//            switch UIDevice.current.orientation {
//            case .portrait:
//                imageOrientation = .up
//            case .portraitUpsideDown:
//                imageOrientation = .down
//            case .landscapeLeft:
//                imageOrientation = .up
//            case .landscapeRight:
//                imageOrientation = .up
//            case .unknown:
//                imageOrientation = .up
//                
//            default:
//                imageOrientation = .up
//            }
            
            predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self, onFpsRate: self)
            currentBuffer = nil
        }
    }
    
    func updateVideoOrientation(orientation:AVCaptureVideoOrientation) {
      guard let connection = videoOutput.connection(with: .video) else { return }

        connection.videoOrientation = orientation
      let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
      if currentInput?.device.position == .front {
        connection.isVideoMirrored = true
      } else {
        connection.isVideoMirrored = false
      }

      self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        predictOnFrame(sampleBuffer: sampleBuffer)
    }
}

extension VideoCapture: AVCapturePhotoCaptureDelegate {
    @available(iOS 11.0, *)
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation(),
              let image =  UIImage(data: data) else {
                return
        }

        self.lastCapturedPhoto = image
    }
}

extension VideoCapture: ResultsListener, InferenceTimeListener, FpsRateListener {
    
    func on(result: YOLOResult) {
        DispatchQueue.main.async {
            self.delegate?.onPredict(result: result)
        }
    }
    
    func on(inferenceTime: Double) {
        
    }
    
    func on(fpsRate: Double) {
        
    }
    
}


