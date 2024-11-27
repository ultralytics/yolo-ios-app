//
//  ModelSelect.swift
//  YOLO
//
//  Created by 間嶋大輔 on 2024/11/26.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

import Foundation
import Vision
import UIKit

extension ViewController {
    func predict(sampleBuffer: CMSampleBuffer) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
            }
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
        DispatchQueue.main.async {
            switch self.task {
            case .detect:
                self.postProcessDetect(request: request)
            case .classify:
                self.postProcessClassify(request: request)
            case .segment:
                self.postProcessSegment(request: request)
            case .pose:
                self.postProcessPose(request: request)
            default:
                break
            }
            // Measure FPS
            if self.t1 < 10.0 {  // valid dt
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
            self.t3 = CACurrentMediaTime()
        }
    }
    
    func setModel() {
        
        /// Switch model
        switch task {
        case .detect:
            switch modelSegmentedControl.selectedSegmentIndex {
            case 0:
                self.labelName.text = "YOLOv11n"
                mlModel = try! yolo11n(configuration: .init()).model
            case 1:
                self.labelName.text = "YOLOv11s"
                mlModel = try! yolo11s(configuration: .init()).model
            case 2:
                self.labelName.text = "YOLOv11m"
                mlModel = try! yolo11m(configuration: .init()).model
            case 3:
                self.labelName.text = "YOLOv11l"
                mlModel = try! yolo11l(configuration: .init()).model
            case 4:
                self.labelName.text = "YOLOv11x"
                mlModel = try! yolo11x(configuration: .init()).model
            default:
                break
            }
        case .classify:
            switch modelSegmentedControl.selectedSegmentIndex {
            case 0:
                self.labelName.text = "YOLO11n"
                mlModel = try! yolo11n_cls(configuration: .init()).model
            default: break
            }
        case .segment:
            switch modelSegmentedControl.selectedSegmentIndex {
            case 0:
                self.labelName.text = "YOLO11n"
                mlModel = try! yolo11n_seg(configuration: .init()).model
            default: break
            }
        case .pose:
            switch modelSegmentedControl.selectedSegmentIndex {
            case 0:
                self.labelName.text = "YOLO11n"
                mlModel = try! yolo11n_pose(configuration: .init()).model
            default: break
            }
        default:
            break
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
}
