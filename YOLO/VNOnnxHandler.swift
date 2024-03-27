//
//  VNOnnxHandler.swift
//  YOLO
//
//  Created by Pradeep Banavara on 25/03/24.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

import Foundation
import Vision
import CoreGraphics
import VideoToolbox
import UIKit

class VNOnnxHandler {
    var sampleBuffer: CVImageBuffer?
    var ortSession: ORTSession
    init(cvImageBufffer: CVImageBuffer, session: ORTSession) {
        sampleBuffer = cvImageBufffer
        ortSession = session
        
    }
    
    private func convertImageBufferToData(sampleBuffer: CVPixelBuffer) -> NSData {
        let imageBuffer = sampleBuffer
        CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
        let height = CVPixelBufferGetHeight(imageBuffer)
        let src_buff = CVPixelBufferGetBaseAddress(imageBuffer)
        let data = NSData(bytes: src_buff, length: bytesPerRow * height)
        CVPixelBufferUnlockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))
        return data
    }
    
    /// This is a method for extracting the scaled output from the input image. The result is an output tensor which can be used to draw bounding boxes.
    func perform() throws -> ORTValue {
        var result: ORTValue?
        let uiImage = UIImage(cgImage: CGImage.create(from: sampleBuffer!)!)
        let inputData = uiImage.pngData()
        let inputDataLength = inputData?.count
        let inputShape = [NSNumber(integerLiteral: inputDataLength!)]
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData!), elementType:ORTTensorElementDataType.uInt8, shape:inputShape)
        let inputNames = try ortSession.inputNames()    // The input names should match the model input names. Visualize the model in Netron
        let outputNames = try ortSession.outputNames()  // Check the model outnames in Netron
        let outputs = try ortSession.run(
            withInputs: [inputNames[0]: inputTensor], outputNames: Set(outputNames), runOptions: nil)
        guard let outputTensor = outputs[outputNames[0]] else {
            fatalError("Failed to get model keypoint output from inference.")
        }
        result = outputTensor
        return result!
    }
    
    /// This is a handler method to use image layer instead of super imposing the boundingbox on the videoPreviewLayer
    /// That function is done by the perform method listed above
    func performImage(poseUtil: OnnxPoseUtils) throws -> UIImage{
        var result: UIImage?
        let uiImage = UIImage(cgImage: CGImage.create(from: sampleBuffer!)!)
        result = poseUtil.plotPose(image: uiImage)
        return result!
    }
}

extension CGImage {
    static func create(from cvPixelBuffer: CVPixelBuffer?) -> CGImage? {
        guard let pixelBuffer = cvPixelBuffer else {
            return nil
        }
        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(
            pixelBuffer,
            options: nil,
            imageOut: &image)
        return image
    }
}

