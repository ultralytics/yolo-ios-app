//
//  Pose.swift
//  YOLO
//
//  Created by 間嶋大輔 on 2024/11/27.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

import Foundation
import UIKit
import Vision

extension ViewController {
    
    func postProcessPose(request: VNRequest) {
        if let results = request.results as? [VNCoreMLFeatureValueObservation] {
            DispatchQueue.main.async { [self] in
                
                if let prediction = results.first?.featureValue.multiArrayValue {
                    
                    let preds = getBoundingBoxesAndKeypoints(
                        prediction: prediction, confidenceThreshold: self.confidenceThreshold,
                        iouThreshold: self.iouThreshold)
                    var boxes = [(CGRect, Float)]()
                    var kpts = [[Float]]()
                    var predictions = [DetectionResult]()
                    for pred in preds {
                        let box = pred.0
                        let rect = CGRect(
                          x: box.minX / 640, y: box.minY / 640, width: box.width / 640, height: box.height / 640)

                        let detectionResult = DetectionResult(rect: rect,label: "person", confidence: pred.1)
                        predictions.append(detectionResult)
                        
                        boxes.append((pred.0, pred.1))
                        kpts.append(pred.2)
                    }
                    self.showBoundingBoxes(predictions: predictions)
                    self.segmentPoseOverlay.sublayers?.forEach { $0.removeFromSuperlayer() }
                    
                    self.drawKeypoints(
                        keypointsList: kpts, boundingBoxes: boxes, on: segmentPoseOverlay,
                        imageViewSize: segmentPoseOverlay.bounds.size, originalImageSize: segmentPoseOverlay.bounds.size)
                    
                } else {
                    self.showBoundingBoxes(predictions: [])
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
    
    func getBoundingBoxesAndKeypoints(prediction: MLMultiArray, confidenceThreshold: Float, iouThreshold: Float)
    -> [(CGRect, Float, [Float])]
    {
        let numAnchors = prediction.shape[2].intValue
        let featureCount = prediction.shape[1].intValue - 5
        var boxes = [CGRect]()
        var scores = [Float]()
        var features = [[Float]]()
        let featurePointer = UnsafeMutablePointer<Float>(OpaquePointer(prediction.dataPointer))
        let lock = DispatchQueue(label: "com.example.lock")
        
        DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
            let confIndex = 4 * numAnchors + j
            let confidence = featurePointer[confIndex]
            
            if confidence > confidenceThreshold {
                let x = featurePointer[j]
                let y = featurePointer[numAnchors + j]
                let width = featurePointer[2 * numAnchors + j]
                let height = featurePointer[3 * numAnchors + j]
                
                let boxWidth = CGFloat(width)
                let boxHeight = CGFloat(height)
                let boxX = CGFloat(x - width / 2)
                let boxY = CGFloat(y - height / 2)
                
                let boundingBox = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)
                
                var boxFeatures = [Float](repeating: 0, count: featureCount)
                for k in 0..<featureCount {
                    let key = (5 + k) * numAnchors + j
                    boxFeatures[k] = featurePointer[key]
                }
                
                lock.sync {
                    boxes.append(boundingBox)
                    scores.append(confidence)
                    features.append(boxFeatures)
                }
            }
        }
        
        let selectedIndices = nonMaxSuppression(boxes: boxes, scores: scores, threshold: iouThreshold)
        
        let filteredBoxes = selectedIndices.map { boxes[$0] }
        let filteredScores = selectedIndices.map { scores[$0] }
        let filteredFeatures = selectedIndices.map { features[$0] }
        
        return zip(zip(filteredBoxes, filteredScores), filteredFeatures).map { ($0.0, $0.1, $1) }
    }
    
    func drawKeypoints(
        keypointsList: [[Float]],
        boundingBoxes: [(CGRect, Float)],
        on layer: CALayer,
        imageViewSize: CGSize,
        originalImageSize: CGSize,
        radius: CGFloat = 5,
        confThreshold: Float = 0.25,
        drawSkeleton: Bool = true
    ) {
        for (i, keypoints) in keypointsList.enumerated() {
            
            drawSinglePersonKeypoints(
                keypoints: keypoints, boundingBox: boundingBoxes[i],
                on: layer,
                imageViewSize: imageViewSize,
                originalImageSize: originalImageSize,
                radius: radius,
                confThreshold: confThreshold,
                drawSkeleton: drawSkeleton
            )
        }
    }
    
    func drawSinglePersonKeypoints(
        keypoints: [Float],
        boundingBox: (CGRect, Float),
        on layer: CALayer,
        imageViewSize: CGSize,
        originalImageSize: CGSize,
        radius: CGFloat,
        confThreshold: Float,
        drawSkeleton: Bool
    ) {
        guard keypoints.count == 51 else {
            print("Keypoints array must have 51 elements.")
            return
        }
        
        let scaleXToOriginal = Float(originalImageSize.width / 640)
        let scaleYToOriginal = Float(originalImageSize.height / 640)
        
        let scaleXToView = Float(imageViewSize.width / originalImageSize.width)
        let scaleYToView = Float(imageViewSize.height / originalImageSize.height)
        
        var points: [(CGPoint, Float)] = Array(repeating: (CGPoint.zero, 0), count: 17)
        
        for i in 0..<17 {
            let x = keypoints[i * 3] * scaleXToOriginal * scaleXToView
            let y = keypoints[i * 3 + 1] * scaleYToOriginal * scaleYToView
            let conf = keypoints[i * 3 + 2]
            
            let point = CGPoint(x: CGFloat(x), y: CGFloat(y))
            let box = boundingBox.0
            
            if conf >= confThreshold
                && box.contains(CGPoint(x: CGFloat(keypoints[i * 3]), y: CGFloat(keypoints[i * 3 + 1])))
            {
                points[i] = (point, conf)
                
                drawCircle(on: layer, at: point, radius: radius, color: kptColorIndices[i])
            }
        }
        
        if drawSkeleton {
            for (index, bone) in skeleton.enumerated() {
                let (startIdx, endIdx) = (bone[0] - 1, bone[1] - 1)
                
                guard startIdx < points.count, endIdx < points.count else {
                    print("Invalid skeleton indices: \(startIdx), \(endIdx)")
                    continue
                }
                
                let startPoint = points[startIdx].0
                let endPoint = points[endIdx].0
                let startConf = points[startIdx].1
                let endConf = points[endIdx].1
                
                if startConf >= confThreshold && endConf >= confThreshold {
                    drawLine(on: layer, from: startPoint, to: endPoint, color: limbColorIndices[index])
                }
            }
        }
    }
    
    func drawCircle(on layer: CALayer, at point: CGPoint, radius: CGFloat, color index: Int) {
        let circleLayer = CAShapeLayer()
        circleLayer.path =
        UIBezierPath(
            arcCenter: point,
            radius: radius,
            startAngle: 0,
            endAngle: .pi * 2,
            clockwise: true
        ).cgPath
        
        let color = posePalette[index].map { $0 / 255.0 }
        circleLayer.fillColor =
        UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor
        
        layer.addSublayer(circleLayer)
    }
    
    func drawLine(on layer: CALayer, from start: CGPoint, to end: CGPoint, color index: Int) {
        let lineLayer = CAShapeLayer()
        let path = UIBezierPath()
        path.move(to: start)
        path.addLine(to: end)
        
        lineLayer.path = path.cgPath
        lineLayer.lineWidth = 2
        
        let color = posePalette[index].map { $0 / 255.0 }
        lineLayer.strokeColor =
        UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor
        
        layer.addSublayer(lineLayer)
    }
}

let posePalette: [[CGFloat]] = [
    [255, 128, 0],
    [255, 153, 51],
    [255, 178, 102],
    [230, 230, 0],
    [255, 153, 255],
    [153, 204, 255],
    [255, 102, 255],
    [255, 51, 255],
    [102, 178, 255],
    [51, 153, 255],
    [255, 153, 153],
    [255, 102, 102],
    [255, 51, 51],
    [153, 255, 153],
    [102, 255, 102],
    [51, 255, 51],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 255],
]

let limbColorIndices = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
let kptColorIndices = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

let skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]
