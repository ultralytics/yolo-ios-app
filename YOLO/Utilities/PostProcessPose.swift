import Foundation
import UIKit
import CoreML

@available(iOS 15.0, *)
extension ViewController {
    func PostProcessPose(prediction: MLMultiArray, confidenceThreshold: Float, iouThreshold: Float) -> [(CGRect, Float, [Float])] {
        let numAnchors = prediction.shape[2].intValue
        let featureCount = prediction.shape[1].intValue - 5 // 56個のうち、ボックス(4)と信頼度(1)を除いた51個の特徴量
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
    
    func drawKeypoints(keypoints: [Float], originalSize: CGSize, confidenceThreshold: Float = 0.25) {
        let path = UIBezierPath()
        let keypointTuples = stride(from: 0, to: keypoints.count, by: 3).map {
            (keypoints[$0], keypoints[$0 + 1], keypoints[$0 + 2])
        }
        
        guard keypointTuples.count == 17 else { return }
        
        // リサイズスケールファクター
        let scaleX = view.bounds.width / originalSize.width
        let scaleY = view.bounds.height / originalSize.height
        
        // キーポイントカラーの設定
        let kptColor: [UIColor] = [
            UIColor(red: 255/255, green: 0/255, blue: 0/255, alpha: 1),
            UIColor(red: 255/255, green: 85/255, blue: 0/255, alpha: 1),
            UIColor(red: 255/255, green: 170/255, blue: 0/255, alpha: 1),
            UIColor(red: 255/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 170/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 85/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 85/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 170/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 255/255, alpha: 1),
            UIColor(red: 0/255, green: 170/255, blue: 255/255, alpha: 1),
            UIColor(red: 0/255, green: 85/255, blue: 255/255, alpha: 1),
            UIColor(red: 0/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 85/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 170/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 255/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 255/255, green: 0/255, blue: 170/255, alpha: 1)
        ]
        
        // キーポイントの描画
        for (index, (x, y, conf)) in keypointTuples.enumerated() {
            if conf < confidenceThreshold { continue }
            
            let xPos = CGFloat(x) * scaleX
            let yPos = CGFloat(y) * scaleY
            
            // キーポイントの円を描画
            let circlePath = UIBezierPath(arcCenter: CGPoint(x: xPos, y: yPos), radius: 5, startAngle: 0, endAngle: CGFloat.pi * 2, clockwise: true)
            let circleLayer = CAShapeLayer()
            circleLayer.path = circlePath.cgPath
            circleLayer.fillColor = kptColor[index % kptColor.count].cgColor
            overlayLayer.addSublayer(circleLayer)
        }
        
        // スケルトンラインの描画
        let skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16)
        ]
        
        let limbColor: [UIColor] = [
            UIColor(red: 255/255, green: 0/255, blue: 0/255, alpha: 1),
            UIColor(red: 255/255, green: 85/255, blue: 0/255, alpha: 1),
            UIColor(red: 255/255, green: 170/255, blue: 0/255, alpha: 1),
            UIColor(red: 255/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 170/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 85/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 0/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 85/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 170/255, alpha: 1),
            UIColor(red: 0/255, green: 255/255, blue: 255/255, alpha: 1),
            UIColor(red: 0/255, green: 170/255, blue: 255/255, alpha: 1),
            UIColor(red: 0/255, green: 85/255, blue: 255/255, alpha: 1),
            UIColor(red: 0/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 85/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 170/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 255/255, green: 0/255, blue: 255/255, alpha: 1),
            UIColor(red: 255/255, green: 0/255, blue: 170/255, alpha: 1)
        ]
        
        for (index, (start, end)) in skeleton.enumerated() {
            let (startX, startY, startConf) = keypointTuples[start]
            let (endX, endY, endConf) = keypointTuples[end]
            
            if startConf < confidenceThreshold || endConf < confidenceThreshold { continue }
            
            let startXPos = CGFloat(startX) * scaleX
            let startYPos = CGFloat(startY) * scaleY
            let endXPos = CGFloat(endX) * scaleX
            let endYPos = CGFloat(endY) * scaleY
            
            // スケルトンラインを描画
            let linePath = UIBezierPath()
            linePath.move(to: CGPoint(x: startXPos, y: startYPos))
            linePath.addLine(to: CGPoint(x: endXPos, y: endYPos))
            
            let lineLayer = CAShapeLayer()
            lineLayer.path = linePath.cgPath
            lineLayer.strokeColor = limbColor[index % limbColor.count].cgColor
            lineLayer.lineWidth = 2.0
            overlayLayer.addSublayer(lineLayer)
        }
    }
}
