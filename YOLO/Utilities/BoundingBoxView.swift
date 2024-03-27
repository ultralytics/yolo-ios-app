//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  BoundingBoxView for Ultralytics YOLO App
//  This class is designed to visualize bounding boxes and labels for detected objects in the YOLOv8 models within the Ultralytics YOLO app.
//  It leverages Core Animation layers to draw the bounding boxes and text labels dynamically on the detection video feed.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  BoundingBoxView facilitates the clear representation of detection results, improving user interaction with the app by
//  providing immediate visual feedback on detected objects, including their classification and confidence level.

import Foundation
import UIKit
import SwiftUI

/// Manages the visualization of bounding boxes and associated labels for object detection results.
class BoundingBoxView {
    /// The layer that draws the bounding box around a detected object.
    let shapeLayer: CAShapeLayer

    /// The layer that displays the label and confidence score for the detected object.
    let textLayer: CATextLayer
    
    /// The layer that displays the pose
    let lineLayer: CAShapeLayer

    /// The parent layer
    var parentLayer: CALayer?
    /// Initializes a new BoundingBoxView with configured shape and text layers.
    init() {
        shapeLayer = CAShapeLayer()
        shapeLayer.fillColor = UIColor.clear.cgColor  // No fill to only show the bounding outline
        shapeLayer.lineWidth = 2  // Set the stroke line width
        shapeLayer.isHidden = true  // Initially hidden; shown when a detection occurs

        textLayer = CATextLayer()
        textLayer.isHidden = true  // Initially hidden; shown with label when a detection occurs
        textLayer.contentsScale = UIScreen.main.scale  // Ensure the text is sharp on retina displays
        textLayer.fontSize = 14  // Set font size for the label text
        textLayer.font = UIFont(name: "Avenir", size: textLayer.fontSize)  // Use Avenir font for labels
        textLayer.alignmentMode = .center  // Center-align the text within the layer
        
        lineLayer = CAShapeLayer()
        lineLayer.fillColor = UIColor.clear.cgColor
        lineLayer.lineWidth = 2
        lineLayer.isHidden = true
    }

    /// Adds the bounding box and text layers to a specified parent layer.
    /// - Parameter parent: The CALayer to which the bounding box and text layers will be added.
    func addToLayer(_ parent: CALayer) {
        parentLayer = parent
        parentLayer!.addSublayer(shapeLayer)
        parentLayer!.addSublayer(textLayer)
        parentLayer!.addSublayer(lineLayer)
    }
    
    /// Updates the bounding box and label to be visible with specified properties.
    /// - Parameters:
    ///   - frame: The CGRect frame defining the bounding box's size and position.
    ///   - keypoints: The pose keypoints
    ///   - widthRatio: To scale the keypoiints x co-ord
    ///   - heightRatio: To scale the keypoints y co-ord
    ///
    func showOnnx(frame: CGRect, keypoints: [Float32], widthRatio: Float, heightRatio: Float ) {
        CATransaction.setDisableActions(true)  // Disable implicit animations

        let path = UIBezierPath(roundedRect: frame, cornerRadius: 6.0)  // Rounded rectangle for the bounding box
        shapeLayer.path = path.cgPath
        shapeLayer.strokeColor = Color.black.cgColor  // Apply color and alpha to the stroke
        shapeLayer.lineWidth = 4
        shapeLayer.isHidden = false // Make the shape layer visible
        parentLayer?.addSublayer(shapeLayer)
        
        // This loop has drawbacks. The layer that is drawn cannot be erased, so the previous keypoints remain.
        // Also the scaling of keypoints is questionable at best. Just a placeholder method for now.
        for i in stride(from: 0, through: keypoints.count-1, by: 3) {
            let keyPointsLayer = CAShapeLayer()
            let kp_x = keypoints[i] * widthRatio
            let kp_y = keypoints[i+1] * heightRatio
            let confidence = keypoints[i+2]
            if (confidence < 0.5) { // Can potentially remove hardcoding and make the confidence configurable
                continue
            }
            let rFrame = CGRect(x: Double(kp_x), y: Double(kp_y), width: 10, height: 10)
            let pointPath = UIBezierPath(roundedRect: rFrame, cornerRadius: 6.0)
            keyPointsLayer.path = pointPath.cgPath
            keyPointsLayer.isHidden = true
            parentLayer?.addSublayer(keyPointsLayer)
            keyPointsLayer.isHidden = false
        }
    }
    /// Hides the bounding box and text layers.
    func hide() {
        shapeLayer.isHidden = true
        textLayer.isHidden = true
        lineLayer.isHidden = true
    }
}
