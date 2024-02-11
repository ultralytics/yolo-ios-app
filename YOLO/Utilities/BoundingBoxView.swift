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

/// Manages the visualization of bounding boxes and associated labels for object detection results.
class BoundingBoxView {
    /// The layer that draws the bounding box around a detected object.
    let shapeLayer: CAShapeLayer

    /// The layer that displays the label and confidence score for the detected object.
    let textLayer: CATextLayer

    /// Initializes a new BoundingBoxView with configured shape and text layers.
    init() {
        shapeLayer = CAShapeLayer()
        shapeLayer.fillColor = UIColor.clear.cgColor  // No fill to only show the bounding outline
        shapeLayer.lineWidth = 4  // Set the stroke line width
        shapeLayer.isHidden = true  // Initially hidden; shown when a detection occurs

        textLayer = CATextLayer()
        textLayer.isHidden = true  // Initially hidden; shown with label when a detection occurs
        textLayer.contentsScale = UIScreen.main.scale  // Ensure the text is sharp on retina displays
        textLayer.fontSize = 14  // Set font size for the label text
        textLayer.font = UIFont(name: "Avenir", size: textLayer.fontSize)  // Use Avenir font for labels
        textLayer.alignmentMode = .center  // Center-align the text within the layer
    }

    /// Adds the bounding box and text layers to a specified parent layer.
    /// - Parameter parent: The CALayer to which the bounding box and text layers will be added.
    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }

    /// Updates the bounding box and label to be visible with specified properties.
    /// - Parameters:
    ///   - frame: The CGRect frame defining the bounding box's size and position.
    ///   - label: The text label to display (e.g., object class and confidence).
    ///   - color: The color of the bounding box stroke and label background.
    ///   - alpha: The opacity level for the bounding box stroke and label background.
    func show(frame: CGRect, label: String, color: UIColor, alpha: CGFloat) {
        CATransaction.setDisableActions(true)  // Disable implicit animations

        let path = UIBezierPath(roundedRect: frame, cornerRadius: 6.0)  // Rounded rectangle for the bounding box
        shapeLayer.path = path.cgPath
        shapeLayer.strokeColor = color.withAlphaComponent(alpha).cgColor  // Apply color and alpha to the stroke
        shapeLayer.isHidden = false  // Make the shape layer visible

        textLayer.string = label  // Set the label text
        textLayer.backgroundColor = color.withAlphaComponent(alpha).cgColor  // Apply color and alpha to the background
        textLayer.isHidden = false  // Make the text layer visible
        textLayer.foregroundColor = UIColor.white.withAlphaComponent(alpha).cgColor  // Set text color

        // Calculate the text size and position based on the label content
        let attributes = [NSAttributedString.Key.font: textLayer.font as Any]
        let textRect = label.boundingRect(with: CGSize(width: 400, height: 100),
                options: .truncatesLastVisibleLine,
                attributes: attributes, context: nil)
        let textSize = CGSize(width: textRect.width + 12, height: textRect.height)  // Add padding to the text size
        let textOrigin = CGPoint(x: frame.origin.x - 2, y: frame.origin.y - textSize.height - 2)  // Position above the bounding box
        textLayer.frame = CGRect(origin: textOrigin, size: textSize)  // Set the text layer frame
    }

    /// Hides the bounding box and text layers.
    func hide() {
        shapeLayer.isHidden = true
        textLayer.isHidden = true
    }
}
