// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  BoundingBoxView for the Ultralytics YOLO SDK
//  Visualizes bounding boxes and labels for detected objects using Core Animation layers drawn dynamically on the
//  detection video feed.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  BoundingBoxView renders detection results — bounding box, class label, and confidence — for immediate visual
//  feedback in camera and image previews.

import Foundation
import UIKit

enum DetectionLabelStyle {
  static let fontName = "Avenir"
  static let maxTextSize = CGSize(width: 400, height: 100)
  static let horizontalPadding: CGFloat = 12
  static let verticalOffset: CGFloat = 2
  static let cornerRadius: CGFloat = 3

  static func text(className: String, confidence: CGFloat) -> String {
    String(format: "%@ %.1f", className, confidence * 100)
  }

  static func alpha(confidence: CGFloat) -> CGFloat {
    max(0.6, (confidence - 0.2) / (1.0 - 0.2) * 0.9)
  }

  static func font(size: CGFloat) -> UIFont {
    UIFont(name: fontName, size: size) ?? UIFont.systemFont(ofSize: size)
  }

  static func configure(_ textLayer: CATextLayer, fontSize: CGFloat) {
    let font = font(size: fontSize)
    textLayer.contentsScale = UIScreen.main.scale
    textLayer.fontSize = fontSize
    textLayer.font = font
    textLayer.alignmentMode = .center
    textLayer.cornerRadius = cornerRadius
    textLayer.masksToBounds = true
  }

  static func attributes(fontSize: CGFloat, alpha: CGFloat = 1) -> [NSAttributedString.Key: Any] {
    [
      .font: font(size: fontSize),
      .foregroundColor: UIColor.white.withAlphaComponent(alpha),
    ]
  }

  static func size(for label: String, fontSize: CGFloat) -> CGSize {
    let textRect = label.boundingRect(
      with: maxTextSize,
      options: .truncatesLastVisibleLine,
      attributes: attributes(fontSize: fontSize),
      context: nil
    )
    return CGSize(width: textRect.width + horizontalPadding, height: textRect.height)
  }

  static func frame(for label: String, fontSize: CGFloat, anchor: CGPoint) -> CGRect {
    let textSize = size(for: label, fontSize: fontSize)
    let origin = CGPoint(
      x: anchor.x - verticalOffset,
      y: anchor.y - textSize.height - verticalOffset
    )
    return CGRect(origin: origin, size: textSize)
  }
}

/// Manages the visualization of bounding boxes and associated labels for object detection results.
@MainActor
public final class BoundingBoxView {
  /// The layer that draws the bounding box around a detected object.
  let shapeLayer: CAShapeLayer

  /// The layer that displays the label and confidence score for the detected object.
  let textLayer: CATextLayer

  /// The base font size that can be scaled for external displays
  private var baseFontSize: CGFloat = 14

  /// Initializes a new BoundingBoxView with configured shape and text layers.
  init() {
    shapeLayer = CAShapeLayer()
    shapeLayer.fillColor = UIColor.clear.cgColor  // outline only, no fill
    shapeLayer.lineWidth = 4
    shapeLayer.isHidden = true  // shown when a detection occurs

    textLayer = CATextLayer()
    textLayer.isHidden = true  // shown when a detection occurs
    DetectionLabelStyle.configure(textLayer, fontSize: baseFontSize)
  }

  /// Sets the font size for the text layer (useful for external displays)
  func setFontSize(_ size: CGFloat) {
    baseFontSize = size
    DetectionLabelStyle.configure(textLayer, fontSize: size)
  }

  /// Sets the line width for the bounding box (useful for external displays)
  func setLineWidth(_ width: CGFloat) {
    shapeLayer.lineWidth = width
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
  ///   - angle: Optional rotation angle in radians for oriented boxes.
  func show(frame: CGRect, label: String, color: UIColor, alpha: CGFloat, angle: CGFloat? = nil) {
    CATransaction.begin()
    CATransaction.setDisableActions(true)  // disable implicit animations

    let path = UIBezierPath(roundedRect: frame, cornerRadius: 6.0)
    if let angle {
      var transform = CGAffineTransform(translationX: frame.midX, y: frame.midY)
      transform = transform.rotated(by: angle)
      transform = transform.translatedBy(x: -frame.midX, y: -frame.midY)
      path.apply(transform)
    }
    shapeLayer.path = path.cgPath
    shapeLayer.strokeColor = color.withAlphaComponent(alpha).cgColor
    shapeLayer.isHidden = false

    textLayer.string = label
    textLayer.backgroundColor = color.withAlphaComponent(alpha).cgColor
    textLayer.isHidden = false
    textLayer.foregroundColor = UIColor.white.withAlphaComponent(alpha).cgColor

    textLayer.frame = DetectionLabelStyle.frame(
      for: label,
      fontSize: textLayer.fontSize,
      anchor: angle == nil ? frame.origin : path.bounds.origin
    )
    CATransaction.commit()
  }

  /// Hides the bounding box and text layers.
  func hide() {
    shapeLayer.isHidden = true
    textLayer.isHidden = true
  }
}
