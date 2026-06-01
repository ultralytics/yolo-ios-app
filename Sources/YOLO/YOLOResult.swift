// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, defining data structures for model inference results.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLOResult and its supporting structs hold the output of a YOLO inference: bounding boxes for detection,
//  instance and semantic masks for segmentation, top-k probabilities for classification, keypoints for pose, and
//  oriented bounding boxes for OBB. The shared shape keeps result handling consistent across tasks.

import CoreGraphics
import Foundation
import UIKit

/// The complete output of a single YOLO inference, with optional task-specific fields populated as needed.
///
/// Holds bounding boxes, instance and semantic masks, classification probabilities, keypoints, oriented bounding
/// boxes, an annotated preview image, and timing metrics.
///
/// - Note: Not every field is populated for every task — detection populates `boxes`, segmentation also populates
///   `masks`, classification populates `probs`, pose populates `keypointsList`, and OBB populates `obb`.
/// - Important: Marked `@unchecked Sendable` so results can cross actor boundaries; fields are written once during
///   construction and treated as read-only thereafter.
public struct YOLOResult: @unchecked Sendable {
  /// The original dimensions of the input image that was processed.
  public let orig_shape: CGSize

  /// Array of detected bounding boxes with class information and confidence scores.
  public let boxes: [Box]

  /// Optional segmentation masks for instance segmentation results.
  public var masks: Masks?

  /// Optional dense class map for semantic segmentation results.
  public var semanticMask: SemanticMask?

  /// Optional probability distribution for classification results.
  public var probs: Probs?

  /// Array of keypoint sets for pose estimation results.
  public var keypointsList: [Keypoints] = []

  /// Array of oriented bounding box results for rotated object detection.
  public var obb: [OBBResult] = []

  /// Image with detection visualizations overlaid on the original input.
  public var annotatedImage: UIImage?

  /// Time taken (in seconds) to perform the inference operation.
  public var speed: Double

  /// Optional frames per second rate for real-time processing.
  public var fps: Double?

  /// Array of class label names used by the model.
  public var names: [String]

  /// An empty result with zeroed metrics, used as a default return value before inference runs.
  public static let empty = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
}

/// A single bounding-box detection from a YOLO model.
///
/// Holds the class index and label, the confidence score, and the rectangle in both image-space (`xywh`) and
/// normalized (`xywhn`) coordinates.
public struct Box: @unchecked Sendable {
  /// The index of the class in the model's class list.
  public let index: Int

  /// The class label (category name) of the detected object.
  public let cls: String

  /// The confidence score (0.0 to 1.0) for the detection.
  public let conf: Float

  /// The bounding box in image coordinates (x, y, width, height).
  public let xywh: CGRect

  /// The bounding box in normalized coordinates (0.0 to 1.0).
  public let xywhn: CGRect
}

/// Instance-segmentation mask data from a YOLO segmentation model.
///
/// Holds raw per-instance probability masks plus an optional pre-rendered composite for display.
public struct Masks: @unchecked Sendable {
  /// The raw mask data as a 3D array [instance][height][width] with float values.
  public let masks: [[[Float]]]

  /// Pre-rendered combined mask image for visualization.
  public let combinedMask: CGImage?
}

/// Semantic-segmentation mask data from a YOLO semantic model.
///
/// `classMap` holds one class index per output pixel after letterbox padding has been removed. `maskImage` is a
/// pre-rendered color overlay for display.
public struct SemanticMask: @unchecked Sendable {
  /// Dense class IDs in row-major order with `width * height` elements.
  public let classMap: [Int]

  /// Width of the dense class map.
  public let width: Int

  /// Height of the dense class map.
  public let height: Int

  /// Pre-rendered color overlay image for visualization.
  public let maskImage: CGImage?
}

/// Classification probability results from a YOLO classification model.
///
/// Holds both the single highest-confidence prediction and the top-5 predictions with their scores.
public struct Probs: @unchecked Sendable {
  /// The class label with the highest confidence score.
  public var top1: String

  /// The top 5 class labels by confidence score.
  public var top5: [String]

  /// The confidence score (0.0 to 1.0) for the top prediction.
  public var top1Conf: Float

  /// The confidence scores (0.0 to 1.0) for the top 5 predictions.
  public var top5Confs: [Float]
}

/// Keypoint detection results from a YOLO pose estimation model.
///
/// Holds the detected body joints for a single subject in both normalized (`xyn`) and image-space (`xy`)
/// coordinates, along with a confidence score per keypoint.
public struct Keypoints {
  /// The keypoint coordinates in normalized space (0.0 to 1.0).
  public let xyn: [(x: Float, y: Float)]

  /// The keypoint coordinates in image space (pixels).
  public let xy: [(x: Float, y: Float)]

  /// The confidence scores (0.0 to 1.0) for each keypoint.
  public let conf: [Float]
}

/// A single oriented bounding-box detection.
///
/// Holds the rotated box, its confidence, and the detected class as both an index and a label.
public struct OBBResult {
  /// The oriented bounding box parameters.
  public var box: OBB

  /// The confidence score (0.0 to 1.0) for the detection.
  public var confidence: Float

  /// The class label (category name) of the detected object.
  public var cls: String

  /// The index of the class in the model's class list.
  public var index: Int
}

/// An oriented (rotated) bounding box stored as center, size, and rotation angle.
///
/// OBBs fit objects that aren't axis-aligned more tightly than a standard `Box`.
public struct OBB {
  /// The x-coordinate of the center of the box.
  public var cx: Float

  /// The y-coordinate of the center of the box.
  public var cy: Float

  /// The width of the box.
  public var w: Float

  /// The height of the box.
  public var h: Float

  /// The rotation angle of the box in radians.
  public var angle: Float

  /// Creates an oriented bounding box.
  ///
  /// - Parameters:
  ///   - cx: Center x-coordinate.
  ///   - cy: Center y-coordinate.
  ///   - w: Box width.
  ///   - h: Box height.
  ///   - angle: Rotation angle in radians.
  public init(cx: Float, cy: Float, w: Float, h: Float, angle: Float) {
    self.cx = cx
    self.cy = cy
    self.w = w
    self.h = h
    self.angle = angle
  }
}

/// A polygon represented as an array of `CGPoint` corners.
///
/// Used to express OBB corners after converting from the center/size/angle form.
public typealias Polygon = [CGPoint]

extension OBB {
  /// Converts the OBB to its four corner points in normalized space.
  ///
  /// Use this overload for NMS/IoU work where the result must be scale-invariant. For rendering on a non-square
  /// image, prefer `toPolygon(imageSize:)` to avoid aspect-ratio distortion.
  ///
  /// - Returns: The four corners of the oriented bounding box, in clockwise order.
  public func toPolygon() -> Polygon {
    // Half extents
    let halfW = w / 2
    let halfH = h / 2

    // Local corners (center at (0,0))
    //  0: (-w/2, -h/2)
    //  1: ( w/2, -h/2)
    //  2: ( w/2,  h/2)
    //  3: (-w/2,  h/2)
    let localCorners = [
      CGPoint(x: -CGFloat(halfW), y: -CGFloat(halfH)),
      CGPoint(x: CGFloat(halfW), y: -CGFloat(halfH)),
      CGPoint(x: CGFloat(halfW), y: CGFloat(halfH)),
      CGPoint(x: -CGFloat(halfW), y: CGFloat(halfH)),
    ]

    // Rotation (angle in radians)
    let cosA = cos(Double(angle))
    let sinA = sin(Double(angle))

    // Compute final corners
    let worldCorners = localCorners.map { pt -> CGPoint in
      // Rotation
      let rx = CGFloat(cosA) * pt.x - CGFloat(sinA) * pt.y
      let ry = CGFloat(sinA) * pt.x + CGFloat(cosA) * pt.y
      // Translation
      let finalX = rx + CGFloat(cx)
      let finalY = ry + CGFloat(cy)
      return CGPoint(x: finalX, y: finalY)
    }
    return worldCorners
  }

  /// Converts the OBB to pixel-space corner points.
  ///
  /// OBB values are normalized to the input image after preprocessing padding is removed, so the stored angle can
  /// be applied directly in pixel space without aspect-ratio correction.
  ///
  /// - Parameter imageSize: Target image or view size in pixels.
  /// - Returns: The four corners of the oriented bounding box in pixel coordinates.
  public func toPolygon(imageSize: CGSize) -> Polygon {
    let cosA = CGFloat(cos(Double(angle)))
    let sinA = CGFloat(sin(Double(angle)))
    let halfWPx = CGFloat(w) * imageSize.width / 2
    let halfHPx = CGFloat(h) * imageSize.height / 2
    let cxPx = CGFloat(cx) * imageSize.width
    let cyPx = CGFloat(cy) * imageSize.height

    let localCorners = [
      CGPoint(x: -halfWPx, y: -halfHPx),
      CGPoint(x: halfWPx, y: -halfHPx),
      CGPoint(x: halfWPx, y: halfHPx),
      CGPoint(x: -halfWPx, y: halfHPx),
    ]

    return localCorners.map { pt -> CGPoint in
      let rx = cosA * pt.x - sinA * pt.y
      let ry = sinA * pt.x + cosA * pt.y
      return CGPoint(x: rx + cxPx, y: ry + cyPx)
    }
  }

  /// The area of the oriented bounding box (`w * h`, rotation-independent).
  public var area: CGFloat {
    return CGFloat(w * h)
  }
}
