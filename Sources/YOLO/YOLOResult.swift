// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, defining data structures for model inference results.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOResult and related structures define the data models for storing and processing
//  the output from YOLO model inference. This includes bounding boxes for object detection,
//  masks for segmentation, probability distributions for classification, keypoints for pose estimation,
//  and oriented bounding boxes for rotated object detection. These structures maintain the
//  results in a consistent format across different tasks, making it easier to process and
//  visualize the information in the application's UI components.

import CoreGraphics
import Foundation
import UIKit

/// Represents the complete results from a YOLO model inference, containing task-specific outputs.
///
/// This structure consolidates all outputs from YOLO model inference across different task types.
/// It can store bounding boxes for object detection, masks for segmentation, probability
/// distributions for classification, keypoints for pose estimation, and oriented
/// bounding boxes for rotated object detection. The structure also maintains performance
/// metrics and visualization data.
///
/// - Note: Not all fields will be populated for every task type. For example, object detection
///   models will only populate the `boxes` array, while segmentation models will populate
///   both `boxes` and `masks`.
/// - Important: This structure is marked as `@unchecked Sendable` to support concurrent operations.
public struct YOLOResult: @unchecked Sendable {
  /// The original dimensions of the input image that was processed.
  public let orig_shape: CGSize

  /// Array of detected bounding boxes with class information and confidence scores.
  public let boxes: [Box]

  /// Optional segmentation masks for instance segmentation results.
  public var masks: Masks?

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
}

/// Represents a single bounding box detection from a YOLO model.
///
/// This structure contains the information for a single detected object,
/// including its class, confidence score, and location in both normalized
/// and image coordinates.
///
/// - Note: This structure is marked as `@unchecked Sendable` to support concurrent operations.
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

/// Represents segmentation mask data from a YOLO segmentation model.
///
/// This structure contains both the raw per-pixel mask data for each detected object
/// and an optional pre-rendered combined mask as a CGImage for visualization.
///
/// - Note: This structure is marked as `@unchecked Sendable` to support concurrent operations.
public struct Masks: @unchecked Sendable {
  /// The raw mask data as a 3D array [instance][height][width] with float values.
  public let masks: [[[Float]]]

  /// Pre-rendered combined mask image for visualization.
  public let combinedMask: CGImage?
}

/// Represents classification probability results from a YOLO classification model.
///
/// This structure contains the top predicted classes and their confidence scores,
/// providing both the single highest confidence prediction and the top 5 predictions.
///
/// - Note: This structure is marked as `@unchecked Sendable` to support concurrent operations.
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

/// Represents keypoint detection results from a YOLO pose estimation model.
///
/// This structure contains the detected keypoints (e.g., body joints) for a human figure,
/// storing both normalized and image-space coordinates along with confidence values.
public struct Keypoints {
  /// The keypoint coordinates in normalized space (0.0 to 1.0).
  public let xyn: [(x: Float, y: Float)]

  /// The keypoint coordinates in image space (pixels).
  public let xy: [(x: Float, y: Float)]

  /// The confidence scores (0.0 to 1.0) for each keypoint.
  public let conf: [Float]
}

/// Represents a single oriented bounding box detection result.
///
/// This structure contains the information for a single detected object with an oriented
/// (rotated) bounding box, including its class, confidence score, and OBB parameters.
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

/// Represents an oriented (rotated) bounding box.
///
/// This structure defines a bounding box with rotation, storing the center coordinates,
/// width, height, and rotation angle. OBBs provide better fitting boundaries for objects
/// that are not aligned with the image axes.
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

  /// Creates a new oriented bounding box with the specified parameters.
  ///
  /// - Parameters:
  ///   - cx: The x-coordinate of the center of the box.
  ///   - cy: The y-coordinate of the center of the box.
  ///   - w: The width of the box.
  ///   - h: The height of the box.
  ///   - angle: The rotation angle of the box in radians.
  public init(cx: Float, cy: Float, w: Float, h: Float, angle: Float) {
    self.cx = cx
    self.cy = cy
    self.w = w
    self.h = h
    self.angle = angle
  }
}

/// A polygon represented as an array of points (x, y).
///
/// This type is used to represent the corners of an oriented bounding box
/// after conversion from the center/width/height/angle representation.
public typealias Polygon = [CGPoint]

/// Extension to provide additional functionality for oriented bounding boxes.
extension OBB {
  /// Converts the OBB to an array of 4 corner points (polygon).
  ///
  /// This method transforms the center, width, height, and angle representation
  /// of an oriented bounding box into the four corner points of the rectangle.
  /// The corners are returned in clockwise or counter-clockwise order.
  ///
  /// - Returns: An array of four CGPoints representing the corners of the oriented bounding box.
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

  /// Converts the OBB to pixel-space corner points, correcting for aspect ratio.
  ///
  /// The model operates in a square coordinate space (e.g. 640x640), so the angle and
  /// extents need adjustment when rendering on a non-square display. This method maps
  /// the OBB from normalized (square) space to pixel (non-square) space, keeping
  /// rectangles as rectangles with the correct visual angle.
  ///
  /// - Parameter imageSize: The target image/view size in pixels.
  /// - Returns: An array of four CGPoints in pixel coordinates.
  public func toPolygon(imageSize: CGSize) -> Polygon {
    let W = Double(imageSize.width)
    let H = Double(imageSize.height)
    let cosA = cos(Double(angle))
    let sinA = sin(Double(angle))

    // Adjust angle from square model space to non-square display space
    let adjAngle = atan2(H * sinA, W * cosA)
    let cosAdj = CGFloat(cos(adjAngle))
    let sinAdj = CGFloat(sin(adjAngle))

    // Adjust extents: width direction (cosA, sinA) scales non-uniformly
    let halfWPx = CGFloat(w) / 2 * CGFloat(sqrt(pow(cosA * W, 2) + pow(sinA * H, 2)))
    // Height direction (-sinA, cosA) scales non-uniformly
    let halfHPx = CGFloat(h) / 2 * CGFloat(sqrt(pow(sinA * W, 2) + pow(cosA * H, 2)))

    let cxPx = CGFloat(cx) * CGFloat(W)
    let cyPx = CGFloat(cy) * CGFloat(H)

    let localCorners = [
      CGPoint(x: -halfWPx, y: -halfHPx),
      CGPoint(x: halfWPx, y: -halfHPx),
      CGPoint(x: halfWPx, y: halfHPx),
      CGPoint(x: -halfWPx, y: halfHPx),
    ]

    return localCorners.map { pt -> CGPoint in
      let rx = cosAdj * pt.x - sinAdj * pt.y
      let ry = sinAdj * pt.x + cosAdj * pt.y
      return CGPoint(x: rx + cxPx, y: ry + cyPx)
    }
  }

  /// Calculates the area of the oriented bounding box.
  ///
  /// The area of an oriented bounding box is simply the product of its width and height,
  /// regardless of the rotation angle.
  ///
  /// - Returns: The area of the bounding box in square pixels.
  public var area: CGFloat {
    return CGFloat(w * h)
  }
}
