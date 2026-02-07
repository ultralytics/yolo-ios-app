// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import CoreGraphics
import Foundation

/// Complete results from a YOLO model inference.
public struct YOLOResult: Sendable {
  /// The original dimensions of the input image.
  public let orig_shape: CGSize

  /// Detected bounding boxes with class information and confidence scores.
  public let boxes: [Box]

  /// Segmentation masks for instance segmentation results.
  public var masks: Masks?

  /// Classification probability results.
  public var probs: Probs?

  /// Keypoint sets for pose estimation results.
  public var keypointsList: [Keypoints]

  /// Oriented bounding box results.
  public var obb: [OBBResult]

  /// Image with detection visualizations overlaid.
  public var annotatedImage: CGImage?

  /// Time taken (in seconds) to perform inference.
  public var speed: Double

  /// Frames per second rate for real-time processing.
  public var fps: Double?

  /// Array of class label names used by the model.
  public var names: [String]

  public init(
    orig_shape: CGSize,
    boxes: [Box],
    masks: Masks? = nil,
    probs: Probs? = nil,
    keypointsList: [Keypoints] = [],
    obb: [OBBResult] = [],
    annotatedImage: CGImage? = nil,
    speed: Double = 0,
    fps: Double? = nil,
    names: [String] = []
  ) {
    self.orig_shape = orig_shape
    self.boxes = boxes
    self.masks = masks
    self.probs = probs
    self.keypointsList = keypointsList
    self.obb = obb
    self.annotatedImage = annotatedImage
    self.speed = speed
    self.fps = fps
    self.names = names
  }
}

/// A single bounding box detection.
public struct Box: Sendable {
  /// The index of the class in the model's class list.
  public let index: Int

  /// The class label of the detected object.
  public let cls: String

  /// The confidence score (0.0 to 1.0).
  public let conf: Float

  /// The bounding box in image coordinates.
  public let xywh: CGRect

  /// The bounding box in normalized coordinates (0.0 to 1.0).
  public let xywhn: CGRect

  public init(index: Int, cls: String, conf: Float, xywh: CGRect, xywhn: CGRect) {
    self.index = index
    self.cls = cls
    self.conf = conf
    self.xywh = xywh
    self.xywhn = xywhn
  }
}

/// Segmentation mask data.
public struct Masks: Sendable {
  /// Raw mask data as a 3D array [instance][height][width].
  public let masks: [[[Float]]]

  /// Pre-rendered combined mask image.
  public let combinedMask: CGImage?

  public init(masks: [[[Float]]], combinedMask: CGImage?) {
    self.masks = masks
    self.combinedMask = combinedMask
  }
}

/// Classification probability results.
public struct Probs: Sendable {
  /// The class label with the highest confidence.
  public var top1: String

  /// The top 5 class labels by confidence.
  public var top5: [String]

  /// The confidence score for the top prediction.
  public var top1Conf: Float

  /// The confidence scores for the top 5 predictions.
  public var top5Confs: [Float]

  public init(top1: String, top5: [String], top1Conf: Float, top5Confs: [Float]) {
    self.top1 = top1
    self.top5 = top5
    self.top1Conf = top1Conf
    self.top5Confs = top5Confs
  }
}

/// Keypoint detection results for pose estimation.
public struct Keypoints: Sendable {
  /// Keypoint coordinates in normalized space (0.0 to 1.0).
  public let xyn: [(x: Float, y: Float)]

  /// Keypoint coordinates in image space (pixels).
  public let xy: [(x: Float, y: Float)]

  /// Confidence scores for each keypoint.
  public let conf: [Float]

  public init(xyn: [(x: Float, y: Float)], xy: [(x: Float, y: Float)], conf: [Float]) {
    self.xyn = xyn
    self.xy = xy
    self.conf = conf
  }
}

/// A single oriented bounding box detection result.
public struct OBBResult: Sendable {
  public var box: OBB
  public var confidence: Float
  public var cls: String
  public var index: Int

  public init(box: OBB, confidence: Float, cls: String, index: Int) {
    self.box = box
    self.confidence = confidence
    self.cls = cls
    self.index = index
  }
}

/// An oriented (rotated) bounding box.
public struct OBB: Sendable {
  public var cx: Float
  public var cy: Float
  public var w: Float
  public var h: Float
  public var angle: Float

  public init(cx: Float, cy: Float, w: Float, h: Float, angle: Float) {
    self.cx = cx
    self.cy = cy
    self.w = w
    self.h = h
    self.angle = angle
  }

  /// Converts the OBB to an array of 4 corner points.
  public func toPolygon() -> Polygon {
    let halfW = w / 2
    let halfH = h / 2
    let localCorners = [
      CGPoint(x: -CGFloat(halfW), y: -CGFloat(halfH)),
      CGPoint(x: CGFloat(halfW), y: -CGFloat(halfH)),
      CGPoint(x: CGFloat(halfW), y: CGFloat(halfH)),
      CGPoint(x: -CGFloat(halfW), y: CGFloat(halfH)),
    ]
    let cosA = cos(Double(angle))
    let sinA = sin(Double(angle))
    return localCorners.map { pt in
      let rx = CGFloat(cosA) * pt.x - CGFloat(sinA) * pt.y
      let ry = CGFloat(sinA) * pt.x + CGFloat(cosA) * pt.y
      return CGPoint(x: rx + CGFloat(cx), y: ry + CGFloat(cy))
    }
  }

  /// The area of the bounding box.
  public var area: CGFloat {
    CGFloat(w * h)
  }

  /// Axis-aligned bounding box around this rotated box.
  public func toAABB() -> CGRect {
    let poly = toPolygon()
    var minX = CGFloat.infinity, maxX = -CGFloat.infinity
    var minY = CGFloat.infinity, maxY = -CGFloat.infinity
    for p in poly {
      if p.x < minX { minX = p.x }
      if p.x > maxX { maxX = p.x }
      if p.y < minY { minY = p.y }
      if p.y > maxY { maxY = p.y }
    }
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }
}

/// A polygon represented as an array of points.
public typealias Polygon = [CGPoint]
