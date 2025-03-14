import CoreGraphics
import Foundation
import UIKit

public struct YOLOResult: @unchecked Sendable {
  public let orig_shape: CGSize
  public let boxes: [Box]
  public var masks: Masks?
  public var probs: Probs?
  public var keypointsList: [Keypoints] = []
  public var obb: [OBBResult] = []
  public var annotatedImage: UIImage?
  public var speed: Double
  public var fps: Double?
  public var originalImage: UIImage?
  public var names: [String]
  //    let keypoints: [Keypoint]
}

public struct Box: @unchecked Sendable {
  public let index: Int
  public let cls: String
  public let conf: Float
  public let xywh: CGRect
  public let xywhn: CGRect
}

public struct Masks: @unchecked Sendable {
  public let masks: [[[Float]]]
  public let combinedMask: CGImage?
}

public struct Probs: @unchecked Sendable {
  public var top1: String
  public var top5: [String]
  public var top1Conf: Float
  public var top5Confs: [Float]
}

public struct Keypoints {
  public let xyn: [(x: Float, y: Float)]
  public let xy: [(x: Float, y: Float)]
  public let conf: [Float]
}

public struct OBBResult {
  var box: OBB
  var confidence: Float
  var cls: String
  var index: Int
}

public struct OBB {
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
}

/// A polygon is just an array of points (x, y).
public typealias Polygon = [CGPoint]

extension OBB {
  /// Convert the OBB to an array of 4 corner points (polygon).
  /// Corners are returned in clockwise or counter-clockwise order.
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

  /// Convenient area for an OBB is just w * h (no rotation needed).
  public var area: CGFloat {
    return CGFloat(w * h)
  }
}
