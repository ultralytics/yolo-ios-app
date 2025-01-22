import Foundation
import CoreGraphics
import UIKit

public struct YOLOResult:@unchecked Sendable  {
    public let orig_shape: CGSize
    public let boxes: [Box]
    public var masks: Masks?
    public var probs: Probs?
    public var keypointsList: [Keypoints] = []
    public var annotatedImage: UIImage?
    public var speed: Double
    public var fps: Double?
    public var originalImage: UIImage?
    public var names: [String]
//    let keypoints: [Keypoint]
}

public struct Box:@unchecked Sendable  {
    public let index: Int
    public let cls: String
    public let conf: Float
    public let xywh: CGRect
    public let xywhn: CGRect
}

public struct Masks:@unchecked Sendable  {
    public let masks: [[[Float]]]
    public let combinedMask: CGImage?
}

public struct Probs:@unchecked Sendable  {
    public var top1: String
    public var top5: [String]
    public var top1Conf: Float
    public var top5Confs: [Float]
}

public struct Keypoints {
    public let xyn: [(x:Float, y:Float)]
    public let xy: [(x:Float, y:Float)]
    public let conf: [Float]
}
