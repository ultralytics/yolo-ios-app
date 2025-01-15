import Foundation
import CoreGraphics
import UIKit

public struct YOLOResult:@unchecked Sendable  {
    public let orig_shape: CGSize
    public let boxes: [Box]
    public var annotatedImage: UIImage?
    public var speed: Double
    public var fps: Double?
    public var originalImage: UIImage?
//    public var speed: Float
//    let masks: [CGImage]
//    let keypoints: [Keypoint]
}

public struct Box:@unchecked Sendable  {
    public let index: Int
    public let cls: String
    public let conf: Float
    public let xywh: CGRect
    public let xywhn: CGRect
    
}

public struct Keypoint {
    public let position: CGPoint
    public let confidence: Float
}
