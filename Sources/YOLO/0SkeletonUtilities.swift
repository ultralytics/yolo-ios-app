// 0SkeletonUtilities.swift
// Common protocol and utilities for skeleton visualization
// Note: File name prefixed with '0' to ensure it compiles before dependent files

import Foundation
import SpriteKit
#if canImport(UIKit)
import UIKit
#endif

/// Skeleton type options
public enum SkeletonType {
    case articulated   
}

/// Protocol for skeleton visualization implementations
public protocol SkeletonMask {
    /// Create a skeleton scene for the given keypoints and bounding boxes
    func createSkeletonScene(
        keypointsList: [[(x: Float, y: Float)]],
        confsList: [[Float]],
        boundingBoxes: [Box],
        sceneSize: CGSize,
        confThreshold: Float
    ) -> SKScene
}

/// Shared utilities for skeleton rendering
public enum SkeletonUtilities {
    
    // MARK: - Constants
    public struct Constants {
        public static let defaultConfThreshold: Float = 0.25
        public static let defaultAlpha: CGFloat = 0.9
        public static let minAlpha: CGFloat = 0.75
        public static let maxAlpha: CGFloat = 0.95
        public static let animationDuration: TimeInterval = 1.5
        
        // Body proportions
        public static let headScale: CGFloat = 0.8
        public static let bodyScale: CGFloat = 1.2
        public static let pelvicScale: CGFloat = 1.2
        public static let handScale: CGFloat = 0.5
        public static let footScale: CGFloat = 0.5
        
        // Vignette effect
        public static let vignetteLineWidth: CGFloat = 80
        public static let vignetteAlpha: CGFloat = 0.25
        
        // X-ray effect colors
        public static let xrayTintRed: CGFloat = 0.85
        public static let xrayTintGreen: CGFloat = 0.92
        public static let xrayTintBlue: CGFloat = 1.0
        public static let xrayTintIntensity: CGFloat = 0.3
    }
    
    // MARK: - YOLO Keypoint Indices
    public enum KeypointIndex: Int {
        case nose = 0
        case leftEye = 1
        case rightEye = 2
        case leftEar = 3
        case rightEar = 4
        case leftShoulder = 5
        case rightShoulder = 6
        case leftElbow = 7
        case rightElbow = 8
        case leftWrist = 9
        case rightWrist = 10
        case leftHip = 11
        case rightHip = 12
        case leftKnee = 13
        case rightKnee = 14
        case leftAnkle = 15
        case rightAnkle = 16
    }
    
    // MARK: - Coordinate Conversion
    
    /// Convert normalized keypoints to scene coordinates (flipping Y for SpriteKit)
    public static func convertToSceneCoordinates(
        keypoints: [(x: Float, y: Float)],
        sceneSize: CGSize
    ) -> [CGPoint] {
        return keypoints.map { kp in
            CGPoint(
                x: CGFloat(kp.x) * sceneSize.width,
                y: sceneSize.height - (CGFloat(kp.y) * sceneSize.height)
            )
        }
    }
    
    // MARK: - Geometric Calculations
    
    /// Calculate distance between two points
    public static func distance(from start: CGPoint, to end: CGPoint) -> CGFloat {
        let dx = end.x - start.x
        let dy = end.y - start.y
        return sqrt(dx * dx + dy * dy)
    }
    
    /// Calculate angle between two points (for rotation)
    /// Returns angle in radians, adjusted for vertical sprite orientation
    public static func angle(from start: CGPoint, to end: CGPoint) -> CGFloat {
        let dx = end.x - start.x
        let dy = end.y - start.y
        // Subtract π/2 because sprites are oriented vertically by default
        return atan2(dy, dx) - .pi / 2
    }
    
    /// Calculate center point between two points
    public static func centerPoint(_ p1: CGPoint, _ p2: CGPoint) -> CGPoint {
        return CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
    }
    
    /// Calculate average point from an array of points
    public static func averagePoint(_ points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
    }
    
    // MARK: - Scene Effects
    
    /// Add atmospheric vignette effect to scene
    public static func addVignetteEffect(to scene: SKScene) {
        let vignette = SKShapeNode(rect: scene.frame)
        vignette.fillColor = .clear
        vignette.strokeColor = .black
        vignette.lineWidth = Constants.vignetteLineWidth
        vignette.alpha = Constants.vignetteAlpha
        vignette.zPosition = 100
        vignette.blendMode = .multiply
        scene.addChild(vignette)
    }
    
    /// Create X-ray glow effect node
    public static func createXRayEffectNode() -> SKEffectNode {
        let effectNode = SKEffectNode()
        effectNode.shouldRasterize = true
        
        let tintFilter = CIFilter(name: "CIColorMonochrome", parameters: [
            "inputColor": CIColor(
                red: Constants.xrayTintRed,
                green: Constants.xrayTintGreen,
                blue: Constants.xrayTintBlue
            ),
            "inputIntensity": Constants.xrayTintIntensity
        ])
        effectNode.filter = tintFilter
        effectNode.alpha = Constants.defaultAlpha
        
        return effectNode
    }
    
    /// Add pulsing animation to a node
    public static func addPulsingAnimation(to node: SKNode) {
        let fadeAction = SKAction.sequence([
            SKAction.fadeAlpha(to: Constants.minAlpha, duration: Constants.animationDuration),
            SKAction.fadeAlpha(to: Constants.maxAlpha, duration: Constants.animationDuration)
        ])
        node.run(SKAction.repeatForever(fadeAction))
    }
    
    // MARK: - Texture Loading
    
    /// Load texture from bundle with error handling
    public static func loadTexture(named name: String) -> SKTexture? {
        // Try loading from main bundle first (for app assets)
        if let image = UIImage(named: name, in: Bundle.main, compatibleWith: nil) {
            return SKTexture(image: image)
        }
        // Fallback to default bundle
        else if let image = UIImage(named: name) {
            return SKTexture(image: image)
        }
        return nil
    }
    
    // MARK: - Image Combining
    
    /// Combine background and overlay images
    public static func combineImages(background: UIImage?, overlay: UIImage?) -> UIImage? {
        guard let bg = background, let over = overlay else { return background }
        
        UIGraphicsBeginImageContextWithOptions(bg.size, false, 0)
        defer { UIGraphicsEndImageContext() }
        
        bg.draw(at: .zero)
        over.draw(at: .zero, blendMode: .normal, alpha: 1.0)
        
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

