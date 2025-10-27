// RealisticSkeletonMask.swift
// Full skeleton visualization using a single skeleton image

import SpriteKit
import UIKit
import CoreGraphics

/// Skeleton type options
public enum SkeletonType {
    case full       // Regular full skeleton
    case silly      // Silly/funny skeleton
}

/// Creates a realistic human skeleton visualization using a full skeleton image
public class RealisticSkeletonMask {
    
    private var fullSkeletonTexture: SKTexture?
    private var sillySkeletonTexture: SKTexture?
    
    /// Current skeleton type to display
    public var skeletonType: SkeletonType = .full
    
    public init() {
        loadSkeletonTextures()
    }
    
    /// Load both skeleton textures
    private func loadSkeletonTextures() {
        // Load full skeleton
        if let image = UIImage(named: "full-skeleton") {
            fullSkeletonTexture = SKTexture(image: image)
        } else {
            print("⚠️ Failed to load full-skeleton image")
            fullSkeletonTexture = generateFallbackSkeleton()
        }
        
        // Load silly skeleton
        if let image = UIImage(named: "silly-skeleton") {
            sillySkeletonTexture = SKTexture(image: image)
        } else {
            print("⚠️ Failed to load silly-skeleton image")
            sillySkeletonTexture = fullSkeletonTexture // Fallback to full skeleton
        }
    }
    
    /// Get the currently selected skeleton texture
    private var currentSkeletonTexture: SKTexture? {
        switch skeletonType {
        case .full:
            return fullSkeletonTexture
        case .silly:
            return sillySkeletonTexture
        }
    }
    
    /// Generate a simple fallback skeleton if image is missing
    private func generateFallbackSkeleton() -> SKTexture {
        let size = CGSize(width: 327, height: 762)
        
        UIGraphicsBeginImageContextWithOptions(size, false, 0)
        guard let context = UIGraphicsGetCurrentContext() else {
            return SKTexture()
        }
        
        // Draw a simple skeleton shape
        context.setStrokeColor(UIColor(red: 0.95, green: 0.92, blue: 0.88, alpha: 1.0).cgColor)
        context.setLineWidth(8)
        
        // Spine
        context.move(to: CGPoint(x: size.width / 2, y: 50))
        context.addLine(to: CGPoint(x: size.width / 2, y: size.height - 100))
        context.strokePath()
        
        // Skull
        let skullRect = CGRect(x: size.width / 2 - 40, y: 10, width: 80, height: 80)
        context.addEllipse(in: skullRect)
        context.strokePath()
        
        // Ribcage
        for i in 0..<5 {
            let y = CGFloat(120 + i * 30)
            context.move(to: CGPoint(x: size.width / 2 - 50, y: y))
            context.addLine(to: CGPoint(x: size.width / 2 + 50, y: y))
            context.strokePath()
        }
        
        let image = UIGraphicsGetImageFromCurrentImageContext() ?? UIImage()
        UIGraphicsEndImageContext()
        
        return SKTexture(image: image)
    }
    
    /// Create the skeleton scene with full skeleton images
    public func createRealisticSkeletonScene(
        keypointsList: [[(x: Float, y: Float)]],
        confsList: [[Float]],
        boundingBoxes: [Box],
        sceneSize: CGSize,
        confThreshold: Float = 0.25
    ) -> SKScene {
        let scene = SKScene(size: sceneSize)
        scene.backgroundColor = .clear
        scene.scaleMode = .aspectFill
        
        // Add atmospheric background
        addAtmosphericEffects(to: scene)
        
        // Process each detected person
        for (personIndex, keypoints) in keypointsList.enumerated() {
            guard personIndex < confsList.count else { continue }
            
            let confs = confsList[personIndex]
            
            // Add full skeleton for this person
            addFullSkeletonToScene(
                scene: scene,
                keypoints: keypoints,
                confs: confs,
                personIndex: personIndex,
                sceneSize: sceneSize,
                confThreshold: confThreshold
            )
        }
        
        return scene
    }
    
    /// Add a full skeleton image positioned and scaled for a detected person
    private func addFullSkeletonToScene(
        scene: SKScene,
        keypoints: [(x: Float, y: Float)],
        confs: [Float],
        personIndex: Int,
        sceneSize: CGSize,
        confThreshold: Float
    ) {
        guard let texture = currentSkeletonTexture else { return }
        
        // YOLO keypoints: 0=nose, 5-6=shoulders, 11-12=hips, 15-16=ankles
        guard keypoints.count >= 17 else { return }
        
        // Convert normalized keypoints to scene coordinates
        let sceneKeypoints = keypoints.map { kp in
            CGPoint(
                x: CGFloat(kp.x) * sceneSize.width,
                y: sceneSize.height - (CGFloat(kp.y) * sceneSize.height)
            )
        }
        
        // Calculate skeleton position and scale based on key body points
        guard let (position, scale, rotation) = calculateSkeletonTransform(
            keypoints: sceneKeypoints,
            confs: confs,
            confThreshold: confThreshold
        ) else {
            return
        }
        
        // Create skeleton sprite
        let skeletonSprite = SKSpriteNode(texture: texture)
        
        // Adjust position: lower by 30 pixels to reach legs better
        skeletonSprite.position = CGPoint(x: position.x, y: position.y - 75)
        
        // Set base scale
        skeletonSprite.setScale(scale)
        
        // Extend vertical scale to reach the legs (1.15 = 15% taller)
        skeletonSprite.yScale *= 1.15
        
        // Fix upside-down issue: Flip the sprite vertically
        skeletonSprite.yScale *= -1
        
        // Apply rotation to match body orientation
        skeletonSprite.zRotation = rotation
        
        skeletonSprite.alpha = 0.9
        skeletonSprite.name = "full_skeleton_\(personIndex)"
        
        // Add color tint for X-ray effect (removed blur for clarity)
        let tintFilter = CIFilter(name: "CIColorMonochrome", parameters: [
            "inputColor": CIColor(red: 0.85, green: 0.92, blue: 1.0),
            "inputIntensity": 0.3
        ])
        
        let effectNode = SKEffectNode()
        effectNode.filter = tintFilter
        effectNode.shouldRasterize = true
        effectNode.addChild(skeletonSprite)
        
        // Add subtle animation
        let fadeAction = SKAction.sequence([
            SKAction.fadeAlpha(to: 0.75, duration: 1.5),
            SKAction.fadeAlpha(to: 0.95, duration: 1.5)
        ])
        effectNode.run(SKAction.repeatForever(fadeAction))
        
        scene.addChild(effectNode)
    }
    
    /// Calculate the position, scale, and rotation for the full skeleton
    private func calculateSkeletonTransform(
        keypoints: [CGPoint],
        confs: [Float],
        confThreshold: Float
    ) -> (position: CGPoint, scale: CGFloat, rotation: CGFloat)? {
        
        // Key indices: 0=nose, 5=L.shoulder, 6=R.shoulder, 11=L.hip, 12=R.hip, 15=L.ankle, 16=R.ankle
        let noseIdx = 0
        let leftShoulderIdx = 5
        let rightShoulderIdx = 6
        let leftHipIdx = 11
        let rightHipIdx = 12
        let leftAnkleIdx = 15
        let rightAnkleIdx = 16
        
        // Check if we have sufficient confident keypoints
        guard confs[leftShoulderIdx] >= confThreshold || confs[rightShoulderIdx] >= confThreshold,
              confs[leftHipIdx] >= confThreshold || confs[rightHipIdx] >= confThreshold else {
            return nil
        }
        
        // Calculate shoulder center
        var shoulderCenter = CGPoint.zero
        var shoulderCount = 0
        if confs[leftShoulderIdx] >= confThreshold {
            shoulderCenter.x += keypoints[leftShoulderIdx].x
            shoulderCenter.y += keypoints[leftShoulderIdx].y
            shoulderCount += 1
        }
        if confs[rightShoulderIdx] >= confThreshold {
            shoulderCenter.x += keypoints[rightShoulderIdx].x
            shoulderCenter.y += keypoints[rightShoulderIdx].y
            shoulderCount += 1
        }
        if shoulderCount > 0 {
            shoulderCenter.x /= CGFloat(shoulderCount)
            shoulderCenter.y /= CGFloat(shoulderCount)
        }
        
        // Calculate hip center
        var hipCenter = CGPoint.zero
        var hipCount = 0
        if confs[leftHipIdx] >= confThreshold {
            hipCenter.x += keypoints[leftHipIdx].x
            hipCenter.y += keypoints[leftHipIdx].y
            hipCount += 1
        }
        if confs[rightHipIdx] >= confThreshold {
            hipCenter.x += keypoints[rightHipIdx].x
            hipCenter.y += keypoints[rightHipIdx].y
            hipCount += 1
        }
        if hipCount > 0 {
            hipCenter.x /= CGFloat(hipCount)
            hipCenter.y /= CGFloat(hipCount)
        }
        
        // Calculate ankle center (for full body height)
        var ankleCenter = CGPoint.zero
        var ankleCount = 0
        if confs[leftAnkleIdx] >= confThreshold {
            ankleCenter.x += keypoints[leftAnkleIdx].x
            ankleCenter.y += keypoints[leftAnkleIdx].y
            ankleCount += 1
        }
        if confs[rightAnkleIdx] >= confThreshold {
            ankleCenter.x += keypoints[rightAnkleIdx].x
            ankleCenter.y += keypoints[rightAnkleIdx].y
            ankleCount += 1
        }
        if ankleCount > 0 {
            ankleCenter.x /= CGFloat(ankleCount)
            ankleCenter.y /= CGFloat(ankleCount)
        }
        
        // Calculate skeleton position (center between shoulders and hips)
        let torsoCenter = CGPoint(
            x: (shoulderCenter.x + hipCenter.x) / 2,
            y: (shoulderCenter.y + hipCenter.y) / 2
        )
        
        // Calculate torso height for scaling
        let torsoHeight = abs(shoulderCenter.y - hipCenter.y)
        
        // Full body height (if ankles are detected)
        var bodyHeight = torsoHeight * 2.5 // Default multiplier
        if ankleCount > 0 && confs[noseIdx] >= confThreshold {
            let fullHeight = abs(keypoints[noseIdx].y - ankleCenter.y)
            if fullHeight > torsoHeight {
                bodyHeight = fullHeight
            }
        }
        
        // Calculate scale based on detected body height
        // Skeleton image is 762 tall, torso is roughly 40% of that
        let skeletonHeight: CGFloat = 762
        let scale = bodyHeight / skeletonHeight
        
        // Calculate rotation based on shoulder-hip alignment
        let dx = hipCenter.x - shoulderCenter.x
        let dy = hipCenter.y - shoulderCenter.y
        let rotation = atan2(dx, dy) // Rotation in radians
        
        return (position: torsoCenter, scale: max(scale, 0.1), rotation: rotation)
    }
    
    /// Add atmospheric effects for X-ray appearance
    private func addAtmosphericEffects(to scene: SKScene) {
        // Add subtle dark vignette
        let vignette = SKShapeNode(rect: scene.frame)
        vignette.fillColor = .clear
        vignette.strokeColor = .black
        vignette.lineWidth = 80
        vignette.alpha = 0.25
        vignette.zPosition = 100
        vignette.blendMode = .multiply
        
        scene.addChild(vignette)
    }
}
