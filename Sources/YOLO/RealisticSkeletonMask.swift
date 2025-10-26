// RealisticSkeletonMask.swift
// Add this to your Sources/YOLO directory

import SpriteKit
import UIKit
import CoreGraphics

/// Creates a realistic human skeleton visualization using actual bone images
public class RealisticSkeletonMask {
    
    // Define bone mappings for a realistic skeleton
    private struct SkeletonBone {
        let imageName: String
        let startKeypoint: Int
        let endKeypoint: Int
        let width: CGFloat // Relative width
        let zPosition: CGFloat // Layer ordering
    }
    
    // Map YOLO keypoints to skeleton bones
    // YOLO keypoints: 0=nose, 1-2=eyes, 3-4=ears, 5-6=shoulders, 7-8=elbows, 
    // 9-10=wrists, 11-12=hips, 13-14=knees, 15-16=ankles
    private let skeletonBones: [SkeletonBone] = [
        // Skull
        SkeletonBone(imageName: "skull", startKeypoint: 0, endKeypoint: 0, width: 80, zPosition: 10),
        
        // Spine and ribcage
        SkeletonBone(imageName: "ribcage", startKeypoint: 5, endKeypoint: 11, width: 100, zPosition: 5),
        SkeletonBone(imageName: "spine", startKeypoint: 5, endKeypoint: 11, width: 30, zPosition: 4),
        
        // Arms - Right
        SkeletonBone(imageName: "clavicle_right", startKeypoint: 5, endKeypoint: 6, width: 25, zPosition: 6),
        SkeletonBone(imageName: "humerus", startKeypoint: 6, endKeypoint: 8, width: 20, zPosition: 7),
        SkeletonBone(imageName: "radius_ulna", startKeypoint: 8, endKeypoint: 10, width: 15, zPosition: 8),
        SkeletonBone(imageName: "hand", startKeypoint: 10, endKeypoint: 10, width: 20, zPosition: 9),
        
        // Arms - Left
        SkeletonBone(imageName: "clavicle_left", startKeypoint: 5, endKeypoint: 5, width: 25, zPosition: 6),
        SkeletonBone(imageName: "humerus", startKeypoint: 5, endKeypoint: 7, width: 20, zPosition: 7),
        SkeletonBone(imageName: "radius_ulna", startKeypoint: 7, endKeypoint: 9, width: 15, zPosition: 8),
        SkeletonBone(imageName: "hand", startKeypoint: 9, endKeypoint: 9, width: 20, zPosition: 9),
        
        // Pelvis
        SkeletonBone(imageName: "pelvis", startKeypoint: 11, endKeypoint: 12, width: 80, zPosition: 5),
        
        // Legs - Right
        SkeletonBone(imageName: "femur", startKeypoint: 12, endKeypoint: 14, width: 25, zPosition: 7),
        SkeletonBone(imageName: "tibia_fibula", startKeypoint: 14, endKeypoint: 16, width: 20, zPosition: 8),
        SkeletonBone(imageName: "foot", startKeypoint: 16, endKeypoint: 16, width: 25, zPosition: 9),
        
        // Legs - Left
        SkeletonBone(imageName: "femur", startKeypoint: 11, endKeypoint: 13, width: 25, zPosition: 7),
        SkeletonBone(imageName: "tibia_fibula", startKeypoint: 13, endKeypoint: 15, width: 20, zPosition: 8),
        SkeletonBone(imageName: "foot", startKeypoint: 15, endKeypoint: 15, width: 25, zPosition: 9)
    ]
    
    private var boneTextures: [String: SKTexture] = [:]
    
    public init() {
        loadBoneTextures()
    }
    
    /// Load or generate bone textures
    private func loadBoneTextures() {
        // Try to load actual bone images from bundle
        let boneNames = ["skull", "ribcage", "spine", "clavicle_right", "clavicle_left",
                        "humerus", "radius_ulna", "hand", "pelvis", "femur", 
                        "tibia_fibula", "foot"]
        
        for boneName in boneNames {
            if let image = UIImage(named: "skeleton_\(boneName)") {
                boneTextures[boneName] = SKTexture(image: image)
            } else {
                // Generate procedural bone texture if image not available
                boneTextures[boneName] = generateProceduralBoneTexture(for: boneName)
            }
        }
    }
    
    /// Generate procedural bone textures when images aren't available
    private func generateProceduralBoneTexture(for boneName: String) -> SKTexture {
        let size: CGSize
        switch boneName {
        case "skull":
            size = CGSize(width: 100, height: 120)
        case "ribcage":
            size = CGSize(width: 150, height: 180)
        case "pelvis":
            size = CGSize(width: 120, height: 80)
        case "hand", "foot":
            size = CGSize(width: 40, height: 50)
        default:
            size = CGSize(width: 30, height: 100)
        }
        
        UIGraphicsBeginImageContextWithOptions(size, false, 0)
        guard let context = UIGraphicsGetCurrentContext() else {
            return SKTexture()
        }
        
        // Draw bone-like shape
        context.setFillColor(UIColor(white: 0.95, alpha: 1.0).cgColor)
        
        switch boneName {
        case "skull":
            drawSkull(in: context, size: size)
        case "ribcage":
            drawRibcage(in: context, size: size)
        case "pelvis":
            drawPelvis(in: context, size: size)
        case "hand", "foot":
            drawExtremity(in: context, size: size)
        default:
            drawLongBone(in: context, size: size)
        }
        
        let image = UIGraphicsGetImageFromCurrentImageContext() ?? UIImage()
        UIGraphicsEndImageContext()
        
        return SKTexture(image: image)
    }
    
    /// Draw skull shape
    private func drawSkull(in context: CGContext, size: CGSize) {
        let skullPath = UIBezierPath(ovalIn: CGRect(x: size.width * 0.1, 
                                                    y: 0, 
                                                    width: size.width * 0.8, 
                                                    height: size.height * 0.7))
        
        // Jaw
        let jawPath = UIBezierPath()
        jawPath.move(to: CGPoint(x: size.width * 0.2, y: size.height * 0.6))
        jawPath.addQuadCurve(to: CGPoint(x: size.width * 0.8, y: size.height * 0.6),
                            controlPoint: CGPoint(x: size.width * 0.5, y: size.height * 0.9))
        
        // Eye sockets
        let leftEye = UIBezierPath(ovalIn: CGRect(x: size.width * 0.25, 
                                                  y: size.height * 0.3,
                                                  width: size.width * 0.2, 
                                                  height: size.height * 0.15))
        let rightEye = UIBezierPath(ovalIn: CGRect(x: size.width * 0.55, 
                                                   y: size.height * 0.3,
                                                   width: size.width * 0.2, 
                                                   height: size.height * 0.15))
        
        // Draw with bone color
        context.setFillColor(UIColor(red: 0.95, green: 0.92, blue: 0.88, alpha: 1.0).cgColor)
        context.addPath(skullPath.cgPath)
        context.fillPath()
        context.addPath(jawPath.cgPath)
        context.fillPath()
        
        // Draw eye sockets (dark)
        context.setFillColor(UIColor(white: 0.2, alpha: 0.8).cgColor)
        context.addPath(leftEye.cgPath)
        context.fillPath()
        context.addPath(rightEye.cgPath)
        context.fillPath()
    }
    
    /// Draw ribcage shape
    private func drawRibcage(in context: CGContext, size: CGSize) {
        context.setStrokeColor(UIColor(red: 0.95, green: 0.92, blue: 0.88, alpha: 1.0).cgColor)
        context.setLineWidth(3)
        
        // Draw ribs
        for i in 0..<8 {
            let y = CGFloat(i) * size.height / 8 + 10
            let width = size.width * (0.9 - CGFloat(i) * 0.05)
            let xOffset = (size.width - width) / 2
            
            let ribPath = UIBezierPath()
            ribPath.move(to: CGPoint(x: xOffset, y: y))
            ribPath.addQuadCurve(to: CGPoint(x: xOffset + width, y: y),
                                controlPoint: CGPoint(x: size.width / 2, y: y + 15))
            
            context.addPath(ribPath.cgPath)
            context.strokePath()
        }
        
        // Sternum
        context.move(to: CGPoint(x: size.width / 2, y: 0))
        context.addLine(to: CGPoint(x: size.width / 2, y: size.height * 0.8))
        context.strokePath()
    }
    
    /// Draw pelvis shape
    private func drawPelvis(in context: CGContext, size: CGSize) {
        let pelvisPath = UIBezierPath()
        
        // Ilium (hip bones)
        pelvisPath.move(to: CGPoint(x: 0, y: size.height * 0.3))
        pelvisPath.addCurve(to: CGPoint(x: size.width, y: size.height * 0.3),
                          controlPoint1: CGPoint(x: size.width * 0.2, y: 0),
                          controlPoint2: CGPoint(x: size.width * 0.8, y: 0))
        pelvisPath.addCurve(to: CGPoint(x: size.width * 0.7, y: size.height),
                          controlPoint1: CGPoint(x: size.width * 0.9, y: size.height * 0.6),
                          controlPoint2: CGPoint(x: size.width * 0.8, y: size.height * 0.9))
        pelvisPath.addLine(to: CGPoint(x: size.width * 0.3, y: size.height))
        pelvisPath.addCurve(to: CGPoint(x: 0, y: size.height * 0.3),
                          controlPoint1: CGPoint(x: size.width * 0.2, y: size.height * 0.9),
                          controlPoint2: CGPoint(x: size.width * 0.1, y: size.height * 0.6))
        
        context.setFillColor(UIColor(red: 0.95, green: 0.92, blue: 0.88, alpha: 1.0).cgColor)
        context.addPath(pelvisPath.cgPath)
        context.fillPath()
    }
    
    /// Draw long bone (femur, humerus, etc.)
    private func drawLongBone(in context: CGContext, size: CGSize) {
        let bonePath = UIBezierPath()
        
        // Bone shaft with enlarged ends (epiphyses)
        bonePath.move(to: CGPoint(x: size.width * 0.3, y: 0))
        bonePath.addLine(to: CGPoint(x: size.width * 0.7, y: 0))
        bonePath.addQuadCurve(to: CGPoint(x: size.width * 0.6, y: size.height * 0.1),
                              controlPoint: CGPoint(x: size.width * 0.8, y: size.height * 0.05))
        bonePath.addLine(to: CGPoint(x: size.width * 0.6, y: size.height * 0.9))
        bonePath.addQuadCurve(to: CGPoint(x: size.width * 0.7, y: size.height),
                              controlPoint: CGPoint(x: size.width * 0.8, y: size.height * 0.95))
        bonePath.addLine(to: CGPoint(x: size.width * 0.3, y: size.height))
        bonePath.addQuadCurve(to: CGPoint(x: size.width * 0.4, y: size.height * 0.9),
                              controlPoint: CGPoint(x: size.width * 0.2, y: size.height * 0.95))
        bonePath.addLine(to: CGPoint(x: size.width * 0.4, y: size.height * 0.1))
        bonePath.addQuadCurve(to: CGPoint(x: size.width * 0.3, y: 0),
                              controlPoint: CGPoint(x: size.width * 0.2, y: size.height * 0.05))
        bonePath.close()
        
        // Bone color with slight gradient
        context.setFillColor(UIColor(red: 0.95, green: 0.92, blue: 0.88, alpha: 1.0).cgColor)
        context.addPath(bonePath.cgPath)
        context.fillPath()
        
        // Add bone texture lines
        context.setStrokeColor(UIColor(red: 0.85, green: 0.82, blue: 0.78, alpha: 0.5).cgColor)
        context.setLineWidth(0.5)
        for i in stride(from: size.height * 0.2, to: size.height * 0.8, by: 5) {
            context.move(to: CGPoint(x: size.width * 0.4, y: i))
            context.addLine(to: CGPoint(x: size.width * 0.6, y: i))
        }
        context.strokePath()
    }
    
    /// Draw hand or foot
    private func drawExtremity(in context: CGContext, size: CGSize) {
        context.setFillColor(UIColor(red: 0.95, green: 0.92, blue: 0.88, alpha: 1.0).cgColor)
        
        // Palm/sole
        let mainPart = UIBezierPath(roundedRect: CGRect(x: size.width * 0.2, 
                                                       y: size.height * 0.3,
                                                       width: size.width * 0.6, 
                                                       height: size.height * 0.5),
                                   cornerRadius: 5)
        context.addPath(mainPart.cgPath)
        context.fillPath()
        
        // Fingers/toes (simplified)
        for i in 0..<5 {
            let fingerRect = CGRect(x: size.width * (0.15 + CGFloat(i) * 0.15),
                                   y: 0,
                                   width: size.width * 0.1,
                                   height: size.height * 0.35)
            let fingerPath = UIBezierPath(roundedRect: fingerRect, cornerRadius: 2)
            context.addPath(fingerPath.cgPath)
            context.fillPath()
        }
    }
    
    /// Create the skeleton scene
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
        
        // Add atmospheric background (optional)
        addAtmosphericEffects(to: scene)
        
        // Process each detected person
        for (personIndex, keypoints) in keypointsList.enumerated() {
            guard personIndex < confsList.count else { continue }
            
            let confs = confsList[personIndex]
            
            // Create skeleton for this person
            addRealisticSkeletonToScene(
                scene: scene,
                keypoints: keypoints,
                confs: confs,
                personIndex: personIndex,
                confThreshold: confThreshold
            )
        }
        
        return scene
    }
    
    /// Add a realistic skeleton to the scene
    private func addRealisticSkeletonToScene(
        scene: SKScene,
        keypoints: [(x: Float, y: Float)],
        confs: [Float],
        personIndex: Int,
        confThreshold: Float
    ) {
        // Convert keypoints to scene coordinates
        let sceneKeypoints = keypoints.map { kp in
            CGPoint(
                x: CGFloat(kp.x) * scene.size.width,
                y: scene.size.height - (CGFloat(kp.y) * scene.size.height)
            )
        }
        
        // Create skeleton container
        let skeletonNode = SKNode()
        skeletonNode.name = "realistic_skeleton_\(personIndex)"
        
        // Sort bones by z-position for proper layering
        let sortedBones = skeletonBones.sorted { $0.zPosition < $1.zPosition }
        
        // Add each bone
        for bone in sortedBones {
            guard bone.startKeypoint < sceneKeypoints.count,
                  bone.endKeypoint < sceneKeypoints.count,
                  bone.startKeypoint < confs.count,
                  bone.endKeypoint < confs.count else { continue }
            
            // Check confidence
            if bone.startKeypoint == bone.endKeypoint {
                // Single point bone (skull, hand, foot)
                if confs[bone.startKeypoint] >= confThreshold {
                    addSinglePointBone(bone, at: sceneKeypoints[bone.startKeypoint], to: skeletonNode)
                }
            } else {
                // Two-point bone
                if confs[bone.startKeypoint] >= confThreshold && 
                   confs[bone.endKeypoint] >= confThreshold {
                    addTwoPointBone(bone, 
                                  from: sceneKeypoints[bone.startKeypoint],
                                  to: sceneKeypoints[bone.endKeypoint],
                                  to: skeletonNode)
                }
            }
        }
        
        // Add X-ray effect
        addXRayEffect(to: skeletonNode)
        
        scene.addChild(skeletonNode)
    }
    
    /// Add single-point bone (skull, hands, feet)
    private func addSinglePointBone(_ bone: SkeletonBone, at point: CGPoint, to parent: SKNode) {
        guard let texture = boneTextures[bone.imageName] else { return }
        
        let boneSprite = SKSpriteNode(texture: texture)
        boneSprite.position = point
        boneSprite.size = CGSize(width: bone.width, height: bone.width * 1.2)
        boneSprite.zPosition = bone.zPosition
        boneSprite.alpha = 0.9
        
        // Add glow effect
        let glowNode = SKEffectNode()
        glowNode.shouldRasterize = true
        glowNode.filter = CIFilter(name: "CIGaussianBlur", parameters: ["inputRadius": 2.0])
        glowNode.addChild(boneSprite)
        
        parent.addChild(glowNode)
    }
    
    /// Add two-point bone (long bones)
    private func addTwoPointBone(_ bone: SkeletonBone, from start: CGPoint, to end: CGPoint, to parent: SKNode) {
        guard let texture = boneTextures[bone.imageName] else { return }
        
        let distance = hypot(end.x - start.x, end.y - start.y)
        let angle = atan2(end.y - start.y, end.x - start.x) - .pi / 2
        
        let boneSprite = SKSpriteNode(texture: texture)
        boneSprite.position = CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2)
        boneSprite.size = CGSize(width: bone.width, height: distance)
        boneSprite.zRotation = angle
        boneSprite.zPosition = bone.zPosition
        boneSprite.alpha = 0.9
        
        // Add subtle animation
        let fadeAction = SKAction.sequence([
            SKAction.fadeAlpha(to: 0.7, duration: 1.0),
            SKAction.fadeAlpha(to: 0.9, duration: 1.0)
        ])
        boneSprite.run(SKAction.repeatForever(fadeAction))
        
        parent.addChild(boneSprite)
    }
    
    /// Add X-ray/ghostly effect
    private func addXRayEffect(to node: SKNode) {
        // Add a subtle blue-white tint
        let tintNode = SKEffectNode()
        tintNode.filter = CIFilter(name: "CIColorMonochrome", parameters: [
            "inputColor": CIColor(red: 0.8, green: 0.9, blue: 1.0),
            "inputIntensity": 0.3
        ])
        
        // Move all children to effect node
        let children = node.children
        for child in children {
            child.removeFromParent()
            tintNode.addChild(child)
        }
        
        node.addChild(tintNode)
    }
    
    /// Add atmospheric effects to make it look more X-ray like
    private func addAtmosphericEffects(to scene: SKScene) {
        // Add subtle vignette
        let vignette = SKShapeNode(rect: scene.frame)
        vignette.fillColor = .clear
        vignette.strokeColor = .black
        vignette.lineWidth = 100
        vignette.alpha = 0.3
        vignette.zPosition = 100
        vignette.blendMode = .multiply
        
        scene.addChild(vignette)
    }
}