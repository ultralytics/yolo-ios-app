// ArticulatedSkeletonMask.swift
// Articulated skeleton visualization using separate body part images

import SpriteKit
import UIKit
import CoreGraphics

/// Creates a realistic articulated human skeleton visualization using separate body part sprites
public class ArticulatedSkeletonMask {
    
    // Textures for body parts
    private var headTexture: SKTexture?
    private var bodyTexture: SKTexture?
    private var pelvicTexture: SKTexture?
    private var leftArmTexture: SKTexture?
    private var rightArmTexture: SKTexture?
    private var leftForearmTexture: SKTexture?
    private var rightForearmTexture: SKTexture?
    private var leftHandTexture: SKTexture?
    private var rightHandTexture: SKTexture?
    private var leftThighTexture: SKTexture?
    private var rightThighTexture: SKTexture?
    private var leftShinTexture: SKTexture?
    private var rightShinTexture: SKTexture?
    private var leftFootTexture: SKTexture?
    private var rightFootTexture: SKTexture?
    
    public init() {
        loadBodyPartTextures()
    }
    
    /// Load all body part textures from Assets
    private func loadBodyPartTextures() {
        headTexture = loadTexture(named: "Head")
        bodyTexture = loadTexture(named: "Body")
        pelvicTexture = loadTexture(named: "Pelvic-bone")
        leftArmTexture = loadTexture(named: "Left-Arm")
        rightArmTexture = loadTexture(named: "Right-Arm")
        leftForearmTexture = loadTexture(named: "Left-Forearm")
        rightForearmTexture = loadTexture(named: "Right-Forearm")
        leftHandTexture = loadTexture(named: "Left-Hand")
        rightHandTexture = loadTexture(named: "Right-Hand")
        leftThighTexture = loadTexture(named: "Left-Thigh")
        rightThighTexture = loadTexture(named: "Right-Thigh")
        leftShinTexture = loadTexture(named: "Left-Shin")
        rightShinTexture = loadTexture(named: "Right-Shin")
        leftFootTexture = loadTexture(named: "Left-Foot")
        rightFootTexture = loadTexture(named: "Right-Foot")
    }
    
    /// Helper to load a texture with error handling
    private func loadTexture(named name: String) -> SKTexture? {
        // Try loading from main bundle first (for app assets)
        if let image = UIImage(named: name, in: Bundle.main, compatibleWith: nil) {
            print("✅ Loaded texture from main bundle: \(name) - size: \(image.size)")
            return SKTexture(image: image)
        }
        // Fallback to default bundle
        else if let image = UIImage(named: name) {
            print("✅ Loaded texture from default: \(name) - size: \(image.size)")
            return SKTexture(image: image)
        } else {
            print("❌ Failed to load texture: \(name) - checked main bundle and default")
            return nil
        }
    }
    
    /// Create the articulated skeleton scene
    public func createArticulatedSkeletonScene(
        keypointsList: [[(x: Float, y: Float)]],
        confsList: [[Float]],
        boundingBoxes: [Box],
        sceneSize: CGSize,
        confThreshold: Float = 0.25
    ) -> SKScene {
        print("🦴 Creating articulated skeleton scene for \(keypointsList.count) person(s)")
        let scene = SKScene(size: sceneSize)
        scene.backgroundColor = .clear
        scene.scaleMode = .aspectFill
        
        // Add atmospheric effects
        addAtmosphericEffects(to: scene)
        
        // Process each detected person
        for (personIndex, keypoints) in keypointsList.enumerated() {
            guard personIndex < confsList.count else { continue }
            
            let confs = confsList[personIndex]
            print("🦴 Processing person \(personIndex) with \(keypoints.count) keypoints")
            
            // Build articulated skeleton for this person
            addArticulatedSkeletonToScene(
                scene: scene,
                keypoints: keypoints,
                confs: confs,
                personIndex: personIndex,
                sceneSize: sceneSize,
                confThreshold: confThreshold
            )
        }
        
        print("🦴 Skeleton scene created with \(scene.children.count) child nodes")
        return scene
    }
    
    /// Add an articulated skeleton for a detected person
    private func addArticulatedSkeletonToScene(
        scene: SKScene,
        keypoints: [(x: Float, y: Float)],
        confs: [Float],
        personIndex: Int,
        sceneSize: CGSize,
        confThreshold: Float
    ) {
        // YOLO Pose keypoint indices:
        // 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        // 5=left_shoulder, 6=right_shoulder
        // 7=left_elbow, 8=right_elbow
        // 9=left_wrist, 10=right_wrist
        // 11=left_hip, 12=right_hip
        // 13=left_knee, 14=right_knee
        // 15=left_ankle, 16=right_ankle
        
        guard keypoints.count >= 17 else { return }
        
        // Convert normalized keypoints to scene coordinates (flip Y for SpriteKit)
        let sceneKeypoints = keypoints.map { kp in
            CGPoint(
                x: CGFloat(kp.x) * sceneSize.width,
                y: sceneSize.height - (CGFloat(kp.y) * sceneSize.height)
            )
        }
        
        // Extract key points
        let nose = sceneKeypoints[0]
        let leftShoulder = sceneKeypoints[5]
        let rightShoulder = sceneKeypoints[6]
        let leftElbow = sceneKeypoints[7]
        let rightElbow = sceneKeypoints[8]
        let leftWrist = sceneKeypoints[9]
        let rightWrist = sceneKeypoints[10]
        let leftHip = sceneKeypoints[11]
        let rightHip = sceneKeypoints[12]
        let leftKnee = sceneKeypoints[13]
        let rightKnee = sceneKeypoints[14]
        let leftAnkle = sceneKeypoints[15]
        let rightAnkle = sceneKeypoints[16]
        
        // Calculate centers
        let shoulderCenter = CGPoint(
            x: (leftShoulder.x + rightShoulder.x) / 2,
            y: (leftShoulder.y + rightShoulder.y) / 2
        )
        
        let hipCenter = CGPoint(
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        )
        
        let torsoCenter = CGPoint(
            x: (shoulderCenter.x + hipCenter.x) / 2,
            y: (shoulderCenter.y + hipCenter.y) / 2
        )
        
        // Calculate torso height for scaling
        let torsoHeight = distance(from: shoulderCenter, to: hipCenter)
        let shoulderWidth = distance(from: leftShoulder, to: rightShoulder)
        
        // Create root container node for the entire skeleton
        let skeletonRoot = SKNode()
        skeletonRoot.position = torsoCenter
        skeletonRoot.name = "skeleton_\(personIndex)"
        
        // 1. BODY (center torso)
        if let bodyTexture = bodyTexture, confs[5] >= confThreshold || confs[6] >= confThreshold {
            let bodyNode = createBodyPart(
                texture: bodyTexture,
                position: .zero, // relative to root
                scale: torsoHeight / bodyTexture.size().height * 1.2,
                rotation: calculateAngle(from: shoulderCenter, to: hipCenter),
                name: "body"
            )
            skeletonRoot.addChild(bodyNode)
        }
        
        // 2. HEAD
        if let headTexture = headTexture, confs[0] >= confThreshold {
            let headOffset = CGPoint(
                x: nose.x - torsoCenter.x,
                y: nose.y - torsoCenter.y
            )
            let headNode = createBodyPart(
                texture: headTexture,
                position: headOffset,
                scale: shoulderWidth / headTexture.size().width * 0.8,
                rotation: 0,
                name: "head"
            )
            skeletonRoot.addChild(headNode)
        }
        
        // 3. PELVIC BONE
        if let pelvicTexture = pelvicTexture, confs[11] >= confThreshold || confs[12] >= confThreshold {
            let pelvicOffset = CGPoint(
                x: hipCenter.x - torsoCenter.x,
                y: hipCenter.y - torsoCenter.y
            )
            let hipWidth = distance(from: leftHip, to: rightHip)
            let pelvicNode = createBodyPart(
                texture: pelvicTexture,
                position: pelvicOffset,
                scale: hipWidth / pelvicTexture.size().width * 1.2,
                rotation: 0,
                name: "pelvis"
            )
            skeletonRoot.addChild(pelvicNode)
        }
        
        // 4. LEFT ARM CHAIN
        if confs[5] >= confThreshold {
            // Upper arm
            if let leftArmTexture = leftArmTexture, confs[7] >= confThreshold {
                let upperArmLength = distance(from: leftShoulder, to: leftElbow)
                let leftArmNode = createLimbNode(
                    texture: leftArmTexture,
                    startPoint: leftShoulder,
                    endPoint: leftElbow,
                    rootPosition: torsoCenter,
                    limbLength: upperArmLength,
                    name: "left_upper_arm"
                )
                skeletonRoot.addChild(leftArmNode)
                
                // Forearm
                if let leftForearmTexture = leftForearmTexture, confs[9] >= confThreshold {
                    let forearmLength = distance(from: leftElbow, to: leftWrist)
                    let leftForearmNode = createLimbNode(
                        texture: leftForearmTexture,
                        startPoint: leftElbow,
                        endPoint: leftWrist,
                        rootPosition: torsoCenter,
                        limbLength: forearmLength,
                        name: "left_forearm"
                    )
                    skeletonRoot.addChild(leftForearmNode)
                    
                    // Hand
                    if let leftHandTexture = leftHandTexture {
                        let handOffset = CGPoint(
                            x: leftWrist.x - torsoCenter.x,
                            y: leftWrist.y - torsoCenter.y
                        )
                        let handNode = createBodyPart(
                            texture: leftHandTexture,
                            position: handOffset,
                            scale: forearmLength / leftHandTexture.size().height * 0.5,
                            rotation: calculateAngle(from: leftElbow, to: leftWrist),
                            name: "left_hand"
                        )
                        skeletonRoot.addChild(handNode)
                    }
                }
            }
        }
        
        // 5. RIGHT ARM CHAIN
        if confs[6] >= confThreshold {
            // Upper arm
            if let rightArmTexture = rightArmTexture, confs[8] >= confThreshold {
                let upperArmLength = distance(from: rightShoulder, to: rightElbow)
                let rightArmNode = createLimbNode(
                    texture: rightArmTexture,
                    startPoint: rightShoulder,
                    endPoint: rightElbow,
                    rootPosition: torsoCenter,
                    limbLength: upperArmLength,
                    name: "right_upper_arm"
                )
                skeletonRoot.addChild(rightArmNode)
                
                // Forearm
                if let rightForearmTexture = rightForearmTexture, confs[10] >= confThreshold {
                    let forearmLength = distance(from: rightElbow, to: rightWrist)
                    let rightForearmNode = createLimbNode(
                        texture: rightForearmTexture,
                        startPoint: rightElbow,
                        endPoint: rightWrist,
                        rootPosition: torsoCenter,
                        limbLength: forearmLength,
                        name: "right_forearm"
                    )
                    skeletonRoot.addChild(rightForearmNode)
                    
                    // Hand
                    if let rightHandTexture = rightHandTexture {
                        let handOffset = CGPoint(
                            x: rightWrist.x - torsoCenter.x,
                            y: rightWrist.y - torsoCenter.y
                        )
                        let handNode = createBodyPart(
                            texture: rightHandTexture,
                            position: handOffset,
                            scale: forearmLength / rightHandTexture.size().height * 0.5,
                            rotation: calculateAngle(from: rightElbow, to: rightWrist),
                            name: "right_hand"
                        )
                        skeletonRoot.addChild(handNode)
                    }
                }
            }
        }
        
        // 6. LEFT LEG CHAIN
        if confs[11] >= confThreshold {
            // Thigh
            if let leftThighTexture = leftThighTexture, confs[13] >= confThreshold {
                let thighLength = distance(from: leftHip, to: leftKnee)
                let leftThighNode = createLimbNode(
                    texture: leftThighTexture,
                    startPoint: leftHip,
                    endPoint: leftKnee,
                    rootPosition: torsoCenter,
                    limbLength: thighLength,
                    name: "left_thigh"
                )
                skeletonRoot.addChild(leftThighNode)
                
                // Shin
                if let leftShinTexture = leftShinTexture, confs[15] >= confThreshold {
                    let shinLength = distance(from: leftKnee, to: leftAnkle)
                    let leftShinNode = createLimbNode(
                        texture: leftShinTexture,
                        startPoint: leftKnee,
                        endPoint: leftAnkle,
                        rootPosition: torsoCenter,
                        limbLength: shinLength,
                        name: "left_shin"
                    )
                    skeletonRoot.addChild(leftShinNode)
                    
                    // Foot
                    if let leftFootTexture = leftFootTexture {
                        let footOffset = CGPoint(
                            x: leftAnkle.x - torsoCenter.x,
                            y: leftAnkle.y - torsoCenter.y
                        )
                        let footNode = createBodyPart(
                            texture: leftFootTexture,
                            position: footOffset,
                            scale: shinLength / leftFootTexture.size().height * 0.5,
                            rotation: calculateAngle(from: leftKnee, to: leftAnkle),
                            name: "left_foot"
                        )
                        skeletonRoot.addChild(footNode)
                    }
                }
            }
        }
        
        // 7. RIGHT LEG CHAIN
        if confs[12] >= confThreshold {
            // Thigh
            if let rightThighTexture = rightThighTexture, confs[14] >= confThreshold {
                let thighLength = distance(from: rightHip, to: rightKnee)
                let rightThighNode = createLimbNode(
                    texture: rightThighTexture,
                    startPoint: rightHip,
                    endPoint: rightKnee,
                    rootPosition: torsoCenter,
                    limbLength: thighLength,
                    name: "right_thigh"
                )
                skeletonRoot.addChild(rightThighNode)
                
                // Shin
                if let rightShinTexture = rightShinTexture, confs[16] >= confThreshold {
                    let shinLength = distance(from: rightKnee, to: rightAnkle)
                    let rightShinNode = createLimbNode(
                        texture: rightShinTexture,
                        startPoint: rightKnee,
                        endPoint: rightAnkle,
                        rootPosition: torsoCenter,
                        limbLength: shinLength,
                        name: "right_shin"
                    )
                    skeletonRoot.addChild(rightShinNode)
                    
                    // Foot
                    if let rightFootTexture = rightFootTexture {
                        let footOffset = CGPoint(
                            x: rightAnkle.x - torsoCenter.x,
                            y: rightAnkle.y - torsoCenter.y
                        )
                        let footNode = createBodyPart(
                            texture: rightFootTexture,
                            position: footOffset,
                            scale: shinLength / rightFootTexture.size().height * 0.5,
                            rotation: calculateAngle(from: rightKnee, to: rightAnkle),
                            name: "right_foot"
                        )
                        skeletonRoot.addChild(footNode)
                    }
                }
            }
        }
        
        // Add X-ray glow effect
        let effectNode = SKEffectNode()
        effectNode.shouldRasterize = true
        
        // Color tint for X-ray effect
        let tintFilter = CIFilter(name: "CIColorMonochrome", parameters: [
            "inputColor": CIColor(red: 0.9, green: 0.95, blue: 1.0),
            "inputIntensity": 0.4
        ])
        effectNode.filter = tintFilter
        
        // Move skeleton to effect node
        effectNode.addChild(skeletonRoot)
        effectNode.alpha = 0.95
        
        // Add subtle pulsing animation
        let fadeAction = SKAction.sequence([
            SKAction.fadeAlpha(to: 0.85, duration: 1.5),
            SKAction.fadeAlpha(to: 0.95, duration: 1.5)
        ])
        effectNode.run(SKAction.repeatForever(fadeAction))
        
        scene.addChild(effectNode)
    }
    
    /// Create a simple body part sprite node
    private func createBodyPart(
        texture: SKTexture,
        position: CGPoint,
        scale: CGFloat,
        rotation: CGFloat,
        name: String
    ) -> SKSpriteNode {
        let node = SKSpriteNode(texture: texture)
        node.position = position
        node.setScale(scale)
        node.zRotation = rotation
        node.name = name
        return node
    }
    
    /// Create a limb node (bones that connect joints)
    private func createLimbNode(
        texture: SKTexture,
        startPoint: CGPoint,
        endPoint: CGPoint,
        rootPosition: CGPoint,
        limbLength: CGFloat,
        name: String
    ) -> SKSpriteNode {
        let node = SKSpriteNode(texture: texture)
        
        // Position at the midpoint of the limb
        let midPoint = CGPoint(
            x: (startPoint.x + endPoint.x) / 2 - rootPosition.x,
            y: (startPoint.y + endPoint.y) / 2 - rootPosition.y
        )
        node.position = midPoint
        
        // Scale to match limb length
        let textureHeight = texture.size().height
        let scale = limbLength / textureHeight
        node.setScale(scale)
        
        // Rotate to match limb angle
        let angle = calculateAngle(from: startPoint, to: endPoint)
        node.zRotation = angle
        
        node.name = name
        return node
    }
    
    /// Calculate angle between two points (for rotation)
    private func calculateAngle(from start: CGPoint, to end: CGPoint) -> CGFloat {
        let dx = end.x - start.x
        let dy = end.y - start.y
        // Subtract π/2 because sprites are oriented vertically by default
        return atan2(dy, dx) - .pi / 2
    }
    
    /// Calculate distance between two points
    private func distance(from start: CGPoint, to end: CGPoint) -> CGFloat {
        let dx = end.x - start.x
        let dy = end.y - start.y
        return sqrt(dx * dx + dy * dy)
    }
    
    /// Add atmospheric effects for X-ray appearance
    private func addAtmosphericEffects(to scene: SKScene) {
        // Add subtle dark vignette
        let vignette = SKShapeNode(rect: scene.frame)
        vignette.fillColor = .clear
        vignette.strokeColor = .black
        vignette.lineWidth = 60
        vignette.alpha = 0.2
        vignette.zPosition = 100
        vignette.blendMode = .multiply
        
        scene.addChild(vignette)
    }
}

