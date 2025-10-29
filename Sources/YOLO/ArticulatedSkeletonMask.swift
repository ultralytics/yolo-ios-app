// ArticulatedSkeletonMask.swift
// Articulated skeleton visualization using separate body part images

import Foundation
import SpriteKit
#if canImport(UIKit)
import UIKit
#endif

/// Creates a realistic articulated human skeleton visualization using separate body part sprites
public class ArticulatedSkeletonMask: SkeletonMask {
    
    // MARK: - Body Part Textures
    private struct BodyPartTextures {
        var head: SKTexture?
        var body: SKTexture?
        var pelvic: SKTexture?
        var leftArm: SKTexture?
        var rightArm: SKTexture?
        var leftForearm: SKTexture?
        var rightForearm: SKTexture?
        var leftHand: SKTexture?
        var rightHand: SKTexture?
        var leftThigh: SKTexture?
        var rightThigh: SKTexture?
        var leftShin: SKTexture?
        var rightShin: SKTexture?
        var leftFoot: SKTexture?
        var rightFoot: SKTexture?
    }
    
    private var textures = BodyPartTextures()
    
    public init() {
        loadBodyPartTextures()
    }
    
    /// Load all body part textures from Assets
    private func loadBodyPartTextures() {
        let textureNames = [
            ("Head", \BodyPartTextures.head),
            ("Body", \BodyPartTextures.body),
            ("Pelvic-bone", \BodyPartTextures.pelvic),
            ("Left-Arm", \BodyPartTextures.leftArm),
            ("Right-Arm", \BodyPartTextures.rightArm),
            ("Left-Forearm", \BodyPartTextures.leftForearm),
            ("Right-Forearm", \BodyPartTextures.rightForearm),
            ("Left-Hand", \BodyPartTextures.leftHand),
            ("Right-Hand", \BodyPartTextures.rightHand),
            ("Left-Thigh", \BodyPartTextures.leftThigh),
            ("Right-Thigh", \BodyPartTextures.rightThigh),
            ("Left-Shin", \BodyPartTextures.leftShin),
            ("Right-Shin", \BodyPartTextures.rightShin),
            ("Left-Foot", \BodyPartTextures.leftFoot),
            ("Right-Foot", \BodyPartTextures.rightFoot)
        ]
        
        for (name, keyPath) in textureNames {
            textures[keyPath: keyPath] = SkeletonUtilities.loadTexture(named: name)
        }
    }
    
    // MARK: - SkeletonMask Protocol
    
    /// Create a skeleton scene conforming to the SkeletonMask protocol
    public func createSkeletonScene(
        keypointsList: [[(x: Float, y: Float)]],
        confsList: [[Float]],
        boundingBoxes: [Box],
        sceneSize: CGSize,
        confThreshold: Float
    ) -> SKScene {
        return createArticulatedSkeletonScene(
            keypointsList: keypointsList,
            confsList: confsList,
            boundingBoxes: boundingBoxes,
            sceneSize: sceneSize,
            confThreshold: confThreshold
        )
    }
    
    /// Create the articulated skeleton scene
    public func createArticulatedSkeletonScene(
        keypointsList: [[(x: Float, y: Float)]],
        confsList: [[Float]],
        boundingBoxes: [Box],
        sceneSize: CGSize,
        confThreshold: Float = 0.25
    ) -> SKScene {
        let scene = SKScene(size: sceneSize)
        scene.backgroundColor = .clear
        scene.scaleMode = .aspectFill
        
        // Add atmospheric effects
        SkeletonUtilities.addVignetteEffect(to: scene)
        
        // Process each detected person
        for (personIndex, keypoints) in keypointsList.enumerated() {
            guard personIndex < confsList.count else { continue }
            
            addArticulatedSkeletonToScene(
                scene: scene,
                keypoints: keypoints,
                confs: confsList[personIndex],
                personIndex: personIndex,
                sceneSize: sceneSize,
                confThreshold: confThreshold
            )
        }
        
        return scene
    }
    
    // MARK: - Body Part Assembly
    
    /// Structure to hold extracted keypoint positions
    private struct KeypointPositions {
        let nose, leftShoulder, rightShoulder: CGPoint
        let leftElbow, rightElbow, leftWrist, rightWrist: CGPoint
        let leftHip, rightHip, leftKnee, rightKnee: CGPoint
        let leftAnkle, rightAnkle: CGPoint
        let shoulderCenter, hipCenter, torsoCenter: CGPoint
        let torsoHeight, shoulderWidth, hipWidth: CGFloat
    }
    
    /// Extract and calculate keypoint positions from raw keypoints
    private func extractKeypointPositions(
        from keypoints: [CGPoint]
    ) -> KeypointPositions {
        typealias K = SkeletonUtilities.KeypointIndex
        
        let nose = keypoints[K.nose.rawValue]
        let leftShoulder = keypoints[K.leftShoulder.rawValue]
        let rightShoulder = keypoints[K.rightShoulder.rawValue]
        let leftElbow = keypoints[K.leftElbow.rawValue]
        let rightElbow = keypoints[K.rightElbow.rawValue]
        let leftWrist = keypoints[K.leftWrist.rawValue]
        let rightWrist = keypoints[K.rightWrist.rawValue]
        let leftHip = keypoints[K.leftHip.rawValue]
        let rightHip = keypoints[K.rightHip.rawValue]
        let leftKnee = keypoints[K.leftKnee.rawValue]
        let rightKnee = keypoints[K.rightKnee.rawValue]
        let leftAnkle = keypoints[K.leftAnkle.rawValue]
        let rightAnkle = keypoints[K.rightAnkle.rawValue]
        
        let shoulderCenter = SkeletonUtilities.centerPoint(leftShoulder, rightShoulder)
        let hipCenter = SkeletonUtilities.centerPoint(leftHip, rightHip)
        let torsoCenter = SkeletonUtilities.centerPoint(shoulderCenter, hipCenter)
        
        let torsoHeight = SkeletonUtilities.distance(from: shoulderCenter, to: hipCenter)
        let shoulderWidth = SkeletonUtilities.distance(from: leftShoulder, to: rightShoulder)
        let hipWidth = SkeletonUtilities.distance(from: leftHip, to: rightHip)
        
        return KeypointPositions(
            nose: nose, leftShoulder: leftShoulder, rightShoulder: rightShoulder,
            leftElbow: leftElbow, rightElbow: rightElbow, leftWrist: leftWrist, rightWrist: rightWrist,
            leftHip: leftHip, rightHip: rightHip, leftKnee: leftKnee, rightKnee: rightKnee,
            leftAnkle: leftAnkle, rightAnkle: rightAnkle,
            shoulderCenter: shoulderCenter, hipCenter: hipCenter, torsoCenter: torsoCenter,
            torsoHeight: torsoHeight, shoulderWidth: shoulderWidth, hipWidth: hipWidth
        )
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
        guard keypoints.count >= 17 else { return }
        
        // Convert normalized keypoints to scene coordinates
        let sceneKeypoints = SkeletonUtilities.convertToSceneCoordinates(
            keypoints: keypoints,
            sceneSize: sceneSize
        )
        
        // Extract keypoint positions
        let positions = extractKeypointPositions(from: sceneKeypoints)
        
        // Create root container node for the entire skeleton
        let skeletonRoot = SKNode()
        skeletonRoot.position = positions.torsoCenter
        skeletonRoot.name = "skeleton_\(personIndex)"
        
        // Add torso components
        addTorsoComponents(to: skeletonRoot, positions: positions, confs: confs, confThreshold: confThreshold)
        
        // Add arms
        addArmChain(to: skeletonRoot, positions: positions, confs: confs, confThreshold: confThreshold, isLeft: true)
        addArmChain(to: skeletonRoot, positions: positions, confs: confs, confThreshold: confThreshold, isLeft: false)
        
        // Add legs
        addLegChain(to: skeletonRoot, positions: positions, confs: confs, confThreshold: confThreshold, isLeft: true)
        addLegChain(to: skeletonRoot, positions: positions, confs: confs, confThreshold: confThreshold, isLeft: false)
        
        // Wrap in X-ray effect and add to scene
        let effectNode = SkeletonUtilities.createXRayEffectNode()
        effectNode.addChild(skeletonRoot)
        SkeletonUtilities.addPulsingAnimation(to: effectNode)
        scene.addChild(effectNode)
    }
    
    // MARK: - Body Part Creation Helpers
    
    /// Add torso components (body, head, pelvis)
    private func addTorsoComponents(
        to parent: SKNode,
        positions: KeypointPositions,
        confs: [Float],
        confThreshold: Float
    ) {
        typealias K = SkeletonUtilities.KeypointIndex
        typealias C = SkeletonUtilities.Constants
        
        // Body (center torso)
        if let bodyTexture = textures.body,
           confs[K.leftShoulder.rawValue] >= confThreshold || confs[K.rightShoulder.rawValue] >= confThreshold {
            let bodyNode = createBodyPart(
                texture: bodyTexture,
                position: .zero,
                scale: positions.torsoHeight / bodyTexture.size().height * C.bodyScale,
                rotation: SkeletonUtilities.angle(from: positions.shoulderCenter, to: positions.hipCenter),
                name: "body"
            )
            parent.addChild(bodyNode)
        }
        
        // Head
        if let headTexture = textures.head, confs[K.nose.rawValue] >= confThreshold {
            let headOffset = CGPoint(
                x: positions.nose.x - positions.torsoCenter.x,
                y: positions.nose.y - positions.torsoCenter.y
            )
            let headNode = createBodyPart(
                texture: headTexture,
                position: headOffset,
                scale: positions.shoulderWidth / headTexture.size().width * C.headScale,
                rotation: 0,
                name: "head"
            )
            parent.addChild(headNode)
        }
        
        // Pelvic bone
        if let pelvicTexture = textures.pelvic,
           confs[K.leftHip.rawValue] >= confThreshold || confs[K.rightHip.rawValue] >= confThreshold {
            let pelvicOffset = CGPoint(
                x: positions.hipCenter.x - positions.torsoCenter.x,
                y: positions.hipCenter.y - positions.torsoCenter.y
            )
            let pelvicNode = createBodyPart(
                texture: pelvicTexture,
                position: pelvicOffset,
                scale: positions.hipWidth / pelvicTexture.size().width * C.pelvicScale,
                rotation: 0,
                name: "pelvis"
            )
            parent.addChild(pelvicNode)
        }
    }
    
    /// Add arm chain (upper arm, forearm, hand)
    private func addArmChain(
        to parent: SKNode,
        positions: KeypointPositions,
        confs: [Float],
        confThreshold: Float,
        isLeft: Bool
    ) {
        typealias K = SkeletonUtilities.KeypointIndex
        typealias C = SkeletonUtilities.Constants
        
        let shoulderIdx = isLeft ? K.leftShoulder.rawValue : K.rightShoulder.rawValue
        let elbowIdx = isLeft ? K.leftElbow.rawValue : K.rightElbow.rawValue
        let wristIdx = isLeft ? K.leftWrist.rawValue : K.rightWrist.rawValue
        
        let shoulder = isLeft ? positions.leftShoulder : positions.rightShoulder
        let elbow = isLeft ? positions.leftElbow : positions.rightElbow
        let wrist = isLeft ? positions.leftWrist : positions.rightWrist
        
        let armTexture = isLeft ? textures.leftArm : textures.rightArm
        let forearmTexture = isLeft ? textures.leftForearm : textures.rightForearm
        let handTexture = isLeft ? textures.leftHand : textures.rightHand
        
        let prefix = isLeft ? "left" : "right"
        
        guard confs[shoulderIdx] >= confThreshold else { return }
        
        // Upper arm
        if let armTexture = armTexture, confs[elbowIdx] >= confThreshold {
            let upperArmLength = SkeletonUtilities.distance(from: shoulder, to: elbow)
            let armNode = createLimbNode(
                texture: armTexture,
                startPoint: shoulder,
                endPoint: elbow,
                rootPosition: positions.torsoCenter,
                limbLength: upperArmLength,
                name: "\(prefix)_upper_arm"
            )
            parent.addChild(armNode)
            
            // Forearm
            if let forearmTexture = forearmTexture, confs[wristIdx] >= confThreshold {
                let forearmLength = SkeletonUtilities.distance(from: elbow, to: wrist)
                let forearmNode = createLimbNode(
                    texture: forearmTexture,
                    startPoint: elbow,
                    endPoint: wrist,
                    rootPosition: positions.torsoCenter,
                    limbLength: forearmLength,
                    name: "\(prefix)_forearm"
                )
                parent.addChild(forearmNode)
                
                // Hand
                if let handTexture = handTexture {
                    let handOffset = CGPoint(
                        x: wrist.x - positions.torsoCenter.x,
                        y: wrist.y - positions.torsoCenter.y
                    )
                    let handNode = createBodyPart(
                        texture: handTexture,
                        position: handOffset,
                        scale: forearmLength / handTexture.size().height * C.handScale,
                        rotation: SkeletonUtilities.angle(from: elbow, to: wrist),
                        name: "\(prefix)_hand"
                    )
                    parent.addChild(handNode)
                }
            }
        }
    }
    
    /// Add leg chain (thigh, shin, foot)
    private func addLegChain(
        to parent: SKNode,
        positions: KeypointPositions,
        confs: [Float],
        confThreshold: Float,
        isLeft: Bool
    ) {
        typealias K = SkeletonUtilities.KeypointIndex
        typealias C = SkeletonUtilities.Constants
        
        let hipIdx = isLeft ? K.leftHip.rawValue : K.rightHip.rawValue
        let kneeIdx = isLeft ? K.leftKnee.rawValue : K.rightKnee.rawValue
        let ankleIdx = isLeft ? K.leftAnkle.rawValue : K.rightAnkle.rawValue
        
        let hip = isLeft ? positions.leftHip : positions.rightHip
        let knee = isLeft ? positions.leftKnee : positions.rightKnee
        let ankle = isLeft ? positions.leftAnkle : positions.rightAnkle
        
        let thighTexture = isLeft ? textures.leftThigh : textures.rightThigh
        let shinTexture = isLeft ? textures.leftShin : textures.rightShin
        let footTexture = isLeft ? textures.leftFoot : textures.rightFoot
        
        let prefix = isLeft ? "left" : "right"
        
        guard confs[hipIdx] >= confThreshold else { return }
        
        // Thigh
        if let thighTexture = thighTexture, confs[kneeIdx] >= confThreshold {
            let thighLength = SkeletonUtilities.distance(from: hip, to: knee)
            let thighNode = createLimbNode(
                texture: thighTexture,
                startPoint: hip,
                endPoint: knee,
                rootPosition: positions.torsoCenter,
                limbLength: thighLength,
                name: "\(prefix)_thigh"
            )
            parent.addChild(thighNode)
            
            // Shin
            if let shinTexture = shinTexture, confs[ankleIdx] >= confThreshold {
                let shinLength = SkeletonUtilities.distance(from: knee, to: ankle)
                let shinNode = createLimbNode(
                    texture: shinTexture,
                    startPoint: knee,
                    endPoint: ankle,
                    rootPosition: positions.torsoCenter,
                    limbLength: shinLength,
                    name: "\(prefix)_shin"
                )
                parent.addChild(shinNode)
                
                // Foot
                if let footTexture = footTexture {
                    let footOffset = CGPoint(
                        x: ankle.x - positions.torsoCenter.x,
                        y: ankle.y - positions.torsoCenter.y
                    )
                    let footNode = createBodyPart(
                        texture: footTexture,
                        position: footOffset,
                        scale: shinLength / footTexture.size().height * C.footScale,
                        rotation: SkeletonUtilities.angle(from: knee, to: ankle),
                        name: "\(prefix)_foot"
                    )
                    parent.addChild(footNode)
                }
            }
        }
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
        let scale = limbLength / texture.size().height
        node.setScale(scale)
        
        // Rotate to match limb angle
        node.zRotation = SkeletonUtilities.angle(from: startPoint, to: endPoint)
        node.name = name
        
        return node
    }
}

