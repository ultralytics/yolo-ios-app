// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
import UIKit
@testable import YOLO

/// Comprehensive tests for error handling and edge cases
class YOLOErrorHandlingTests: XCTestCase {
    
    func testPredictorErrorTypes() {
        """Test PredictorError enum cases"""
        let errors: [PredictorError] = [
            .invalidTask,
            .noLabelsFound,
            .invalidUrl,
            .modelFileNotFound
        ]
        
        for error in errors {
            XCTAssertNotNil(error)
            // Test that errors can be used in switch statements
            switch error {
            case .invalidTask: break
            case .noLabelsFound: break
            case .invalidUrl: break
            case .modelFileNotFound: break
            }
        }
    }
    
    func testYOLOResultWithAllFields() {
        """Test YOLOResult with all optional fields populated"""
        let boxes = [Box(index: 0, cls: "person", conf: 0.9, xywh: CGRect(), xywhn: CGRect())]
        let masks = Masks(masks: [[[0.5]]], combinedMask: nil)
        let probs = Probs(top1: "cat", top5: ["cat"], top1Conf: 0.95, top5Confs: [0.95])
        let keypoints = Keypoints(xyn: [(0.5, 0.5)], xy: [(100, 100)], conf: [0.9])
        let obb = OBBResult(box: OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.1), confidence: 0.8, cls: "ship", index: 1)
        
        let result = YOLOResult(
            orig_shape: CGSize(width: 640, height: 480),
            boxes: boxes,
            masks: masks,
            probs: probs,
            keypointsList: [keypoints],
            obb: [obb],
            annotatedImage: nil,
            speed: 0.05,
            fps: 20.0,
            originalImage: nil,
            names: ["person", "ship", "cat"]
        )
        
        XCTAssertEqual(result.boxes.count, 1)
        XCTAssertNotNil(result.masks)
        XCTAssertNotNil(result.probs)
        XCTAssertEqual(result.keypointsList.count, 1)
        XCTAssertEqual(result.obb.count, 1)
        XCTAssertEqual(result.speed, 0.05, accuracy: 0.001)
        XCTAssertEqual(result.fps, 20.0, accuracy: 0.001)
        XCTAssertEqual(result.names.count, 3)
    }
    
    func testBoxNormalizedCoordinates() {
        """Test Box with normalized coordinates"""
        let box = Box(
            index: 2,
            cls: "bicycle",
            conf: 0.75,
            xywh: CGRect(x: 100, y: 50, width: 200, height: 150),
            xywhn: CGRect(x: 0.156, y: 0.104, width: 0.313, height: 0.313)
        )
        
        // Test that normalized coordinates are between 0 and 1
        XCTAssertLessThanOrEqual(box.xywhn.minX, 1.0)
        XCTAssertLessThanOrEqual(box.xywhn.minY, 1.0)
        XCTAssertLessThanOrEqual(box.xywhn.maxX, 1.0)
        XCTAssertLessThanOrEqual(box.xywhn.maxY, 1.0)
        XCTAssertGreaterThanOrEqual(box.xywhn.minX, 0.0)
        XCTAssertGreaterThanOrEqual(box.xywhn.minY, 0.0)
    }
    
    func testProbsEdgeCases() {
        """Test Probs with edge cases"""
        // Test with single class
        let singleProbs = Probs(top1: "only_class", top5: ["only_class"], top1Conf: 1.0, top5Confs: [1.0])
        XCTAssertEqual(singleProbs.top5.count, 1)
        XCTAssertEqual(singleProbs.top5Confs.count, 1)
        
        // Test with low confidence
        let lowConfProbs = Probs(top1: "uncertain", top5: ["uncertain"], top1Conf: 0.01, top5Confs: [0.01])
        XCTAssertEqual(lowConfProbs.top1Conf, 0.01, accuracy: 0.001)
        
        // Test modification
        var mutableProbs = Probs(top1: "initial", top5: ["initial"], top1Conf: 0.5, top5Confs: [0.5])
        mutableProbs.top1 = "modified"
        mutableProbs.top1Conf = 0.8
        XCTAssertEqual(mutableProbs.top1, "modified")
        XCTAssertEqual(mutableProbs.top1Conf, 0.8, accuracy: 0.001)
    }
    
    func testKeypointsWithMultiplePoints() {
        """Test Keypoints with realistic human pose data"""
        // 17 keypoints for human pose (COCO format)
        let xyn: [(x: Float, y: Float)] = (0..<17).map { i in (Float(i) * 0.05, Float(i) * 0.05) }
        let xy: [(x: Float, y: Float)] = (0..<17).map { i in (Float(i * 20), Float(i * 25)) }
        let conf: [Float] = (0..<17).map { _ in Float.random(in: 0.5...1.0) }
        
        let keypoints = Keypoints(xyn: xyn, xy: xy, conf: conf)
        
        XCTAssertEqual(keypoints.xyn.count, 17)
        XCTAssertEqual(keypoints.xy.count, 17)
        XCTAssertEqual(keypoints.conf.count, 17)
        
        // Test that normalized coordinates are within bounds
        for point in keypoints.xyn {
            XCTAssertGreaterThanOrEqual(point.x, 0.0)
            XCTAssertLessThanOrEqual(point.x, 1.0)
            XCTAssertGreaterThanOrEqual(point.y, 0.0)
            XCTAssertLessThanOrEqual(point.y, 1.0)
        }
        
        // Test that confidences are within bounds
        for confidence in keypoints.conf {
            XCTAssertGreaterThanOrEqual(confidence, 0.0)
            XCTAssertLessThanOrEqual(confidence, 1.0)
        }
    }
    
    func testOBBPolygonConversionWithRotation() {
        """Test OBB polygon conversion with various rotations"""
        let rotations: [Float] = [0, Float.pi/4, Float.pi/2, Float.pi, 3*Float.pi/2]
        
        for angle in rotations {
            let obb = OBB(cx: 0.5, cy: 0.5, w: 0.4, h: 0.2, angle: angle)
            let polygon = obb.toPolygon()
            
            XCTAssertEqual(polygon.count, 4, "OBB should always produce 4 corners")
            
            // Test that polygon points are different (not all the same)
            let uniquePoints = Set(polygon.map { "\($0.x),\($0.y)" })
            XCTAssertGreaterThan(uniquePoints.count, 1, "Polygon should have distinct corners")
        }
    }
    
    func testOBBAreaCalculation() {
        """Test OBB area calculation"""
        let testCases: [(w: Float, h: Float, expectedArea: Float)] = [
            (1.0, 1.0, 1.0),
            (2.0, 3.0, 6.0),
            (0.5, 0.8, 0.4),
            (10.0, 5.0, 50.0)
        ]
        
        for (w, h, expectedArea) in testCases {
            let obb = OBB(cx: 0, cy: 0, w: w, h: h, angle: 0)
            XCTAssertEqual(Float(obb.area), expectedArea, accuracy: 0.001)
        }
    }
    
    func testMasksWithEmptyData() {
        """Test Masks with edge cases"""
        // Empty masks
        let emptyMasks = Masks(masks: [], combinedMask: nil)
        XCTAssertEqual(emptyMasks.masks.count, 0)
        XCTAssertNil(emptyMasks.combinedMask)
        
        // Single pixel mask
        let singlePixelMasks = Masks(masks: [[[1.0]]], combinedMask: nil)
        XCTAssertEqual(singlePixelMasks.masks.count, 1)
        XCTAssertEqual(singlePixelMasks.masks[0].count, 1)
        XCTAssertEqual(singlePixelMasks.masks[0][0].count, 1)
        XCTAssertEqual(singlePixelMasks.masks[0][0][0], 1.0, accuracy: 0.001)
    }
    
    func testThresholdProviderEdgeValues() {
        """Test ThresholdProvider with edge values"""
        let minProvider = ThresholdProvider(iouThreshold: 0.0, confidenceThreshold: 0.0)
        let maxProvider = ThresholdProvider(iouThreshold: 1.0, confidenceThreshold: 1.0)
        
        XCTAssertEqual(minProvider.featureValue(for: "iouThreshold")?.doubleValue, 0.0)
        XCTAssertEqual(minProvider.featureValue(for: "confidenceThreshold")?.doubleValue, 0.0)
        
        XCTAssertEqual(maxProvider.featureValue(for: "iouThreshold")?.doubleValue, 1.0)
        XCTAssertEqual(maxProvider.featureValue(for: "confidenceThreshold")?.doubleValue, 1.0)
    }
    
    func testNonMaxSuppressionEdgeCases() {
        """Test NonMaxSuppression with edge cases"""
        // Single box
        let singleBox = [CGRect(x: 0, y: 0, width: 10, height: 10)]
        let singleScore: [Float] = [0.9]
        let singleResult = nonMaxSuppression(boxes: singleBox, scores: singleScore, threshold: 0.5)
        XCTAssertEqual(singleResult, [0])
        
        // No boxes
        let emptyResult = nonMaxSuppression(boxes: [], scores: [], threshold: 0.5)
        XCTAssertEqual(emptyResult.count, 0)
        
        // All boxes with same score
        let sameScoreBoxes = [
            CGRect(x: 0, y: 0, width: 10, height: 10),
            CGRect(x: 5, y: 5, width: 10, height: 10)
        ]
        let sameScores: [Float] = [0.5, 0.5]
        let sameScoreResult = nonMaxSuppression(boxes: sameScoreBoxes, scores: sameScores, threshold: 0.3)
        XCTAssertGreaterThan(sameScoreResult.count, 0)
    }
    
    func testYOLOTasksEquality() {
        """Test YOLOTask equality and hashing"""
        let allTasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        // Test equality
        for task in allTasks {
            XCTAssertEqual(task, task)
        }
        
        // Test inequality
        for i in 0..<allTasks.count {
            for j in 0..<allTasks.count {
                if i != j {
                    XCTAssertNotEqual(allTasks[i], allTasks[j])
                }
            }
        }
        
        // Test in Set (requires Hashable)
        let taskSet = Set(allTasks)
        XCTAssertEqual(taskSet.count, allTasks.count)
    }
    
    func testCGRectAreaExtension() {
        """Test CGRect area extension with various sizes"""
        let testRects = [
            (CGRect(x: 0, y: 0, width: 10, height: 5), 50.0),
            (CGRect(x: 5, y: 5, width: 3, height: 4), 12.0),
            (CGRect(x: 0, y: 0, width: 1, height: 1), 1.0),
            (CGRect.zero, 0.0)
        ]
        
        for (rect, expectedArea) in testRects {
            XCTAssertEqual(rect.area, expectedArea, accuracy: 0.001)
        }
    }
}

/// Tests for protocol conformance and interface compliance
class YOLOProtocolTests: XCTestCase {
    
    func testResultsListenerProtocol() {
        """Test ResultsListener protocol can be implemented"""
        class TestListener: ResultsListener {
            var receivedResult: YOLOResult?
            
            func on(result: YOLOResult) {
                receivedResult = result
            }
        }
        
        let listener = TestListener()
        let testResult = YOLOResult(orig_shape: CGSize(width: 100, height: 100), boxes: [], speed: 0.1, names: [])
        
        listener.on(result: testResult)
        XCTAssertNotNil(listener.receivedResult)
    }
    
    func testInferenceTimeListenerProtocol() {
        """Test InferenceTimeListener protocol can be implemented"""
        class TestListener: InferenceTimeListener {
            var lastInferenceTime: Double = 0
            var lastFpsRate: Double = 0
            
            func on(inferenceTime: Double, fpsRate: Double) {
                lastInferenceTime = inferenceTime
                lastFpsRate = fpsRate
            }
        }
        
        let listener = TestListener()
        listener.on(inferenceTime: 25.5, fpsRate: 30.0)
        
        XCTAssertEqual(listener.lastInferenceTime, 25.5)
        XCTAssertEqual(listener.lastFpsRate, 30.0)
    }
    
    func testPredictorProtocol() {
        """Test Predictor protocol interface"""
        class TestPredictor: Predictor {
            var labels: [String] = ["test"]
            var isUpdating: Bool = false
            
            func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?) {
                // Mock implementation
            }
            
            func predictOnImage(image: CIImage) -> YOLOResult {
                return YOLOResult(orig_shape: image.extent.size, boxes: [], speed: 0, names: labels)
            }
        }
        
        let predictor = TestPredictor()
        XCTAssertEqual(predictor.labels, ["test"])
        XCTAssertFalse(predictor.isUpdating)
        
        let testImage = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let result = predictor.predictOnImage(image: testImage)
        XCTAssertEqual(result.names, ["test"])
    }
}
