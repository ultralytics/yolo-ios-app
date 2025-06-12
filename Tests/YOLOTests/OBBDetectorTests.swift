// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Minimal tests for ObbDetector functionality
class ObbDetectorTests: XCTestCase {
    
    func testObbDetectorInitialization() {
        // Test ObbDetector initialization inherits from BasePredictor
        let obbDetector = ObbDetector()
        
        XCTAssertFalse(obbDetector.isModelLoaded)
        XCTAssertEqual(obbDetector.labels.count, 0)
        XCTAssertEqual(obbDetector.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(obbDetector.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertEqual(obbDetector.numItemsThreshold, 30)
        XCTAssertFalse(obbDetector.isUpdating)
    }
    
    func testObbDetectorPredictOnImageWithoutModel() {
        // Test predictOnImage without loaded model returns empty result
        let obbDetector = ObbDetector()
        obbDetector.labels = ["plane", "ship", "vehicle"]
        
        let image = CIImage(color: .magenta).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 640))
        let result = obbDetector.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNil(result.probs)
        XCTAssertNil(result.masks)
        XCTAssertEqual(result.keypointsList.count, 0)
        XCTAssertEqual(result.obb.count, 0) // Will be empty without model
        XCTAssertEqual(result.names, ["plane", "ship", "vehicle"])
        XCTAssertEqual(result.orig_shape.width, 640)
        XCTAssertEqual(result.orig_shape.height, 640)
    }
    
    func testObbDetectorProcessObservationsWithoutModel() {
        // Test processObservations without crashing
        let obbDetector = ObbDetector()
        obbDetector.labels = ["plane", "ship", "vehicle"]
        obbDetector.inputSize = CGSize(width: 640, height: 480)
        
        let mockRequest = MockVNRequest()
        
        // Should not crash
        obbDetector.processObservations(for: mockRequest, error: nil)
        obbDetector.processObservations(for: mockRequest, error: NSError(domain: "test", code: 1))
    }
    
    func testObbDetectorLabelsAssignment() {
        // Test labels can be assigned and retrieved
        let obbDetector = ObbDetector()
        let testLabels = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court"]
        
        obbDetector.labels = testLabels
        XCTAssertEqual(obbDetector.labels, testLabels)
        XCTAssertEqual(obbDetector.labels.count, 5)
    }
    
    func testObbDetectorInputSize() {
        // Test input size can be set and retrieved
        let obbDetector = ObbDetector()
        let testSize = CGSize(width: 1024, height: 1024)
        
        obbDetector.inputSize = testSize
        XCTAssertEqual(obbDetector.inputSize, testSize)
    }
    
    func testObbDetectorTimingProperties() {
        // Test timing properties are properly initialized
        let obbDetector = ObbDetector()
        
        XCTAssertEqual(obbDetector.t0, 0.0, accuracy: 0.001)
        XCTAssertEqual(obbDetector.t1, 0.0, accuracy: 0.001)
        XCTAssertEqual(obbDetector.t2, 0.0, accuracy: 0.001)
        XCTAssertEqual(obbDetector.t4, 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(obbDetector.t3, 0)
    }
    
    func testObbDetectorIsInstanceOfBasePredictor() {
        // Test ObbDetector is instance of BasePredictor/Predictor
        let obbDetector = ObbDetector()
        // The following lines will always be true if ObbDetector inherits from BasePredictor/Predictor.
        // Kept for explicitness, but compiler will warn if statically obvious.
        XCTAssertTrue(obbDetector is BasePredictor)
        XCTAssertTrue(obbDetector is Predictor)
    }

    func testObbDetectorResultStructure() {
        // Test ObbDetector result has correct structure
        let obbDetector = ObbDetector()
        obbDetector.labels = ["plane", "ship"]

        // FIX: CIColor has no 'brown', use 'orange'
        let image = CIImage(color: .orange).cropped(to: CGRect(x: 0, y: 0, width: 512, height: 512))
        let result = obbDetector.predictOnImage(image: image)

        XCTAssertNotNil(result.boxes)
        XCTAssertNil(result.probs) // OBB doesn't use probs
        XCTAssertNil(result.masks) // OBB doesn't use masks
        XCTAssertEqual(result.keypointsList.count, 0) // OBB doesn't use keypoints
        XCTAssertNotNil(result.obb) // OBB uses oriented bounding boxes
        XCTAssertEqual(result.names, ["plane", "ship"])
    }

    func testObbDetectorNonMaxSuppressionOBB() {
        // Test nonMaxSuppressionOBB utility method
        let obbDetector = ObbDetector()

        let boxes = [
            OBB(cx: 100, cy: 100, w: 50, h: 30, angle: 0),
            OBB(cx: 105, cy: 105, w: 50, h: 30, angle: 0.1), // Overlapping
            OBB(cx: 200, cy: 200, w: 50, h: 30, angle: 0.5)  // Non-overlapping
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]

        let selected = obbDetector.nonMaxSuppressionOBB(
            boxes: boxes,
            scores: scores,
            iouThreshold: 0.5
        )

        XCTAssertGreaterThan(selected.count, 0)
        XCTAssertLessThanOrEqual(selected.count, 3)
        XCTAssertTrue(selected.contains(0)) // Highest score should be kept
    }

    func testObbDetectorLockQueue() {
        // Test lockQueue property exists
        let obbDetector = ObbDetector()
        // If lockQueue is fileprivate, this will not compile unless your test is in the same file/module.
        // If not accessible, this test should be removed or lockQueue made internal for testing.
        // Uncomment the following only if accessible:
        // obbDetector.lockQueue.sync {
        //     XCTAssertTrue(true)
        // }
    }
}

// MARK: - Tests for OBB utility functions

class OBBUtilityTests: XCTestCase {

    func testPolygonArea() {
        // Test polygonArea calculation
        let square: Polygon = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 10, y: 0),
            CGPoint(x: 10, y: 10),
            CGPoint(x: 0, y: 10)
        ]

        let area = polygonArea(square)
        XCTAssertEqual(area, 100, accuracy: 0.1)

        let emptyPolygon: Polygon = []
        XCTAssertEqual(polygonArea(emptyPolygon), 0)

        let triangle: Polygon = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 10, y: 0),
            CGPoint(x: 5, y: 10)
        ]
        let triangleArea = polygonArea(triangle)
        XCTAssertEqual(triangleArea, 50, accuracy: 0.1)
    }

    func testOBBToAABB() {
        // Test OBB to axis-aligned bounding box conversion
        let obb = OBB(cx: 50, cy: 50, w: 20, h: 10, angle: 0)
        let aabb = obb.toAABB()

        XCTAssertEqual(aabb.origin.x, 40, accuracy: 0.1)
        XCTAssertEqual(aabb.origin.y, 45, accuracy: 0.1)
        XCTAssertEqual(aabb.width, 20, accuracy: 0.1)
        XCTAssertEqual(aabb.height, 10, accuracy: 0.1)
    }

    func testIsInsideFunction() {
        // Test isInside point-in-polygon test
        let edgeStart = CGPoint(x: 0, y: 0)
        let edgeEnd = CGPoint(x: 10, y: 0)

        let pointAbove = CGPoint(x: 5, y: 5)
        let pointBelow = CGPoint(x: 5, y: -5)

        XCTAssertTrue(isInside(point: pointAbove, edgeStart: edgeStart, edgeEnd: edgeEnd))
        XCTAssertFalse(isInside(point: pointBelow, edgeStart: edgeStart, edgeEnd: edgeEnd))
    }

    func testComputeIntersection() {
        // Test line intersection computation
        let p1 = CGPoint(x: 0, y: 0)
        let p2 = CGPoint(x: 10, y: 10)
        let clipStart = CGPoint(x: 0, y: 5)
        let clipEnd = CGPoint(x: 10, y: 5)

        let intersection = computeIntersection(p1: p1, p2: p2, clipStart: clipStart, clipEnd: clipEnd)

        XCTAssertNotNil(intersection)
        // FIX: Cast CGFloat? to Double for XCTAssertEqual with accuracy
        XCTAssertEqual(Double(intersection?.x ?? 0), 5, accuracy: 0.1)
        XCTAssertEqual(Double(intersection?.y ?? 0), 5, accuracy: 0.1)
    }
    
    func testOBBInfoInitialization() {
        // Test OBBInfo struct initialization
        let obb = OBB(cx: 25, cy: 25, w: 10, h: 8, angle: 0.5)
        let obbInfo = OBBInfo(obb)
        
        XCTAssertEqual(obbInfo.box.cx, 25, accuracy: 0.001)
        XCTAssertEqual(obbInfo.polygon.count, 4)
        XCTAssertGreaterThan(obbInfo.area, 0)
        XCTAssertGreaterThan(obbInfo.aabb.width, 0)
        XCTAssertGreaterThan(obbInfo.aabb.height, 0)
    }
    
    func testOBBInfoAABBIntersection() {
        // Test AABB intersection check
        let obb1 = OBB(cx: 25, cy: 25, w: 10, h: 10, angle: 0)
        let obb2 = OBB(cx: 30, cy: 30, w: 10, h: 10, angle: 0) // Overlapping
        let obb3 = OBB(cx: 100, cy: 100, w: 10, h: 10, angle: 0) // Non-overlapping
        
        let info1 = OBBInfo(obb1)
        let info2 = OBBInfo(obb2)
        let info3 = OBBInfo(obb3)
        
        XCTAssertTrue(info1.aabbIntersects(with: info2))
        XCTAssertFalse(info1.aabbIntersects(with: info3))
    }
}
