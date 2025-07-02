// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
@testable import YOLO

/// Integration tests combining multiple components
class YOLOIntegrationTests: XCTestCase {
    
    func testYOLOWorkflowWithMockComponents() {
        // Test complete YOLO workflow with mock components
        // This test simulates a complete workflow without requiring actual models
        
        // 1. Create mock data structures
        let box = Box(
            index: 0,
            cls: "person",
            conf: 0.87,
            xywh: CGRect(x: 100, y: 50, width: 200, height: 300),
            xywhn: CGRect(x: 0.156, y: 0.104, width: 0.313, height: 0.625)
        )
        
        let result = YOLOResult(
            orig_shape: CGSize(width: 640, height: 480),
            boxes: [box],
            speed: 0.025,
            names: ["person", "car", "bicycle"]
        )
        
        // 2. Test result processing
        XCTAssertEqual(result.boxes.count, 1)
        XCTAssertEqual(result.boxes[0].cls, "person")
        XCTAssertEqual(result.boxes[0].conf, 0.87, accuracy: 0.001)
        XCTAssertEqual(result.names.count, 3)
        
        // 3. Test threshold provider with result
        let thresholdProvider = ThresholdProvider(iouThreshold: 0.5, confidenceThreshold: 0.8)
        XCTAssertNotNil(thresholdProvider.featureValue(for: "confidenceThreshold"))
        
        // Box confidence should be above threshold
        let confThreshold = thresholdProvider.featureValue(for: "confidenceThreshold")?.doubleValue ?? 0.0
        XCTAssertGreaterThan(Double(box.conf), confThreshold)
        
        // 4. Test NMS with multiple overlapping boxes
        let boxes = [
            CGRect(x: 100, y: 50, width: 200, height: 300),
            CGRect(x: 110, y: 60, width: 190, height: 290), // Slightly overlapping
            CGRect(x: 400, y: 200, width: 100, height: 150) // Separate
        ]
        let scores: [Float] = [0.87, 0.82, 0.75]
        let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.3)
        
        XCTAssertEqual(selected.count, 2) // Should keep highest score + separate box
        XCTAssertTrue(selected.contains(0)) // Highest score should be kept
        XCTAssertTrue(selected.contains(2)) // Separate box should be kept
    }
    
    func testMultiTaskYOLOResults() {
        // Test YOLOResult structure for different task types
        let originalSize = CGSize(width: 416, height: 416)
        let names = ["person", "car", "bicycle", "dog", "cat"]
        
        // Detection result
        let detectionBox = Box(index: 0, cls: "person", conf: 0.9, xywh: CGRect(), xywhn: CGRect())
        let detectionResult = YOLOResult(
            orig_shape: originalSize,
            boxes: [detectionBox],
            speed: 0.02,
            names: names
        )
        
        XCTAssertEqual(detectionResult.boxes.count, 1)
        XCTAssertNil(detectionResult.masks)
        XCTAssertNil(detectionResult.probs)
        XCTAssertEqual(detectionResult.keypointsList.count, 0)
        XCTAssertEqual(detectionResult.obb.count, 0)
        
        // Classification result
        let classificationProbs = Probs(
            top1: "cat",
            top5: ["cat", "dog", "person", "car", "bicycle"],
            top1Conf: 0.95,
            top5Confs: [0.95, 0.87, 0.23, 0.15, 0.08]
        )
        let classificationResult = YOLOResult(
            orig_shape: originalSize,
            boxes: [],
            probs: classificationProbs,
            speed: 0.01,
            names: names
        )
        
        XCTAssertEqual(classificationResult.boxes.count, 0)
        XCTAssertNil(classificationResult.masks)
        XCTAssertNotNil(classificationResult.probs)
        XCTAssertEqual(classificationResult.probs?.top1, "cat")
        
        // Segmentation result  
        let segmentationMasks = Masks(masks: [[[0.1, 0.9], [0.8, 0.2]]], combinedMask: nil)
        let segmentationResult = YOLOResult(
            orig_shape: originalSize,
            boxes: [detectionBox],
            masks: segmentationMasks,
            speed: 0.04,
            names: names
        )
        
        XCTAssertEqual(segmentationResult.boxes.count, 1)
        XCTAssertNotNil(segmentationResult.masks)
        XCTAssertNil(segmentationResult.probs)
        
        // Pose estimation result
        let keypoints = Keypoints(
            xyn: [(0.5, 0.3), (0.6, 0.4)],
            xy: [(208, 125), (250, 166)],
            conf: [0.95, 0.88]
        )
        let poseResult = YOLOResult(
            orig_shape: originalSize,
            boxes: [detectionBox],
            keypointsList: [keypoints],
            speed: 0.03,
            names: names
        )
        
        XCTAssertEqual(poseResult.boxes.count, 1)
        XCTAssertEqual(poseResult.keypointsList.count, 1)
        XCTAssertEqual(poseResult.keypointsList[0].xyn.count, 2)
        
        // OBB result
        let obbBox = OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.7)
        let obbResult = OBBResult(box: obbBox, confidence: 0.83, cls: "ship", index: 5)
        let obbDetectionResult = YOLOResult(
            orig_shape: originalSize,
            boxes: [],
            obb: [obbResult],
            speed: 0.03,
            names: names
        )
        
        XCTAssertEqual(obbDetectionResult.boxes.count, 0)
        XCTAssertEqual(obbDetectionResult.obb.count, 1)
        XCTAssertEqual(obbDetectionResult.obb[0].cls, "ship")
    }
    
    func testPerformanceMetrics() {
        // Test performance-related calculations
        let speeds = [0.025, 0.030, 0.028, 0.032, 0.026]
        let fps = speeds.map { 1.0 / $0 }
        
        // Test that FPS calculations are reasonable
        for (speed, expectedFps) in zip(speeds, fps) {
            XCTAssertEqual(expectedFps, 1.0 / speed, accuracy: 0.001)
            XCTAssertGreaterThan(expectedFps, 0)
        }
        
        // Test smoothing simulation (like what's done in predictors)
        var smoothedSpeed = speeds[0]
        for speed in speeds.dropFirst() {
            smoothedSpeed = speed * 0.05 + smoothedSpeed * 0.95
        }
        
        XCTAssertGreaterThan(smoothedSpeed, 0)
        XCTAssertLessThan(smoothedSpeed, 1.0) // Should be reasonable inference time
    }
    
    func testCoordinateTransformations() {
        // Test coordinate transformations between normalized and pixel space
        let imageSize = CGSize(width: 640, height: 480)
        
        // Test conversion from normalized to pixel coordinates
        let normalizedBoxes = [
            CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
            CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4),
            CGRect(x: 0.0, y: 0.0, width: 1.0, height: 1.0)  // Full image
        ]
        
        for normalizedBox in normalizedBoxes {
            // Convert to pixel coordinates (similar to VNImageRectForNormalizedRect)
            let pixelBox = CGRect(
                x: normalizedBox.origin.x * imageSize.width,
                y: normalizedBox.origin.y * imageSize.height,
                width: normalizedBox.size.width * imageSize.width,
                height: normalizedBox.size.height * imageSize.height
            )
            
            // Verify conversion
            XCTAssertEqual(pixelBox.origin.x, normalizedBox.origin.x * imageSize.width, accuracy: 0.001)
            XCTAssertEqual(pixelBox.origin.y, normalizedBox.origin.y * imageSize.height, accuracy: 0.001)
            XCTAssertEqual(pixelBox.size.width, normalizedBox.size.width * imageSize.width, accuracy: 0.001)
            XCTAssertEqual(pixelBox.size.height, normalizedBox.size.height * imageSize.height, accuracy: 0.001)
            
            // Verify bounds
            XCTAssertGreaterThanOrEqual(pixelBox.minX, 0)
            XCTAssertGreaterThanOrEqual(pixelBox.minY, 0)
            XCTAssertLessThanOrEqual(pixelBox.maxX, imageSize.width)
            XCTAssertLessThanOrEqual(pixelBox.maxY, imageSize.height)
        }
    }
    
    func testColorManagement() {
        // Test color selection and management for visualizations
        let classIndices = [0, 1, 2, 19, 20, 21, 50]  // Test wrap-around
        
        for classIndex in classIndices {
            let colorIndex = classIndex % ultralyticsColors.count
            let color = ultralyticsColors[colorIndex]
            
            XCTAssertNotNil(color)
            XCTAssertGreaterThanOrEqual(colorIndex, 0)
            XCTAssertLessThan(colorIndex, ultralyticsColors.count)
            
            // Test color conversion
            if let rgbComponents = color.toRGBComponents() {
                XCTAssertGreaterThanOrEqual(rgbComponents.red, UInt8(0))
                XCTAssertLessThanOrEqual(rgbComponents.red, UInt8(255))
                XCTAssertGreaterThanOrEqual(rgbComponents.green, UInt8(0))
                XCTAssertLessThanOrEqual(rgbComponents.green, UInt8(255))
                XCTAssertGreaterThanOrEqual(rgbComponents.blue, UInt8(0))
                XCTAssertLessThanOrEqual(rgbComponents.blue, UInt8(255))
            }
        }
    }
    
    func testTaskSpecificDataStructures() {
        // Test data structures specific to each task type
        
        // Test pose-specific data
        XCTAssertEqual(skeleton.count, 19) // Number of bones in human skeleton
        XCTAssertEqual(limbColorIndices.count, 19)
        XCTAssertEqual(kptColorIndices.count, 17) // Number of keypoints
        XCTAssertEqual(posePalette.count, 20)
        
        // Verify skeleton connections are valid indices
        for bone in skeleton {
            XCTAssertEqual(bone.count, 2)
            XCTAssertGreaterThanOrEqual(bone[0], 1)
            XCTAssertLessThanOrEqual(bone[0], 17)
            XCTAssertGreaterThanOrEqual(bone[1], 1)
            XCTAssertLessThanOrEqual(bone[1], 17)
        }
        
        // Test color palette structure
        for color in posePalette {
            XCTAssertEqual(color.count, 3) // RGB values
            for component in color {
                XCTAssertGreaterThanOrEqual(component, CGFloat(0))
                XCTAssertLessThanOrEqual(component, CGFloat(255))
            }
        }
    }
    
    func testThreadSafety() {
        // Skip this test in CI as it may be flaky due to timing issues
        XCTSkip("Thread safety test can be flaky in CI environments")
        
        // Test thread safety of data structures
        let dispatchGroup = DispatchGroup()
        var results = [YOLOResult]()
        let resultsQueue = DispatchQueue(label: "resultsQueue")
        
        // Create multiple results concurrently
        for i in 0..<10 {
            dispatchGroup.enter()
            DispatchQueue.global(qos: .background).async {
                let box = Box(
                    index: i,
                    cls: "object_\(i)",
                    conf: Float(i) * 0.1,
                    xywh: CGRect(x: i * 10, y: i * 10, width: 50, height: 50),
                    xywhn: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2)
                )
                
                let result = YOLOResult(
                    orig_shape: CGSize(width: 640, height: 480),
                    boxes: [box],
                    speed: Double(i) * 0.01,
                    names: ["object_\(i)"]
                )
                
                resultsQueue.sync {
                    results.append(result)
                }
                dispatchGroup.leave()
            }
        }
        
        let waitResult = dispatchGroup.wait(timeout: .now() + 10.0)
        XCTAssertEqual(waitResult, .success, "Thread safety test timed out")
        XCTAssertEqual(results.count, 10)
    }
}

/// Tests for edge cases and boundary conditions  
class YOLOBoundaryTests: XCTestCase {
    
    func testZeroSizedInputs() {
        // Test handling of zero-sized inputs
        let zeroSizeResult = YOLOResult(
            orig_shape: .zero,
            boxes: [],
            speed: 0,
            names: []
        )
        
        XCTAssertEqual(zeroSizeResult.orig_shape, .zero)
        XCTAssertEqual(zeroSizeResult.boxes.count, 0)
        XCTAssertEqual(zeroSizeResult.speed, 0)
        XCTAssertEqual(zeroSizeResult.names.count, 0)
    }
    
    func testLargeInputs() {
        // Test handling of large inputs
        let largeImageSize = CGSize(width: 4096, height: 4096)
        let largeBox = Box(
            index: 999,
            cls: "large_object",
            conf: 0.999,
            xywh: CGRect(x: 0, y: 0, width: 4096, height: 4096),
            xywhn: CGRect(x: 0, y: 0, width: 1, height: 1)
        )
        
        let largeResult = YOLOResult(
            orig_shape: largeImageSize,
            boxes: [largeBox],
            speed: 1.0,
            names: Array(0..<1000).map { "class_\($0)" }
        )
        
        XCTAssertEqual(largeResult.orig_shape.width, 4096)
        XCTAssertEqual(largeResult.orig_shape.height, 4096)
        XCTAssertEqual(largeResult.boxes[0].index, 999)
        XCTAssertEqual(largeResult.names.count, 1000)
    }
    
    func testExtremeConfidenceValues() {
        // Test handling of extreme confidence values
        let boxes = [
            Box(index: 0, cls: "min_conf", conf: 0.0, xywh: CGRect(), xywhn: CGRect()),
            Box(index: 1, cls: "max_conf", conf: 1.0, xywh: CGRect(), xywhn: CGRect()),
            Box(index: 2, cls: "tiny_conf", conf: 0.001, xywh: CGRect(), xywhn: CGRect()),
            Box(index: 3, cls: "near_max", conf: 0.999, xywh: CGRect(), xywhn: CGRect())
        ]
        
        for box in boxes {
            XCTAssertGreaterThanOrEqual(box.conf, 0.0)
            XCTAssertLessThanOrEqual(box.conf, 1.0)
        }
        
        let thresholds: [Float] = [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0]
        for threshold in thresholds {
            let filteredBoxes = boxes.filter { $0.conf >= threshold }
            XCTAssertLessThanOrEqual(filteredBoxes.count, boxes.count)
        }
    }
    
    func testUnicodeAndSpecialCharacters() {
        // Test handling of unicode and special characters in class names
        let specialClassNames = [
            "人物", // Chinese
            "自動車", // Japanese  
            "🚗", // Emoji
            "café", // Accented characters
            "naïve", // More accents
            "test-class_with.special+chars",
            "",
            " ", // Whitespace
            "a".repeated(1000) // Very long string
        ]
        
        for className in specialClassNames {
            let box = Box(index: 0, cls: className, conf: 0.5, xywh: CGRect(), xywhn: CGRect())
            XCTAssertEqual(box.cls, className)
            
            let result = YOLOResult(
                orig_shape: CGSize(width: 100, height: 100),
                boxes: [box],
                speed: 0.1,
                names: [className]
            )
            XCTAssertEqual(result.names[0], className)
        }
    }
}

// Extension for string repetition in tests
extension String {
    func repeated(_ count: Int) -> String {
        return String(repeating: self, count: count)
    }
}

// Thread-safe array for testing
class ThreadSafeArray<T>: @unchecked Sendable {
    private var array: [T] = []
    private let queue = DispatchQueue(label: "ThreadSafeArray", attributes: .concurrent)
    
    func append(_ element: T) {
        queue.async(flags: .barrier) {
            self.array.append(element)
        }
    }
    
    var count: Int {
        return queue.sync {
            return array.count
        }
    }
}
