// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest

@testable import YOLOCore

/// Integration tests combining multiple YOLOCore components
class YOLOIntegrationTests: XCTestCase {

  func testWorkflowWithMockData() {
    let box = Box(
      index: 0, cls: "person", conf: 0.87,
      xywh: CGRect(x: 100, y: 50, width: 200, height: 300),
      xywhn: CGRect(x: 0.156, y: 0.104, width: 0.313, height: 0.625)
    )

    let result = YOLOResult(
      orig_shape: CGSize(width: 640, height: 480),
      boxes: [box],
      speed: 0.025,
      names: ["person", "car", "bicycle"]
    )

    XCTAssertEqual(result.boxes.count, 1)
    XCTAssertEqual(result.boxes[0].cls, "person")
    XCTAssertEqual(result.boxes[0].conf, 0.87, accuracy: 0.001)
    XCTAssertEqual(result.names.count, 3)

    // Test threshold provider with result
    let thresholdProvider = ThresholdProvider(iouThreshold: 0.5, confidenceThreshold: 0.8)
    let confThreshold =
      thresholdProvider.featureValue(for: "confidenceThreshold")?.doubleValue ?? 0.0
    XCTAssertGreaterThan(Double(box.conf), confThreshold)

    // Test NMS with overlapping boxes
    let boxes = [
      CGRect(x: 100, y: 50, width: 200, height: 300),
      CGRect(x: 110, y: 60, width: 190, height: 290),
      CGRect(x: 400, y: 200, width: 100, height: 150),
    ]
    let scores: [Float] = [0.87, 0.82, 0.75]
    let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.3)

    XCTAssertEqual(selected.count, 2)
    XCTAssertTrue(selected.contains(0))
    XCTAssertTrue(selected.contains(2))
  }

  func testMultiTaskResults() {
    let originalSize = CGSize(width: 416, height: 416)
    let names = ["person", "car", "bicycle", "dog", "cat"]

    // Detection
    let detectionBox = Box(index: 0, cls: "person", conf: 0.9, xywh: CGRect(), xywhn: CGRect())
    let detectionResult = YOLOResult(
      orig_shape: originalSize, boxes: [detectionBox], speed: 0.02, names: names)
    XCTAssertEqual(detectionResult.boxes.count, 1)
    XCTAssertNil(detectionResult.masks)
    XCTAssertNil(detectionResult.probs)

    // Classification
    let probs = Probs(
      top1: "cat", top5: ["cat", "dog", "person", "car", "bicycle"],
      top1Conf: 0.95, top5Confs: [0.95, 0.87, 0.23, 0.15, 0.08])
    let classResult = YOLOResult(
      orig_shape: originalSize, boxes: [], probs: probs, speed: 0.01, names: names)
    XCTAssertNotNil(classResult.probs)
    XCTAssertEqual(classResult.probs?.top1, "cat")

    // Segmentation
    let masks = Masks(masks: [[[0.1, 0.9], [0.8, 0.2]]], combinedMask: nil)
    let segResult = YOLOResult(
      orig_shape: originalSize, boxes: [detectionBox], masks: masks, speed: 0.04, names: names)
    XCTAssertNotNil(segResult.masks)

    // Pose
    let keypoints = Keypoints(
      xyn: [(0.5, 0.3), (0.6, 0.4)], xy: [(208, 125), (250, 166)], conf: [0.95, 0.88])
    let poseResult = YOLOResult(
      orig_shape: originalSize, boxes: [detectionBox], keypointsList: [keypoints],
      speed: 0.03, names: names)
    XCTAssertEqual(poseResult.keypointsList.count, 1)

    // OBB
    let obb = OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.7)
    let obbResult = OBBResult(box: obb, confidence: 0.83, cls: "ship", index: 5)
    let obbDetResult = YOLOResult(
      orig_shape: originalSize, boxes: [], obb: [obbResult], speed: 0.03, names: names)
    XCTAssertEqual(obbDetResult.obb.count, 1)
    XCTAssertEqual(obbDetResult.obb[0].cls, "ship")
  }

  func testCoordinateTransformations() {
    let imageSize = CGSize(width: 640, height: 480)
    let normalizedBoxes = [
      CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
      CGRect(x: 0.0, y: 0.0, width: 1.0, height: 1.0),
    ]

    for normalizedBox in normalizedBoxes {
      let pixelBox = CGRect(
        x: normalizedBox.origin.x * imageSize.width,
        y: normalizedBox.origin.y * imageSize.height,
        width: normalizedBox.size.width * imageSize.width,
        height: normalizedBox.size.height * imageSize.height
      )

      XCTAssertGreaterThanOrEqual(pixelBox.minX, 0)
      XCTAssertGreaterThanOrEqual(pixelBox.minY, 0)
      XCTAssertLessThanOrEqual(pixelBox.maxX, imageSize.width)
      XCTAssertLessThanOrEqual(pixelBox.maxY, imageSize.height)
    }
  }

  func testOBBInfoGeometry() {
    let obb = OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.5)
    let info = OBBInfo(obb)

    XCTAssertEqual(info.polygon.count, 4)
    XCTAssertEqual(info.area, CGFloat(0.3 * 0.2), accuracy: 0.001)
    XCTAssertGreaterThan(info.aabb.width, 0)
    XCTAssertGreaterThan(info.aabb.height, 0)
  }
}

/// Tests for edge cases and boundary conditions
class YOLOBoundaryTests: XCTestCase {

  func testZeroSizedInputs() {
    let result = YOLOResult(orig_shape: .zero, boxes: [], names: [])
    XCTAssertEqual(result.orig_shape, .zero)
    XCTAssertEqual(result.boxes.count, 0)
  }

  func testLargeInputs() {
    let largeBox = Box(
      index: 999, cls: "large_object", conf: 0.999,
      xywh: CGRect(x: 0, y: 0, width: 4096, height: 4096),
      xywhn: CGRect(x: 0, y: 0, width: 1, height: 1))

    let result = YOLOResult(
      orig_shape: CGSize(width: 4096, height: 4096),
      boxes: [largeBox],
      names: Array(0..<1000).map { "class_\($0)" })

    XCTAssertEqual(result.orig_shape.width, 4096)
    XCTAssertEqual(result.boxes[0].index, 999)
    XCTAssertEqual(result.names.count, 1000)
  }

  func testExtremeConfidenceValues() {
    let boxes = [
      Box(index: 0, cls: "min", conf: 0.0, xywh: CGRect(), xywhn: CGRect()),
      Box(index: 1, cls: "max", conf: 1.0, xywh: CGRect(), xywhn: CGRect()),
    ]

    for box in boxes {
      XCTAssertGreaterThanOrEqual(box.conf, 0.0)
      XCTAssertLessThanOrEqual(box.conf, 1.0)
    }

    let filtered = boxes.filter { $0.conf >= 0.5 }
    XCTAssertEqual(filtered.count, 1)
    XCTAssertEqual(filtered[0].cls, "max")
  }

  func testUnicodeClassNames() {
    let names = ["person", "自動車", "café"]
    for name in names {
      let box = Box(index: 0, cls: name, conf: 0.5, xywh: CGRect(), xywhn: CGRect())
      XCTAssertEqual(box.cls, name)
    }
  }
}
