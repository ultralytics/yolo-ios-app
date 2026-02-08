// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreGraphics
import UIKit
import XCTest

@testable import YOLO

/// Minimal tests for YOLOResult data structures
class YOLOResultTests: XCTestCase {

  func testBoxCreation() {
    // Test Box struct initialization and properties
    let box = Box(
      index: 5,
      cls: "person",
      conf: 0.85,
      xywh: CGRect(x: 10, y: 20, width: 100, height: 50),
      xywhn: CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.25)
    )

    XCTAssertEqual(box.index, 5)
    XCTAssertEqual(box.cls, "person")
    XCTAssertEqual(box.conf, 0.85, accuracy: 0.001)
    XCTAssertEqual(box.xywh.width, 100)
    XCTAssertEqual(box.xywhn.minX, 0.1, accuracy: 0.001)
  }

  func testYOLOResultCreation() {
    // Test YOLOResult struct initialization
    let boxes = [Box(index: 0, cls: "cat", conf: 0.9, xywh: CGRect(), xywhn: CGRect())]
    let result = YOLOResult(
      orig_shape: CGSize(width: 640, height: 480),
      boxes: boxes,
      speed: 0.05,
      names: ["cat", "dog"]
    )

    XCTAssertEqual(result.orig_shape.width, 640)
    XCTAssertEqual(result.boxes.count, 1)
    XCTAssertEqual(result.boxes[0].cls, "cat")
    XCTAssertEqual(result.speed, 0.05, accuracy: 0.001)
    XCTAssertEqual(result.names, ["cat", "dog"])
  }

  func testProbsCreation() {
    // Test Probs struct for classification results
    var probs = Probs(
      top1: "dog",
      top5: ["dog", "cat", "bird", "fish", "mouse"],
      top1Conf: 0.95,
      top5Confs: [0.95, 0.8, 0.6, 0.4, 0.2]
    )

    XCTAssertEqual(probs.top1, "dog")
    XCTAssertEqual(probs.top5.count, 5)
    XCTAssertEqual(probs.top1Conf, 0.95, accuracy: 0.001)
    XCTAssertEqual(probs.top5Confs[2], 0.6, accuracy: 0.001)

    probs.top1 = "cat"
    XCTAssertEqual(probs.top1, "cat")
  }

  func testMasksCreation() {
    // Test Masks struct for segmentation results
    let maskData: [[[Float]]] = [[[0.1, 0.9], [0.8, 0.2]]]
    let masks = Masks(masks: maskData, combinedMask: nil)

    XCTAssertEqual(masks.masks.count, 1)
    XCTAssertEqual(masks.masks[0][0][1], 0.9, accuracy: 0.001)
    XCTAssertNil(masks.combinedMask)
  }

  func testKeypointsCreation() {
    // Test Keypoints struct for pose estimation
    let keypoints = Keypoints(
      xyn: [(0.5, 0.3), (0.6, 0.4)],
      xy: [(320, 240), (384, 288)],
      conf: [0.9, 0.8]
    )

    XCTAssertEqual(keypoints.xyn.count, 2)
    XCTAssertEqual(keypoints.xy[0].0, 320, accuracy: 0.001)
    XCTAssertEqual(keypoints.conf[1], 0.8, accuracy: 0.001)
  }

  func testOBBCreation() {
    // Test OBB struct for oriented bounding boxes
    let obb = OBB(cx: 100, cy: 50, w: 80, h: 40, angle: 0.5)

    XCTAssertEqual(obb.cx, 100, accuracy: 0.001)
    XCTAssertEqual(obb.w, 80, accuracy: 0.001)
    XCTAssertEqual(obb.angle, 0.5, accuracy: 0.001)
  }

  func testOBBToPolygon() {
    // Test OBB conversion to polygon points
    let obb = OBB(cx: 0, cy: 0, w: 4, h: 2, angle: 0)
    let polygon = obb.toPolygon()

    XCTAssertEqual(polygon.count, 4)
    XCTAssertEqual(polygon[0].x, -2, accuracy: 0.001)
    XCTAssertEqual(polygon[1].x, 2, accuracy: 0.001)
  }

  func testOBBArea() {
    // Test OBB area calculation
    let obb = OBB(cx: 0, cy: 0, w: 10, h: 5, angle: 1.0)
    XCTAssertEqual(obb.area, 50, accuracy: 0.001)
  }

  func testOBBResultCreation() {
    // Test OBBResult struct
    let obb = OBB(cx: 50, cy: 25, w: 30, h: 20, angle: 0.3)
    var obbResult = OBBResult(box: obb, confidence: 0.8, cls: "car", index: 2)

    XCTAssertEqual(obbResult.confidence, 0.8, accuracy: 0.001)
    XCTAssertEqual(obbResult.cls, "car")
    XCTAssertEqual(obbResult.index, 2)

    obbResult.confidence = 0.9
    XCTAssertEqual(obbResult.confidence, 0.9, accuracy: 0.001)
  }
}
