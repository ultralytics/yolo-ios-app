// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import XCTest

@testable import UltralyticsYOLO

final class ObjectDetectorTests: XCTestCase {
  private func makeDetector() -> ObjectDetector {
    let predictor = ObjectDetector()
    predictor.inputSize = CGSize(width: 4, height: 4)
    predictor.modelInputSize = (width: 4, height: 4)
    return predictor
  }

  /// End2end format is [1, max_det, 6]; max_det must exceed the 6 fields per detection for the shape-based
  /// end2end/traditional heuristic in `processRawResults` to route here.
  private func fillEnd2EndOutput(_ output: MLMultiArray, detections: [[Float]]) {
    for (i, detection) in detections.enumerated() {
      for (j, value) in detection.enumerated() {
        output[i * 6 + j] = NSNumber(value: value)
      }
    }
  }

  func testProcessRawResultsReadsFloat32End2EndOutput() throws {
    // [1, max_det, 6] = [x1, y1, x2, y2, conf, class_id], one detection above threshold, the rest below/zeroed.
    let maxDet = 10
    let output = try MLMultiArray(shape: [1, NSNumber(value: maxDet), 6], dataType: .float32)
    fillEnd2EndOutput(
      output,
      detections: [
        [0, 0, 2, 2, 0.9, 1],
        [1, 1, 3, 3, 0.1, 0],
      ])

    let boxes = makeDetector().processRawResults(output)
    XCTAssertEqual(boxes.count, 1)
    XCTAssertEqual(boxes.first?.index, 1)
    XCTAssertEqual(boxes.first?.conf, 0.9)
  }

  func testProcessRawResultsReadsFloat16End2EndOutput() throws {
    // Regression test: some NMS-free end2end exports produce a .float16 raw output tensor rather than .float32.
    // Reinterpreting that buffer as Float via `assumingMemoryBound` reads the wrong byte offsets and can run past
    // the tensor's actual allocation as detections accumulate.
    let maxDet = 10
    let output = try MLMultiArray(shape: [1, NSNumber(value: maxDet), 6], dataType: .float16)
    fillEnd2EndOutput(output, detections: [[0, 0, 2, 2, 0.9, 1]])

    let boxes = makeDetector().processRawResults(output)
    XCTAssertEqual(boxes.count, 1)
    XCTAssertEqual(boxes.first?.index, 1)
    XCTAssertEqual(boxes.first?.conf ?? 0, 0.9, accuracy: 0.001)  // float16 loses some precision
    XCTAssertEqual(boxes.first?.xywh, CGRect(x: 0, y: 0, width: 2, height: 2))
  }

  func testProcessRawResultsReadsPaddedFloat32End2EndOutput() throws {
    // Regression test: a valid .float32 tensor whose storage is NOT densely packed (e.g. row-padded for
    // alignment) must not be flat-copied — that would read across padding gaps and misalign every subsequent
    // detection. Simulate 2 floats of padding after each 6-field detection row (real stride 8, not 6).
    guard #available(iOS 18.0, *) else {
      throw XCTSkip("MLMultiArray(shape:dataType:strides:) requires iOS 18+")
    }
    // Two detections, so a naive flat copy that ignores the padding after row 0 would misalign row 1 (and
    // everything after it), corrupting or dropping the second detection.
    let maxDet = 10
    let output = try MLMultiArray(shape: [1, maxDet, 6], dataType: .float32, strides: [maxDet * 8, 8, 1])
    fillEnd2EndOutput(
      output,
      detections: [
        [0, 0, 2, 2, 0.9, 1],
        [1, 1, 3, 3, 0.85, 2],
      ])

    let boxes = makeDetector().processRawResults(output)
    XCTAssertEqual(boxes.count, 2)
    XCTAssertTrue(boxes.contains { $0.index == 1 && $0.conf == 0.9 && $0.xywh == CGRect(x: 0, y: 0, width: 2, height: 2) })
    XCTAssertTrue(boxes.contains { $0.index == 2 && $0.conf == 0.85 && $0.xywh == CGRect(x: 1, y: 1, width: 2, height: 2) })
  }

  func testProcessRawResultsReadsFloat16TraditionalOutput() throws {
    // Traditional [1, 4+nc, num_anchors] layout requires num_anchors > 4+nc for the format heuristic to route
    // here (real models have far more anchors than features). Also exercised with a non-float32 backing type.
    let numClasses = 2
    let numAnchors = 10
    let output = try MLMultiArray(
      shape: [1, NSNumber(value: 4 + numClasses), NSNumber(value: numAnchors)], dataType: .float16)
    // feature layout: [x, y, w, h, class0, class1] per anchor, anchor-major via strides.
    var anchors = Array(repeating: [Float](repeating: 0, count: 4 + numClasses), count: numAnchors)
    anchors[0] = [1, 1, 2, 2, 0.05, 0.9]
    anchors[1] = [3, 3, 1, 1, 0.02, 0.01]
    anchors[2] = [10, 10, 2, 2, 0.8, 0.1]
    for anchor in 0..<numAnchors {
      for feature in 0..<(4 + numClasses) {
        let flatIndex = feature * numAnchors + anchor
        output[flatIndex] = NSNumber(value: anchors[anchor][feature])
      }
    }

    let boxes = makeDetector().processRawResults(output)
    XCTAssertEqual(boxes.count, 2)
    XCTAssertTrue(boxes.contains { $0.index == 1 })
    XCTAssertTrue(boxes.contains { $0.index == 0 })
  }

  func testProcessRawResultsRejectsInvalidShape() throws {
    let output = try MLMultiArray(shape: [2, 3], dataType: .float32)
    XCTAssertTrue(makeDetector().processRawResults(output).isEmpty)
  }

  func testProcessRawResultsRejectsTooFewTraditionalFeatures() throws {
    // Fewer than 5 features (i.e. no class scores) would make numClasses <= 0; must not trap on `0..<numClasses`.
    let output = try MLMultiArray(shape: [1, 3, 20], dataType: .float32)
    XCTAssertTrue(makeDetector().processRawResults(output).isEmpty)
  }
}
