// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

import CoreML
import XCTest

@testable import YOLO

/// Tests for Classifier softmax + top-5 selection.
final class ClassifierTests: XCTestCase {

  /// The linear top-5 selection must match a full-sort reference (lowest class index wins ties).
  func testTopKSelectionMatchesFullSort() {
    let count = 1000
    let classifier = Classifier()
    classifier.labels = (0..<count).map { "class\($0)" }

    for _ in 0..<50 {
      let logits = (0..<count).map { _ in Float.random(in: -10...10) }
      let arr = try! MLMultiArray(shape: [NSNumber(value: count)], dataType: .float32)
      let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count { p[i] = logits[i] }

      // Reference top-5 from softmax of the same logits via a full sort.
      let mx = logits.max()!
      let exps = logits.map { expf($0 - mx) }
      let s = exps.reduce(0, +)
      let probs = exps.map { $0 / s }
      let expected = probs.enumerated().sorted { $0.element > $1.element }.prefix(5)

      let result = classifier.softmaxProbs(from: arr)

      XCTAssertEqual(result.top5.count, 5)
      XCTAssertEqual(result.top5Confs.count, 5)
      XCTAssertEqual(result.top1, "class\(expected.first!.offset)")
      XCTAssertEqual(result.top1Conf, expected.first!.element, accuracy: 1e-6)
      for (got, ref) in zip(result.top5Confs, expected.map { $0.element }) {
        XCTAssertEqual(got, ref, accuracy: 1e-6)
      }
    }
  }
}
