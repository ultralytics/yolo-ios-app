// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import XCTest

@testable import UltralyticsYOLO

/// Tests for Classifier top-5 selection.
final class ClassifierTests: XCTestCase {

  /// The linear top-5 selection must match a full-sort reference (lowest class index wins ties).
  func testTopKSelectionMatchesFullSort() {
    let count = 1000
    let classifier = Classifier()
    classifier.labels = (0..<count).map { "class\($0)" }

    for _ in 0..<50 {
      // Classify exports apply softmax inside the model, so the tensor already holds probabilities.
      let raw = (0..<count).map { _ in Float.random(in: 0...1) }
      let total = raw.reduce(0, +)
      let probs = raw.map { $0 / total }
      let arr = try! MLMultiArray(shape: [NSNumber(value: count)], dataType: .float32)
      let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count { p[i] = probs[i] }

      // Reference top-5 via a full sort.
      let expected = probs.enumerated().sorted { $0.element > $1.element }.prefix(5)

      let result = classifier.topProbs(from: arr)

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
