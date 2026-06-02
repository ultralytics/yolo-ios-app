// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import XCTest

@testable import UltralyticsYOLO

/// Tests for SemanticSegmenter logits post-processing (per-pixel class argmax).
final class SemanticSegmenterTests: XCTestCase {

  func testPostProcessSemanticArgmaxNCHW() {
    // Logits [1, C=3, H=2, W=2] in NCHW. Layout index = c*HW + y*W + x, HW = 4.
    // Pixel argmax (lowest class index wins ties):
    //   (0,0): c=[0.1, 0.9, 0.2] -> 1
    //   (0,1): c=[0.5, 0.1, 0.3] -> 0
    //   (1,0): c=[0.2, 0.2, 0.8] -> 2
    //   (1,1): c=[0.7, 0.7, 0.1] -> 0 (tie c0/c1 -> c0)
    let segmenter = SemanticSegmenter()
    segmenter.labels = ["a", "b", "c"]  // makes the NCHW axis heuristic unambiguous (shape[1] == labels.count)

    let logits = try! MLMultiArray(shape: [1, 3, 2, 2] as [NSNumber], dataType: .float32)
    let values: [Float] = [
      0.1, 0.5, 0.2, 0.7,  // class 0 plane
      0.9, 0.1, 0.2, 0.7,  // class 1 plane
      0.2, 0.3, 0.8, 0.1,  // class 2 plane
    ]
    let p = logits.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<values.count { p[i] = values[i] }

    guard let mask = segmenter.postProcessSemantic(logits) else {
      XCTFail("postProcessSemantic returned nil")
      return
    }

    XCTAssertEqual(mask.width, 2)
    XCTAssertEqual(mask.height, 2)
    XCTAssertEqual(mask.classMap, [1, 0, 2, 0])
  }
}
