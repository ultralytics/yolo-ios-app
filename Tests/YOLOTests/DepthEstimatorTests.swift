// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import XCTest

@testable import UltralyticsYOLO

final class DepthEstimatorTests: XCTestCase {
  func testPostProcessDepthCropsLetterboxPadding() throws {
    let output = try MLMultiArray(shape: [1, 1, 4, 4], dataType: .float32)
    let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)
    for index in 0..<16 {
      pointer[index] = Float(index + 1)
    }

    let predictor = DepthEstimator()
    predictor.inputSize = CGSize(width: 4, height: 2)
    predictor.modelInputSize = (width: 4, height: 4)
    let result = try XCTUnwrap(predictor.postProcessDepth(output))

    XCTAssertEqual(result.width, 4)
    XCTAssertEqual(result.height, 2)
    XCTAssertEqual(result.values, [5, 6, 7, 8, 9, 10, 11, 12])
    XCTAssertEqual(result.minDepth, 5)
    XCTAssertEqual(result.maxDepth, 12)
    XCTAssertNotNil(result.image)
  }

  func testPostProcessDepthRejectsInvalidShape() throws {
    let output = try MLMultiArray(shape: [2, 3, 4], dataType: .float32)
    XCTAssertNil(DepthEstimator().postProcessDepth(output))
  }

  func testPostProcessDepthReadsFloat16Output() throws {
    let output = try MLMultiArray(shape: [1, 1, 1, 2], dataType: .float16)
    output[0] = 1.5
    output[1] = 2.5

    let result = try XCTUnwrap(DepthEstimator().postProcessDepth(output))
    XCTAssertEqual(result.values, [1.5, 2.5])
  }

  func testPostProcessDepthIgnoresInvalidZeroForRange() throws {
    let output = try MLMultiArray(shape: [1, 1, 1, 3], dataType: .float32)
    let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)
    pointer[0] = 0
    pointer[1] = 2
    pointer[2] = 5

    let result = try XCTUnwrap(DepthEstimator().postProcessDepth(output))
    XCTAssertEqual(result.values, [0, 2, 5])
    XCTAssertEqual(result.minDepth, 2)
    XCTAssertEqual(result.maxDepth, 5)
    XCTAssertNotNil(result.image)
  }

  func testDepthImageColorsNearAndFarPixels() throws {
    let output = try MLMultiArray(shape: [1, 1, 1, 2], dataType: .float32)
    let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)
    pointer[0] = 1
    pointer[1] = 10

    let result = try XCTUnwrap(DepthEstimator().postProcessDepth(output))
    let image = try XCTUnwrap(result.image)
    let data = try XCTUnwrap(image.dataProvider?.data as Data?)

    XCTAssertEqual(Array(data[0..<4]), [180, 20, 40, 255])
    XCTAssertEqual(Array(data[4..<8]), [48, 18, 59, 255])
  }
}
