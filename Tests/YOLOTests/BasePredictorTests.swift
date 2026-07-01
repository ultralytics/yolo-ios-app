// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Vision
import XCTest

@testable import UltralyticsYOLO

/// Minimal tests for BasePredictor functionality
class BasePredictorTests: XCTestCase {

  func testInitialization() {
    // Test BasePredictor initialization
    let predictor = BasePredictor()

    XCTAssertFalse(predictor.isModelLoaded)
    XCTAssertEqual(predictor.labels.count, 0)
    XCTAssertEqual(predictor.confidenceThreshold, 0.25, accuracy: 0.001)
    XCTAssertEqual(predictor.iouThreshold, 0.7, accuracy: 0.001)
    XCTAssertEqual(predictor.numItemsThreshold, 30)
    XCTAssertFalse(predictor.isUpdating)
  }

  func testConfidenceThresholdSetting() {
    // Test confidence threshold configuration
    let predictor = BasePredictor()

    predictor.setConfidenceThreshold(confidence: 0.8)
    XCTAssertEqual(predictor.confidenceThreshold, 0.8, accuracy: 0.001)

    predictor.setConfidenceThreshold(confidence: 0.1)
    XCTAssertEqual(predictor.confidenceThreshold, 0.1, accuracy: 0.001)
  }

  func testIoUThresholdSetting() {
    // Test IoU threshold configuration
    let predictor = BasePredictor()

    predictor.setIouThreshold(iou: 0.7)
    XCTAssertEqual(predictor.iouThreshold, 0.7, accuracy: 0.001)

    predictor.setIouThreshold(iou: 0.2)
    XCTAssertEqual(predictor.iouThreshold, 0.2, accuracy: 0.001)
  }

  func testNumItemsThresholdSetting() {
    // Test number of items threshold configuration
    let predictor = BasePredictor()

    predictor.setNumItemsThreshold(numItems: 50)
    XCTAssertEqual(predictor.numItemsThreshold, 50)

    predictor.setNumItemsThreshold(numItems: 10)
    XCTAssertEqual(predictor.numItemsThreshold, 10)
  }

  func testBasePredictOnImage() {
    // Test base predictOnImage method returns empty result
    let predictor = BasePredictor()
    let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

    let result = predictor.predictOnImage(image: image)

    XCTAssertEqual(result.orig_shape, .zero)
    XCTAssertEqual(result.boxes.count, 0)
    XCTAssertEqual(result.speed, 0, accuracy: 0.001)
    XCTAssertEqual(result.names.count, 0)
  }

  func testBaseProcessObservations() {
    // Test base processObservations method does nothing
    let predictor = BasePredictor()

    // Should not crash when called
    predictor.processObservations(for: MockVNRequest(), nil)
    predictor.processObservations(for: MockVNRequest(), NSError(domain: "test", code: 0))
  }

  func testLabelsProperty() {
    // Test labels property can be read and written
    let predictor = BasePredictor()
    let testLabels = ["person", "car", "dog"]

    predictor.labels = testLabels
    XCTAssertEqual(predictor.labels, testLabels)

    predictor.labels.append("cat")
    XCTAssertEqual(predictor.labels.count, 4)
    XCTAssertEqual(predictor.labels.last, "cat")
  }

  func testIsUpdatingFlag() {
    // Test isUpdating flag can be set and read
    let predictor = BasePredictor()

    XCTAssertFalse(predictor.isUpdating)

    predictor.isUpdating = true
    XCTAssertTrue(predictor.isUpdating)

    predictor.isUpdating = false
    XCTAssertFalse(predictor.isUpdating)
  }

  func testModelInputSizeInitialization() {
    // Test model input size has proper default values
    let predictor = BasePredictor()

    XCTAssertEqual(predictor.modelInputSize.width, 0)
    XCTAssertEqual(predictor.modelInputSize.height, 0)
  }

  func testTimingProperties() {
    // Test timing properties initialization
    let predictor = BasePredictor()

    XCTAssertEqual(predictor.t0, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t1, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t2, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t4, 1.0, accuracy: 0.001)  // non-zero to avoid infinity FPS on first frame
    XCTAssertGreaterThan(predictor.t3, 0)  // Should be initialized with current time
  }

  func testLetterboxRectMappingLandscapePadsTopBottom() {
    let predictor = BasePredictor()
    predictor.modelInputSize = (width: 640, height: 640)
    predictor.inputSize = CGSize(width: 1920, height: 1080)

    let rect = predictor.inputRect(fromModelRect: CGRect(x: 320, y: 300, width: 64, height: 32))

    XCTAssertEqual(rect.minX, 960, accuracy: 0.001)
    XCTAssertEqual(rect.minY, 480, accuracy: 0.001)
    XCTAssertEqual(rect.width, 192, accuracy: 0.001)
    XCTAssertEqual(rect.height, 96, accuracy: 0.001)
  }

  func testLetterboxRectMappingPortraitPadsLeftRight() {
    let predictor = BasePredictor()
    predictor.modelInputSize = (width: 640, height: 640)
    predictor.inputSize = CGSize(width: 1080, height: 1920)

    let rect = predictor.inputRect(fromModelRect: CGRect(x: 300, y: 320, width: 32, height: 64))

    XCTAssertEqual(rect.minX, 480, accuracy: 0.001)
    XCTAssertEqual(rect.minY, 960, accuracy: 0.001)
    XCTAssertEqual(rect.width, 96, accuracy: 0.001)
    XCTAssertEqual(rect.height, 192, accuracy: 0.001)
  }

  func testLetterboxRectMappingNonSquareModelsUseModelAspect() {
    let predictor = BasePredictor()

    predictor.modelInputSize = (width: 640, height: 384)
    predictor.inputSize = CGSize(width: 1920, height: 1080)
    let landscape = predictor.inputRect(
      fromModelRect: CGRect(x: 320, y: 192, width: 64, height: 32))

    XCTAssertEqual(landscape.minX, 960, accuracy: 0.001)
    XCTAssertEqual(landscape.minY, 540, accuracy: 0.001)
    XCTAssertEqual(landscape.width, 192, accuracy: 0.001)
    XCTAssertEqual(landscape.height, 96, accuracy: 0.001)

    predictor.modelInputSize = (width: 384, height: 640)
    predictor.inputSize = CGSize(width: 1080, height: 1920)
    let portrait = predictor.inputRect(fromModelRect: CGRect(x: 192, y: 320, width: 32, height: 64))

    XCTAssertEqual(portrait.minX, 540, accuracy: 0.001)
    XCTAssertEqual(portrait.minY, 960, accuracy: 0.001)
    XCTAssertEqual(portrait.width, 96, accuracy: 0.001)
    XCTAssertEqual(portrait.height, 192, accuracy: 0.001)
  }

  func testLetterboxOBBMappingKeepsAngleInInputSpace() {
    let predictor = BasePredictor()
    predictor.modelInputSize = (width: 640, height: 640)
    predictor.inputSize = CGSize(width: 1920, height: 1080)

    let box = predictor.inputOBB(
      fromModelOBB: OBB(cx: 0.5, cy: 0.5, w: 0.25, h: 0.125, angle: 0.7))

    XCTAssertEqual(box.cx, 0.5, accuracy: 0.001)
    XCTAssertEqual(box.cy, 0.5, accuracy: 0.001)
    XCTAssertEqual(box.w, 0.25, accuracy: 0.001)
    XCTAssertEqual(box.h, 0.222_222, accuracy: 0.001)
    XCTAssertEqual(box.angle, 0.7, accuracy: 0.001)
  }

  func testLetterboxMaskCropRectMatchesPaddingAxis() {
    let predictor = BasePredictor()
    predictor.modelInputSize = (width: 640, height: 640)

    let landscape = predictor.inputMaskCropRect(
      maskWidth: 160, maskHeight: 160,
      inputSize: CGSize(width: 1920, height: 1080),
      modelInputSize: predictor.modelInputSize)
    XCTAssertEqual(landscape?.minX ?? -1, 0, accuracy: 0.001)
    XCTAssertEqual(landscape?.minY ?? -1, 35, accuracy: 0.001)
    XCTAssertEqual(landscape?.width ?? -1, 160, accuracy: 0.001)
    XCTAssertEqual(landscape?.height ?? -1, 90, accuracy: 0.001)

    let portrait = predictor.inputMaskCropRect(
      maskWidth: 160, maskHeight: 160,
      inputSize: CGSize(width: 1080, height: 1920),
      modelInputSize: predictor.modelInputSize)
    XCTAssertEqual(portrait?.minX ?? -1, 35, accuracy: 0.001)
    XCTAssertEqual(portrait?.minY ?? -1, 0, accuracy: 0.001)
    XCTAssertEqual(portrait?.width ?? -1, 90, accuracy: 0.001)
    XCTAssertEqual(portrait?.height ?? -1, 160, accuracy: 0.001)
  }

  func testInvalidLetterboxTransformReturnsEmptyGeometry() {
    let predictor = BasePredictor()

    XCTAssertEqual(
      predictor.inputRect(fromModelRect: CGRect(x: 1, y: 2, width: 3, height: 4)), .zero)
    XCTAssertEqual(predictor.inputPoint(fromModelPoint: CGPoint(x: 1, y: 2)), .zero)

    let box = predictor.inputOBB(fromModelOBB: OBB(cx: 1, cy: 2, w: 3, h: 4, angle: 5))
    XCTAssertEqual(box.cx, 0, accuracy: 0.001)
    XCTAssertEqual(box.cy, 0, accuracy: 0.001)
    XCTAssertEqual(box.w, 0, accuracy: 0.001)
    XCTAssertEqual(box.h, 0, accuracy: 0.001)
    XCTAssertEqual(box.angle, 0, accuracy: 0.001)

    predictor.inputSize = .zero
    XCTAssertEqual(
      predictor.normalizedRect(fromInputRect: CGRect(x: 1, y: 2, width: 3, height: 4)), .zero)
    XCTAssertEqual(predictor.normalizedPoint(fromInputPoint: CGPoint(x: 1, y: 2)), .zero)
  }

  func testSemanticPostProcessBuildsDenseClassMap() throws {
    let predictor = SemanticSegmenter()
    predictor.modelInputSize = (width: 2, height: 2)
    predictor.inputSize = CGSize(width: 2, height: 2)
    predictor.labels = ["road", "car", "sky"]

    let logits = try MLMultiArray(shape: [1, 3, 2, 2], dataType: .float32)
    let values: [Float] = [
      9, 1, 1, 1,
      1, 8, 1, 7,
      1, 1, 6, 1,
    ]
    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    for (index, value) in values.enumerated() {
      pointer[index] = value
    }

    let mask = predictor.postProcessSemantic(logits)

    XCTAssertEqual(mask?.width, 2)
    XCTAssertEqual(mask?.height, 2)
    XCTAssertEqual(mask?.classMap, [0, 1, 2, 1])
    XCTAssertNotNil(mask?.maskImage)
  }

  func testSemanticPostProcessRemovesLetterboxPadding() throws {
    let predictor = SemanticSegmenter()
    predictor.modelInputSize = (width: 4, height: 4)
    predictor.inputSize = CGSize(width: 4, height: 2)
    predictor.labels = ["pad", "scene"]

    let logits = try MLMultiArray(shape: [1, 2, 4, 4], dataType: .float32)
    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    let strides = logits.strides.map { $0.intValue }
    for y in 0..<4 {
      for x in 0..<4 {
        let classOneWins = y == 1 || y == 2
        pointer[y * strides[2] + x * strides[3]] = classOneWins ? 0 : 10
        pointer[strides[1] + y * strides[2] + x * strides[3]] = classOneWins ? 10 : 0
      }
    }

    let mask = predictor.postProcessSemantic(logits)

    XCTAssertEqual(mask?.width, 4)
    XCTAssertEqual(mask?.height, 2)
    XCTAssertEqual(mask?.classMap, Array(repeating: 1, count: 8))
  }

  func testSemanticPostProcessSingleChannelThresholdsForeground() throws {
    let predictor = SemanticSegmenter()
    predictor.modelInputSize = (width: 2, height: 2)
    predictor.inputSize = CGSize(width: 2, height: 2)
    predictor.labels = ["foreground"]

    let logits = try MLMultiArray(shape: [1, 1, 2, 2], dataType: .float32)
    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    for index in 0..<4 {
      pointer[index] = index.isMultiple(of: 2) ? 1 : -1
    }

    let mask = predictor.postProcessSemantic(logits)

    XCTAssertEqual(mask?.classMap, [1, 0, 1, 0])
  }

  func testObjectDetectorDecodesTraditionalTensorWithNMS() throws {
    let detector = ObjectDetector()
    detector.labels = ["person", "car"]
    detector.modelInputSize = (width: 100, height: 100)
    detector.inputSize = CGSize(width: 100, height: 100)

    let prediction = try makeArray(shape: [1, 6, 8]) { write in
      write([0, 0, 0], 50)
      write([0, 1, 0], 50)
      write([0, 2, 0], 20)
      write([0, 3, 0], 20)
      write([0, 4, 0], 0.9)

      write([0, 0, 1], 52)
      write([0, 1, 1], 50)
      write([0, 2, 1], 20)
      write([0, 3, 1], 20)
      write([0, 4, 1], 0.8)

      write([0, 0, 2], 80)
      write([0, 1, 2], 80)
      write([0, 2, 2], 10)
      write([0, 3, 2], 10)
      write([0, 5, 2], 0.7)
    }

    let boxes = detector.processRawResults(prediction)

    XCTAssertEqual(boxes.count, 2)
    XCTAssertEqual(boxes.map(\.cls), ["person", "car"])
    XCTAssertEqual(boxes[0].xywh, CGRect(x: 40, y: 40, width: 20, height: 20))
    XCTAssertEqual(boxes[1].xywh, CGRect(x: 75, y: 75, width: 10, height: 10))
  }

  func testObjectDetectorDecodesEndToEndTensorAndAppliesLimit() throws {
    let detector = ObjectDetector()
    detector.labels = ["person", "car"]
    detector.modelInputSize = (width: 100, height: 100)
    detector.inputSize = CGSize(width: 100, height: 100)
    detector.setNumItemsThreshold(numItems: 1)

    let prediction = try makeArray(shape: [1, 8, 6]) { write in
      write([0, 0, 0], 10)
      write([0, 0, 1], 20)
      write([0, 0, 2], 40)
      write([0, 0, 3], 60)
      write([0, 0, 4], 0.7)
      write([0, 0, 5], 1)

      write([0, 1, 0], 50)
      write([0, 1, 1], 50)
      write([0, 1, 2], 70)
      write([0, 1, 3], 70)
      write([0, 1, 4], 0.6)
      write([0, 1, 5], 0)
    }

    let boxes = detector.processRawResults(prediction)

    XCTAssertEqual(boxes.count, 1)
    let box = try XCTUnwrap(boxes.first)
    XCTAssertEqual(box.cls, "car")
    XCTAssertEqual(box.xywh, CGRect(x: 10, y: 20, width: 30, height: 40))
    XCTAssertEqual(box.xywhn, CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4))
  }

  func testSegmenterDecodesTraditionalAndEndToEndTensors() throws {
    let segmenter = Segmenter()

    let traditional = try makeArray(shape: [1, 38, 40]) { write in
      write([0, 0, 0], 20)
      write([0, 1, 0], 20)
      write([0, 2, 0], 10)
      write([0, 3, 0], 10)
      write([0, 4, 0], 0.9)
      write([0, 6, 0], 0.1)

      write([0, 0, 1], 21)
      write([0, 1, 1], 20)
      write([0, 2, 1], 10)
      write([0, 3, 1], 10)
      write([0, 4, 1], 0.8)

      write([0, 0, 2], 80)
      write([0, 1, 2], 80)
      write([0, 2, 2], 8)
      write([0, 3, 2], 8)
      write([0, 5, 2], 0.7)
      for i in 0..<32 { write([0, 6 + i, 2], Float(i)) }
    }

    let traditionalResults = segmenter.postProcessSegment(
      feature: traditional, confidenceThreshold: 0.25, iouThreshold: 0.5)

    XCTAssertEqual(traditionalResults.count, 2)
    let classZero = try XCTUnwrap(traditionalResults.first { $0.1 == 0 })
    XCTAssertEqual(classZero.2, 0.9, accuracy: 0.001)
    let classOne = try XCTUnwrap(traditionalResults.first { $0.1 == 1 })
    XCTAssertEqual(classOne.3[31], 31)

    let endToEnd = try makeArray(shape: [1, 40, 38]) { write in
      write([0, 0, 0], 1)
      write([0, 0, 1], 2)
      write([0, 0, 2], 5)
      write([0, 0, 3], 8)
      write([0, 0, 4], 0.8)
      write([0, 0, 5], 3)
      write([0, 0, 6], 0.25)
    }

    let endToEndResults = segmenter.postProcessSegment(
      feature: endToEnd, confidenceThreshold: 0.25, iouThreshold: 0.5)

    XCTAssertEqual(endToEndResults.count, 1)
    let endToEndResult = try XCTUnwrap(endToEndResults.first)
    XCTAssertEqual(endToEndResult.0, CGRect(x: 1, y: 2, width: 4, height: 6))
    XCTAssertEqual(endToEndResult.1, 3)
    XCTAssertEqual(endToEndResult.3[0], 0.25, accuracy: 0.001)
  }

  func testPoseEstimatorDecodesTraditionalTensor() throws {
    let pose = PoseEstimator()
    pose.labels = ["person"]
    pose.modelInputSize = (width: 100, height: 100)
    pose.inputSize = CGSize(width: 100, height: 100)

    let prediction = try makeArray(shape: [1, 56, 80]) { write in
      write([0, 0, 0], 50)
      write([0, 1, 0], 50)
      write([0, 2, 0], 20)
      write([0, 3, 0], 10)
      write([0, 4, 0], 0.9)
      for keypoint in 0..<17 {
        let base = 5 + keypoint * 3
        write([0, base, 0], Float(10 + keypoint))
        write([0, base + 1, 0], Float(20 + keypoint))
        write([0, base + 2, 0], 0.8)
      }
    }

    let results = pose.PostProcessPose(
      prediction: prediction, confidenceThreshold: 0.25, iouThreshold: 0.5)

    XCTAssertEqual(results.count, 1)
    let result = try XCTUnwrap(results.first)
    XCTAssertEqual(result.box.xywh, CGRect(x: 40, y: 45, width: 20, height: 10))
    XCTAssertEqual(result.keypoints.xy.count, 17)
    XCTAssertEqual(result.keypoints.xyn[16].x, 0.26, accuracy: 0.001)
    XCTAssertEqual(result.keypoints.conf[16], 0.8, accuracy: 0.001)
  }

  func testObbDetectorDecodesTraditionalTensorAndRunsRotatedNMS() throws {
    let detector = ObbDetector()
    detector.modelInputSize = (width: 100, height: 100)

    let prediction = try makeArray(shape: [1, 7, 40]) { write in
      write([0, 0, 0], 50)
      write([0, 1, 0], 50)
      write([0, 2, 0], 20)
      write([0, 3, 0], 10)
      write([0, 4, 0], 0.9)
      write([0, 6, 0], 0.2)

      write([0, 0, 1], 51)
      write([0, 1, 1], 50)
      write([0, 2, 1], 20)
      write([0, 3, 1], 10)
      write([0, 4, 1], 0.8)
      write([0, 6, 1], 0.2)
    }

    let results = detector.postProcessOBB(
      feature: prediction, confidenceThreshold: 0.25, iouThreshold: 0.3)

    XCTAssertEqual(results.count, 1)
    let result = try XCTUnwrap(results.first)
    XCTAssertEqual(result.box.cx, 0.5, accuracy: 0.001)
    XCTAssertEqual(result.box.w, 0.2, accuracy: 0.001)
    XCTAssertEqual(result.score, 0.9, accuracy: 0.001)
  }
}

private func makeArray(
  shape: [Int],
  writeValues: (_ write: (_ indices: [Int], _ value: Float) -> Void) -> Void
) throws -> MLMultiArray {
  let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
  let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
  pointer.initialize(repeating: 0, count: array.count)
  let strides = array.strides.map { $0.intValue }

  writeValues { indices, value in
    let offset = zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }
    pointer[offset] = value
  }

  return array
}

// MARK: - Mock Classes

class MockVNRequest: VNRequest, @unchecked Sendable {
  init() {
    super.init(completionHandler: nil)
  }

  @available(*, unavailable)
  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }
}
