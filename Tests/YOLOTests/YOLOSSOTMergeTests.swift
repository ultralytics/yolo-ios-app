// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import UIKit
import XCTest

@testable import UltralyticsYOLO

final class YOLOSSOTMergeTests: XCTestCase {
  private final class AsyncResultBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, Error>?

    func store(_ result: Result<T, Error>) {
      lock.lock()
      defer { lock.unlock() }
      self.result = result
    }

    func load() -> Result<T, Error>? {
      lock.lock()
      defer { lock.unlock() }
      return result
    }
  }

  private func modelURL(_ name: String) throws -> URL {
    let testsDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    let url = testsDirectory.appendingPathComponent("Resources/\(name).mlpackage")
    XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "Missing \(url.path)")
    return url
  }

  private func loadPredictor(
    _ name: String,
    task: YOLOTask,
    useGpu: Bool = false,
    numItemsThreshold: Int = 30
  ) throws -> BasePredictor {
    let expectation = XCTestExpectation(description: "Load \(name)")
    let url = try modelURL(name)
    let loaded = AsyncResultBox<BasePredictor>()

    BasePredictor.create(
      for: task,
      modelURL: url,
      useGpu: useGpu,
      numItemsThreshold: numItemsThreshold
    ) { result in
      loaded.store(result)
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 30)
    let predictor = try XCTUnwrap(loaded.load()).get()
    let expectedInputSize = task == .classify ? 224 : 640
    XCTAssertEqual(predictor.modelInputSize.width, expectedInputSize)
    XCTAssertEqual(predictor.modelInputSize.height, expectedInputSize)
    return predictor
  }

  private func testImage(width: CGFloat = 640, height: CGFloat = 640) -> CIImage {
    CIImage(color: CIColor(red: 0.2, green: 0.3, blue: 0.4))
      .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
  }

  private func featureValue(_ predictor: BasePredictor, _ name: String) -> Double? {
    predictor.detector?.featureProvider?.featureValue(for: name)?.doubleValue
  }

  func testYOLO26NMSFreeEffectiveIoUStaysOne() throws {
    let predictor = try loadPredictor("yolo26n", task: .detect)

    XCTAssertFalse(predictor.requiresNMS)
    XCTAssertEqual(featureValue(predictor, "iouThreshold") ?? -1, 1.0, accuracy: 0.001)

    predictor.setConfidenceThreshold(confidence: 0.35)
    XCTAssertEqual(featureValue(predictor, "iouThreshold") ?? -1, 1.0, accuracy: 0.001)
    XCTAssertEqual(featureValue(predictor, "confidenceThreshold") ?? -1, 0.35, accuracy: 0.001)

    predictor.setIouThreshold(iou: 0.2)
    XCTAssertEqual(predictor.iouThreshold, 0.2, accuracy: 0.001)
    XCTAssertEqual(featureValue(predictor, "iouThreshold") ?? -1, 1.0, accuracy: 0.001)
  }

  func testMetadataLabelParsingAndFallbacks() {
    XCTAssertEqual(
      BasePredictor.parseLabels(from: ["classes": "person, car ,dog"]),
      ["person", "car", "dog"])
    XCTAssertEqual(
      BasePredictor.parseLabels(from: ["names": "{0: 'person', 1: 'car'}"]),
      ["person", "car"])
    XCTAssertEqual(
      BasePredictor.parseLabels(from: ["names": "{0: 'person', 2: 'dog'}"]),
      ["person", "", "dog"])
    XCTAssertEqual(BasePredictor.parseLabels(from: ["names": ""]), [])
    XCTAssertEqual(BasePredictor.parseLabels(from: [:]), [])

    let predictor = BasePredictor()
    predictor.labels = ["person", "", "dog"]
    XCTAssertEqual(predictor.labelName(for: 0), "person")
    XCTAssertEqual(predictor.labelName(for: 1), "class 1")
    XCTAssertEqual(predictor.labelName(for: 9), "class 9")
    XCTAssertEqual(predictor.labelName(for: -1), "class -1")
  }

  func testYOLOTaskFromStringAcceptsBridgeAliases() {
    XCTAssertEqual(YOLOTask.fromString("detect"), .detect)
    XCTAssertEqual(YOLOTask.fromString("object_detection"), .detect)
    XCTAssertEqual(YOLOTask.fromString("seg"), .segment)
    XCTAssertEqual(YOLOTask.fromString("instance-segmentation"), .segment)
    XCTAssertEqual(YOLOTask.fromString("semantic_segmentation"), .semantic)
    XCTAssertEqual(YOLOTask.fromString("depth_estimation"), .depth)
    XCTAssertEqual(YOLOTask.fromString("cls"), .classify)
    XCTAssertEqual(YOLOTask.fromString("classification"), .classify)
    XCTAssertEqual(YOLOTask.fromString("keypoints"), .pose)
    XCTAssertEqual(YOLOTask.fromString("oriented_bounding_box"), .obb)
    XCTAssertEqual(YOLOTask.fromString("unknown"), .detect)
  }

  func testProbsAliasesAndOriginalImage() {
    var probs = Probs(
      top1: "dog",
      top5: ["dog", "cat"],
      top1Conf: 0.9,
      top5Confs: [0.9, 0.7])

    XCTAssertEqual(probs.top1Label, "dog")
    XCTAssertEqual(probs.top5Labels, ["dog", "cat"])

    probs.top1Label = "cat"
    probs.top5Labels = ["cat", "dog"]
    XCTAssertEqual(probs.top1, "cat")
    XCTAssertEqual(probs.top5, ["cat", "dog"])

    var result = YOLOResult(
      orig_shape: CGSize(width: 10, height: 20), boxes: [], speed: 0, names: [])
    result.originalImage = UIImage()
    XCTAssertNotNil(result.originalImage)
  }

  func testNumItemsThresholdAtCreateAndLoad() throws {
    let predictor = try loadPredictor("yolo26n", task: .detect, numItemsThreshold: 3)
    XCTAssertEqual(predictor.numItemsThreshold, 3)

    let expectation = XCTestExpectation(description: "Load YOLO wrapper")
    let modelPath = try modelURL("yolo26n").path
    var yolo: YOLO?
    var threshold: Int?

    yolo = YOLO(modelPath, task: .detect, useGpu: false, numItemsThreshold: 4) { result in
      if case .success(let yolo) = result {
        threshold = yolo.getNumItemsThreshold()
      }
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 30)
    XCTAssertNotNil(yolo)
    XCTAssertEqual(threshold, 4)
  }

  func testRepresentativeStaticImageOutputShapes() throws {
    let image = testImage()
    let imageSize = image.extent.size

    let detect = try loadPredictor("yolo26n", task: .detect, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(detect.orig_shape, imageSize)
    XCTAssertLessThanOrEqual(detect.boxes.count, 5)
    XCTAssertNil(detect.masks)
    XCTAssertNil(detect.probs)
    XCTAssertNil(detect.semanticMask)

    let segment = try loadPredictor("yolo26n-seg", task: .segment, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(segment.orig_shape, imageSize)
    XCTAssertLessThanOrEqual(segment.boxes.count, 5)
    XCTAssertNotNil(segment.masks)
    XCTAssertNil(segment.probs)

    let classify = try loadPredictor("yolo26n-cls", task: .classify, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(classify.orig_shape, imageSize)
    XCTAssertNotNil(classify.probs)
    XCTAssertLessThanOrEqual(classify.probs?.top5.count ?? 0, 5)

    let pose = try loadPredictor("yolo26n-pose", task: .pose, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(pose.orig_shape, imageSize)
    XCTAssertLessThanOrEqual(pose.boxes.count, 5)
    XCTAssertLessThanOrEqual(pose.keypointsList.count, 5)

    let obb = try loadPredictor("yolo26n-obb", task: .obb, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(obb.orig_shape, imageSize)
    XCTAssertLessThanOrEqual(obb.obb.count, 5)

    let semantic = try loadPredictor("yolo26n-sem", task: .semantic, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(semantic.orig_shape, imageSize)
    XCTAssertNotNil(semantic.semanticMask)

    let depth = try loadPredictor("yolo26n-depth", task: .depth, numItemsThreshold: 5)
      .predictOnImage(image: image)
    XCTAssertEqual(depth.orig_shape, imageSize)
    let depthMap = try XCTUnwrap(depth.depthMap)
    XCTAssertEqual(depthMap.values.count, depthMap.width * depthMap.height)
    XCTAssertLessThan(depthMap.minDepth, depthMap.maxDepth)
  }
}
