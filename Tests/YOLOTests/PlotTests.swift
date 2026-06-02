// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import CoreML
import QuartzCore
import UIKit
import XCTest

@testable import YOLO

/// Minimal tests for Plot visualization functions
class PlotTests: XCTestCase {

  func testUltralyticsColorsExist() {
    // Test ultralyticsColors array is populated
    XCTAssertGreaterThan(ultralyticsColors.count, 0)
    XCTAssertEqual(ultralyticsColors.count, 20)

    // Test first and last colors exist
    XCTAssertNotNil(ultralyticsColors.first)
    XCTAssertNotNil(ultralyticsColors.last)
  }

  func testPosePaletteAndConstants() {
    // Test pose-related constants are properly initialized
    XCTAssertGreaterThan(posePalette.count, 0)
    XCTAssertEqual(posePalette.count, 20)
    XCTAssertEqual(posePalette[0].count, 3)  // RGB values

    XCTAssertGreaterThan(limbColorIndices.count, 0)
    XCTAssertGreaterThan(kptColorIndices.count, 0)
    XCTAssertGreaterThan(skeleton.count, 0)
    XCTAssertEqual(skeleton[0].count, 2)  // Each bone connects 2 points
  }

  func testUIColorRGBComponentsExtension() {
    // Test UIColor RGB components extension
    let redColor = UIColor.red
    let components = redColor.toRGBComponents()

    XCTAssertNotNil(components)
    XCTAssertEqual(components?.red, 255)
    XCTAssertEqual(components?.green, 0)
    XCTAssertEqual(components?.blue, 0)

    let blueColor = UIColor.blue
    let blueComponents = blueColor.toRGBComponents()
    XCTAssertEqual(blueComponents?.blue, 255)

    let clearColor = UIColor.clear
    let clearComponents = clearColor.toRGBComponents()
    XCTAssertNotNil(clearComponents)  // Clear color still resolves to RGB components
  }

  func testDrawYOLODetections() {
    // Test drawYOLODetections function creates image
    let inputImage = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

    let box = Box(
      index: 0,
      cls: "person",
      conf: 0.85,
      xywh: CGRect(x: 10, y: 10, width: 30, height: 40),
      xywhn: CGRect(x: 0.1, y: 0.1, width: 0.3, height: 0.4)
    )

    let result = YOLOResult(
      orig_shape: CGSize(width: 100, height: 100),
      boxes: [box],
      speed: 0.1,
      names: ["person"]
    )

    let outputImage = drawYOLODetections(on: inputImage, result: result)

    XCTAssertNotNil(outputImage)
    XCTAssertGreaterThan(outputImage.size.width, 0)
    XCTAssertGreaterThan(outputImage.size.height, 0)
  }

  func testDrawYOLODetectionsEmptyBoxes() {
    // Test drawYOLODetections with empty boxes
    let inputImage = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 150))

    let result = YOLOResult(
      orig_shape: CGSize(width: 200, height: 150),
      boxes: [],
      speed: 0.05,
      names: []
    )

    let outputImage = drawYOLODetections(on: inputImage, result: result)

    XCTAssertNotNil(outputImage)
    XCTAssertEqual(outputImage.size.width, 200)
    XCTAssertEqual(outputImage.size.height, 150)
  }

  func testDrawYOLOClassifications() {
    // Test drawYOLOClassifications function
    let inputImage = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 224, height: 224))

    let probs = Probs(
      top1: "cat",
      top5: ["cat", "dog", "bird", "fish", "mouse"],
      top1Conf: 0.95,
      top5Confs: [0.95, 0.8, 0.6, 0.4, 0.2]
    )

    let result = YOLOResult(
      orig_shape: CGSize(width: 224, height: 224),
      boxes: [],
      probs: probs,
      speed: 0.02,
      names: ["cat", "dog", "bird", "fish", "mouse"]
    )

    let outputImage = drawYOLOClassifications(on: inputImage, result: result)

    XCTAssertNotNil(outputImage)
    XCTAssertGreaterThan(outputImage.size.width, 0)
    XCTAssertGreaterThan(outputImage.size.height, 0)
  }

  func testDrawYOLOClassificationsWithoutProbs() {
    // Test drawYOLOClassifications without probs returns original
    let inputImage = CIImage(color: .yellow).cropped(
      to: CGRect(x: 0, y: 0, width: 100, height: 100))

    let result = YOLOResult(
      orig_shape: CGSize(width: 100, height: 100),
      boxes: [],
      probs: nil,
      speed: 0.02,
      names: []
    )

    let outputImage = drawYOLOClassifications(on: inputImage, result: result)

    XCTAssertNotNil(outputImage)
    // Falls back to the original image when probs are absent
    XCTAssertGreaterThan(outputImage.size.width, 0)
  }

  func testDrawOBBsOnCIImageWithEmptyDetections() {
    // Test drawOBBsOnCIImage with empty detections
    let inputImage = CIImage(color: CIColor(red: 0.5, green: 0.0, blue: 0.5, alpha: 1.0)).cropped(
      to: CGRect(x: 0, y: 0, width: 100, height: 100))
    let emptyDetections: [OBBResult] = []

    let outputImage = drawOBBsOnCIImage(ciImage: inputImage, obbDetections: emptyDetections)

    XCTAssertNotNil(outputImage)
    XCTAssertGreaterThan(outputImage?.size.width ?? 0, 0)
  }

  func testDrawOBBsOnCIImageWithDetections() {
    // Test drawOBBsOnCIImage with actual detections
    let inputImage = CIImage(color: CIColor(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0)).cropped(
      to: CGRect(x: 0, y: 0, width: 200, height: 200))

    let obb = OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.5)
    let obbResult = OBBResult(box: obb, confidence: 0.8, cls: "ship", index: 1)

    let outputImage = drawOBBsOnCIImage(ciImage: inputImage, obbDetections: [obbResult])

    XCTAssertNotNil(outputImage)
    XCTAssertEqual(outputImage?.size.width, 200)
    XCTAssertEqual(outputImage?.size.height, 200)
  }

  func testOBBPolygonPixelConversion() {
    let obb = OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.2)
    let obbResult = OBBResult(box: obb, confidence: 0.9, cls: "plane", index: 0)
    XCTAssertEqual(obbResult.cls, "plane")
    XCTAssertEqual(obbResult.confidence, 0.9, accuracy: 0.001)

    // Verify polygon corners are computed correctly in pixel space
    let corners = obb.toPolygon(imageSize: CGSize(width: 100, height: 100))
    XCTAssertEqual(corners.count, 4)

    // All corners should be near the center of the 100x100 image
    for corner in corners {
      XCTAssertGreaterThan(corner.x, 0)
      XCTAssertLessThan(corner.x, 100)
      XCTAssertGreaterThan(corner.y, 0)
      XCTAssertLessThan(corner.y, 100)
    }
  }

  func testDrawYOLOPoseWithBoxes() {
    // Test integrated pose with boxes rendering
    let inputImage = CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 400, height: 300))

    let keypoints: [(x: Float, y: Float)] = Array(repeating: (0.5, 0.5), count: 17)
    let confs: [Float] = Array(repeating: 0.8, count: 17)
    let box = Box(
      index: 0, cls: "person", conf: 0.9,
      xywh: CGRect(x: 100, y: 50, width: 200, height: 200),
      xywhn: CGRect(x: 0.25, y: 0.17, width: 0.5, height: 0.67)
    )

    let outputImage = drawYOLOPoseWithBoxes(
      ciImage: inputImage,
      keypointsList: [keypoints],
      confsList: [confs],
      boundingBoxes: [box]
    )

    XCTAssertNotNil(outputImage)
    XCTAssertGreaterThan(outputImage?.size.width ?? 0, 0)
  }

  func testGenerateCombinedMaskImageProducesCorrectPerInstanceMasks() {
    // Prototype masks: shape [1, C=2, H=4, W=4]. Channel 0 = index value, channel 1 = constant 0.
    let C = 2
    let H = 4
    let W = 4
    let HW = 16
    let protos = try! MLMultiArray(shape: [1, C, H, W] as [NSNumber], dataType: .float32)
    let pPtr = protos.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<HW { pPtr[i] = Float(i) }  // channel 0: 0..15
    for i in 0..<HW { pPtr[HW + i] = 0 }  // channel 1: zeros

    // One detection with coefficients [1, 0] -> combinedMask == channel 0 of protos.
    let detected: [(CGRect, Int, Float, [Float])] = [
      (CGRect(x: 0, y: 0, width: 4, height: 4), 0, 0.9, [1, 0])
    ]

    guard
      let result = generateCombinedMaskImage(
        detectedObjects: detected, protos: protos,
        inputWidth: W, inputHeight: H, threshold: 0.5,
        cropRect: nil, returnIndividualMasks: true) as? (CGImage?, [[[Float]]]?),
      let masks = result.1
    else {
      XCTFail("generateCombinedMaskImage returned nil or wrong type")
      return
    }

    XCTAssertEqual(masks.count, 1)
    XCTAssertEqual(masks[0].count, H)
    XCTAssertEqual(masks[0][0].count, W)
    // Each entry must equal the matmul result (channel 0 = linear index).
    for y in 0..<H {
      for x in 0..<W {
        XCTAssertEqual(masks[0][y][x], Float(y * W + x), accuracy: 1e-4)
      }
    }
  }

  func testGenerateCombinedMaskImageCropsPerInstanceMasksToDetectionBox() {
    let C = 1
    let H = 4
    let W = 4
    let protos = try! MLMultiArray(shape: [1, C, H, W] as [NSNumber], dataType: .float32)
    let pPtr = protos.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<(H * W) { pPtr[i] = 1 }

    let detected: [(CGRect, Int, Float, [Float])] = [
      (CGRect(x: 1, y: 1, width: 2, height: 2), 0, 0.9, [1])
    ]

    guard
      let result = generateCombinedMaskImage(
        detectedObjects: detected, protos: protos,
        inputWidth: W, inputHeight: H,
        cropRect: nil, returnIndividualMasks: true) as? (CGImage?, [[[Float]]]?),
      let masks = result.1
    else {
      XCTFail("generateCombinedMaskImage returned nil or wrong type")
      return
    }

    XCTAssertEqual(masks.count, 1)
    for y in 0..<H {
      for x in 0..<W {
        let expected: Float = (1..<3).contains(x) && (1..<3).contains(y) ? 1 : 0
        XCTAssertEqual(masks[0][y][x], expected, accuracy: 1e-4)
      }
    }
  }

  func testGenerateCombinedMaskImageEmptyDetectionsReturnsGracefully() {
    // Zero detections must not crash on the unsafe-buffer paths; it returns no image and empty masks.
    let protos = try! MLMultiArray(shape: [1, 2, 4, 4] as [NSNumber], dataType: .float32)
    let result =
      generateCombinedMaskImage(
        detectedObjects: [], protos: protos,
        inputWidth: 4, inputHeight: 4, threshold: 0.5,
        cropRect: nil, returnIndividualMasks: true) as? (CGImage?, [[[Float]]]?)
    XCTAssertNotNil(result)
    XCTAssertNil(result?.0)
    XCTAssertEqual(result?.1?.count, 0)
  }

  func testDrawYOLOSegmentationWithBoxes() {
    // Test integrated segmentation with boxes rendering
    let inputImage = CIImage(color: .magenta).cropped(
      to: CGRect(x: 0, y: 0, width: 300, height: 300))

    let box = Box(
      index: 1, cls: "car", conf: 0.85,
      xywh: CGRect(x: 50, y: 50, width: 100, height: 80),
      xywhn: CGRect(x: 0.17, y: 0.17, width: 0.33, height: 0.27)
    )

    let outputImage = drawYOLOSegmentationWithBoxes(
      ciImage: inputImage,
      boxes: [box],
      maskImage: nil
    )

    XCTAssertNotNil(outputImage)
    XCTAssertGreaterThan(outputImage?.size.width ?? 0, 0)
  }
}
