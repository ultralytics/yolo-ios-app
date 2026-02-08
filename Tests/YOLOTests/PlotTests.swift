// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import QuartzCore
import UIKit
import XCTest

@testable import YOLO

/// Minimal tests for Plot visualization functions
@MainActor class PlotTests: XCTestCase {

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
    XCTAssertNotNil(clearComponents)  // Should still work
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
    // Should return an image based on the original
    XCTAssertGreaterThan(outputImage.size.width, 0)
  }

  func testComposeImageWithMask() {
    // Test composeImageWithMask function
    // Create a simple test image
    UIGraphicsBeginImageContextWithOptions(CGSize(width: 50, height: 50), false, 1.0)
    UIColor.red.setFill()
    UIRectFill(CGRect(x: 0, y: 0, width: 50, height: 50))
    let baseUIImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    // Create a mask image
    UIGraphicsBeginImageContextWithOptions(CGSize(width: 50, height: 50), false, 1.0)
    UIColor.blue.setFill()
    UIRectFill(CGRect(x: 0, y: 0, width: 50, height: 50))
    let maskUIImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    let baseImage = baseUIImage.cgImage!
    let maskImage = maskUIImage.cgImage!

    let composedImage = composeImageWithMask(baseImage: baseImage, maskImage: maskImage)

    XCTAssertNotNil(composedImage)
    XCTAssertEqual(composedImage?.size.width, 50)
    XCTAssertEqual(composedImage?.size.height, 50)
  }

  func testOBBShapeLayerBundleInitialization() {
    // Test OBBShapeLayerBundle initialization
    let bundle = OBBShapeLayerBundle()

    XCTAssertNotNil(bundle.shapeLayer)
    XCTAssertNotNil(bundle.textLayer)
    XCTAssertEqual(bundle.shapeLayer.strokeColor, UIColor.red.cgColor)
    XCTAssertEqual(bundle.shapeLayer.fillColor, UIColor.clear.cgColor)
    XCTAssertEqual(bundle.textLayer.fontSize, 14)
  }

  func testOBBRendererInitialization() {
    // Test OBBRenderer initialization
    let renderer = OBBRenderer()

    // Should not crash on initialization
    XCTAssertNotNil(renderer)
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

  func testDrawPoseOnCIImageWithEmptyKeypoints() {
    // Test drawPoseOnCIImage with empty keypoints
    let inputImage = CIImage(color: .cyan).cropped(to: CGRect(x: 0, y: 0, width: 300, height: 200))
    let emptyKeypoints: [[(x: Float, y: Float)]] = []
    let emptyConfs: [[Float]] = []
    let emptyBoxes: [Box] = []

    let outputImage = drawPoseOnCIImage(
      ciImage: inputImage,
      keypointsList: emptyKeypoints,
      confsList: emptyConfs,
      boundingBoxes: emptyBoxes,
      originalImageSize: CGSize(width: 300, height: 200)
    )

    XCTAssertNotNil(outputImage)
    XCTAssertGreaterThan(outputImage?.size.width ?? 0, 0)
  }

  func testDrawObbDetectionsWithReuse() {
    // Test OBBRenderer drawObbDetectionsWithReuse method
    let renderer = OBBRenderer()
    let layer = CALayer()
    layer.frame = CGRect(x: 0, y: 0, width: 100, height: 100)

    let obb = OBB(cx: 0.5, cy: 0.5, w: 0.3, h: 0.2, angle: 0.2)
    let obbResult = OBBResult(box: obb, confidence: 0.9, cls: "plane", index: 0)

    // Should not crash
    renderer.drawObbDetectionsWithReuse(
      obbDetections: [obbResult],
      on: layer,
      imageViewSize: CGSize(width: 100, height: 100)
    )

    XCTAssertTrue(true)  // Test passes if no crash
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
