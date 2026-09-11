// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreGraphics
import XCTest

@testable import UltralyticsYOLO

/// Verifies the model-input letterbox is robust across camera aspect ratios (16:9 and 4:3, portrait and landscape).
///
/// The camera capture preset determines the frame aspect (e.g. `.photo`/`.vga640x480` are 4:3, `.hd1280x720` is
/// 16:9). The letterbox scales the frame to fit the square model input and pads the short axis, so the bars land
/// left/right for tall frames and top/bottom for wide frames. `inputRect(fromModelRect:)` must invert that for any
/// aspect so detections map back onto the full camera frame regardless of where the bars fall.
final class LetterboxTests: XCTestCase {

  /// A predictor configured for a 640×640 model input and a given camera frame size.
  private func predictor(_ cameraWidth: CGFloat, _ cameraHeight: CGFloat) -> BasePredictor {
    let p = BasePredictor()
    p.modelInputSize = (width: 640, height: 640)
    p.inputSize = CGSize(width: cameraWidth, height: cameraHeight)
    return p
  }

  /// The active (non-padded) model rect for each format must map back to the full camera frame.
  ///
  /// `activeRect` is the letterboxed image region inside the 640×640 input, derived from `gain = min(640/W, 640/H)`
  /// and centered padding — tall frames pad x (left/right bars), wide frames pad y (top/bottom bars).
  func testActiveRegionMapsToFullFrame() {
    let cases: [(w: CGFloat, h: CGFloat, activeRect: CGRect, desc: String)] = [
      (720, 1280, CGRect(x: 140, y: 0, width: 360, height: 640), "16:9 portrait → left/right bars"),
      (480, 640, CGRect(x: 80, y: 0, width: 480, height: 640), "4:3 portrait → left/right bars"),
      (
        1280, 720, CGRect(x: 0, y: 140, width: 640, height: 360), "16:9 landscape → top/bottom bars"
      ),
      (640, 480, CGRect(x: 0, y: 80, width: 640, height: 480), "4:3 landscape → top/bottom bars"),
    ]
    for c in cases {
      let mapped = predictor(c.w, c.h).inputRect(fromModelRect: c.activeRect)
      XCTAssertEqual(mapped.minX, 0, accuracy: 1, "\(c.desc): minX")
      XCTAssertEqual(mapped.minY, 0, accuracy: 1, "\(c.desc): minY")
      XCTAssertEqual(mapped.width, c.w, accuracy: 1, "\(c.desc): width")
      XCTAssertEqual(mapped.height, c.h, accuracy: 1, "\(c.desc): height")
    }
  }

  /// The model-space center must map to the camera-frame center for any aspect ratio.
  func testCenterMapsToCenter() {
    for (w, h) in [(720.0, 1280.0), (480.0, 640.0), (1280.0, 720.0), (640.0, 480.0)] {
      let mapped = predictor(CGFloat(w), CGFloat(h))
        .inputRect(fromModelRect: CGRect(x: 319, y: 319, width: 2, height: 2))
      XCTAssertEqual(mapped.midX, CGFloat(w) / 2, accuracy: 2, "center X for \(w)x\(h)")
      XCTAssertEqual(mapped.midY, CGFloat(h) / 2, accuracy: 2, "center Y for \(w)x\(h)")
    }
  }
}
