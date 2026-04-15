// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import UIKit
import XCTest

@testable import YOLO

final class UIImageOrientationTests: XCTestCase {

  func testUprightForYOLONormalizesCIImageBackedOrientation() {
    let source = UIImage(
      ciImage: CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 40, height: 20)),
      scale: 1,
      orientation: .right
    )

    XCTAssertNil(source.cgImage)
    XCTAssertEqual(source.imageOrientation, .right)

    let normalized = source.uprightForYOLO()

    XCTAssertEqual(normalized.imageOrientation, .up)
    XCTAssertNotNil(normalized.cgImage)
  }
}
