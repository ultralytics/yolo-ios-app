// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing image-orientation utilities.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import CoreImage
import ImageIO
import UIKit

extension CGImagePropertyOrientation {
  /// Maps a `UIImage.Orientation` to its equivalent `CGImagePropertyOrientation`.
  ///
  /// The two enums use different raw values and need an explicit mapping.
  public init(_ uiOrientation: UIImage.Orientation) {
    switch uiOrientation {
    case .up: self = .up
    case .down: self = .down
    case .left: self = .left
    case .right: self = .right
    case .upMirrored: self = .upMirrored
    case .downMirrored: self = .downMirrored
    case .leftMirrored: self = .leftMirrored
    case .rightMirrored: self = .rightMirrored
    @unknown default: self = .up
    }
  }
}

extension UIImage {
  /// Returns an upright copy of this image with `imageOrientation == .up`.
  ///
  /// Photos from the camera roll and camera API often carry a non-`.up` `imageOrientation`
  /// (for example `.right` for a portrait photo from the rear camera). `CIImage(image:)` and
  /// many Core Graphics APIs ignore that metadata, so inference would run against the raw,
  /// rotated pixels. Call this to normalize the image before handing it off to CV pipelines.
  ///
  /// - Returns: A new `UIImage` whose pixels are visually upright, or `self` when already upright.
  public func uprightForYOLO() -> UIImage {
    guard imageOrientation != .up else { return self }
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = scale
    return UIGraphicsImageRenderer(size: size, format: format).image { _ in
      draw(at: .zero)
    }
  }
}
