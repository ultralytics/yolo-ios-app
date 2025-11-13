// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing image processing utilities.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This file provides utility functions for image processing, including orientation correction
//  for images loaded from the photo library or other sources that may have orientation metadata.

import UIKit

/// Normalizes image orientation to `.up` for proper YOLO model processing.
///
/// Images from the photo library often have orientation metadata that isn't applied to pixel data
/// when converting to `CIImage`. This function normalizes the image by applying the orientation
/// transformation to the actual pixels.
///
/// - Parameter uiImage: The input image that may have orientation metadata.
/// - Returns: A `UIImage` with `.up` orientation. Returns the original image if already normalized.
public func normalizeImageOrientation(_ uiImage: UIImage) -> UIImage {
  guard uiImage.imageOrientation != .up else { return uiImage }

  UIGraphicsBeginImageContextWithOptions(uiImage.size, false, uiImage.scale)
  uiImage.draw(in: CGRect(origin: .zero, size: uiImage.size))
  let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()

  return normalizedImage ?? uiImage
}

