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
  guard let cgImage = uiImage.cgImage else { return uiImage }

  let width = CGFloat(cgImage.width)
  let height = CGFloat(cgImage.height)

  var transform = CGAffineTransform.identity
  var contextSize = CGSize(width: width, height: height)

  // Rotation
  switch uiImage.imageOrientation {
  case .down, .downMirrored:
    transform = transform.translatedBy(x: width, y: height).rotated(by: .pi)
  case .left, .leftMirrored:
    transform = transform.translatedBy(x: width, y: 0).rotated(by: .pi / 2)
    contextSize = CGSize(width: height, height: width)
  case .right, .rightMirrored:
    transform = transform.translatedBy(x: 0, y: height).rotated(by: -.pi / 2)
    contextSize = CGSize(width: height, height: width)
  default:
    break
  }

  // Mirroring
  switch uiImage.imageOrientation {
  case .upMirrored, .downMirrored:
    transform = transform.translatedBy(x: width, y: 0).scaledBy(x: -1, y: 1)
  case .leftMirrored, .rightMirrored:
    transform = transform.translatedBy(x: height, y: 0).scaledBy(x: -1, y: 1)
  default:
    break
  }

  let colorSpace = cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB()

  guard
    let context = CGContext(
      data: nil,
      width: Int(contextSize.width),
      height: Int(contextSize.height),
      bitsPerComponent: cgImage.bitsPerComponent,
      bytesPerRow: 0,
      space: colorSpace,
      bitmapInfo: cgImage.bitmapInfo.rawValue
    )
  else {
    return uiImage
  }

  context.concatenate(transform)

  let drawRect: CGRect
  switch uiImage.imageOrientation {
  case .left, .leftMirrored, .right, .rightMirrored:
    drawRect = CGRect(x: 0, y: 0, width: height, height: width)
  default:
    drawRect = CGRect(x: 0, y: 0, width: width, height: height)
  }

  context.draw(cgImage, in: drawRect)

  guard let normalizedCGImage = context.makeImage() else { return uiImage }
  return UIImage(cgImage: normalizedCGImage, scale: uiImage.scale, orientation: .up)
}
