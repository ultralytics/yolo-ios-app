//
//  Detect.swift
//  YOLO

import Foundation
import UIKit
import Vision

extension ViewController {
  func postProcessDetect(request: VNRequest) {
    if let results = request.results as? [VNRecognizedObjectObservation] {
      var predictions = [DetectionResult]()
      for result in results {
        let prediction = DetectionResult(
          rect: result.boundingBox, label: result.labels[0].identifier,
          confidence: result.labels[0].confidence)
        predictions.append(prediction)
      }
      self.showBoundingBoxes(predictions: predictions)
    } else {
      self.showBoundingBoxes(predictions: [])
    }
  }

  func showBoundingBoxes(predictions: [DetectionResult]) {
    var str = ""
    // date
    let date = Date()
    let calendar = Calendar.current
    let hour = calendar.component(.hour, from: date)
    let minutes = calendar.component(.minute, from: date)
    let seconds = calendar.component(.second, from: date)
    let nanoseconds = calendar.component(.nanosecond, from: date)
    let sec_day =
      Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day

    self.labelSlider.text =
      String(predictions.count) + " items (max " + String(Int(slider.value)) + ")"
    let width = videoPreview.bounds.width  // 375 pix
    let height = videoPreview.bounds.height  // 812 pix

    if UIDevice.current.orientation == .portrait {

      // ratio = videoPreview AR divided by sessionPreset AR
      var ratio: CGFloat = 1.0
      if videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)  // .photo
      } else {
        ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
      }

      for i in 0..<boundingBoxViews.count {
        if i < predictions.count && i < Int(slider.value) {
          let prediction = predictions[i]

          var rect = prediction.rect  // normalized xywh, origin lower left
          switch UIDevice.current.orientation {
          case .portraitUpsideDown:
            rect = CGRect(
              x: 1.0 - rect.origin.x - rect.width,
              y: 1.0 - rect.origin.y - rect.height,
              width: rect.width,
              height: rect.height)
          case .landscapeLeft:
            rect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .landscapeRight:
            rect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .unknown:
            print("The device orientation is unknown, the predictions may be affected")
            fallthrough
          default: break
          }

          if ratio >= 1 {  // iPhone ratio = 1.218
            let offset = (1 - ratio) * (0.5 - rect.minX)
            if task == .detect {
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
              rect = rect.applying(transform)
            } else {
              let transform = CGAffineTransform(translationX: offset, y: 0)
              rect = rect.applying(transform)
            }
            rect.size.width *= ratio
          } else {  // iPad ratio = 0.75
            let offset = (ratio - 1) * (0.5 - rect.maxY)
            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
            rect = rect.applying(transform)
            ratio = (height / width) / (3.0 / 4.0)
            rect.size.height /= ratio
          }

          // Scale normalized to pixels [375, 812] [width, height]
          rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

          // The labels array is a list of VNClassificationObservation objects,
          // with the highest scoring class first in the list.
          let bestClass = prediction.label
          let confidence = prediction.confidence
          // print(confidence, rect)  // debug (confidence, xywh) with xywh origin top left (pixels)
          let label = String(format: "%@ %.1f", bestClass, confidence * 100)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          // Show the bounding box.
          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: colors[bestClass] ?? UIColor.white,
            alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)

          if developerMode {
            // Write
            if save_detections {
              str += String(
                format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
            }
          }
        } else {
          boundingBoxViews[i].hide()
        }
      }
    } else {
      let frameAspectRatio = longSide / shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / shortSide
        scaleX = scaleY
        offsetX = (longSide * scaleX - width) / 2
      } else {
        scaleX = width / longSide
        scaleY = scaleX
        offsetY = (shortSide * scaleY - height) / 2
      }

      for i in 0..<boundingBoxViews.count {
        if i < predictions.count {
          let prediction = predictions[i]

          var rect = prediction.rect
          if task == .detect {
            rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
            rect.origin.y =
              height
              - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide
                * scaleY)
            rect.size.width *= longSide * scaleX
            rect.size.height *= shortSide * scaleY
          } else {

            rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
            rect.origin.y = rect.origin.y * shortSide * scaleY - offsetY
            rect.size.width *= longSide * scaleX
            rect.size.height *= shortSide * scaleY
          }

          let bestClass = prediction.label
          let confidence = prediction.confidence

          let label = String(format: "%@ %.1f", bestClass, confidence * 100)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          // Show the bounding box.
          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: colors[bestClass] ?? UIColor.white,
            alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
        } else {
          boundingBoxViews[i].hide()
        }
      }
    }
    // Write
    if developerMode {
      if save_detections {
        saveText(text: str, file: "detections.txt")  // Write stats for each detection
      }
      if save_frames {
        str = String(
          format: "%.3f %.3f %.3f %.3f %.1f %.1f %.1f\n",
          sec_day, freeSpace(), memoryUsage(), UIDevice.current.batteryLevel,
          self.t1 * 1000, self.t2 * 1000, 1 / self.t4)
        saveText(text: str, file: "frames.txt")  // Write stats for each image
      }
    }

    // Debug
    // print(str)
    // print(UIDevice.current.identifierForVendor!)
    // saveImage()
  }

  func hideBoundingBoxes() {
    for box in boundingBoxViews {
      box.hide()
    }
  }
}

struct DetectionResult {
  let rect: CGRect
  let label: String
  let confidence: Float
}
