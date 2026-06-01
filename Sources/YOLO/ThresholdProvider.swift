// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, supplying runtime IoU and confidence thresholds to YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  ThresholdProvider conforms to MLFeatureProvider so it can inject Intersection over Union (IoU) and confidence
//  thresholds into a Core ML model at inference time. Adjusting these values tunes detection sensitivity and the
//  trade-off between recall and false positives.

import CoreML

/// Provides custom IoU and confidence thresholds for adjusting model predictions.
public final class ThresholdProvider: MLFeatureProvider {
  /// Stores IoU and confidence thresholds as MLFeatureValue objects.
  let values: [String: MLFeatureValue]

  /// The set of feature names provided by this provider.
  public var featureNames: Set<String> {
    return Set(values.keys)
  }

  /// Creates a provider with the given IoU and confidence thresholds.
  /// - Parameters:
  ///   - iouThreshold: IoU threshold used by NMS to merge overlapping detections.
  ///   - confidenceThreshold: Minimum confidence required for a detection to be kept.
  init(iouThreshold: Double = 0.7, confidenceThreshold: Double = 0.25) {
    values = [
      "iouThreshold": MLFeatureValue(double: iouThreshold),
      "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
    ]
  }

  /// Returns the feature value for the given feature name.
  /// - Parameter featureName: The feature name to look up (`iouThreshold` or `confidenceThreshold`).
  /// - Returns: The matching `MLFeatureValue`, or `nil` if the name is unknown.
  public func featureValue(for featureName: String) -> MLFeatureValue? {
    return values[featureName]
  }
}
