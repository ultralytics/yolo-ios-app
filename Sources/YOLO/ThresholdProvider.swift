// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  Threshold Provider for Ultralytics YOLO App
//  This class is designed to supply custom Intersection Over Union (IoU) and confidence thresholds
//  for the YOLOv8 object detection models within the Ultralytics YOLO app. It conforms to the MLFeatureProvider protocol,
//  allowing these thresholds to be dynamically adjusted and applied to model predictions.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ThresholdProvider class enables fine-tuning of detection sensitivity and precision by modifying the
//  IoU and confidence thresholds, which are crucial for balancing the detection accuracy and the number of false positives.

import CoreML

/// Provides custom IoU and confidence thresholds for adjusting model predictions.
class ThresholdProvider: MLFeatureProvider {
  /// Stores IoU and confidence thresholds as MLFeatureValue objects.
  var values: [String: MLFeatureValue]

  /// The set of feature names provided by this provider.
  var featureNames: Set<String> {
    return Set(values.keys)
  }

  /// Initializes the provider with specified IoU and confidence thresholds.
  /// - Parameters:
  ///   - iouThreshold: The IoU threshold for determining object overlap.
  ///   - confidenceThreshold: The minimum confidence for considering a detection valid.
  init(iouThreshold: Double = 0.45, confidenceThreshold: Double = 0.25) {
    values = [
      "iouThreshold": MLFeatureValue(double: iouThreshold),
      "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
    ]
  }

  /// Returns the feature value for the given feature name.
  /// - Parameter featureName: The name of the feature.
  /// - Returns: The MLFeatureValue object corresponding to the feature name.
  func featureValue(for featureName: String) -> MLFeatureValue? {
    return values[featureName]
  }
}
