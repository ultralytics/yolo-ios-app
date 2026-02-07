// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import CoreML

/// Provides custom IoU and confidence thresholds for Vision model predictions.
public final class ThresholdProvider: MLFeatureProvider, @unchecked Sendable {
  private let values: [String: MLFeatureValue]

  public var featureNames: Set<String> {
    Set(values.keys)
  }

  init(iouThreshold: Double = 0.45, confidenceThreshold: Double = 0.25) {
    values = [
      "iouThreshold": MLFeatureValue(double: iouThreshold),
      "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
    ]
  }

  public func featureValue(for featureName: String) -> MLFeatureValue? {
    values[featureName]
  }
}
