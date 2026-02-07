// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOCore

/// Optional slider controls bound to YOLOConfiguration thresholds.
public struct ThresholdControls: View {
  @Binding public var configuration: YOLOConfiguration

  public init(configuration: Binding<YOLOConfiguration>) {
    self._configuration = configuration
  }

  public var body: some View {
    VStack(spacing: 8) {
      HStack {
        Text("Confidence")
          .font(.caption)
          .foregroundStyle(.white)
        Slider(value: $configuration.confidenceThreshold, in: 0...1)
        Text(String(format: "%.2f", configuration.confidenceThreshold))
          .font(.caption.monospacedDigit())
          .foregroundStyle(.white)
          .frame(width: 36)
      }

      HStack {
        Text("IoU")
          .font(.caption)
          .foregroundStyle(.white)
        Slider(value: $configuration.iouThreshold, in: 0...1)
        Text(String(format: "%.2f", configuration.iouThreshold))
          .font(.caption.monospacedDigit())
          .foregroundStyle(.white)
          .frame(width: 36)
      }
    }
    .padding(.horizontal)
    .padding(.vertical, 8)
    .background(.ultraThinMaterial)
    .clipShape(RoundedRectangle(cornerRadius: 10))
    .padding(.horizontal)
  }
}
