// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOCore

/// SwiftUI overlay for classification results.
public struct ClassificationBanner: View {
  public let probs: Probs?

  public init(probs: Probs?) {
    self.probs = probs
  }

  public var body: some View {
    if let probs {
      VStack(alignment: .leading, spacing: 4) {
        ForEach(Array(zip(probs.top5, probs.top5Confs)), id: \.0) { label, conf in
          HStack {
            Text(label)
              .font(.system(size: 14, weight: label == probs.top1 ? .bold : .regular))
              .foregroundStyle(.white)
            Spacer()
            Text(String(format: "%.1f%%", conf * 100))
              .font(.system(size: 14, weight: .medium).monospacedDigit())
              .foregroundStyle(.white.opacity(0.8))
          }
        }
      }
      .padding(12)
      .background(.ultraThinMaterial)
      .clipShape(RoundedRectangle(cornerRadius: 12))
      .padding()
      .frame(maxHeight: .infinity, alignment: .bottom)
    }
  }
}
