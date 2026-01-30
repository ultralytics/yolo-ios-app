// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation
import UIKit
import YOLO

struct ModelSelectionManager {
  enum ModelSize: String, CaseIterable {
    case n, s, m, l, x

    var displayName: String {
      switch self {
      case .n: return "nano"
      case .s: return "small"
      case .m: return "medium"
      case .l: return "large"
      case .x: return "xlarge"
      }
    }
  }

  struct ModelInfo {
    let name: String
    let url: URL?
    let isLocal: Bool
    let size: ModelSize?
  }

  private static let modelSizeRegex: NSRegularExpression = {
    do {
      return try NSRegularExpression(pattern: "^\\d+([nsmxl])", options: [])
    } catch {
      print("Failed to create model size regex: \(error)")
      return try! NSRegularExpression(pattern: "^$", options: [])
    }
  }()

  static func categorizeModels(
    from models: [(name: String, url: URL?, isLocal: Bool)],
    preferYOLO26: Bool = true
  ) -> [ModelSize: ModelInfo] {
    var standardModels: [ModelSize: ModelInfo] = [:]
    let targetVersion = preferYOLO26 ? 26 : 11

    for model in models {
      let baseName = (model.name as NSString).deletingPathExtension.lowercased()
      guard baseName.hasPrefix("yolo") else { continue }

      guard
        let sizeChar = extractSizeFromModelName(baseName),
        let size = ModelSize(rawValue: String(sizeChar))
      else { continue }

      let currentVersion = extractVersionNumber(from: baseName)

      func isBetterCandidate(new: ModelInfo, current: ModelInfo?) -> Bool {
        guard let current = current else { return true }
        let currentName = (current.name as NSString).deletingPathExtension.lowercased()
        let existingVersion = extractVersionNumber(from: currentName)

        let newMatchesTarget = currentVersion == targetVersion
        let existingMatchesTarget = existingVersion == targetVersion

        if newMatchesTarget != existingMatchesTarget {
          return newMatchesTarget
        }

        if new.isLocal != current.isLocal {
          return new.isLocal
        }

        if let newVer = currentVersion, let existingVer = existingVersion, newVer != existingVer {
          return newVer > existingVer
        }

        return false
      }

      let info = ModelInfo(
        name: model.name,
        url: model.url,
        isLocal: model.isLocal,
        size: size
      )

      if isBetterCandidate(new: info, current: standardModels[size]) {
        standardModels[size] = info
      }
    }

    return standardModels
  }

  private static func extractVersionNumber(from name: String) -> Int? {
    // Extract version number from names like "yolo11n", "yolo26n", etc.
    let pattern = "^yolo(\\d+)"
    if let regex = try? NSRegularExpression(pattern: pattern, options: []),
      let match = regex.firstMatch(
        in: name, options: [], range: NSRange(location: 0, length: name.count)),
      match.numberOfRanges > 1
    {
      let versionRange = match.range(at: 1)
      if let range = Range(versionRange, in: name) {
        return Int(String(name[range]))
      }
    }
    return nil
  }

  private static func extractSizeFromModelName(_ baseName: String) -> Character? {
    let taskSuffixes = ["-seg", "-cls", "-pose", "-obb"]

    var nameWithoutSuffix = baseName
    for suffix in taskSuffixes {
      if baseName.hasSuffix(suffix) {
        nameWithoutSuffix = String(baseName.dropLast(suffix.count))
        break
      }
    }

    if nameWithoutSuffix.hasPrefix("yolo") && !nameWithoutSuffix.dropFirst(4).isEmpty {
      let afterYolo = nameWithoutSuffix.dropFirst(4)
      let afterYoloString = String(afterYolo)
      let range = NSRange(location: 0, length: afterYoloString.count)

      if let match = modelSizeRegex.firstMatch(in: afterYoloString, options: [], range: range),
        match.numberOfRanges > 1
      {
        let sizeRange = match.range(at: 1)
        if let range = Range(sizeRange, in: afterYoloString) {
          return afterYoloString[range].first
        }
      }
    }

    return nil
  }

  private static func removeTaskSuffix(from name: String) -> String {
    let taskSuffixes = ["-seg", "-cls", "-pose", "-obb"]

    var result = name
    for suffix in taskSuffixes {
      if name.hasSuffix(suffix) {
        result = String(name.dropLast(suffix.count))
        break
      }
    }

    if result.lowercased().hasPrefix("yolo") {
      let afterYolo = result.dropFirst(4)
      result = "YOLO" + afterYolo
    }

    return result
  }

  static func setupSegmentedControl(
    _ control: UISegmentedControl, standardModels: [ModelSize: ModelInfo], currentTask: YOLOTask,
    preserveSelection: Bool = false
  ) {
    let previousSelection = preserveSelection ? control.selectedSegmentIndex : -1

    control.removeAllSegments()

    for (index, size) in ModelSize.allCases.enumerated() {
      if let model = standardModels[size] {
        let fullName = (model.name as NSString).deletingPathExtension
        let displayTitle = removeTaskSuffix(from: fullName)

        // Check if model is downloaded using ModelCacheManager for remote models
        // Use the model name without extension as the key (e.g., "yolo11n", "yolo11m-seg")
        let modelKey = (model.name as NSString).deletingPathExtension
        let isDownloaded =
          model.isLocal
          || (model.url != nil && ModelCacheManager.shared.isModelDownloaded(key: modelKey))

        let titleWithIcon: String
        if isDownloaded {
          titleWithIcon = displayTitle
        } else {
          titleWithIcon = "⤓ \(displayTitle)"
        }
        control.insertSegment(withTitle: titleWithIcon, at: index, animated: false)
        control.setEnabled(true, forSegmentAt: index)
      } else {
        control.insertSegment(withTitle: size.displayName, at: index, animated: false)
        control.setEnabled(false, forSegmentAt: index)
      }
    }

    if preserveSelection && previousSelection >= 0 && previousSelection < control.numberOfSegments {
      control.selectedSegmentIndex = previousSelection
    } else {
      control.selectedSegmentIndex = 0
    }

    setupResponsiveFontSize(for: control)

    control.setNeedsLayout()
    control.layoutIfNeeded()

    DispatchQueue.main.async {
      updateSegmentAppearance(control, standardModels: standardModels, currentTask: currentTask)
    }
  }

  static func updateSegmentAppearance(
    _ control: UISegmentedControl, standardModels: [ModelSize: ModelInfo], currentTask: YOLOTask
  ) {
    for (index, size) in ModelSize.allCases.enumerated() {
      guard index < control.numberOfSegments else { break }

      if let model = standardModels[size] {
        // Check if model is downloaded using ModelCacheManager for remote models
        // Use the model name without extension as the key (e.g., "yolo11n", "yolo11m-seg")
        let modelKey = (model.name as NSString).deletingPathExtension
        let isDownloaded =
          model.isLocal
          || (model.url != nil && ModelCacheManager.shared.isModelDownloaded(key: modelKey))

        if !isDownloaded {
          setSegmentTextColor(control, at: index, color: .systemGray)
        } else {
          setSegmentTextColor(control, at: index, color: .white)
        }
      } else if !control.isEnabledForSegment(at: index) {
        setSegmentTextColor(control, at: index, color: .gray)
      }
    }
  }

  private static func setSegmentTextColor(
    _ control: UISegmentedControl, at index: Int, color: UIColor
  ) {
    if #available(iOS 13.0, *) {
      if let title = control.titleForSegment(at: index) {
        control.setTitle(title, forSegmentAt: index)

        if let image = control.imageForSegment(at: index) {
          control.setImage(
            image.withTintColor(color, renderingMode: .alwaysOriginal), forSegmentAt: index)
        }

        control.subviews.forEach { subview in
          if subview.subviews.count > 0 {
            subview.subviews.forEach { label in
              if let label = label as? UILabel, label.text == title {
                label.textColor = color
              }
            }
          }
        }
      }
    }
  }

  static func getModelForSelection(size: ModelSize, standardModels: [ModelSize: ModelInfo])
    -> ModelInfo?
  {
    return standardModels[size]
  }

  private static func setupResponsiveFontSize(for control: UISegmentedControl) {
    let screenWidth = UIScreen.main.bounds.width

    let baseFontSize: CGFloat = 8
    let baseScreenWidth: CGFloat = 375
    let scaleFactor = screenWidth / baseScreenWidth
    let responsiveFontSize = max(7, min(12, baseFontSize * scaleFactor))

    control.setTitleTextAttributes(
      [
        .font: UIFont.systemFont(ofSize: responsiveFontSize, weight: .medium),
        .foregroundColor: UIColor.white,
      ], for: .normal)

    control.setTitleTextAttributes(
      [
        .font: UIFont.systemFont(ofSize: responsiveFontSize, weight: .semibold),
        .foregroundColor: UIColor.white,
      ], for: .selected)
  }
}
