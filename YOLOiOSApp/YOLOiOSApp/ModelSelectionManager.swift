// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation
import UIKit
import UltralyticsYOLO

struct ModelSelectionManager {
  private static let taskSuffixes = ["-seg", "-sem", "-depth", "-cls", "-pose", "-obb"]

  enum ModelSize: String, CaseIterable {
    case n, s, m, l, x
  }

  struct ModelInfo {
    let name: String
    let url: URL?
    let isLocal: Bool
  }

  private static let modelSizeRegex = try! NSRegularExpression(
    pattern: "^\\d+([nsmxl])", options: [])

  static func categorizeModels(from models: [(name: String, url: URL?, isLocal: Bool)])
    -> [ModelSize: ModelInfo]
  {
    var standardModels: [ModelSize: ModelInfo] = [:]

    for model in models {
      let baseName = (model.name as NSString).deletingPathExtension.lowercased()

      if baseName.hasPrefix("yolo") {
        let sizeChar = extractSizeFromModelName(baseName)

        if let char = sizeChar,
          let size = ModelSize(rawValue: String(char))
        {
          standardModels[size] = ModelInfo(
            name: model.name,
            url: model.url,
            isLocal: model.isLocal
          )
        }
      }
    }

    return standardModels
  }

  private static func extractSizeFromModelName(_ baseName: String) -> Character? {
    var nameWithoutSuffix = baseName
    for suffix in taskSuffixes {
      if baseName.hasSuffix(suffix) {
        nameWithoutSuffix = String(baseName.dropLast(suffix.count))
        break
      }
    }

    if nameWithoutSuffix.hasPrefix("yolo") && !nameWithoutSuffix.dropFirst(4).isEmpty {
      let afterYOLO = nameWithoutSuffix.dropFirst(4)
      let afterYOLOString = String(afterYOLO)
      let range = NSRange(location: 0, length: afterYOLOString.count)

      if let match = modelSizeRegex.firstMatch(in: afterYOLOString, options: [], range: range),
        match.numberOfRanges > 1
      {
        let sizeRange = match.range(at: 1)
        if let range = Range(sizeRange, in: afterYOLOString) {
          return afterYOLOString[range].first
        }
      }
    }

    return nil
  }

  private static func removeTaskSuffix(from name: String) -> String {
    var result = name
    for suffix in taskSuffixes {
      if name.hasSuffix(suffix) {
        result = String(name.dropLast(suffix.count))
        break
      }
    }

    if result.lowercased().hasPrefix("yolo") {
      let afterYOLO = result.dropFirst(4)
      result = "YOLO" + afterYOLO
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

        // For remote models, check the on-disk cache. The key is the model name without extension (e.g. "yolo26n").
        let modelKey = (model.name as NSString).deletingPathExtension
        let isDownloaded =
          model.isLocal
          || (model.url != nil
            && ModelCacheManager.shared.isModelDownloaded(
              key: ModelEntry.cacheKey(for: modelKey, remoteURL: model.url)))

        let titleWithIcon: String
        if isDownloaded {
          titleWithIcon = displayTitle
        } else {
          titleWithIcon = "⤓ \(displayTitle)"
        }
        control.insertSegment(withTitle: titleWithIcon, at: index, animated: false)
        control.setEnabled(true, forSegmentAt: index)
      } else {
        control.insertSegment(withTitle: "YOLO26\(size.rawValue)", at: index, animated: false)
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
        // For remote models, check the on-disk cache. The key is the model name without extension (e.g. "yolo26n").
        let modelKey = (model.name as NSString).deletingPathExtension
        let isDownloaded =
          model.isLocal
          || (model.url != nil
            && ModelCacheManager.shared.isModelDownloaded(
              key: ModelEntry.cacheKey(for: modelKey, remoteURL: model.url)))

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
    guard let title = control.titleForSegment(at: index) else { return }
    control.setTitle(title, forSegmentAt: index)

    if let image = control.imageForSegment(at: index) {
      control.setImage(
        image.withTintColor(color, renderingMode: .alwaysOriginal), forSegmentAt: index)
    }

    control.subviews.forEach { subview in
      subview.subviews.forEach { label in
        if let label = label as? UILabel, label.text == title {
          label.textColor = color
        }
      }
    }
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
