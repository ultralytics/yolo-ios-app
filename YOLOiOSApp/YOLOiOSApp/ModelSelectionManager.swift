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
        let isCustom: Bool
    }

    static func categorizeModels(from models: [(name: String, url: URL?, isLocal: Bool)]) -> (standard: [ModelSize: ModelInfo], custom: [ModelInfo]) {
        var standardModels: [ModelSize: ModelInfo] = [:]
        var customModels: [ModelInfo] = []

        for model in models {
            let baseName = (model.name as NSString).deletingPathExtension.lowercased()

            if baseName.hasPrefix("yolo") {
                let sizeChar = extractSizeFromModelName(baseName)

                if let char = sizeChar,
                   let size = ModelSize(rawValue: String(char)) {
                    standardModels[size] = ModelInfo(
                        name: model.name,
                        url: model.url,
                        isLocal: model.isLocal,
                        size: size,
                        isCustom: false
                    )
                } else {
                    customModels.append(ModelInfo(
                        name: model.name,
                        url: model.url,
                        isLocal: model.isLocal,
                        size: nil,
                        isCustom: true
                    ))
                }
            } else {
                customModels.append(ModelInfo(
                    name: model.name,
                    url: model.url,
                    isLocal: model.isLocal,
                    size: nil,
                    isCustom: true
                ))
            }
        }

        return (standardModels, customModels)
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
            let pattern = "^\\d+([nsmxl])"
            if let regex = try? NSRegularExpression(pattern: pattern, options: []),
               let match = regex.firstMatch(in: String(afterYolo), options: [], range: NSRange(location: 0, length: afterYolo.count)),
               match.numberOfRanges > 1 {
                let sizeRange = match.range(at: 1)
                if let range = Range(sizeRange, in: String(afterYolo)) {
                    return String(afterYolo)[range].first
                }
            }
        }

        return nil
    }

    private static func removeTaskSuffix(from name: String) -> String {
        let taskSuffixes = ["-seg", "-cls", "-pose", "-obb"]

        for suffix in taskSuffixes {
            if name.hasSuffix(suffix) {
                return String(name.dropLast(suffix.count))
            }
        }

        return name
    }

    static func setupSegmentedControl(_ control: UISegmentedControl, standardModels: [ModelSize: ModelInfo], hasCustomModels: Bool, currentTask: YOLOTask, preserveSelection: Bool = false) {
        let previousSelection = preserveSelection ? control.selectedSegmentIndex : -1

        control.removeAllSegments()

        for (index, size) in ModelSize.allCases.enumerated() {
            if let model = standardModels[size] {
                let fullName = (model.name as NSString).deletingPathExtension
                let displayTitle = removeTaskSuffix(from: fullName)

                let isDownloaded = model.isLocal ||
                    (model.url != nil && YOLOModelCache.shared.isCached(url: model.url!, task: currentTask))

                let titleWithIcon: String
                if isDownloaded {
                    titleWithIcon = displayTitle
                } else {
                    titleWithIcon = "↓ \(displayTitle)"
                }
                control.insertSegment(withTitle: titleWithIcon, at: index, animated: false)
                control.setEnabled(true, forSegmentAt: index)
            } else {
                control.insertSegment(withTitle: size.displayName, at: index, animated: false)
                control.setEnabled(false, forSegmentAt: index)
            }
        }

        if hasCustomModels {
            control.insertSegment(withTitle: "Custom", at: control.numberOfSegments, animated: false)
        }

        if preserveSelection && previousSelection >= 0 && previousSelection < control.numberOfSegments {
            control.selectedSegmentIndex = previousSelection
        } else {
            control.selectedSegmentIndex = 0
        }

        control.setNeedsLayout()
        control.layoutIfNeeded()

        DispatchQueue.main.async {
            updateSegmentAppearance(control, standardModels: standardModels, currentTask: currentTask)
        }
    }

    static func updateSegmentAppearance(_ control: UISegmentedControl, standardModels: [ModelSize: ModelInfo], currentTask: YOLOTask) {
        for (index, size) in ModelSize.allCases.enumerated() {
            guard index < control.numberOfSegments else { break }

            if let model = standardModels[size] {
                let isDownloaded = model.isLocal ||
                    (model.url != nil && YOLOModelCache.shared.isCached(url: model.url!, task: currentTask))

                if !isDownloaded {
                    setSegmentTextColor(control, at: index, color: .systemGray)
                } else {
                    setSegmentTextColor(control, at: index, color: .white)
                }
            } else if !control.isEnabledForSegment(at: index) {
                setSegmentTextColor(control, at: index, color: .gray)
            }
        }

        if control.numberOfSegments > ModelSize.allCases.count {
            setSegmentTextColor(control, at: control.numberOfSegments - 1, color: .white)
        }
    }

    private static func setSegmentTextColor(_ control: UISegmentedControl, at index: Int, color: UIColor) {
        if #available(iOS 13.0, *) {
            if let title = control.titleForSegment(at: index) {
                let attributes: [NSAttributedString.Key: Any] = [.foregroundColor: color]
                control.setTitle(title, forSegmentAt: index)

                if let image = control.imageForSegment(at: index) {
                    control.setImage(image.withTintColor(color, renderingMode: .alwaysOriginal), forSegmentAt: index)
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

    static func getModelForSelection(size: ModelSize, standardModels: [ModelSize: ModelInfo]) -> ModelInfo? {
        return standardModels[size]
    }
}