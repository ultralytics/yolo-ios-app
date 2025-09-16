import Foundation
import UIKit

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

    static func setupSegmentedControl(_ control: UISegmentedControl, hasCustomModels: Bool) {
        control.removeAllSegments()

        for (index, size) in ModelSize.allCases.enumerated() {
            control.insertSegment(withTitle: size.displayName, at: index, animated: false)
        }

        if hasCustomModels {
            control.insertSegment(withTitle: "Custom", at: control.numberOfSegments, animated: false)
        }

        control.selectedSegmentIndex = 0
    }

    static func getModelForSelection(size: ModelSize, standardModels: [ModelSize: ModelInfo]) -> ModelInfo? {
        return standardModels[size]
    }
}