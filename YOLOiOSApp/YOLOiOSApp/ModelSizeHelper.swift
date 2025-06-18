// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

struct ModelSizeHelper {
    static func getModelSize(from modelName: String) -> String {
        let lowercased = modelName.lowercased()
        
        // Check for size indicators in the model name
        if lowercased.contains("nano") || lowercased.hasSuffix("n") {
            return "NANO"
        } else if lowercased.contains("small") || lowercased.hasSuffix("s") {
            return "SMALL"
        } else if lowercased.contains("medium") || lowercased.hasSuffix("m") {
            return "MEDIUM"
        } else if lowercased.contains("large") || lowercased.hasSuffix("l") {
            return "LARGE"
        } else if lowercased.contains("extra") || lowercased.hasSuffix("x") {
            return "XLARGE"
        }
        
        // Default for custom models
        return "CUSTOM"
    }
}