// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

struct ModelSizeHelper {
    static func getModelSize(from modelName: String) -> String {
        let lowercased = modelName.lowercased()
        
        // Check for size indicators in the model name
        // Look for patterns like "yolo11n", "yolo11n-seg", "yolo11n-pose", etc.
        if lowercased.contains("nano") || lowercased.contains("11n") || lowercased.contains("8n") || lowercased.contains("5n") {
            return "NANO"
        } else if lowercased.contains("small") || lowercased.contains("11s") || lowercased.contains("8s") || lowercased.contains("5s") {
            return "SMALL"
        } else if lowercased.contains("medium") || lowercased.contains("11m") || lowercased.contains("8m") || lowercased.contains("5m") {
            return "MEDIUM"
        } else if lowercased.contains("large") || lowercased.contains("11l") || lowercased.contains("8l") || lowercased.contains("5l") {
            return "LARGE"
        } else if lowercased.contains("extra") || lowercased.contains("11x") || lowercased.contains("8x") || lowercased.contains("5x") {
            return "XLARGE"
        }
        
        // Also check if it ends with size indicator (for detect models)
        if lowercased.hasSuffix("n") {
            return "NANO"
        } else if lowercased.hasSuffix("s") {
            return "SMALL"
        } else if lowercased.hasSuffix("m") {
            return "MEDIUM"
        } else if lowercased.hasSuffix("l") {
            return "LARGE"
        } else if lowercased.hasSuffix("x") {
            return "XLARGE"
        }
        
        // Default for custom models
        return "CUSTOM"
    }
}