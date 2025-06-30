// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Foundation

struct ModelMetadataHelper {
    
    /// Extracts model size from CoreML model metadata
    /// Looks for patterns like "YOLO11n", "YOLOv8s", etc. in model description
    static func extractModelSize(from modelPath: String) -> String? {
        do {
            let modelURL: URL
            if modelPath.hasPrefix("/") {
                // Full path
                modelURL = URL(fileURLWithPath: modelPath)
            } else {
                // Bundle resource
                guard let bundleURL = Bundle.main.url(forResource: modelPath, withExtension: nil) else {
                    return nil
                }
                modelURL = bundleURL
            }
            
            // Load the model to access metadata
            let mlModel = try MLModel(contentsOf: modelURL)
            let modelDescription = mlModel.modelDescription
            
            // Check metadata dictionary
            let metadata = modelDescription.metadata
            
            // Check creator-defined metadata
            if let userDefined = metadata[.creatorDefinedKey] as? [String: String] {
                // Check various fields that might contain model info
                for (_, value) in userDefined {
                    if let size = extractSizeFromDescription(value) {
                        return size
                    }
                }
            }
            
            // Try to extract from model description text
            let modelDescText = String(describing: modelDescription)
            if let size = extractSizeFromDescription(modelDescText) {
                return size
            }
            
        } catch {
            // Ignore error silently
        }
        
        return nil
    }
    
    /// Extracts size indicator from a description string
    /// Looks for patterns like "YOLO11n", "YOLOv8s", "yolo5m" in the text
    private static func extractSizeFromDescription(_ description: String) -> String? {
        let lowercased = description.lowercased()
        
        // Define patterns to search for YOLO model sizes
        let patterns = [
            // YOLO11 patterns
            "yolo11n": "n", "yolo11s": "s", "yolo11m": "m", "yolo11l": "l", "yolo11x": "x",
            "yolo 11n": "n", "yolo 11s": "s", "yolo 11m": "m", "yolo 11l": "l", "yolo 11x": "x",
            // YOLOv8 patterns
            "yolov8n": "n", "yolov8s": "s", "yolov8m": "m", "yolov8l": "l", "yolov8x": "x",
            "yolo v8n": "n", "yolo v8s": "s", "yolo v8m": "m", "yolo v8l": "l", "yolo v8x": "x",
            // YOLOv5 patterns
            "yolov5n": "n", "yolov5s": "s", "yolov5m": "m", "yolov5l": "l", "yolov5x": "x",
            "yolo v5n": "n", "yolo v5s": "s", "yolo v5m": "m", "yolo v5l": "l", "yolo v5x": "x",
            // YOLO5 patterns (without v)
            "yolo5n": "n", "yolo5s": "s", "yolo5m": "m", "yolo5l": "l", "yolo5x": "x",
            "yolo 5n": "n", "yolo 5s": "s", "yolo 5m": "m", "yolo 5l": "l", "yolo 5x": "x",
        ]
        
        // Check each pattern
        for (pattern, size) in patterns {
            if lowercased.contains(pattern) {
                return size
            }
        }
        
        // Check for size words in description
        if lowercased.contains("nano") {
            return "n"
        } else if lowercased.contains("small") && !lowercased.contains("xlarge") {
            return "s"
        } else if lowercased.contains("medium") {
            return "m"
        } else if lowercased.contains("large") && !lowercased.contains("xlarge") && !lowercased.contains("extra") {
            return "l"
        } else if lowercased.contains("xlarge") || lowercased.contains("extra large") || lowercased.contains("extra-large") {
            return "x"
        }
        
        return nil
    }
    
    /// Gets all available metadata from a model (for debugging purposes)
    static func getAllMetadata(from modelPath: String) -> [String: Any]? {
        do {
            let modelURL: URL
            if modelPath.hasPrefix("/") {
                modelURL = URL(fileURLWithPath: modelPath)
            } else {
                guard let bundleURL = Bundle.main.url(forResource: modelPath, withExtension: nil) else {
                    return nil
                }
                modelURL = bundleURL
            }
            
            let mlModel = try MLModel(contentsOf: modelURL)
            let modelDescription = mlModel.modelDescription
            
            var result: [String: Any] = [:]
            
            // Add creator-defined metadata if available
            if let userDefined = modelDescription.metadata[.creatorDefinedKey] as? [String: Any] {
                result["userDefined"] = userDefined
            }
            
            // Add basic model info
            result["inputDescription"] = modelDescription.inputDescriptionsByName.keys.joined(separator: ", ")
            result["outputDescription"] = modelDescription.outputDescriptionsByName.keys.joined(separator: ", ")
            
            return result
            
        } catch {
            print("ModelMetadataHelper: Error reading model metadata: \(error)")
            return nil
        }
    }
    
    /// Extract model size from metadata dictionary
    /// - Parameter metadata: Metadata dictionary from model
    /// - Returns: The detected model size or nil
    static func extractModelSizeFromMetadata(_ metadata: [String: String]) -> ModelSizeFilterBar.ModelSize? {
        // Only check the "description" field for size info
        if let description = metadata["description"]?.lowercased() {
            if let sizeString = extractSizeFromDescription(description) {
                // Convert size code to ModelSize enum
                switch sizeString {
                case "n": return .nano
                case "s": return .small
                case "m": return .medium
                case "l": return .large
                case "x": return .xlarge
                default: return nil
                }
            }
        }
        
        return nil
    }
}