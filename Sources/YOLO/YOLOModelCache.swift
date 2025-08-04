// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing model caching functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOModelCache class manages the caching of downloaded YOLO models to improve performance
//  and reduce network usage. It provides functionality to store, retrieve, and manage cached models
//  in the device's documents directory, with URL-based cache key generation for unique identification.

import Foundation
import CryptoKit

/// Manages caching of downloaded YOLO models in the documents directory.
public class YOLOModelCache {
  /// Shared singleton instance
  public static let shared = YOLOModelCache()
  
  /// Cache directory URL
  let cacheDirectory: URL
  
  /// Error types for cache operations
  public enum CacheError: LocalizedError {
    case failedToCreateDirectory
    case failedToSaveModel
    case modelNotFound
    
    public var errorDescription: String? {
      switch self {
      case .failedToCreateDirectory:
        return "Failed to create cache directory"
      case .failedToSaveModel:
        return "Failed to save model to cache"
      case .modelNotFound:
        return "Model not found in cache"
      }
    }
  }
  
  private init() {
    // Create cache directory in Documents/YOLOModels/
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    self.cacheDirectory = documentsDirectory.appendingPathComponent("YOLOModels", isDirectory: true)
    
    // Create directory if it doesn't exist
    if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
      try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true, attributes: nil)
    }
  }
  
  /// Generate a cache key from URL
  public func cacheKey(for url: URL) -> String {
    // Use the last path component and host to create a unique key
    let urlString = url.absoluteString
    let key = urlString.data(using: .utf8)?.sha256() ?? url.lastPathComponent
    // Generated cache key
    return key.replacingOccurrences(of: "/", with: "_")
  }
  
  /// Generate a cache key from URL and task type
  public func cacheKey(for url: URL, task: YOLOTask) -> String {
    // Include task type in the cache key to prevent cross-task contamination
    let taskString = String(describing: task)
    let urlString = url.absoluteString + "_" + taskString
    let key = urlString.data(using: .utf8)?.sha256() ?? (url.lastPathComponent + "_" + taskString)
    // Generated cache key with task
    return key.replacingOccurrences(of: "/", with: "_")
  }
  
  /// Check if a model is cached
  public func isCached(url: URL) -> Bool {
    return getCachedModelPath(url: url) != nil
  }
  
  /// Check if a model is cached for specific task
  public func isCached(url: URL, task: YOLOTask) -> Bool {
    return getCachedModelPath(url: url, task: task) != nil
  }
  
  /// Get cached model path if available
  public func getCachedModelPath(url: URL) -> URL? {
    let key = cacheKey(for: url)
    // Looking for cached model
    
    // Check for compiled model first (.mlmodelc)
    let compiledPath = cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlmodelc")
    if FileManager.default.fileExists(atPath: compiledPath.path) {
      // Found compiled model
      
      return compiledPath
    }
    
    // Check for mlpackage directory
    let packagePath = cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlpackage")
    if FileManager.default.fileExists(atPath: packagePath.path) {
      // Verify it's a valid mlpackage by checking for Manifest.json
      let manifestPath = packagePath.appendingPathComponent("Manifest.json")
      if FileManager.default.fileExists(atPath: manifestPath.path) {
        return packagePath
      }
    }
    
    // Check for mlmodel file
    let modelPath = cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlmodel")
    if FileManager.default.fileExists(atPath: modelPath.path) {
      return modelPath
    }
    
    return nil
  }
  
  /// Get cached model path if available for specific task
  public func getCachedModelPath(url: URL, task: YOLOTask) -> URL? {
    let key = cacheKey(for: url, task: task)
    // Looking for cached model with task
    
    // Check for compiled model first (.mlmodelc)
    let compiledPath = cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlmodelc")
    if FileManager.default.fileExists(atPath: compiledPath.path) {
      // Found compiled model
      
      return compiledPath
    }
    
    // Check for mlpackage directory
    let packagePath = cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlpackage")
    if FileManager.default.fileExists(atPath: packagePath.path) {
      // Verify it's a valid mlpackage by checking for Manifest.json
      let manifestPath = packagePath.appendingPathComponent("Manifest.json")
      if FileManager.default.fileExists(atPath: manifestPath.path) {
        return packagePath
      }
    }
    
    // Check for mlmodel file
    let modelPath = cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlmodel")
    if FileManager.default.fileExists(atPath: modelPath.path) {
      return modelPath
    }
    
    return nil
  }
  
  /// Cache a model from temporary path
  public func cacheModel(from tempPath: URL, for url: URL) throws -> URL {
    let key = cacheKey(for: url)
    let fileExtension = tempPath.pathExtension
    let destinationPath = cacheDirectory.appendingPathComponent(key).appendingPathExtension(fileExtension)
    
    // Remove existing cache if present
    if FileManager.default.fileExists(atPath: destinationPath.path) {
      try FileManager.default.removeItem(at: destinationPath)
    }
    
    // Move model to cache
    try FileManager.default.moveItem(at: tempPath, to: destinationPath)
    
    return destinationPath
  }
  
  /// Clear all cached models
  public func clearCache() throws {
    let contents = try FileManager.default.contentsOfDirectory(at: cacheDirectory, includingPropertiesForKeys: nil)
    for file in contents {
      try FileManager.default.removeItem(at: file)
    }
  }
  
  /// Get cache size in bytes
  public func getCacheSize() throws -> Int64 {
    var size: Int64 = 0
    let contents = try FileManager.default.contentsOfDirectory(at: cacheDirectory, includingPropertiesForKeys: [.fileSizeKey])
    
    for file in contents {
      let fileSize = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
      size += Int64(fileSize)
    }
    
    return size
  }
  
  /// List all cached models
  public func listCachedModels() throws -> [String] {
    let contents = try FileManager.default.contentsOfDirectory(at: cacheDirectory, includingPropertiesForKeys: nil)
    return contents.map { $0.deletingPathExtension().lastPathComponent }
  }
  
  /// Calculate directory size recursively
  private func directorySize(at url: URL) -> Int64 {
    var size: Int64 = 0
    
    let enumerator = FileManager.default.enumerator(
      at: url,
      includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
      options: [.skipsHiddenFiles]
    )
    
    while let fileURL = enumerator?.nextObject() as? URL {
      do {
        let resourceValues = try fileURL.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
        if let isDirectory = resourceValues.isDirectory, !isDirectory,
           let fileSize = resourceValues.fileSize {
          size += Int64(fileSize)
        }
      } catch {
        continue
      }
    }
    
    return size
  }
}

// MARK: - SHA256 Extension for Cache Key Generation
extension Data {
  func sha256() -> String {
    let digest = SHA256.hash(data: self)
    return digest.compactMap { String(format: "%02x", $0) }.joined()
  }
}