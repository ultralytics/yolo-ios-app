// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing model caching functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOModelCache class manages the caching of downloaded YOLO models to improve performance
//  and reduce network usage. It provides functionality to store, retrieve, and manage cached models
//  in the device's documents directory, with URL-based cache key generation for unique identification.

import CryptoKit
import Foundation

/// Manages caching of downloaded YOLO models in the documents directory.
public class YOLOModelCache {
  /// Shared singleton instance
  public static let shared = YOLOModelCache()

  /// Cache directory URL
  let cacheDirectory: URL

  /// Lock for thread-safe file system access
  private let lock = NSLock()

  /// Error types for cache operations
  public enum CacheError: LocalizedError {
    case failedToCreateDirectory
    case failedToSaveModel
    case modelNotFound

    public var errorDescription: String? {
      switch self {
      case .failedToCreateDirectory: return "Failed to create cache directory"
      case .failedToSaveModel: return "Failed to save model to cache"
      case .modelNotFound: return "Model not found in cache"
      }
    }
  }

  private init() {
    // Create cache directory in Library/Caches/YOLOModels/ (Apple storage guidelines)
    let cachesDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    self.cacheDirectory = cachesDirectory.appendingPathComponent("YOLOModels", isDirectory: true)

    // Migrate from old Documents location if it exists
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[
      0]
    let oldCacheDir = documentsDirectory.appendingPathComponent("YOLOModels", isDirectory: true)
    if FileManager.default.fileExists(atPath: oldCacheDir.path) {
      try? FileManager.default.moveItem(at: oldCacheDir, to: cacheDirectory)
    }

    // Create directory if it doesn't exist
    if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
      try? FileManager.default.createDirectory(
        at: cacheDirectory, withIntermediateDirectories: true)
    }
  }

  /// Generate cache key from URL with optional task type
  public func cacheKey(for url: URL, task: YOLOTask? = nil) -> String {
    let urlString =
      task != nil ? url.absoluteString + "_" + String(describing: task!) : url.absoluteString
    let key = urlString.data(using: .utf8)?.sha256() ?? url.lastPathComponent
    return key.replacingOccurrences(of: "/", with: "_")
  }

  /// Check if model is cached
  public func isCached(url: URL, task: YOLOTask? = nil) -> Bool {
    getCachedModelPath(url: url, task: task) != nil
  }

  /// Get cached model path if available
  public func getCachedModelPath(url: URL, task: YOLOTask? = nil) -> URL? {
    lock.lock()
    defer { lock.unlock() }

    let key = cacheKey(for: url, task: task)

    for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
      let path = cacheDirectory.appendingPathComponent(key).appendingPathExtension(ext)

      if FileManager.default.fileExists(atPath: path.path) {
        // For mlpackage, verify it's valid by checking for Manifest.json
        if ext == "mlpackage" {
          let manifestPath = path.appendingPathComponent("Manifest.json")
          guard FileManager.default.fileExists(atPath: manifestPath.path) else { continue }
        }
        return path
      }
    }

    return nil
  }

  /// Clear all cached models
  public func clearCache() throws {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory, includingPropertiesForKeys: nil)
    for file in contents {
      try FileManager.default.removeItem(at: file)
    }
  }

  /// Get cache size in bytes
  public func getCacheSize() throws -> Int64 {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory,
      includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey]
    )

    var size: Int64 = 0
    for url in contents {
      let values = try url.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
      if values.isDirectory == true {
        size += try directorySize(at: url)
      } else {
        size += Int64(values.fileSize ?? 0)
      }
    }
    return size
  }

  /// List all cached models
  public func listCachedModels() throws -> [String] {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory, includingPropertiesForKeys: nil)
    return contents.map { $0.deletingPathExtension().lastPathComponent }
  }

  /// Calculate directory size recursively
  private func directorySize(at url: URL) throws -> Int64 {
    let enumerator = FileManager.default.enumerator(
      at: url,
      includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
      options: [.skipsHiddenFiles]
    )

    var size: Int64 = 0
    while let fileURL = enumerator?.nextObject() as? URL {
      let values = try fileURL.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
      if values.isDirectory != true {
        size += Int64(values.fileSize ?? 0)
      }
    }
    return size
  }
}

// MARK: - SHA256 Extension for Cache Key Generation
extension Data {
  func sha256() -> String {
    SHA256.hash(data: self).compactMap { String(format: "%02x", $0) }.joined()
  }
}
