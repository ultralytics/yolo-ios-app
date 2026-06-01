// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing model caching functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLOModelCache stores compiled YOLO models in `Library/Caches/YOLOModels/` so subsequent launches can skip the
//  download and compilation. Each entry is keyed by the SHA-256 of the source URL (plus optional task) so the same
//  archive can be cached separately per task.

import CryptoKit
import Foundation

/// Manages caching of downloaded YOLO models in `Library/Caches/YOLOModels/`.
public final class YOLOModelCache {
  /// Shared singleton instance.
  public static let shared = YOLOModelCache()

  /// Root cache directory for compiled models.
  let cacheDirectory: URL

  /// Lock that serializes file-system access from concurrent callers.
  private let lock = NSLock()

  /// Errors that can be reported by cache operations.
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

  /// Returns the SHA-256 cache key for the given URL and optional task.
  public func cacheKey(for url: URL, task: YOLOTask? = nil) -> String {
    let urlString =
      task.map { url.absoluteString + "_" + String(describing: $0) } ?? url.absoluteString
    return Data(urlString.utf8).sha256()
  }

  /// Returns `true` if a compiled model for the given URL (and optional task) is already cached.
  public func isCached(url: URL, task: YOLOTask? = nil) -> Bool {
    getCachedModelPath(url: url, task: task) != nil
  }

  /// Returns the cached model URL for the given source URL (and optional task), or `nil` if none is present.
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

  /// Removes every entry from the cache directory.
  public func clearCache() throws {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory, includingPropertiesForKeys: nil)
    for file in contents {
      try FileManager.default.removeItem(at: file)
    }
  }

  /// Returns the total cache size in bytes.
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

  /// Returns the cache keys for every cached model.
  public func listCachedModels() throws -> [String] {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory, includingPropertiesForKeys: nil)
    return contents.map { $0.deletingPathExtension().lastPathComponent }
  }

  /// Recursively sums the file sizes inside `url`.
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
