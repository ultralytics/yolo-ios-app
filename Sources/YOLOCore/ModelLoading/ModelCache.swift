// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import CryptoKit
import Foundation

/// Manages caching of downloaded YOLO models in the documents directory.
public final class ModelCache: Sendable {
  /// Shared singleton instance.
  public static let shared = ModelCache()

  /// Cache directory URL.
  let cacheDirectory: URL

  public enum CacheError: LocalizedError, Sendable {
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
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[
      0]
    self.cacheDirectory = documentsDirectory.appendingPathComponent("YOLOModels", isDirectory: true)
    if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
      try? FileManager.default.createDirectory(
        at: cacheDirectory, withIntermediateDirectories: true)
    }
  }

  /// Generate cache key from URL with optional task type.
  public func cacheKey(for url: URL, task: YOLOTask? = nil) -> String {
    let urlString =
      task != nil ? url.absoluteString + "_" + String(describing: task!) : url.absoluteString
    let key = urlString.data(using: .utf8)?.sha256() ?? url.lastPathComponent
    return key.replacingOccurrences(of: "/", with: "_")
  }

  /// Check if model is cached.
  public func isCached(url: URL, task: YOLOTask? = nil) -> Bool {
    getCachedModelPath(url: url, task: task) != nil
  }

  /// Get cached model path if available.
  public func getCachedModelPath(url: URL, task: YOLOTask? = nil) -> URL? {
    let key = cacheKey(for: url, task: task)
    for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
      let path = cacheDirectory.appendingPathComponent(key).appendingPathExtension(ext)
      if FileManager.default.fileExists(atPath: path.path) {
        if ext == "mlpackage" {
          let manifestPath = path.appendingPathComponent("Manifest.json")
          guard FileManager.default.fileExists(atPath: manifestPath.path) else { continue }
        }
        return path
      }
    }
    return nil
  }

  /// Clear all cached models.
  public func clearCache() throws {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory, includingPropertiesForKeys: nil)
    for file in contents {
      try FileManager.default.removeItem(at: file)
    }
  }

  /// Get cache size in bytes.
  public func getCacheSize() throws -> Int64 {
    let contents = try FileManager.default.contentsOfDirectory(
      at: cacheDirectory, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey])
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

  private func directorySize(at url: URL) throws -> Int64 {
    let enumerator = FileManager.default.enumerator(
      at: url, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
      options: [.skipsHiddenFiles])
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

extension Data {
  func sha256() -> String {
    SHA256.hash(data: self).compactMap { String(format: "%02x", $0) }.joined()
  }
}
