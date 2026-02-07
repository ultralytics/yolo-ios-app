// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Foundation
import ZIPFoundation

/// Handles downloading and processing of YOLO models from remote URLs.
public final class ModelDownloader: Sendable {

  public enum DownloadError: LocalizedError, Sendable {
    case invalidURL
    case invalidZipFile
    case modelNotFoundInArchive
    case downloadFailed(Error)
    case extractionFailed(Error)
    case compilationFailed(Error)

    public var errorDescription: String? {
      switch self {
      case .invalidURL: return "Invalid model URL"
      case .downloadFailed(let error): return "Download failed: \(error.localizedDescription)"
      case .invalidZipFile: return "Invalid or corrupted ZIP file"
      case .modelNotFoundInArchive: return "No valid model file found in archive"
      case .extractionFailed(let error):
        return "Failed to extract archive: \(error.localizedDescription)"
      case .compilationFailed(let error):
        return "Failed to compile model: \(error.localizedDescription)"
      }
    }
  }

  /// Downloads a model from the given URL, returning the compiled model path.
  public func download(from url: URL, task: YOLOTask? = nil) async throws -> URL {
    // Check cache first
    if let cachedPath = ModelCache.shared.getCachedModelPath(url: url, task: task) {
      return cachedPath
    }

    // Download file
    let (location, _) = try await URLSession.shared.download(from: url)

    // Process downloaded file
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let zipPath = tempDir.appendingPathComponent("model.zip")
    try FileManager.default.moveItem(at: location, to: zipPath)

    let extractedPath = tempDir.appendingPathComponent("extracted")
    try unzipSkippingMacOSX(at: zipPath, to: extractedPath)

    let modelPath = try findModelPath(in: extractedPath, for: url, task: task)
    let compiledPath = try compileModelIfNeeded(at: modelPath)
    return try cacheModel(compiledPath, for: url, task: task)
  }

  private func findModelPath(in extractedPath: URL, for url: URL, task: YOLOTask?) throws -> URL {
    let contents = try FileManager.default.contentsOfDirectory(
      at: extractedPath, includingPropertiesForKeys: [.isDirectoryKey])

    let hasManifest = contents.contains { $0.lastPathComponent == "Manifest.json" }
    let hasDataFolder = contents.contains { $0.lastPathComponent == "Data" }

    if hasManifest && hasDataFolder {
      let key = ModelCache.shared.cacheKey(for: url, task: task)
      let mlpackagePath = extractedPath.deletingLastPathComponent().appendingPathComponent(
        "\(key).mlpackage")
      try FileManager.default.moveItem(at: extractedPath, to: mlpackagePath)
      return mlpackagePath
    } else {
      guard let found = try findModelFile(in: extractedPath) else {
        throw DownloadError.modelNotFoundInArchive
      }
      return found
    }
  }

  private func cacheModel(_ compiledPath: URL, for url: URL, task: YOLOTask?) throws -> URL {
    let key = ModelCache.shared.cacheKey(for: url, task: task)
    let cachedPath = ModelCache.shared.cacheDirectory.appendingPathComponent(key)
      .appendingPathExtension("mlmodelc")
    if FileManager.default.fileExists(atPath: cachedPath.path) {
      try FileManager.default.removeItem(at: cachedPath)
    }
    try FileManager.default.moveItem(at: compiledPath, to: cachedPath)
    return cachedPath
  }

  private func unzipSkippingMacOSX(at sourceURL: URL, to destinationURL: URL) throws {
    guard let archive = Archive(url: sourceURL, accessMode: .read) else {
      throw DownloadError.invalidZipFile
    }
    try FileManager.default.createDirectory(at: destinationURL, withIntermediateDirectories: true)
    for entry in archive {
      guard !entry.path.hasPrefix("__MACOSX") && !entry.path.contains("._") else { continue }
      let destinationPath = destinationURL.appendingPathComponent(entry.path)
      let parentDir = destinationPath.deletingLastPathComponent()
      if !FileManager.default.fileExists(atPath: parentDir.path) {
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
      }
      _ = try archive.extract(entry, to: destinationPath)
    }
  }

  private func findModelFile(in directory: URL) throws -> URL? {
    let contents = try FileManager.default.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: [.isDirectoryKey])
    for url in contents {
      if url.pathExtension == "mlpackage" {
        let manifestPath = url.appendingPathComponent("Manifest.json")
        if FileManager.default.fileExists(atPath: manifestPath.path) { return url }
      }
      if ["mlmodel", "mlmodelc"].contains(url.pathExtension) { return url }
    }
    for url in contents {
      let resourceValues = try url.resourceValues(forKeys: [.isDirectoryKey])
      if resourceValues.isDirectory == true && url.pathExtension != "mlpackage" {
        if let found = try findModelFile(in: url) { return found }
      }
    }
    return nil
  }

  private func compileModelIfNeeded(at modelURL: URL) throws -> URL {
    switch modelURL.pathExtension {
    case "mlmodel", "mlpackage":
      do {
        let compiledURL = try MLModel.compileModel(at: modelURL)
        _ = try MLModel(contentsOf: compiledURL)
        return compiledURL
      } catch {
        throw DownloadError.compilationFailed(error)
      }
    case "mlmodelc":
      return modelURL
    default:
      throw DownloadError.modelNotFoundInArchive
    }
  }
}
