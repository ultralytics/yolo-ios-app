// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing model download functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOModelDownloader class handles the downloading, extraction, and processing of YOLO models
//  from remote URLs. It supports progress tracking, ZIP file extraction with proper handling of
//  directory structures, and automatic caching of downloaded models for offline use.

import CoreML
import Foundation
import ZIPFoundation

/// Handles downloading and processing of YOLO models from remote URLs.
public class YOLOModelDownloader: NSObject {

  public typealias ProgressHandler = (Double) -> Void
  public typealias CompletionHandler = (Result<URL, Error>) -> Void

  /// Error types for download operations
  public enum DownloadError: LocalizedError {
    case invalidURL, invalidZipFile, modelNotFoundInArchive
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

  private var downloadTask: URLSessionDownloadTask?
  private var progressHandler: ProgressHandler?
  private var completionHandler: CompletionHandler?
  private var currentTask: YOLOTask?
  private var originalURL: URL?

  private lazy var session: URLSession = {
    URLSession(configuration: .default, delegate: self, delegateQueue: nil)
  }()

  /// Download model from URL with optional task type and progress tracking
  public func download(
    from url: URL, task: YOLOTask? = nil, progress: ProgressHandler? = nil,
    completion: @escaping CompletionHandler
  ) {
    self.progressHandler = progress
    self.completionHandler = completion
    self.currentTask = task
    self.originalURL = url

    // Check cache first
    if let cachedPath = YOLOModelCache.shared.getCachedModelPath(url: url, task: task) {
      completion(.success(cachedPath))
      return
    }

    // No cache found, Start download
    downloadTask = session.downloadTask(with: url)
    downloadTask?.resume()
  }

  /// Cancel current download
  public func cancelDownload() {
    downloadTask?.cancel()
    downloadTask = nil
  }

  /// Process downloaded file
  private func processDownloadedFile(at location: URL, originalURL: URL) {
    // Use the stored originalURL if available (for task-aware downloads)
    let url = self.originalURL ?? originalURL

    do {
      // Create temporary directory for extraction
      let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

      // Cleanup temp directory
      defer { try? FileManager.default.removeItem(at: tempDir) }

      // Move downloaded file to temp location
      let zipPath = tempDir.appendingPathComponent("model.zip")
      try FileManager.default.moveItem(at: location, to: zipPath)

      // Extract ZIP file
      let extractedPath = tempDir.appendingPathComponent("extracted")
      try unzipSkippingMacOSX(at: zipPath, to: extractedPath)

      // Find model file
      let modelPath = try findModelPath(in: extractedPath, for: url)

      // Compile and cache model
      let compiledPath = try compileModelIfNeeded(at: modelPath)
      let cachedPath = try cacheModel(compiledPath, for: url)

      completionHandler?(.success(cachedPath))

    } catch {
      completionHandler?(.failure(error))
    }
  }

  /// Find model file in extracted contents
  private func findModelPath(in extractedPath: URL, for url: URL) throws -> URL {
    let contents = try FileManager.default.contentsOfDirectory(
      at: extractedPath, includingPropertiesForKeys: [.isDirectoryKey])

    // Check if ZIP contains mlpackage contents directly
    let hasManifest = contents.contains { $0.lastPathComponent == "Manifest.json" }
    let hasDataFolder = contents.contains { $0.lastPathComponent == "Data" }

    if hasManifest && hasDataFolder {
      // Create proper .mlpackage directory
      let key = YOLOModelCache.shared.cacheKey(for: url, task: currentTask)
      let mlpackagePath = extractedPath.parent.appendingPathComponent("\(key).mlpackage")
      try FileManager.default.moveItem(at: extractedPath, to: mlpackagePath)
      return mlpackagePath
    } else {
      return try findModelFile(in: extractedPath)
        ?? { throw DownloadError.modelNotFoundInArchive }()
    }
  }

  /// Cache compiled model
  private func cacheModel(_ compiledPath: URL, for url: URL) throws -> URL {
    let key = YOLOModelCache.shared.cacheKey(for: url, task: currentTask)
    let cachedPath = YOLOModelCache.shared.cacheDirectory.appendingPathComponent(key)
      .appendingPathExtension("mlmodelc")

    if FileManager.default.fileExists(atPath: cachedPath.path) {
      try FileManager.default.removeItem(at: cachedPath)
    }

    try FileManager.default.moveItem(at: compiledPath, to: cachedPath)
    return cachedPath
  }

  /// Extract ZIP file while skipping macOS metadata
  private func unzipSkippingMacOSX(at sourceURL: URL, to destinationURL: URL) throws {
    guard let archive = Archive(url: sourceURL, accessMode: .read) else {
      throw DownloadError.invalidZipFile
    }

    try FileManager.default.createDirectory(at: destinationURL, withIntermediateDirectories: true)

    for entry in archive {
      // Skip macOS metadata files
      guard !entry.path.hasPrefix("__MACOSX") && !entry.path.contains("._") else { continue }

      let destinationPath = destinationURL.appendingPathComponent(entry.path)

      // Create parent directory if needed
      let parentDir = destinationPath.deletingLastPathComponent()
      if !FileManager.default.fileExists(atPath: parentDir.path) {
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
      }

      try archive.extract(entry, to: destinationPath)
    }
  }

  /// Recursively find model file in directory
  private func findModelFile(in directory: URL) throws -> URL? {
    let contents = try FileManager.default.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: [.isDirectoryKey])

    // Look for model files in current directory
    for url in contents {
      if url.pathExtension == "mlpackage" {
        let manifestPath = url.appendingPathComponent("Manifest.json")
        if FileManager.default.fileExists(atPath: manifestPath.path) { return url }
      }

      if ["mlmodel", "mlmodelc"].contains(url.pathExtension) { return url }
    }

    // Search subdirectories recursively
    for url in contents {
      let resourceValues = try url.resourceValues(forKeys: [.isDirectoryKey])
      if resourceValues.isDirectory == true && url.pathExtension != "mlpackage" {
        if let found = try findModelFile(in: url) { return found }
      }
    }

    return nil
  }

  /// Compile model if needed
  private func compileModelIfNeeded(at modelURL: URL) throws -> URL {
    switch modelURL.pathExtension {
    case "mlmodel", "mlpackage":
      do {
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let _ = try MLModel(contentsOf: compiledURL)  // Verify compilation
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

// MARK: - URLSessionDownloadDelegate
extension YOLOModelDownloader: URLSessionDownloadDelegate {

  public func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    guard let originalURL = downloadTask.originalRequest?.url else {
      completionHandler?(.failure(DownloadError.invalidURL))
      return
    }

    processDownloadedFile(at: location, originalURL: originalURL)
  }

  public func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
    DispatchQueue.main.async {
      self.progressHandler?(progress)
    }
  }

  public func urlSession(
    _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
  ) {
    if let error = error {
      completionHandler?(.failure(DownloadError.downloadFailed(error)))
    }
  }
}

// MARK: - URL Extension
extension URL {
  fileprivate var parent: URL { deletingLastPathComponent() }
}
