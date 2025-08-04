// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing model download functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOModelDownloader class handles the downloading, extraction, and processing of YOLO models
//  from remote URLs. It supports progress tracking, ZIP file extraction with proper handling of
//  directory structures, and automatic caching of downloaded models for offline use.

import Foundation
import CoreML
import ZIPFoundation

/// Handles downloading and processing of YOLO models from remote URLs.
public class YOLOModelDownloader: NSObject {
  
  /// Progress handler callback type
  public typealias ProgressHandler = (Double) -> Void
  
  /// Completion handler callback type
  public typealias CompletionHandler = (Result<URL, Error>) -> Void
  
  /// Error types for download operations
  public enum DownloadError: LocalizedError {
    case invalidURL
    case downloadFailed(Error)
    case invalidZipFile
    case modelNotFoundInArchive
    case extractionFailed(Error)
    case compilationFailed(Error)
    
    public var errorDescription: String? {
      switch self {
      case .invalidURL:
        return "Invalid model URL"
      case .downloadFailed(let error):
        return "Download failed: \(error.localizedDescription)"
      case .invalidZipFile:
        return "Invalid or corrupted ZIP file"
      case .modelNotFoundInArchive:
        return "No valid model file found in archive"
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
    let config = URLSessionConfiguration.default
    return URLSession(configuration: config, delegate: self, delegateQueue: nil)
  }()
  
  /// Download model from URL with progress tracking
  public func download(
    from url: URL,
    progress: ProgressHandler? = nil,
    completion: @escaping CompletionHandler
  ) {
    self.progressHandler = progress
    self.completionHandler = completion
    self.originalURL = url  // Store the URL for processDownloadedFile
    
    // Check cache first
    if let cachedPath = YOLOModelCache.shared.getCachedModelPath(url: url) {
      // Using cached model
      completion(.success(cachedPath))
      return
    }
    
    // No cache found, downloading
    
    // Start download
    downloadTask = session.downloadTask(with: url)
    downloadTask?.resume()
  }
  
  /// Download model from URL with progress tracking and task type
  public func download(
    from url: URL,
    task: YOLOTask,
    progress: ProgressHandler? = nil,
    completion: @escaping CompletionHandler
  ) {
    self.progressHandler = progress
    self.completionHandler = completion
    self.currentTask = task
    self.originalURL = url
    
    // Check cache first with task type
    if let cachedPath = YOLOModelCache.shared.getCachedModelPath(url: url, task: task) {
      // Using cached model for task
      completion(.success(cachedPath))
      return
    }
    
    // No cache found, downloading for task
    
    // Start download
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
      
      defer {
        // Cleanup temp directory
        try? FileManager.default.removeItem(at: tempDir)
      }
      
      // Move downloaded file to temp location
      let zipPath = tempDir.appendingPathComponent("model.zip")
      try FileManager.default.moveItem(at: location, to: zipPath)
      
      // ZIP file moved to temp location
      
      // Extract ZIP file
      let extractedPath = tempDir.appendingPathComponent("extracted")
      try unzipSkippingMacOSX(at: zipPath, to: extractedPath)
      
      // Files extracted
      
      // Check extracted contents
      let contents = try FileManager.default.contentsOfDirectory(at: extractedPath, includingPropertiesForKeys: [.isDirectoryKey])
      
      var modelPath: URL?
      
      // Check if ZIP contains mlpackage contents directly (Manifest.json + Data folder at root)
      let hasManifest = contents.contains { $0.lastPathComponent == "Manifest.json" }
      let hasDataFolder = contents.contains { $0.lastPathComponent == "Data" }
      
      if hasManifest && hasDataFolder {
        // The extracted directory contains mlpackage contents directly
        // We need to create a proper .mlpackage directory
        // ZIP contains mlpackage contents directly
        
        // Create mlpackage with key as filename (matching merge-backup-changes behavior)
        let key = currentTask != nil ? YOLOModelCache.shared.cacheKey(for: url, task: currentTask!) : YOLOModelCache.shared.cacheKey(for: url)
        let mlpackagePath = tempDir.appendingPathComponent("\(key).mlpackage")
        
        // Move the extracted contents into a .mlpackage directory
        try FileManager.default.moveItem(at: extractedPath, to: mlpackagePath)
        modelPath = mlpackagePath
        
      } else {
        // Normal case: find model file in extracted contents
        modelPath = try findModelFile(in: extractedPath)
      }
      
      guard let finalModelPath = modelPath else {
        throw DownloadError.modelNotFoundInArchive
      }
      
      // Found model
      
      // Compile model if needed
      let compiledPath = try compileModelIfNeeded(at: finalModelPath)
      
      // Model compiled
      
      // Move compiled model to cache with key as filename (matching merge-backup-changes behavior)
      let key = currentTask != nil ? YOLOModelCache.shared.cacheKey(for: url, task: currentTask!) : YOLOModelCache.shared.cacheKey(for: url)
      let cachedPath = YOLOModelCache.shared.cacheDirectory.appendingPathComponent(key).appendingPathExtension("mlmodelc")
      
      // Remove existing cached model if present
      if FileManager.default.fileExists(atPath: cachedPath.path) {
        try FileManager.default.removeItem(at: cachedPath)
      }
      
      // Move compiled model to cache location
      try FileManager.default.moveItem(at: compiledPath, to: cachedPath)
      
      // Model cached
      
      // Complete with success
      completionHandler?(.success(cachedPath))
      
    } catch {
      // Error during processing
      completionHandler?(.failure(error))
    }
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
      
      // Extract entry
      _ = try archive.extract(entry, to: destinationPath)
    }
  }
  
  /// Recursively find model file in directory
  private func findModelFile(in directory: URL) throws -> URL? {
    let contents = try FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: [.isDirectoryKey]
    )
    
    // First, look for model files in current directory
    for url in contents {
      let filename = url.lastPathComponent
      
      // Check for .mlpackage directory (must have Manifest.json inside)
      if url.pathExtension == "mlpackage" {
        let manifestPath = url.appendingPathComponent("Manifest.json")
        if FileManager.default.fileExists(atPath: manifestPath.path) {
          return url
        }
      }
      
      // Check for .mlmodel or .mlmodelc files
      if ["mlmodel", "mlmodelc"].contains(url.pathExtension) {
        return url
      }
    }
    
    // Then search subdirectories recursively
    for url in contents {
      let resourceValues = try url.resourceValues(forKeys: [.isDirectoryKey])
      if resourceValues.isDirectory == true && url.pathExtension != "mlpackage" {
        if let found = try findModelFile(in: url) {
          return found
        }
      }
    }
    
    return nil
  }
  
  /// Compile model if needed
  private func compileModelIfNeeded(at modelURL: URL) throws -> URL {
    let fileExtension = modelURL.pathExtension
    
    switch fileExtension {
    case "mlmodel", "mlpackage":
      // Compile the model
      do {
        // Compiling model
        let compiledURL = try MLModel.compileModel(at: modelURL)
        // Model compiled successfully
        
        // Verify the compiled model can be loaded
        _ = try MLModel(contentsOf: compiledURL)
        // Compiled model verified
        
        return compiledURL
      } catch {
        // Compilation failed
        throw DownloadError.compilationFailed(error)
      }
      
    case "mlmodelc":
      // Already compiled
      // Model is already compiled
      return modelURL
      
    default:
      throw DownloadError.modelNotFoundInArchive
    }
  }
}

// MARK: - URLSessionDownloadDelegate
extension YOLOModelDownloader: URLSessionDownloadDelegate {
  
  public func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    guard let originalURL = downloadTask.originalRequest?.url else {
      completionHandler?(.failure(DownloadError.invalidURL))
      return
    }
    
    processDownloadedFile(at: location, originalURL: originalURL)
  }
  
  public func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) {
    let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
    DispatchQueue.main.async {
      self.progressHandler?(progress)
    }
  }
  
  public func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    didCompleteWithError error: Error?
  ) {
    if let error = error {
      completionHandler?(.failure(DownloadError.downloadFailed(error)))
    }
  }
}