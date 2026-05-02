// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app, handling machine learning model management.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ModelDownloadManager and related classes provide a complete system for managing YOLO models.
//  This includes downloading models from remote URLs, caching loaded models in memory, handling model
//  extraction from ZIP archives, and managing the lifecycle of models on the device. The implementation
//  includes progress tracking for downloads, prioritization of download tasks, and memory management
//  for loaded models to ensure optimal performance on resource-constrained devices. These utilities
//  allow the application to dynamically load models based on user selection while maintaining a responsive
//  user experience.

import CoreML
import Foundation
import ZIPFoundation

/// Shared documents directory accessor.
private let documentsDirectory = FileManager.default.urls(
  for: .documentDirectory, in: .userDomainMask)[0]

/// A structure representing a YOLO model with metadata for display and loading.
struct ModelEntry {
  let displayName: String
  let identifier: String
  let isLocalBundle: Bool
  let remoteURL: URL?
}

class ModelCacheManager {
  static let shared = ModelCacheManager()
  private var modelCache: [String: MLModel] = [:]
  private var accessOrder: [String] = []
  private let cacheLimit = 3

  private init() {}

  private func updateAccessOrder(for key: String) {
    if let index = accessOrder.firstIndex(of: key) {
      accessOrder.remove(at: index)
    }
    accessOrder.append(key)
  }

  private func modelURL(for key: String) -> URL {
    documentsDirectory.appendingPathComponent(key).appendingPathExtension("mlmodelc")
  }

  private func loadLocalModel(key: String, completion: @escaping (MLModel?, String) -> Void) {
    if let cachedModel = modelCache[key] {
      updateAccessOrder(for: key)
      completion(cachedModel, key)
      return
    }

    let localModelURL = modelURL(for: key)
    do {
      let model = try MLModel(contentsOf: localModelURL)
      addModelToCache(model, for: key)
      completion(model, key)
    } catch {
      print("Error loading local model: \(error)")
      completion(nil, key)
    }
  }

  func loadModel(
    from fileName: String, remoteURL: URL, key: String,
    completion: @escaping (MLModel?, String) -> Void
  ) {
    if let cachedModel = modelCache[key] {
      updateAccessOrder(for: key)
      completion(cachedModel, key)
      return
    }

    if FileManager.default.fileExists(atPath: modelURL(for: key).path) {
      loadLocalModel(key: key, completion: completion)
    } else {
      ModelDownloadManager.shared.startDownload(
        url: remoteURL, fileName: fileName, key: key, completion: completion)
    }
  }

  func addModelToCache(_ model: MLModel, for key: String) {
    if modelCache.count >= cacheLimit {
      let oldKey = accessOrder.removeFirst()
      modelCache.removeValue(forKey: oldKey)
    }
    modelCache[key] = model
    accessOrder.append(key)
  }

  func isModelDownloaded(key: String) -> Bool {
    FileManager.default.fileExists(atPath: modelURL(for: key).path)
  }
}

class ModelDownloadManager: NSObject {
  static let shared = ModelDownloadManager()
  private var downloadTasks: [URLSessionDownloadTask: (url: URL, key: String)] = [:]
  private var downloadCompletionHandlers: [URLSessionDownloadTask: (MLModel?, String) -> Void] = [:]
  private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
  var progressHandler: ((Double) -> Void)?

  private override init() {}

  private func completeTask(_ task: URLSessionDownloadTask, model: MLModel?, key: String) {
    downloadCompletionHandlers[task]?(model, key)
    downloadCompletionHandlers.removeValue(forKey: task)
  }

  func startDownload(
    url: URL, fileName: String, key: String, completion: @escaping (MLModel?, String) -> Void
  ) {
    let downloadTask = session.downloadTask(with: url)
    let destinationURL = documentsDirectory.appendingPathComponent(fileName)
    downloadTasks[downloadTask] = (url: destinationURL, key: key)
    downloadCompletionHandlers[downloadTask] = completion
    downloadTask.resume()
  }
}

extension ModelDownloadManager: URLSessionDownloadDelegate {
  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    guard let destinationURL = downloadTasks[downloadTask]?.url,
      let key = downloadTasks[downloadTask]?.key
    else { return }

    do {
      let zipURL = destinationURL
      if FileManager.default.fileExists(atPath: zipURL.path) {
        try FileManager.default.removeItem(at: zipURL)
      }
      try FileManager.default.moveItem(at: location, to: zipURL)
      downloadTasks.removeValue(forKey: downloadTask)

      // Extract to model-specific temporary directory to avoid conflicts
      let tempExtractionURL = documentsDirectory.appendingPathComponent("temp_\(key)")
      if FileManager.default.fileExists(atPath: tempExtractionURL.path) {
        try FileManager.default.removeItem(at: tempExtractionURL)
      }

      try unzipSkippingMacOSX(at: zipURL, to: tempExtractionURL)

      // Find the model file in the extracted contents (search recursively)
      func findModelFile(in directory: URL) throws -> URL? {
        let contents = try FileManager.default.contentsOfDirectory(
          at: directory, includingPropertiesForKeys: [.isDirectoryKey])

        // First, look for model files in current directory
        for url in contents {
          if ["mlmodel", "mlpackage"].contains(url.pathExtension) {
            return url
          }
        }

        // Then search subdirectories
        for url in contents {
          let resourceValues = try url.resourceValues(forKeys: [.isDirectoryKey])
          if resourceValues.isDirectory == true {
            if let found = try findModelFile(in: url) {
              return found
            }
          }
        }

        return nil
      }

      guard let foundModelURL = try findModelFile(in: tempExtractionURL) else {
        throw NSError(
          domain: "ModelDownload", code: 1,
          userInfo: [NSLocalizedDescriptionKey: "No model file found in extracted archive"])
      }

      loadModel(from: foundModelURL, key: key) { model in
        // Clean up temp directory and zip file
        try? FileManager.default.removeItem(at: tempExtractionURL)
        try? FileManager.default.removeItem(at: zipURL)
        self.completeTask(downloadTask, model: model, key: key)
      }
    } catch {
      print("Download processing failed: \(error)")
      completeTask(downloadTask, model: nil, key: key)
    }
  }

  private func loadModel(from url: URL, key: String, completion: @escaping (MLModel?) -> Void) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let compiledModelURL = try MLModel.compileModel(at: url)
        let model = try MLModel(contentsOf: compiledModelURL)
        let localModelURL = documentsDirectory.appendingPathComponent(key).appendingPathExtension(
          "mlmodelc")
        ModelCacheManager.shared.addModelToCache(model, for: key)
        try FileManager.default.moveItem(at: compiledModelURL, to: localModelURL)
        DispatchQueue.main.async { completion(model) }
      } catch {
        print("Failed to load model: \(error)")
        DispatchQueue.main.async { completion(nil) }
      }
    }
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    guard totalBytesExpectedToWrite > 0 else { return }
    let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
    DispatchQueue.main.async { self.progressHandler?(progress) }
  }
}

func unzipSkippingMacOSX(at sourceURL: URL, to destinationURL: URL) throws {
  let archive = try Archive(url: sourceURL, accessMode: .read)

  if !FileManager.default.fileExists(atPath: destinationURL.path) {
    try FileManager.default.createDirectory(
      at: destinationURL, withIntermediateDirectories: true, attributes: nil)
  }

  for entry in archive {
    guard !entry.path.hasPrefix("__MACOSX") && !entry.path.contains("._") else { continue }

    let entryDestinationURL = destinationURL.appendingPathComponent(entry.path)
    let parentDir = entryDestinationURL.deletingLastPathComponent()

    if !FileManager.default.fileExists(atPath: parentDir.path) {
      try FileManager.default.createDirectory(
        at: parentDir, withIntermediateDirectories: true, attributes: nil)
    }

    _ = try archive.extract(entry, to: entryDestinationURL)
  }
}
