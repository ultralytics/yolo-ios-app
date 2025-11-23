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
  let isRemote: Bool
  let remoteURL: URL?

  init(
    displayName: String, identifier: String, isLocalBundle: Bool = false, isRemote: Bool = false,
    remoteURL: URL? = nil
  ) {
    self.displayName = displayName
    self.identifier = identifier
    self.isLocalBundle = isLocalBundle
    self.isRemote = isRemote
    self.remoteURL = remoteURL
  }
}

class ModelCacheManager {
  static let shared = ModelCacheManager()
  var modelCache: [String: MLModel] = [:]
  private var accessOrder: [String] = []
  private let cacheLimit = 3
  private var currentSelectedModelKey: String?

  private init() {}

  /// Update cache access order for key.
  private func updateAccessOrder(for key: String) {
    if let index = accessOrder.firstIndex(of: key) {
      accessOrder.remove(at: index)
    }
    accessOrder.append(key)
  }

  /// Get model URL in documents directory.
  private func modelURL(for key: String) -> URL {
    documentsDirectory.appendingPathComponent(key).appendingPathExtension("mlmodelc")
  }

  func loadBundledModel() {
    guard let url = getModelFileURL(fileName: "yolov8m"),
      let bundledModel = try? MLModel(contentsOf: url)
    else {
      print("Failed to load bundled model")
      return
    }

    addModelToCache(bundledModel, for: "yolov8m")
    let destinationURL = modelURL(for: "yolov8m")

    do {
      if !FileManager.default.fileExists(atPath: destinationURL.path) {
        try FileManager.default.copyItem(at: url, to: destinationURL)
        print("File copied to documents directory: \(destinationURL.path)")
      }
    } catch {
      print("Error copying file: \(error)")
    }
  }

  func loadLocalModel(key: String, completion: @escaping (MLModel?, String) -> Void) {
    if let cachedModel = modelCache[key] {
      updateAccessOrder(for: key)
      completion(cachedModel, key)
      return
    }

    let localModelURL = modelURL(for: key)
    guard FileManager.default.fileExists(atPath: localModelURL.path) else { return }

    do {
      let model = try MLModel(contentsOf: localModelURL)
      addModelToCache(model, for: key)
      completion(model, key)
    } catch {
      print("Error loading local model: \(error)")
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

  func prioritizeDownload(for fileName: String, completion: @escaping (MLModel?, String) -> Void) {
    ModelDownloadManager.shared.prioritizeDownload(for: fileName, completion: completion)
  }

  func setCurrentSelectedModelKey(_ key: String) { currentSelectedModelKey = key }
  func getCurrentSelectedModelKey() -> String? { currentSelectedModelKey }
}

class ModelDownloadManager: NSObject {
  static let shared = ModelDownloadManager()
  private var downloadTasks: [URLSessionDownloadTask: (url: URL, key: String)] = [:]
  private var downloadCompletionHandlers: [URLSessionDownloadTask: (MLModel?, String) -> Void] = [:]
  private var priorityTask: URLSessionDownloadTask?
  var progressHandler: ((Double) -> Void)?

  private override init() {}

  /// Complete download task and cleanup.
  private func completeTask(_ task: URLSessionDownloadTask, model: MLModel?, key: String) {
    downloadCompletionHandlers[task]?(model, key)
    downloadCompletionHandlers.removeValue(forKey: task)
  }

  /// Create priority download task.
  private func createPriorityTask(
    from task: URLSessionDownloadTask, urlKeyPair: (url: URL, key: String),
    completion: @escaping (MLModel?, String) -> Void
  ) {
    let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    let priorityDownloadTask = session.downloadTask(with: task.originalRequest!)
    priorityDownloadTask.priority = URLSessionTask.highPriority
    downloadTasks[priorityDownloadTask] = urlKeyPair
    downloadCompletionHandlers[priorityDownloadTask] = completion
    priorityTask = priorityDownloadTask
    priorityDownloadTask.resume()
  }

  func startDownload(
    url: URL, fileName: String, key: String, completion: @escaping (MLModel?, String) -> Void
  ) {
    let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    let downloadTask = session.downloadTask(with: url)
    let destinationURL = documentsDirectory.appendingPathComponent(fileName)
    downloadTasks[downloadTask] = (url: destinationURL, key: key)
    downloadCompletionHandlers[downloadTask] = completion
    downloadTask.resume()
  }

  func prioritizeDownload(for fileName: String, completion: @escaping (MLModel?, String) -> Void) {
    for (task, urlKeyPair) in downloadTasks {
      guard urlKeyPair.url.lastPathComponent.contains(fileName) else { continue }

      task.cancel(byProducingResumeData: { resumeData in
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        let priorityDownloadTask: URLSessionDownloadTask

        if let resumeData = resumeData {
          priorityDownloadTask = session.downloadTask(withResumeData: resumeData)
        } else {
          priorityDownloadTask = session.downloadTask(with: task.originalRequest!)
        }

        priorityDownloadTask.priority = URLSessionTask.highPriority
        self.downloadTasks[priorityDownloadTask] = urlKeyPair
        self.downloadCompletionHandlers[priorityDownloadTask] = completion
        self.priorityTask = priorityDownloadTask
        priorityDownloadTask.resume()
      })
      break
    }
  }

  func cancelCurrentDownload() {
    priorityTask?.cancel()
    priorityTask = nil
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
      // Verify the downloaded file exists and has content
      let fileAttributes = try FileManager.default.attributesOfItem(atPath: location.path)
      guard let fileSize = fileAttributes[.size] as? Int64, fileSize > 0 else {
        throw NSError(
          domain: "ModelDownload", code: 2,
          userInfo: [NSLocalizedDescriptionKey: "Downloaded file is empty or invalid"])
      }
      
      print("📦 ModelDownloadManager: Downloaded file size: \(fileSize) bytes for key: \(key)")
      
      let zipURL = destinationURL
      if fileExists(at: zipURL) {
        try FileManager.default.removeItem(at: zipURL)
      }
      try FileManager.default.moveItem(at: location, to: zipURL)
      downloadTasks.removeValue(forKey: downloadTask)

      // Verify ZIP file is valid before attempting extraction
      do {
        _ = try Archive(url: zipURL, accessMode: .read)
        print("✅ ModelDownloadManager: ZIP file is valid")
      } catch {
        print("❌ ModelDownloadManager: ZIP file is corrupted: \(error)")
        try? FileManager.default.removeItem(at: zipURL)
        throw NSError(
          domain: "ModelDownload", code: 3,
          userInfo: [NSLocalizedDescriptionKey: "Downloaded ZIP file is corrupted: \(error.localizedDescription)"])
      }

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
      print("❌ ModelDownloadManager: Download processing failed for key '\(key)': \(error)")
      // Clean up any partial files
      let zipURL = downloadTasks[downloadTask]?.url ?? destinationURL
      try? FileManager.default.removeItem(at: zipURL)
      let tempExtractionURL = documentsDirectory.appendingPathComponent("temp_\(key)")
      try? FileManager.default.removeItem(at: tempExtractionURL)
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
    let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
    DispatchQueue.main.async { self.progressHandler?(progress) }
  }
  
  func urlSession(
    _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
  ) {
    guard let error = error else { return }
    guard let downloadTask = task as? URLSessionDownloadTask,
          let key = downloadTasks[downloadTask]?.key else { return }
    
    print("❌ ModelDownloadManager: Download failed for key '\(key)': \(error.localizedDescription)")
    
    // Clean up partial download
    if let destinationURL = downloadTasks[downloadTask]?.url {
      try? FileManager.default.removeItem(at: destinationURL)
    }
    
    downloadTasks.removeValue(forKey: downloadTask)
    completeTask(downloadTask, model: nil, key: key)
  }
}

class ModelFileManager {
  static let shared = ModelFileManager()
  private init() {}

  func deleteAllDownloadedModels() {
    do {
      let fileURLs = try FileManager.default.contentsOfDirectory(
        at: documentsDirectory, includingPropertiesForKeys: nil)
      for fileURL in fileURLs
      where ["mlmodel", "mlmodelc", "mlpackage"].contains(fileURL.pathExtension) {
        try FileManager.default.removeItem(at: fileURL)
        print("Deleted file: \(fileURL.lastPathComponent)")
      }
    } catch {
      print("Error deleting files: \(error)")
    }
  }
}

func getModelFileURL(fileName: String) -> URL? {
  Bundle.main.url(forResource: fileName, withExtension: "mlmodelc")
}

func fileExists(at url: URL) -> Bool {
  FileManager.default.fileExists(atPath: url.path)
}

extension URL {
  func changingFileExtension(to newExtension: String) -> URL? {
    var urlString = self.absoluteString
    if let range = urlString.range(of: "\\.[^./]*$", options: .regularExpression) {
      urlString.replaceSubrange(range, with: ".\(newExtension)")
    } else {
      urlString.append(".\(newExtension)")
    }
    return URL(string: urlString)
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
