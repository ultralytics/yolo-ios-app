// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and manages the machine learning model lifecycle.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  ModelCacheManager and ModelDownloadManager together handle the YOLO model lifecycle: downloading from remote URLs,
//  extracting ZIP archives, compiling MLModels, caching loaded models in memory, and tracking what is on disk. The
//  in-memory cache is bounded to keep memory in check on resource-constrained devices, while progress callbacks keep
//  the UI responsive during downloads.

import CoreML
import Foundation
import UltralyticsYOLO

/// URL of the app's Documents directory; used for storing compiled models and download archives.
private let documentsDirectory = FileManager.default.urls(
  for: .documentDirectory, in: .userDomainMask)[0]

/// Metadata describing a selectable YOLO model entry (local bundle or downloadable remote).
struct ModelEntry {
  let displayName: String
  let identifier: String
  let isLocalBundle: Bool
  let remoteURL: URL?

  var cacheKey: String {
    Self.cacheKey(for: identifier, remoteURL: remoteURL)
  }

  static func cacheKey(for identifier: String, remoteURL: URL?) -> String {
    guard let remoteURL else { return identifier }
    return "\(identifier)-\(remoteURL.deletingLastPathComponent().lastPathComponent)"
  }
}

class ModelCacheManager {
  static let shared = ModelCacheManager()
  private var modelCache: [String: MLModel] = [:]
  private var accessOrder: [String] = []
  private let cacheLimit = 3

  private init() {
    let suffixes = ["", "-seg", "-sem", "-depth", "-cls", "-pose", "-obb"]
    for size in ["n", "s", "m", "l", "x"] {
      for suffix in suffixes {
        try? FileManager.default.removeItem(
          at: modelURL(for: "yolo26\(size)\(suffix)"))
      }
    }
  }

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
    let completion = downloadCompletionHandlers[task]
    downloadCompletionHandlers.removeValue(forKey: task)
    downloadTasks.removeValue(forKey: task)
    completion?(model, key)
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

  func cancelDownload(key: String) {
    let tasks = downloadTasks.filter { $0.value.key == key }.map(\.key)
    tasks.forEach {
      $0.cancel()
      downloadTasks.removeValue(forKey: $0)
      downloadCompletionHandlers.removeValue(forKey: $0)
    }
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

      // Extract to a per-key temp directory to avoid collisions between concurrent downloads.
      let tempExtractionURL = documentsDirectory.appendingPathComponent("temp_\(key)")
      if FileManager.default.fileExists(atPath: tempExtractionURL.path) {
        try FileManager.default.removeItem(at: tempExtractionURL)
      }

      try unzipSkippingMacOSX(at: zipURL, to: tempExtractionURL)

      // Recursively locate the model file inside the extracted contents.
      func findModelFile(in directory: URL) throws -> URL? {
        let contents = try FileManager.default.contentsOfDirectory(
          at: directory, includingPropertiesForKeys: [.isDirectoryKey])

        // Prefer model files in the current directory before descending.
        for url in contents {
          if ["mlmodel", "mlpackage"].contains(url.pathExtension) {
            return url
          }
        }

        // Otherwise search subdirectories.
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
        // Clean up temp extraction directory and downloaded zip.
        try? FileManager.default.removeItem(at: tempExtractionURL)
        try? FileManager.default.removeItem(at: zipURL)
        self.completeTask(downloadTask, model: model, key: key)
      }
    } catch {
      print("Download processing failed: \(error)")
      completeTask(downloadTask, model: nil, key: key)
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    guard let error = error,
      let downloadTask = task as? URLSessionDownloadTask,
      let key = downloadTasks[downloadTask]?.key
    else { return }

    print("Download failed: \(error)")
    completeTask(downloadTask, model: nil, key: key)
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
    guard downloadTasks[downloadTask] != nil else { return }
    guard totalBytesExpectedToWrite > 0 else { return }
    let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
    DispatchQueue.main.async { self.progressHandler?(progress) }
  }
}

func unzipSkippingMacOSX(at sourceURL: URL, to destinationURL: URL) throws {
  // Extract via the SDK's dependency-free MiniZip, skipping macOS resource-fork metadata.
  try MiniZip.extract(at: sourceURL, to: destinationURL) { path in
    path.hasPrefix("__MACOSX") || path.contains("._")
  }
}
