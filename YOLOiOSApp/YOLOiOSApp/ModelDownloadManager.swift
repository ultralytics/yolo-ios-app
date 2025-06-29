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

/// A structure representing a YOLO model with metadata for display and loading.
struct ModelEntry {
  let displayName: String

  let identifier: String

  let isLocalBundle: Bool

  let isRemote: Bool

  let remoteURL: URL?

  init(
    displayName: String,
    identifier: String,
    isLocalBundle: Bool = false,
    isRemote: Bool = false,
    remoteURL: URL? = nil
  ) {
    self.displayName = displayName
    self.identifier = identifier
    self.isLocalBundle = isLocalBundle
    self.isRemote = isRemote
    self.remoteURL = remoteURL
  }
  
  /// Extract the model version (e.g., "YOLO11", "YOLOv8", "YOLOv5", "Custom")
  var modelVersion: String {
    let name = displayName.lowercased()
    if name.hasPrefix("yolo11") {
      return "YOLO11"
    } else if name.hasPrefix("yolov8") {
      return "YOLOv8"
    } else if name.hasPrefix("yolov5") {
      return "YOLOv5"
    } else {
      return "Custom"
    }
  }
  
  /// Extract the model size (e.g., "n", "s", "m", "l", "x", nil for custom)
  var modelSize: String? {
    let name = displayName.lowercased()
    
    // Check if this is a standard YOLO model with size indicator
    // Look for patterns like "yolo11n", "yolov8s-seg", etc.
    let yoloPattern = #"yolo(v?\d+)?([nsmxl])([-_]|$)"#
    if let regex = try? NSRegularExpression(pattern: yoloPattern, options: .caseInsensitive) {
      let matches = regex.matches(in: name, options: [], range: NSRange(location: 0, length: name.count))
      if let match = matches.first, match.numberOfRanges > 2 {
        let sizeRange = match.range(at: 2)
        if let range = Range(sizeRange, in: name) {
          return String(name[range])
        }
      }
    }
    
    // For custom models, try to get size from cached metadata
    if isLocalBundle && modelVersion == "Custom" {
      if let cachedSize = ModelCacheManager.shared.getCachedModelSize(for: identifier) {
        return cachedSize
      }
    }
    
    return nil
  }
}

class ModelCacheManager {
  static let shared = ModelCacheManager()
  var modelCache: [String: MLModel] = [:]
  private var metadataCache: [String: [String: String]] = [:]  // Cache for model metadata
  private var accessOrder: [String] = []
  private let cacheLimit: Int = 3
  private var currentSelectedModelKey: String?

  private init() {
  }

  func loadBundledModel() {
    if let url = getModelFileURL(fileName: "yolov8m"),
      let bundledModel = try? MLModel(contentsOf: url)
    {
      addModelToCache(bundledModel, for: "yolov8m")
      let documentsURL = getDocumentsDirectory()
      let destinationURL = documentsURL.appendingPathComponent("yolov8m").appendingPathExtension(
        "mlmodelc")

      do {
        if !FileManager.default.fileExists(atPath: destinationURL.path) {
          try FileManager.default.copyItem(at: url, to: destinationURL)
          #if DEBUG
          print("File copied to documents directory: \(destinationURL.path)")
          #endif
        } else {
          #if DEBUG
          print("File already exists in documents directory: \(destinationURL.path)")
          #endif
        }
      } catch {
        #if DEBUG
        print("Error copying file: \(error)")
        #endif
      }
    } else {
      #if DEBUG
      print("Failed to load bundled model")
      #endif
    }
  }

  func loadLocalModel(key: String, completion: @escaping (MLModel?, String) -> Void) {
    if let cachedModel = modelCache[key] {
      if let index = accessOrder.firstIndex(of: key) {
        accessOrder.remove(at: index)
      }
      accessOrder.append(key)
      completion(cachedModel, key)
      return
    }

    let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension(
      "mlmodelc")
    if FileManager.default.fileExists(atPath: modelURL.path) {
      do {
        let model = try MLModel(contentsOf: modelURL)
        addModelToCache(model, for: key)
        completion(model, key)
      } catch let error {
        #if DEBUG
        print(error)
        #endif
      }
    }
  }

  func loadModel(
    from fileName: String, remoteURL: URL, key: String,
    completion: @escaping (MLModel?, String) -> Void
  ) {
    if let cachedModel = modelCache[key] {
      if let index = accessOrder.firstIndex(of: key) {
        accessOrder.remove(at: index)
      }
      accessOrder.append(key)
      completion(cachedModel, key)
      return
    }

    let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension(
      "mlmodelc")
    if FileManager.default.fileExists(atPath: modelURL.path) {
      loadLocalModel(key: key) { model, key in
        completion(model, key)
      }
    } else {
      ModelDownloadManager.shared.startDownload(
        url: remoteURL, fileName: fileName, key: key, completion: completion)
    }
  }

  private func compileAndLoadDownloadedModel(
    from url: URL, key: String, completion: @escaping (MLModel?) -> Void
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {

        let compiledModelURL = try MLModel.compileModel(at: url)
        let destinationURL = self.getDocumentsDirectory().appendingPathComponent(
          compiledModelURL.lastPathComponent)

        // Move compiled model to desired location
        try FileManager.default.moveItem(at: compiledModelURL, to: destinationURL)
        let model = try MLModel(contentsOf: destinationURL)
        DispatchQueue.main.async {
          self.addModelToCache(model, for: key)
          completion(model)
        }
      } catch {
        #if DEBUG
        print("Failed to load model: \(error)")
        #endif
        DispatchQueue.main.async {
          completion(nil)
        }
      }
    }
  }

  func addModelToCache(_ model: MLModel, for key: String) {
    if modelCache.count >= cacheLimit {
      let oldKey = accessOrder.removeFirst()
      modelCache.removeValue(forKey: oldKey)
      metadataCache.removeValue(forKey: oldKey)
    }
    modelCache[key] = model
    accessOrder.append(key)
    
    // Cache metadata from model
    if let userDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] {
      metadataCache[key] = userDefined
    }
  }

  func isModelDownloaded(key: String) -> Bool {
    let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension(
      "mlmodelc")
    return FileManager.default.fileExists(atPath: modelURL.path)
  }

  func getDocumentsDirectory() -> URL {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }

  func prioritizeDownload(for fileName: String, completion: @escaping (MLModel?, String) -> Void) {
    ModelDownloadManager.shared.prioritizeDownload(for: fileName, completion: completion)
  }

  func setCurrentSelectedModelKey(_ key: String) {
    currentSelectedModelKey = key
  }

  func getCurrentSelectedModelKey() -> String? {
    return currentSelectedModelKey
  }
  
  /// Get model size from cached metadata
  func getCachedModelSize(for key: String) -> String? {
    guard let metadata = metadataCache[key] else { 
      return nil 
    }
    
    // Use ModelMetadataHelper to extract size from metadata
    if let size = ModelMetadataHelper.extractModelSizeFromMetadata(metadata) {
      return size.rawValue
    }
    return nil
  }
  
  /// Cache metadata for a model
  func cacheMetadata(for key: String, metadata: [String: String]) {
    metadataCache[key] = metadata
  }
}

class ModelDownloadManager: NSObject {
  static let shared = ModelDownloadManager()
  private var downloadTasks: [URLSessionDownloadTask: (url: URL, key: String)] = [:]
  private var priorityTask: URLSessionDownloadTask?
  var progressHandler: ((Double) -> Void)?

  private override init() {}

  func startDownload(
    url: URL, fileName: String, key: String, completion: @escaping (MLModel?, String) -> Void
  ) {
    let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    let downloadTask = session.downloadTask(with: url)
    let destinationURL = getDocumentsDirectory().appendingPathComponent(fileName)
    downloadTasks[downloadTask] = (url: destinationURL, key: key)
    downloadTask.resume()
    downloadCompletionHandlers[downloadTask] = completion
  }

  private var downloadCompletionHandlers: [URLSessionDownloadTask: (MLModel?, String) -> Void] = [:]

  private func getDocumentsDirectory() -> URL {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }

  func prioritizeDownload(for fileName: String, completion: @escaping (MLModel?, String) -> Void) {
    for (task, urlKeyPair) in downloadTasks {
      if urlKeyPair.url.lastPathComponent.contains(fileName) {
        task.cancel(byProducingResumeData: { resumeData in
          if let resumeData = resumeData {
            let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
            let priorityDownloadTask = session.downloadTask(withResumeData: resumeData)
            priorityDownloadTask.priority = URLSessionTask.highPriority
            self.downloadTasks[priorityDownloadTask] = urlKeyPair
            self.downloadCompletionHandlers[priorityDownloadTask] = completion
            self.priorityTask = priorityDownloadTask
            priorityDownloadTask.resume()
          } else {
            let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
            let priorityDownloadTask = session.downloadTask(with: task.originalRequest!)
            priorityDownloadTask.priority = URLSessionTask.highPriority
            self.downloadTasks[priorityDownloadTask] = urlKeyPair
            self.downloadCompletionHandlers[priorityDownloadTask] = completion
            self.priorityTask = priorityDownloadTask
            priorityDownloadTask.resume()
          }
        })
        break
      }
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
      let zipURL = destinationURL
      #if DEBUG
      print("zipURL: \(zipURL)")
      #endif
      if fileExists(at: zipURL) {
        try FileManager.default.removeItem(at: zipURL)
      }
      try FileManager.default.moveItem(at: location, to: zipURL)
      downloadTasks.removeValue(forKey: downloadTask)
      let unzipDestinationURL = destinationURL.deletingPathExtension()
      if FileManager.default.fileExists(atPath: unzipDestinationURL.path) {
        try FileManager.default.removeItem(at: unzipDestinationURL)
      }
      do {
        try unzipSkippingMacOSX(at: zipURL, to: getDocumentsDirectory())
        let modelURL = unzipDestinationURL
        #if DEBUG
        print("modelURL: \(modelURL)")
        #endif
        loadModel(from: modelURL, key: key) { model in
          self.downloadCompletionHandlers[downloadTask]?(model, key)
          self.downloadCompletionHandlers.removeValue(forKey: downloadTask)
        }
      } catch {
        #if DEBUG
        print("Extraction of ZIP archive failed with error: \(error)")
        #endif
        self.downloadCompletionHandlers[downloadTask]?(nil, key)
        self.downloadCompletionHandlers.removeValue(forKey: downloadTask)
      }
    } catch {
      #if DEBUG
      print("Error moving downloaded file: \(error)")
      #endif
      self.downloadCompletionHandlers[downloadTask]?(nil, key)
      self.downloadCompletionHandlers.removeValue(forKey: downloadTask)
    }
  }

  private func loadModel(from url: URL, key: String, completion: @escaping (MLModel?) -> Void) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let compiledModelURL = try MLModel.compileModel(at: url)
        let model = try MLModel(contentsOf: compiledModelURL)
        let localModelURL = self.getDocumentsDirectory().appendingPathComponent(key)
          .appendingPathExtension("mlmodelc")
        ModelCacheManager.shared.addModelToCache(model, for: key)
        try FileManager.default.moveItem(at: compiledModelURL, to: localModelURL)
        print("model copied to document directory")
        DispatchQueue.main.async {
          completion(model)
        }
      } catch {
        #if DEBUG
        print("Failed to load model: \(error)")
        #endif
        DispatchQueue.main.async {
          completion(nil)
        }
      }
    }
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) {
    let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
    DispatchQueue.main.async {
      self.progressHandler?(progress)
    }
  }
}

class ModelFileManager {
  static let shared = ModelFileManager()

  private init() {}

  func deleteAllDownloadedModels() {
    let fileManager = FileManager.default
    let documentsDirectory = getDocumentsDirectory()
    do {
      let fileURLs = try fileManager.contentsOfDirectory(
        at: documentsDirectory, includingPropertiesForKeys: nil)
      for fileURL in fileURLs {
        if fileURL.pathExtension == "mlmodel" || fileURL.pathExtension == "mlmodelc"
          || fileURL.pathExtension == "mlpackage"
        {
          try fileManager.removeItem(at: fileURL)
          print("Deleted file: \(fileURL.lastPathComponent)")
        }
      }
    } catch {
      print("Error deleting files: \(error)")
    }
  }

  private func getDocumentsDirectory() -> URL {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }
}

func getModelFileURL(fileName: String) -> URL? {
  let bundle = Bundle.main
  if let fileURL = bundle.url(forResource: fileName, withExtension: "mlmodelc") {
    return fileURL
  }
  return nil
}

func fileExists(at url: URL) -> Bool {
  let fileManager = FileManager.default
  return fileManager.fileExists(atPath: url.path)
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
      at: destinationURL,
      withIntermediateDirectories: true,
      attributes: nil)
  }

  for entry in archive {
    if entry.path.hasPrefix("__MACOSX") {
      continue
    }
    if entry.path.contains("._") {
      continue
    }

    let entryDestinationURL = destinationURL.appendingPathComponent(entry.path)

    let parentDir = entryDestinationURL.deletingLastPathComponent()
    if !FileManager.default.fileExists(atPath: parentDir.path) {
      try FileManager.default.createDirectory(
        at: parentDir,
        withIntermediateDirectories: true,
        attributes: nil)
    }

    try archive.extract(entry, to: entryDestinationURL)
  }
}
