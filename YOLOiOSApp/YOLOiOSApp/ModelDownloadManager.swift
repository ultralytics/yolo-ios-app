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
    // Clean up incomplete downloads on startup
    cleanupIncompleteDownloads()
  }
  
  private func cleanupIncompleteDownloads() {
    DispatchQueue.global(qos: .background).async {
      do {
        let documentsDir = self.getDocumentsDirectory()
        let contents = try FileManager.default.contentsOfDirectory(at: documentsDir, includingPropertiesForKeys: nil)
        
        for item in contents {
          let filename = item.lastPathComponent
          
          // Process any ZIP files that exist without corresponding mlmodelc
          if item.pathExtension == "zip" && filename.contains(".mlpackage.zip") {
            // Extract key from filename (e.g., "yolo11l-seg.mlpackage.zip" -> "yolo11l-seg")
            let key = filename.replacingOccurrences(of: ".mlpackage.zip", with: "")
            let mlmodelcURL = documentsDir.appendingPathComponent(key).appendingPathExtension("mlmodelc")
            
            if !FileManager.default.fileExists(atPath: mlmodelcURL.path) {
              #if DEBUG
              print("Found unprocessed ZIP at startup: \(filename)")
              #endif
              // Process this ZIP in background
              ModelDownloadManager.shared.processExistingZip(zipURL: item, key: key) { _, _ in
                // Silent processing at startup
              }
            }
          }
          
          // Process any mlpackage files that exist without corresponding mlmodelc
          if item.pathExtension == "mlpackage" {
            let key = item.deletingPathExtension().lastPathComponent
            let mlmodelcURL = documentsDir.appendingPathComponent(key).appendingPathExtension("mlmodelc")
            
            if !FileManager.default.fileExists(atPath: mlmodelcURL.path) {
              #if DEBUG
              print("Found uncompiled mlpackage at startup: \(filename)")
              #endif
              // Compile this model in background
              ModelDownloadManager.shared.compileExistingModel(modelURL: item, key: key) { _, _ in
                // Silent processing at startup
              }
            }
          }
        }
      } catch {
        #if DEBUG
        print("Error during cleanup: \(error)")
        #endif
      }
    }
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
      // Check if ZIP file already exists
      let zipURL = getDocumentsDirectory().appendingPathComponent(fileName)
      let mlpackageURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlpackage")
      
      if FileManager.default.fileExists(atPath: zipURL.path) {
        // ZIP exists, process it
        #if DEBUG
        print("\n=== Found existing ZIP ===")
        print("Processing existing ZIP: \(fileName)")
        #endif
        ModelDownloadManager.shared.processExistingZip(
          zipURL: zipURL, key: key, completion: completion)
      } else if FileManager.default.fileExists(atPath: mlpackageURL.path) {
        // MLPackage exists, compile it
        #if DEBUG
        print("\n=== Found existing MLPackage ===")
        print("Compiling existing MLPackage: \(mlpackageURL.lastPathComponent)")
        #endif
        ModelDownloadManager.shared.compileExistingModel(
          modelURL: mlpackageURL, key: key, completion: completion)
      } else {
        // Nothing exists, start download
        ModelDownloadManager.shared.startDownload(
          url: remoteURL, fileName: fileName, key: key, completion: completion)
      }
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
    // Check for compiled model first
    let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension(
      "mlmodelc")
    let exists = FileManager.default.fileExists(atPath: modelURL.path)
    
    #if DEBUG
    print("\n=== Checking Model Cache ===")
    print("Key: \(key)")
    print("Expected file: \(key).mlmodelc")
    print("Full path: \(modelURL.path)")
    print("File exists: \(exists)")
    
    // List all files in documents directory for debugging
    if !exists {
      let documentsPath = getDocumentsDirectory()
      do {
        let files = try FileManager.default.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: nil)
        print("\nAll files in documents directory:")
        
        var hasZip = false
        var hasMLPackage = false
        
        for file in files {
          print("  - \(file.lastPathComponent)")
          // Check if this might be our model with different naming
          if file.lastPathComponent.contains(key) {
            print("    ^ This file contains the key '\(key)'")
            
            if file.pathExtension == "zip" {
              hasZip = true
            } else if file.pathExtension == "mlpackage" {
              hasMLPackage = true
            }
          }
        }
        
        if hasZip || hasMLPackage {
          print("\nNote: Found related files but model is not compiled yet")
          print("ZIP exists: \(hasZip), MLPackage exists: \(hasMLPackage)")
        }
      } catch {
        print("Error listing files: \(error)")
      }
    }
    print("========================\n")
    #endif
    
    // For UI purposes, we should return true if any related file exists
    // to prevent showing download icon when files are being processed
    if !exists {
      // Check if ZIP or mlpackage exists
      let zipURL = getDocumentsDirectory().appendingPathComponent("\(key).mlpackage.zip")
      let mlpackageURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlpackage")
      
      return FileManager.default.fileExists(atPath: zipURL.path) || 
             FileManager.default.fileExists(atPath: mlpackageURL.path)
    }
    
    return exists
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
    #if DEBUG
    print("\n=== Starting Download ===")
    print("URL: \(url)")
    print("FileName: \(fileName)")
    print("Key: \(key)")
    #endif
    
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
  
  func processExistingZip(zipURL: URL, key: String, completion: @escaping (MLModel?, String) -> Void) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        #if DEBUG
        print("\n=== Processing Existing ZIP ===")
        print("ZIP URL: \(zipURL.lastPathComponent)")
        print("Key: \(key)")
        #endif
        
        // Remove any existing mlpackage directory first
        let mlpackageURL = self.getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlpackage")
        if FileManager.default.fileExists(atPath: mlpackageURL.path) {
          try FileManager.default.removeItem(at: mlpackageURL)
          #if DEBUG
          print("Removed existing mlpackage")
          #endif
        }
        
        // Unzip
        #if DEBUG
        print("Unzipping...")
        #endif
        try unzipSkippingMacOSX(at: zipURL, to: self.getDocumentsDirectory())
        
        // Find and compile the model
        let documentsDir = self.getDocumentsDirectory()
        let contents = try FileManager.default.contentsOfDirectory(at: documentsDir, includingPropertiesForKeys: nil)
        
        #if DEBUG
        print("Looking for mlpackage after unzip:")
        for item in contents {
          if item.pathExtension == "mlpackage" {
            print("  Found: \(item.lastPathComponent)")
          }
        }
        #endif
        
        var modelURL: URL? = nil
        for item in contents {
          if item.pathExtension == "mlpackage" && 
             (item.lastPathComponent.lowercased().contains(key.lowercased()) ||
              item.lastPathComponent.lowercased().replacingOccurrences(of: ".mlpackage", with: "") == key.lowercased()) {
            modelURL = item
            #if DEBUG
            print("Selected mlpackage: \(item.lastPathComponent)")
            #endif
            break
          }
        }
        
        guard let foundModelURL = modelURL else {
          #if DEBUG
          print("ERROR: No mlpackage found for key: \(key)")
          #endif
          throw NSError(domain: "ModelDownloadManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model package not found"])
        }
        
        self.loadModel(from: foundModelURL, key: key) { model in
          DispatchQueue.main.async {
            completion(model, key)
          }
        }
      } catch {
        #if DEBUG
        print("Error processing existing ZIP: \(error)")
        #endif
        DispatchQueue.main.async {
          completion(nil, key)
        }
      }
    }
  }
  
  func compileExistingModel(modelURL: URL, key: String, completion: @escaping (MLModel?, String) -> Void) {
    loadModel(from: modelURL, key: key) { model in
      DispatchQueue.main.async {
        completion(model, key)
      }
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
        
        // After unzipping, find the .mlpackage in the documents directory
        // The unzipped content might have different naming patterns
        let documentsDir = getDocumentsDirectory()
        let contents = try FileManager.default.contentsOfDirectory(at: documentsDir, includingPropertiesForKeys: nil)
        
        #if DEBUG
        print("\n=== After Unzip Debug ===")
        print("Key: \(key)")
        print("Documents directory contents:")
        for item in contents {
          print("  - \(item.lastPathComponent)")
        }
        #endif
        
        // Look for .mlpackage files that match our key
        var modelURL: URL? = nil
        
        // First try exact match
        let exactMatch = documentsDir.appendingPathComponent("\(key).mlpackage")
        if FileManager.default.fileExists(atPath: exactMatch.path) {
          modelURL = exactMatch
          #if DEBUG
          print("Found exact match: \(exactMatch.lastPathComponent)")
          #endif
        } else {
          // Look for any .mlpackage that contains the key
          for item in contents {
            if item.pathExtension == "mlpackage" {
              let filename = item.lastPathComponent
              // Check if this mlpackage is related to our key
              if filename.lowercased().contains(key.lowercased()) ||
                 filename.lowercased().replacingOccurrences(of: ".mlpackage", with: "") == key.lowercased() {
                modelURL = item
                #if DEBUG
                print("Found matching mlpackage: \(filename)")
                #endif
                break
              }
            }
          }
        }
        
        guard let foundModelURL = modelURL else {
          #if DEBUG
          print("ERROR: No .mlpackage found for key: \(key)")
          #endif
          throw NSError(domain: "ModelDownloadManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model package not found after extraction"])
        }
        
        loadModel(from: foundModelURL, key: key) { model in
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
        #if DEBUG
        print("\n=== Model Compilation ===")
        print("Compiling model from: \(url.lastPathComponent)")
        print("Key: \(key)")
        #endif
        
        let compiledModelURL = try MLModel.compileModel(at: url)
        
        #if DEBUG
        print("Compiled model at: \(compiledModelURL.path)")
        #endif
        
        let model = try MLModel(contentsOf: compiledModelURL)
        let localModelURL = self.getDocumentsDirectory().appendingPathComponent(key)
          .appendingPathExtension("mlmodelc")
          
        #if DEBUG
        print("Will save to: \(localModelURL.lastPathComponent)")
        #endif
        
        ModelCacheManager.shared.addModelToCache(model, for: key)
        try FileManager.default.moveItem(at: compiledModelURL, to: localModelURL)
        
        #if DEBUG
        print("Model successfully saved to: \(localModelURL.path)")
        #endif
        
        // Clean up the original .mlpackage after successful compilation
        let mlpackageURL = self.getDocumentsDirectory().appendingPathComponent(key)
          .appendingPathExtension("mlpackage")
        if FileManager.default.fileExists(atPath: mlpackageURL.path) {
          try? FileManager.default.removeItem(at: mlpackageURL)
          print("Cleaned up .mlpackage")
        }
        
        // Clean up the ZIP file
        let zipURL = self.getDocumentsDirectory().appendingPathComponent(key)
          .appendingPathExtension("mlpackage.zip")
        if FileManager.default.fileExists(atPath: zipURL.path) {
          try? FileManager.default.removeItem(at: zipURL)
          print("Cleaned up .zip file")
        }
        
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
