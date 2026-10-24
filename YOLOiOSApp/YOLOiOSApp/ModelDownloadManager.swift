// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and manages the machine learning model lifecycle.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  ModelCacheManager and ModelDownloadManager together handle the YOLO model lifecycle: downloading from remote URLs,
//  extracting ZIP archives, compiling Core ML models (Core AI `.aimodel` assets need no compile step), and tracking
//  what is on disk, while progress callbacks keep the UI responsive during downloads.

import CoreML
import Foundation
import UltralyticsYOLO

/// URL of the app's Documents directory; used for storing compiled models and download archives.
private let documentsDirectory = FileManager.default.urls(
  for: .documentDirectory, in: .userDomainMask)[0]

/// Metadata describing a selectable YOLO model entry (local bundle or downloadable remote).
struct ModelEntry: Codable {
  let displayName: String
  let identifier: String
  let isLocalBundle: Bool
  let remoteURL: URL?
  var task: String? = nil
  var installedExtension: String? = nil

  var installedURL: URL? {
    installedExtension.map {
      documentsDirectory.appendingPathComponent(identifier).appendingPathExtension($0)
    }
  }
}

struct MobileModelImport: Decodable {
  let version: Int
  let exportId: String
  let name: String
  let task: String
  let format: String
  let filename: String
  let size: Int64
  let downloadUrl: URL
  let expiresAt: String

  var entry: ModelEntry {
    ModelEntry(
      displayName: name, identifier: exportId, isLocalBundle: false, remoteURL: nil,
      task: task, installedExtension: format == "coreai" ? "aimodel" : "mlmodelc")
  }

  static func resolve(_ url: URL) async throws -> MobileModelImport {
    let parts = url.pathComponents
    guard url.scheme == "https", url.host == "platform.ultralytics.com", url.port == nil,
      url.user == nil, url.password == nil, parts.count == 4,
      parts[1] == "mobile", parts[2] == "models",
      parts[3].range(of: "^[0-9a-f]{24}$", options: .regularExpression) != nil,
      let fragment = url.fragment, fragment.count <= 128
    else { throw CocoaError(.fileReadUnsupportedScheme) }
    var endpoint = URLComponents(
      string: "https://platform.ultralytics.com/api/mobile/models/\(parts[3])")!
    endpoint.queryItems = URLComponents(string: "?\(fragment)")?.queryItems
    let (data, response) = try await URLSession.shared.data(from: endpoint.url!)
    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
      throw NSError(
        domain: "ModelImport", code: 1,
        userInfo: [
          NSLocalizedDescriptionKey:
            "Import link expired or unavailable. Create a new QR code in Platform."
        ])
    }
    let model = try JSONDecoder().decode(Self.self, from: data)
    let dateFormatter = ISO8601DateFormatter()
    dateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    guard model.version == 1, model.exportId == parts[3],
      ["coreml", "coreai"].contains(model.format),
      model.filename
        == "\(model.exportId).\(model.format == "coreai" ? "aimodel.zip" : "mlpackage.zip")",
      appTasks.contains(where: { String(describing: $0.yoloTask) == model.task }),
      model.downloadUrl.scheme == "https", model.size > 0,
      dateFormatter.date(from: model.expiresAt).map({ $0 > Date() }) == true
    else { throw CocoaError(.fileReadCorruptFile) }
    if model.format == "coreai" && !BasePredictor.isCoreAIAvailable {
      throw NSError(
        domain: "ModelImport", code: 2,
        userInfo: [
          NSLocalizedDescriptionKey:
            "This model requires Core AI on iOS 27 or later. Export Core ML for this device."
        ])
    }
    return model
  }
}

class ModelCacheManager {
  static let shared = ModelCacheManager()
  private static let assetRevision = "mobile-standard-v1"

  private init() {}

  /// Core ML and Core AI downloads of one model are cached side by side under their own extensions.
  func modelURL(for key: String) -> URL {
    documentsDirectory.appendingPathComponent("\(key)-\(Self.assetRevision)")
      .appendingPathExtension(remoteModelExtension == "aimodel" ? "aimodel" : "mlmodelc")
  }

  var customModels: [ModelEntry] {
    guard let data = UserDefaults.standard.data(forKey: "customModels"),
      let entries = try? JSONDecoder().decode([ModelEntry].self, from: data)
    else { return [] }
    return entries.filter {
      $0.installedURL.map { FileManager.default.fileExists(atPath: $0.path) } == true
    }
  }

  func saveCustomModel(_ entry: ModelEntry) throws {
    let entries = customModels.filter { $0.identifier != entry.identifier } + [entry]
    UserDefaults.standard.set(try JSONEncoder().encode(entries), forKey: "customModels")
  }

  func removeCustomModel(_ entry: ModelEntry) throws {
    if let url = entry.installedURL { try FileManager.default.removeItem(at: url) }
    UserDefaults.standard.set(
      try JSONEncoder().encode(customModels.filter { $0.identifier != entry.identifier }),
      forKey: "customModels")
  }

  func loadModel(
    from fileName: String, remoteURL: URL, key: String,
    completion: @escaping (Bool, String) -> Void
  ) {
    let legacyModelURL = documentsDirectory.appendingPathComponent(key)
      .appendingPathExtension("mlmodelc")
    if FileManager.default.fileExists(atPath: legacyModelURL.path) {
      try? FileManager.default.removeItem(at: legacyModelURL)
    }

    if isModelDownloaded(key: key) {
      completion(true, key)
    } else {
      ModelDownloadManager.shared.startDownload(
        url: remoteURL, fileName: fileName, key: key, completion: completion)
    }
  }

  func isModelDownloaded(key: String) -> Bool {
    FileManager.default.fileExists(atPath: modelURL(for: key).path)
  }
}

class ModelDownloadManager: NSObject {
  static let shared = ModelDownloadManager()
  private var downloadTasks:
    [URLSessionDownloadTask: (url: URL, key: String, modelImport: MobileModelImport?)] = [:]
  private var downloadCompletionHandlers: [URLSessionDownloadTask: (Bool, String) -> Void] = [:]
  private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
  var progressHandler: ((Double) -> Void)?

  private override init() {}

  private func completeTask(_ task: URLSessionDownloadTask, success: Bool, key: String) {
    let completion = downloadCompletionHandlers[task]
    downloadCompletionHandlers.removeValue(forKey: task)
    downloadTasks.removeValue(forKey: task)
    completion?(success, key)
  }

  func startDownload(
    url: URL, fileName: String, key: String, modelImport: MobileModelImport? = nil,
    completion: @escaping (Bool, String) -> Void
  ) {
    let downloadTask = session.downloadTask(with: url)
    let destinationURL = documentsDirectory.appendingPathComponent(fileName)
    downloadTasks[downloadTask] = (url: destinationURL, key: key, modelImport: modelImport)
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
      guard (downloadTask.response as? HTTPURLResponse)?.statusCode == 200 else {
        throw CocoaError(.fileReadUnknown)
      }
      let modelImport = downloadTasks[downloadTask]?.modelImport
      if let modelImport {
        let size = try location.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard size.map(Int64.init) == modelImport.size else {
          throw CocoaError(.fileReadCorruptFile)
        }
      }
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
          if ["aimodel", "mlmodel", "mlpackage"].contains(url.pathExtension) {
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

      installModel(from: foundModelURL, key: key, modelImport: modelImport) { success in
        // Clean up temp extraction directory and downloaded zip.
        try? FileManager.default.removeItem(at: tempExtractionURL)
        try? FileManager.default.removeItem(at: zipURL)
        self.completeTask(downloadTask, success: success, key: key)
      }
    } catch {
      print("Download processing failed: \(error)")
      completeTask(downloadTask, success: false, key: key)
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    guard let error = error,
      let downloadTask = task as? URLSessionDownloadTask,
      let key = downloadTasks[downloadTask]?.key
    else { return }

    print("Download failed: \(error.localizedDescription)")
    completeTask(downloadTask, success: false, key: key)
  }

  /// Moves the extracted model into Documents, compiling it first when it is a Core ML model.
  private func installModel(
    from url: URL, key: String, modelImport: MobileModelImport?,
    completion: @escaping (Bool) -> Void
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let isCoreAI = url.pathExtension == "aimodel"
        if let modelImport, isCoreAI != (modelImport.format == "coreai") {
          throw CocoaError(.fileReadCorruptFile)
        }
        let marker = url.appendingPathComponent("metadata.json")
        if isCoreAI && !FileManager.default.fileExists(atPath: marker.path) {
          throw CocoaError(.fileReadCorruptFile)
        }
        let installURL = isCoreAI ? url : try MLModel.compileModel(at: url)
        if let modelImport {
          let metadata: [String: String]
          if isCoreAI {
            let json =
              try JSONSerialization.jsonObject(with: Data(contentsOf: marker)) as? [String: Any]
            metadata = json?["creatorDefinedMetadata"] as? [String: String] ?? [:]
          } else {
            metadata =
              try MLModel(contentsOf: installURL).modelDescription.metadata[.creatorDefinedKey]
              as? [String: String] ?? [:]
          }
          guard metadata["task"] == modelImport.task else { throw CocoaError(.fileReadCorruptFile) }
        }
        // The destination follows the installed artifact, not the live setting, which can change during an install.
        let destination =
          modelImport?.entry.installedURL
          ?? ModelCacheManager.shared.modelURL(for: key).deletingPathExtension()
          .appendingPathExtension(installURL.pathExtension)
        if FileManager.default.fileExists(atPath: destination.path) {
          try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: installURL, to: destination)
        DispatchQueue.main.async { completion(true) }
      } catch {
        print("Failed to install model: \(error)")
        DispatchQueue.main.async { completion(false) }
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
