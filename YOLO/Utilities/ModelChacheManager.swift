import Foundation
import CoreML
import ZIPFoundation

class ModelCacheManager {
    static let shared = ModelCacheManager()
    var modelCache: [String: MLModel] = [:]
    private var accessOrder: [String] = []
    private let cacheLimit: Int = 3
    private var currentSelectedModelKey: String?
    
    private init() {
//        loadBundledModel()
    }
        
    private func getRemoteURL(for key: String) -> URL? {
        for (fileName, url) in fileMappings {
            if fileName == key {
                return url
            }
        }
        return nil
    }

    func loadBundledModel() {
        if let url = getModelFileURL(fileName: "yolov8m"),
           let bundledModel = try? MLModel(contentsOf: url) {
            addModelToCache(bundledModel, for: "yolov8m")
            let documentsURL = getDocumentsDirectory()
            let destinationURL = documentsURL.appendingPathComponent("yolov8m").appendingPathExtension("mlmodelc")
            
            do {
                if !FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.copyItem(at: url, to: destinationURL)
                    print("File copied to documents directory: \(destinationURL.path)")
                } else {
                    print("File already exists in documents directory: \(destinationURL.path)")
                }
            } catch {
                print("Error copying file: \(error)")
            }
        } else {
            print("Failed to load bundled model")
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
        
        let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: modelURL.path) {
            do {
                let model = try MLModel(contentsOf: modelURL)
                addModelToCache(model, for: key)
                completion(model, key)
            } catch let error {
                print(error)
            }
        }
    }
    
    func loadModel(from fileName: String, remoteURL: URL, key: String, completion: @escaping (MLModel?, String) -> Void) {
        if let cachedModel = modelCache[key] {
            if let index = accessOrder.firstIndex(of: key) {
                accessOrder.remove(at: index)
            }
            accessOrder.append(key)
            completion(cachedModel, key)
            return
        }
        
        let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: modelURL.path) {
            loadLocalModel(key: key) { model,key  in
                completion(model, key)
            }
        } else {
            ModelDownloadManager.shared.startDownload(url: remoteURL, fileName: fileName, key: key, completion: completion)
        }
    }
    
    private func compileAndLoadDownloadedModel(from url: URL, key: String, completion: @escaping (MLModel?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                
                let compiledModelURL = try MLModel.compileModel(at: url)
                let destinationURL = self.getDocumentsDirectory().appendingPathComponent(compiledModelURL.lastPathComponent)
                
                // Move compiled model to desired location
                try FileManager.default.moveItem(at: compiledModelURL, to: destinationURL)
                let model = try MLModel(contentsOf: destinationURL)
                DispatchQueue.main.async {
                    self.addModelToCache(model, for: key)
                    completion(model)
                }
            } catch {
                print("Failed to load model: \(error)")
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
        }
        modelCache[key] = model
        accessOrder.append(key)
    }
    
    func isModelDownloaded(key: String) -> Bool {
        let modelURL = getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlmodelc")
        return FileManager.default.fileExists(atPath: modelURL.path)
    }
    
    private func getDocumentsDirectory() -> URL {
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
}

class ModelDownloadManager: NSObject {
    static let shared = ModelDownloadManager()
    private var downloadTasks: [URLSessionDownloadTask: (url: URL, key: String)] = [:]
    private var priorityTask: URLSessionDownloadTask?
    
    private override init() {}
    
    func startDownload(url: URL, fileName: String, key: String, completion: @escaping (MLModel?, String) -> Void) {
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
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let destinationURL = downloadTasks[downloadTask]?.url,
              let key = downloadTasks[downloadTask]?.key else { return }
        do {
            let zipURL = destinationURL
            print("zipURL: \(zipURL)")
            if fileExists(at: zipURL) {
                try FileManager.default.removeItem(at: zipURL)
            }
            try FileManager.default.moveItem(at: location, to: zipURL)
            downloadTasks.removeValue(forKey: downloadTask)
            let unzipDestinationURL = destinationURL.deletingPathExtension()
            
            do {
                try FileManager.default.unzipItem(at: zipURL, to: getDocumentsDirectory(), skipCRC32: true)
                let modelURL = unzipDestinationURL
                print("modelURL: \(modelURL)")
                loadModel(from: modelURL, key: key) { model in
                    self.downloadCompletionHandlers[downloadTask]?(model, key)
                    self.downloadCompletionHandlers.removeValue(forKey: downloadTask)
                }
            } catch {
                print("Extraction of ZIP archive failed with error: \(error)")
                self.downloadCompletionHandlers[downloadTask]?(nil, key)
                self.downloadCompletionHandlers.removeValue(forKey: downloadTask)
            }
        } catch {
            print("Error moving downloaded file: \(error)")
            self.downloadCompletionHandlers[downloadTask]?(nil, key)
            self.downloadCompletionHandlers.removeValue(forKey: downloadTask)
        }
    }
    
    private func loadModel(from url: URL, key: String, completion: @escaping (MLModel?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let compiledModelURL = try MLModel.compileModel(at: url)
                let model = try MLModel(contentsOf: compiledModelURL)
                let localModelURL = self.getDocumentsDirectory().appendingPathComponent(key).appendingPathExtension("mlmodelc")
                ModelCacheManager.shared.addModelToCache(model, for: key)
                try FileManager.default.moveItem(at: compiledModelURL, to: localModelURL)
                print("model copied to document directory")
                DispatchQueue.main.async {
                    completion(model)
                }
            } catch {
                print("Failed to load model: \(error)")
                DispatchQueue.main.async {
                    completion(nil)
                }
            }
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
            let fileURLs = try fileManager.contentsOfDirectory(at: documentsDirectory, includingPropertiesForKeys: nil)
            for fileURL in fileURLs {
                if fileURL.pathExtension == "mlmodel" || fileURL.pathExtension == "mlmodelc" || fileURL.pathExtension == "mlpackage" {
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
