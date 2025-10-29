// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app, providing the main user interface for model selection and visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController serves as the primary interface for users to interact with YOLO models.
//  It provides the ability to select different models, tasks (detection, segmentation, classification, etc.),
//  and visualize results in real-time. The controller manages the loading of local and remote models,
//  handles UI updates during model loading and inference, and provides functionality for capturing
//  and sharing detection results. Advanced features include model download progress
//  tracking, and adaptive UI layout for different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import YOLO

// MARK: - Extensions
extension Result {
  var isSuccess: Bool { if case .success = self { return true } else { return false } }
}

extension Array {
  subscript(safe index: Int) -> Element? {
    return indices.contains(index) ? self[index] : nil
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  // MARK: - External Display Support
  override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
  
    if SceneDelegate.hasExternalDisplay {
      return [.landscapeLeft, .landscapeRight]
    } else {
      return [.portrait, .landscapeLeft, .landscapeRight]
    }
  }

  override var shouldAutorotate: Bool {
    return true
  }

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet weak var View0: UIView!
  @IBOutlet weak var segmentedControl: UISegmentedControl!
  @IBOutlet weak var modelSegmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var logoImage: UIImageView!

  let selection = UISelectionFeedbackGenerator()

  var currentLoadingEntry: ModelEntry?

  var customModelButton: UIButton!

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()

  private var loadingOverlayView: UIView?

  // MARK: - Constants
  private struct Constants {
    static let defaultTaskIndex = 2  // Detect
    static let tableRowHeight: CGFloat = 30
    static let logoURL = "https://www.ultralytics.com"
    static let progressViewWidth: CGFloat = 200
  }

  // MARK: - Loading State Management
  private func setLoadingState(_ loading: Bool, showOverlay: Bool = false) {
    loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
    view.isUserInteractionEnabled = !loading
    if showOverlay && loading { updateLoadingOverlay(true) }
    if !loading { updateLoadingOverlay(false) }
  }

  private func updateLoadingOverlay(_ show: Bool) {
    if show && loadingOverlayView == nil {
      let overlay = UIView(frame: view.bounds)
      overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
      view.addSubview(overlay)
      loadingOverlayView = overlay
      view.bringSubviewToFront(downloadProgressView)
      view.bringSubviewToFront(downloadProgressLabel)
    } else if !show {
      loadingOverlayView?.removeFromSuperview()
      loadingOverlayView = nil
    }
  }

  let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
    ("Classify", "ClassifyModels", .classify),
    ("Segment", "SegmentModels", .segment),
    ("Detect", "DetectModels", .detect),
    ("Pose", "PoseModels", .pose),
    ("OBB", "OBBModels", .obb),
  ]

  private var modelsForTask: [String: [String]] = [:]

  var currentModels: [ModelEntry] = []
  private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] =
    [:]

  var currentTask: String = ""
  var currentModelName: String = ""

  private var isLoadingModel = false

  override func viewDidLoad() {
    super.viewDidLoad()

    debugCheckModelFolders()

    // MARK: External Display Setup (Optional)
  
    setupExternalDisplayNotifications()

    checkForExternalDisplays()

 
    if UIScreen.screens.count > 1 {
      yoloView.isHidden = true
    }


    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    setupModelSegmentedControl()
    setupCustomModelButton()

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name

      reloadModelEntriesAndLoadFirst(for: currentTask)
     
      if UIScreen.screens.count > 1 {
        print("External display may be connected at startup - will be handled by notifications")
      }
    }

    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    [yoloView.labelName, yoloView.labelFPS].forEach { $0?.isHidden = true }

  
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderNumItems.addTarget(
      self, action: #selector(sliderValueChanged), for: .valueChanged)


    [labelName, labelFPS, labelVersion].forEach {
      $0?.textColor = .white
      $0?.overrideUserInterfaceStyle = .dark
    }
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    {
      labelVersion.text = "v\(version) (\(build))"
    }


    [downloadProgressView, downloadProgressLabel].forEach {
      $0.isHidden = true
      $0.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview($0)
    }
    downloadProgressLabel.textAlignment = .center
    downloadProgressLabel.textColor = .systemGray
    downloadProgressLabel.font = .systemFont(ofSize: 14)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: Constants.progressViewWidth),
      downloadProgressView.heightAnchor.constraint(equalToConstant: 2),
      downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      downloadProgressLabel.topAnchor.constraint(
        equalTo: downloadProgressView.bottomAnchor, constant: 8),
    ])

    ModelDownloadManager.shared.progressHandler = { [weak self] progress in
      guard let self = self else { return }
      DispatchQueue.main.async {
        self.downloadProgressView.progress = Float(progress)
        self.downloadProgressLabel.isHidden = false
        let percentage = Int(progress * 100)
        self.downloadProgressLabel.text = "Downloading \(percentage)%"
      }
    }
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    view.overrideUserInterfaceStyle = .dark
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(
        at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
      )
    else { return [] }

    let modelFiles =
      fileURLs
      .filter { ["mlmodel", "mlpackage"].contains($0.pathExtension) }
      .map { $0.lastPathComponent }

    return folderName == "DetectModels" ? reorderDetectionModels(modelFiles) : modelFiles.sorted()
  }

  private func reorderDetectionModels(_ fileNames: [String]) -> [String] {
    let order: [Character: Int] = ["n": 0, "m": 1, "s": 2, "l": 3, "x": 4]
    let (official, custom) = fileNames.reduce(into: ([String](), [String]())) { result, name in
      let base = (name as NSString).deletingPathExtension.lowercased()
      base.hasPrefix("yolo") && order[base.last ?? "z"] != nil
        ? result.0.append(name) : result.1.append(name)
    }
    return custom.sorted()
      + official.sorted {
        order[($0 as NSString).deletingPathExtension.lowercased().last ?? "z"] ?? 99 < order[
          ($1 as NSString).deletingPathExtension.lowercased().last ?? "z"] ?? 99
      }
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)
    let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
    standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)

    let yoloTask = tasks.first(where: { $0.name == taskName })?.yoloTask ?? .detect
    ModelSelectionManager.setupSegmentedControl(
      modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)

    if let firstSize = ModelSelectionManager.ModelSize.allCases.first,
      let model = standardModels[firstSize]
    {
      let entry = ModelEntry(
        displayName: (model.name as NSString).deletingPathExtension,
        identifier: model.name,
        isLocalBundle: model.isLocal,
        isRemote: model.url != nil,
        remoteURL: model.url
      )
      loadModel(entry: entry, forTask: taskName)
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    let localEntries = localFileNames.map { fileName -> ModelEntry in
      let display = (fileName as NSString).deletingPathExtension
      return ModelEntry(
        displayName: display,
        identifier: fileName,
        isLocalBundle: true,
        isRemote: false,
        remoteURL: nil
      )
    }

 
    let localModelNames = Set(localEntries.map { $0.displayName.lowercased() })

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.compactMap { (modelName, url) -> ModelEntry? in
   
      guard !localModelNames.contains(modelName.lowercased()) else { return nil }

      return ModelEntry(
        displayName: modelName,
        identifier: modelName,
        isLocalBundle: false,
        isRemote: true,
        remoteURL: url
      )
    }

    return localEntries + remoteEntries
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      return
    }
    isLoadingModel = true


    let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay


    if !hasExternalDisplay {
      yoloView.resetLayers()
      yoloView.setInferenceFlag(ok: false)
    }

    setLoadingState(true, showOverlay: true)
    resetDownloadProgress()

    print("Start loading model: \(entry.displayName)")

    currentLoadingEntry = entry

    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }

        guard let folderURL = self.tasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          DispatchQueue.main.async { [weak self] in
            self?.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        DispatchQueue.main.async { [weak self] in
          guard let self = self else { return }
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"

         
          let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay

          if hasExternalDisplay {
            
            self.finishLoadingModel(success: true, modelName: entry.displayName)
          } else {
           
            self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
              switch result {
              case .success():
                Task { @MainActor in
                  if yoloTask == .pose, let poseEstimator = self.yoloView.currentPredictor as? PoseEstimator {
                    self.configureSkeletonMode(for: poseEstimator)
                  }
                }
                self.finishLoadingModel(success: true, modelName: entry.displayName)
              case .failure(let error):
                print(error)
                self.finishLoadingModel(success: false, modelName: entry.displayName)
              }
            }
          }
        }
      }
    } else {
      let key = entry.identifier  

      if ModelCacheManager.shared.isModelDownloaded(key: key) {
        loadCachedModelAndSetToYOLOView(
          key: key, yoloTask: yoloTask, displayName: entry.displayName)
      } else {
        guard let remoteURL = entry.remoteURL else {
          self.finishLoadingModel(success: false, modelName: entry.displayName)
          return
        }

        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false

      
        self.downloadProgressLabel.text = "Downloading \(processString(entry.displayName))"

        let localZipFileName = remoteURL.lastPathComponent  

        ModelCacheManager.shared.loadModel(
          from: localZipFileName,
          remoteURL: remoteURL,
          key: key
        ) { [weak self] mlModel, loadedKey in
          guard let self = self else { return }
          if mlModel == nil {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
            return
          }
          self.loadCachedModelAndSetToYOLOView(
            key: loadedKey,
            yoloTask: yoloTask,
            displayName: entry.displayName)
        }
      }
    }
  }

  private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String)
  {
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[
      0]
    let localModelURL = documentsDirectory.appendingPathComponent(key).appendingPathExtension(
      "mlmodelc")

    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"
      let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay

      if hasExternalDisplay {
        self.finishLoadingModel(success: true, modelName: displayName)
      } else {
       
        self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) { result in
          switch result {
          case .success():
            Task { @MainActor in
              if yoloTask == .pose, let poseEstimator = self.yoloView.currentPredictor as? PoseEstimator {
                self.configureSkeletonMode(for: poseEstimator)
              }
            }
            self.finishLoadingModel(success: true, modelName: displayName)
          case .failure(let error):
            print(error)
            self.finishLoadingModel(success: false, modelName: displayName)
          }
        }
      }
    }
  }

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    downloadProgressLabel.text = ""
    [downloadProgressView, downloadProgressLabel].forEach { $0.isHidden = true }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.setLoadingState(false)
      self.isLoadingModel = false
      self.resetDownloadProgress()

      if success {
        let yoloTask = self.tasks.first(where: { $0.name == self.currentTask })?.yoloTask ?? .detect

        ModelSelectionManager.setupSegmentedControl(
          self.modelSegmentedControl,
          standardModels: self.standardModels,
          currentTask: yoloTask,
          preserveSelection: true
        )

        ModelSelectionManager.updateSegmentAppearance(
          self.modelSegmentedControl,
          standardModels: self.standardModels,
          currentTask: yoloTask
        )
      }


      if success {
       
        self.currentModelName = processString(modelName)
        let yoloTask = self.tasks.first(where: { $0.name == self.currentTask })?.yoloTask ?? .detect

        var fullModelPath = ""

        if let entry = self.currentLoadingEntry {
          if entry.isLocalBundle {
            
            if let folderURL = self.tasks.first(where: { $0.name == self.currentTask })?.folder,
              let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
            {
              let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
              fullModelPath = modelURL.path
              print("External display local model path: \(fullModelPath)")
            }
          } else {
            fullModelPath = entry.identifier
            

          
            let documentsDirectory = FileManager.default.urls(
              for: .documentDirectory, in: .userDomainMask)[0]
            let localModelURL =
              documentsDirectory
              .appendingPathComponent(entry.identifier)
              .appendingPathExtension("mlmodelc")

            if !FileManager.default.fileExists(atPath: localModelURL.path) {
              print("Cached model not found at: \(localModelURL.path)")
              return
            }
          }
        }


        if !fullModelPath.isEmpty {
          ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
         
          self.checkAndNotifyExternalDisplayIfReady()
        } else {
          print("Could not determine model path for external display")
        }
      }

      // Check if external display is connected
      let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay

      // Only set inference flag on YOLOView if no external display
      if !hasExternalDisplay {
        self.yoloView.setInferenceFlag(ok: success)
      }

      if success {
        // currentModelName is already set above in the notification section
        self.labelName.text = processString(modelName)
        
        // Enable skeleton mode for pose models (with small delay to ensure predictor is set)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
          if self.currentTask == "Pose", let poseEstimator = self.yoloView.currentPredictor as? PoseEstimator {
            self.configureSkeletonMode(for: poseEstimator)
          }
        }
      }
    }
  }

  @IBAction func vibrate(_ sender: Any) { selection.selectionChanged() }

  @IBAction func indexChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()
    guard tasks.indices.contains(sender.selectedSegmentIndex) else { return }

    let newTask = tasks[sender.selectedSegmentIndex].name

    if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
      let alert = UIAlertController(
        title: "\(newTask) Models not found",
        message: "Please add or define models for \(newTask).", preferredStyle: .alert)
      alert.addAction(
        UIAlertAction(title: "OK", style: .cancel) { _ in alert.dismiss(animated: true) })
      present(alert, animated: true)
      sender.selectedSegmentIndex = tasks.firstIndex { $0.name == currentTask } ?? 0
      return
    }

    currentTask = newTask

    // Notify external display of task change immediately (Optional external display feature)
    NotificationCenter.default.post(
      name: .taskDidChange,
      object: nil,
      userInfo: ["task": newTask]
    )
    reloadModelEntriesAndLoadFirst(for: currentTask)
  }
  
  // MARK: - Skeleton Configuration
  
  /// Configure skeleton mode for the pose estimator (always uses articulated)
  private func configureSkeletonMode(for poseEstimator: PoseEstimator) {
    poseEstimator.useRealisticSkeleton = true
    poseEstimator.skeletonType = .articulated
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: Constants.logoURL) {
      UIApplication.shared.open(link)
    }
  }

  private func setupModelSegmentedControl() {
    modelSegmentedControl.isHidden = false
    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.apportionsSegmentWidthsByContent = true
    modelSegmentedControl.addTarget(
      self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)

    modelSegmentedControl.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      modelSegmentedControl.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
      modelSegmentedControl.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
    ])
  }

  private func setupCustomModelButton() {
    customModelButton = UIButton(type: .system)
    customModelButton.setTitle("Custom", for: .normal)
    customModelButton.titleLabel?.font = UIFont.systemFont(ofSize: 13)
    customModelButton.setTitleColor(.white, for: .normal)
    customModelButton.setTitleColor(.systemBlue, for: .selected)
    customModelButton.backgroundColor = .systemBackground.withAlphaComponent(0.1)
    customModelButton.layer.cornerRadius = 8
    customModelButton.layer.borderWidth = 1
    customModelButton.layer.borderColor = UIColor.systemGray.cgColor
    customModelButton.addTarget(
      self, action: #selector(customModelButtonTapped), for: .touchUpInside)
    customModelButton.translatesAutoresizingMaskIntoConstraints = false

    View0.addSubview(customModelButton)

    modelSegmentedControl.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      modelSegmentedControl.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
      modelSegmentedControl.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
    ])
  }
  
  // MARK: - Actions
  @objc func customModelButtonTapped() {
    selection.selectionChanged()
  }

  func updateModelSegmentedControlAppearance() {
    guard modelSegmentedControl != nil else { return }

    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.backgroundColor = .clear

    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect
    ModelSelectionManager.updateSegmentAppearance(
      modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)
  }

  @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    if sender.selectedSegmentIndex < ModelSelectionManager.ModelSize.allCases.count {
      let size = ModelSelectionManager.ModelSize.allCases[sender.selectedSegmentIndex]
      if let model = standardModels[size] {
        let entry = ModelEntry(
          displayName: (model.name as NSString).deletingPathExtension,
          identifier: model.name,
          isLocalBundle: model.isLocal,
          isRemote: model.url != nil,
          remoteURL: model.url
        )
        loadModel(entry: entry, forTask: currentTask)
      }
    }
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    adjustLayoutForExternalDisplayIfNeeded()
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] image in
      guard let self = self, let image = image else { return print("error capturing photo") }
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        let vc = UIActivityViewController(activityItems: [image], applicationActivities: nil)
        vc.popoverPresentationController?.sourceView = self.View0
        self.present(vc, animated: true)
      }
    }
  }

  @objc func sliderValueChanged(_ sender: UISlider) {
    let conf = Double(round(100 * yoloView.sliderConf.value)) / 100
    let iou = Double(round(100 * yoloView.sliderIoU.value)) / 100
    let maxItems = Int(yoloView.sliderNumItems.value)

    NotificationCenter.default.post(
      name: .thresholdDidChange,
      object: nil,
      userInfo: [
        "conf": conf,
        "iou": iou,
        "maxItems": maxItems,
      ]
    )

    print("Threshold changed - Conf: \(conf), IoU: \(iou), Max items: \(maxItems)")
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }

  private func debugCheckModelFolders() {
    print("\nDEBUG: Checking model folders...")
    let folders = ["DetectModels", "SegmentModels", "ClassifyModels", "PoseModels", "OBBModels"]

    for folder in folders {
      if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
        print("\(folder) found at: \(folderURL.path)")

        do {
          let files = try FileManager.default.contentsOfDirectory(
            at: folderURL, includingPropertiesForKeys: nil)
          let models = files.filter {
            $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage"
          }
          print("Models: \(models.map { $0.lastPathComponent })")
        } catch {
          print("Error reading folder: \(error)")
        }
      } else {
        print("\(folder) NOT FOUND in bundle")
      }
    }
    print("\n")
  }

}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { [weak self] in
      self?.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
      self?.labelFPS.textColor = .white
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async { [weak self] in
      guard self != nil else { return }
      // Share results with external display (Optional external display feature)
      ExternalDisplayManager.shared.shareResults(result)

      // Also send via notification for direct communication (Optional external display feature)
      NotificationCenter.default.post(
        name: .yoloResultsAvailable,
        object: nil,
        userInfo: ["result": result]
      )
    }
  }

}
