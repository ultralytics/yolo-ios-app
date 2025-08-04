// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app, providing the main user interface for model selection and visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController serves as the primary interface for users to interact with YOLO models.
//  It provides the ability to select different models, tasks (detection, segmentation, classification, etc.),
//  and visualize results in real-time. The controller manages the loading of local and remote models,
//  handles UI updates during model loading and inference, and provides functionality for capturing
//  and sharing detection results. Advanced features include screen recording, model download progress
//  tracking, and adaptive UI layout for different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import ReplayKit
import UIKit
import YOLO

// Definition of custom table view cell
class ModelTableViewCell: UITableViewCell {
  static let identifier = "ModelTableViewCell"

  private let modelNameLabel: UILabel = {
    let label = UILabel()
    label.textAlignment = .center
    label.font = UIFont.systemFont(ofSize: 14, weight: .medium)
    label.translatesAutoresizingMaskIntoConstraints = false
    // Configure auto text size adjustment for long content
    label.adjustsFontSizeToFitWidth = true
    label.minimumScaleFactor = 0.7  // Scale down to 70% minimum
    label.lineBreakMode = .byClipping  // Clip text instead of using ellipsis
    return label
  }()

  private let downloadIconImageView: UIImageView = {
    let imageView = UIImageView(image: UIImage(systemName: "icloud.and.arrow.down"))
    imageView.tintColor = .white
    imageView.contentMode = .scaleAspectFit
    imageView.translatesAutoresizingMaskIntoConstraints = false
    imageView.isHidden = true
    return imageView
  }()

  override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
    super.init(style: style, reuseIdentifier: reuseIdentifier)
    setupUI()
  }

  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }

  private func setupUI() {
    backgroundColor = .clear
    selectionStyle = .default

    contentView.addSubview(modelNameLabel)
    contentView.addSubview(downloadIconImageView)

    NSLayoutConstraint.activate([
      // Center the label
      modelNameLabel.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),  // Add center X alignment
      modelNameLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
      // Set margin from the leading edge
      modelNameLabel.leadingAnchor.constraint(
        greaterThanOrEqualTo: contentView.leadingAnchor, constant: 8),
      // Ensure margin between label and download icon
      modelNameLabel.trailingAnchor.constraint(
        lessThanOrEqualTo: downloadIconImageView.leadingAnchor, constant: -4),

      // Position download icon at the trailing edge
      downloadIconImageView.trailingAnchor.constraint(
        equalTo: contentView.trailingAnchor, constant: -4),
      downloadIconImageView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
      downloadIconImageView.widthAnchor.constraint(equalToConstant: 16),  // Slightly smaller to save space
      downloadIconImageView.heightAnchor.constraint(equalToConstant: 16),
    ])

    // Configure the background view for selection state
    let selectedBGView = UIView()
    selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
    selectedBGView.layer.cornerRadius = 5  // More gentle corner radius
    selectedBGView.layer.masksToBounds = true
    selectedBackgroundView = selectedBGView
  }

  // Method to configure the cell
  func configure(with modelName: String, isRemote: Bool, isDownloaded: Bool) {
    modelNameLabel.text = modelName

    // Show download icon only for remote models that are not yet downloaded
    let showDownloadIcon = isRemote && !isDownloaded
    downloadIconImageView.isHidden = !showDownloadIcon

    // Adjust text priorities based on icon visibility
    if showDownloadIcon {
      // Keep center alignment when icon is visible
      modelNameLabel.textAlignment = .center
      // Adjust width to accommodate icon space
      modelNameLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
      // Set priority to ensure center alignment is respected
      modelNameLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
    } else {
      // Keep center alignment when no icon is present
      modelNameLabel.textAlignment = .center
      // Expand to full width
      modelNameLabel.setContentCompressionResistancePriority(.defaultHigh, for: .horizontal)
      // Prioritize center alignment
      modelNameLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
    }
  }

  override func layoutSubviews() {
    super.layoutSubviews()

    // Adjust selection background view size - reduce margins to bring frame closer to table view
    if let selectedBGView = selectedBackgroundView {
      selectedBGView.frame = bounds.insetBy(dx: 2, dy: 1)
    }

    // Final label adjustments - ensure label displays correctly after layout
    let iconSpace = downloadIconImageView.isHidden ? 0 : 20  // Consider icon space if visible
    let availableWidth = bounds.width - 16 - CGFloat(iconSpace)  // Left/right margins(16) + icon space

    // Set maximum label width to ensure center alignment works properly
    modelNameLabel.preferredMaxLayoutWidth = availableWidth

    // Fine-tune frame to enforce center alignment
    if downloadIconImageView.isHidden {
      // Center completely when no icon is present
      modelNameLabel.center.x = bounds.width / 2
    } else {
      // Keep text centered in the cell even when download icon is visible
      modelNameLabel.center.x = bounds.width / 2
    }
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {
  
  // Override supported orientations based on external display connection
  override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
    // Use SceneDelegate's state to determine orientation support
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
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var logoImage: UIImageView!

  var shareButton = UIButton()
  var recordButton = UIButton()
  var externalUIToggleButton = UIButton()
  let selection = UISelectionFeedbackGenerator()
  var firstLoad = true
  
  // External display UI state
  private var isExternalUIVisible = true
  
  // Store current loading entry for external display notification
  private var currentLoadingEntry: ModelEntry?

  private let downloadProgressView: UIProgressView = {
    let pv = UIProgressView(progressViewStyle: .default)
    pv.progress = 0.0
    pv.isHidden = true
    return pv
  }()

  private let downloadProgressLabel: UILabel = {
    let label = UILabel()
    label.text = ""
    label.textAlignment = .center
    label.textColor = .systemGray
    label.font = UIFont.systemFont(ofSize: 14)
    label.isHidden = true
    return label
  }()

  private var loadingOverlayView: UIView?

  func showLoadingOverlay() {
    guard loadingOverlayView == nil else { return }
    let overlay = UIView(frame: view.bounds)
    overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)

    view.addSubview(overlay)
    loadingOverlayView = overlay
    view.bringSubviewToFront(downloadProgressView)
    view.bringSubviewToFront(downloadProgressLabel)

    view.isUserInteractionEnabled = false
  }

  func hideLoadingOverlay() {
    loadingOverlayView?.removeFromSuperview()
    loadingOverlayView = nil
    view.isUserInteractionEnabled = true
  }

  private let tasks: [(name: String, folder: String)] = [
    ("Classify", "ClassifyModels"),  // index 0
    ("Segment", "SegmentModels"),  // index 1
    ("Detect", "DetectModels"),  // index 2
    ("Pose", "PoseModels"),  // index 3
    ("OBB", "OBBModels"),  // index 4
  ]

  private var modelsForTask: [String: [String]] = [:]

  private var currentModels: [ModelEntry] = []

  private var currentTask: String = ""
  private var currentModelName: String = ""

  private var isLoadingModel = false

  private let modelTableView: UITableView = {
    let table = UITableView()
    table.isHidden = true
    table.layer.cornerRadius = 5  // Match corner radius of other elements
    table.clipsToBounds = true
    return table
  }()

  private let tableViewBGView = UIView()

  private var selectedIndexPath: IndexPath?

  override func viewDidLoad() {
    super.viewDidLoad()
    
    // Debug: Check model folders
    debugCheckModelFolders()
    
    // Setup external display notifications
    setupExternalDisplayNotifications()
    
    // Check for already connected external displays
    checkForExternalDisplays()
    
    // If external display is already connected, ensure YOLOView doesn't interfere
    if UIScreen.screens.count > 1 {
      print("External display already connected at startup - deferring camera init")
      yoloView.isHidden = true
    }

    setupTaskSegmentedControl()
    loadModelsForAllTasks()

    // Initialize task selection without loading model on main display
    if tasks.indices.contains(2) {
      segmentedControl.selectedSegmentIndex = 2
      currentTask = tasks[2].name
      
      // Don't load model on main YOLOView if external display is connected
      if UIScreen.screens.count == 1 {
        // Only load model on iPhone if no external display
        if let detectFolderURL = Bundle.main.url(forResource: "DetectModels", withExtension: nil) {
          let yolo11nURL = detectFolderURL.appendingPathComponent("yolo11n.mlmodel")
          if FileManager.default.fileExists(atPath: yolo11nURL.path) {
            print("Setting initial model: yolo11n.mlmodel")
            yoloView.setModel(modelPathOrName: yolo11nURL.path, task: .detect) { result in
              switch result {
              case .success():
                print("Initial model loaded successfully")
              case .failure(let error):
                print("Failed to load initial model: \(error)")
              }
            }
          }
        }
      } else {
        // External display connected - stop main YOLOView
        print("External display detected at startup - stopping main YOLOView")
        yoloView.stop()
        yoloView.isHidden = true
        
        // Ensure camera is fully released
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
          print("Main YOLOView fully stopped, external display can now use camera")
        }
      }
      
      // Always load model list for UI
      currentModels = makeModelEntries(for: currentTask)
      modelTableView.reloadData()
      
      // Show model table view if models are available
      if !currentModels.isEmpty {
        modelTableView.isHidden = false
      }
    }

    setupTableView()
    setupButtons()

    yoloView.delegate = self
    yoloView.labelName.isHidden = true
    yoloView.labelFPS.isHidden = true
    
    // Add target to sliders to monitor changes
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderNumItems.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)

    // Force label text color to white
    labelName.textColor = .white
    labelFPS.textColor = .white
    labelVersion.textColor = .white

    // Set app version  
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
      labelVersion.text = "v\(version)"
      print("DEBUG: App version set to: v\(version)")
    } else {
      labelVersion.text = ""
      print("DEBUG: Could not retrieve app version")
    }

    labelName.overrideUserInterfaceStyle = .dark
    labelFPS.overrideUserInterfaceStyle = .dark
    labelVersion.overrideUserInterfaceStyle = .dark

    downloadProgressView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressView)

    downloadProgressLabel.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressLabel)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: 200),
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

    // Force text color to white whenever the screen appears
    enforceWhiteTextColor()

    // Override system appearance mode setting to ensure consistent styling
    view.overrideUserInterfaceStyle = .dark
  }

  // Called when trait collection changes (dark mode/light mode)
  override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
    super.traitCollectionDidChange(previousTraitCollection)

    // Maintain white text color even when appearance mode changes
    enforceWhiteTextColor()
  }

  // Common method to set label text color to white
  private func enforceWhiteTextColor() {
    labelName.textColor = .white
    labelFPS.textColor = .white
    labelVersion.textColor = .white
  }

  private func setupTaskSegmentedControl() {
    segmentedControl.removeAllSegments()
    for (index, taskInfo) in tasks.enumerated() {
      segmentedControl.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
    }
  }

  private func loadModelsForAllTasks() {
    for taskInfo in tasks {
      let taskName = taskInfo.name
      let folderName = taskInfo.folder
      let modelFiles = getModelFiles(in: folderName)
      modelsForTask[taskName] = modelFiles
    }
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil) else {
      return []
    }
    do {
      let fileURLs = try FileManager.default.contentsOfDirectory(
        at: folderURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
      )
      let modelFiles =
        fileURLs
        .filter { $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage" }
        .map { $0.lastPathComponent }

      if folderName == "DetectModels" {
        return reorderDetectionModels(modelFiles)
      } else {
        return modelFiles.sorted()
      }

    } catch {
      print("Error reading contents of folder \(folderName): \(error)")
      return []
    }
  }

  private func reorderDetectionModels(_ fileNames: [String]) -> [String] {
    let officialOrder: [Character: Int] = ["n": 0, "m": 1, "s": 2, "l": 3, "x": 4]

    var customModels: [String] = []
    var officialModels: [String] = []

    for fileName in fileNames {
      let baseName = (fileName as NSString).deletingPathExtension.lowercased()

      if baseName.hasPrefix("yolo"),
        let lastChar = baseName.last,
        officialOrder.keys.contains(lastChar)
      {
        officialModels.append(fileName)
      } else {
        customModels.append(fileName)
      }
    }

    customModels.sort { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }

    officialModels.sort { fileA, fileB in
      let baseA = (fileA as NSString).deletingPathExtension.lowercased()
      let baseB = (fileB as NSString).deletingPathExtension.lowercased()
      guard let lastA = baseA.last, let lastB = baseB.last,
        let indexA = officialOrder[lastA], let indexB = officialOrder[lastB]
      else {
        return baseA < baseB
      }
      return indexA < indexB
    }

    return customModels + officialModels
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)

    if !currentModels.isEmpty {
      modelTableView.isHidden = false
      modelTableView.reloadData()

      DispatchQueue.main.async {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        let firstModel = self.currentModels[0]
        self.loadModel(entry: firstModel, forTask: taskName)
      }
    } else {
      print("No models found for task: \(taskName)")
      modelTableView.isHidden = true
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

    // Get local model names for filtering
    let localModelNames = Set(localEntries.map { $0.displayName.lowercased() })

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.compactMap { (modelName, url) -> ModelEntry? in
      // Only include remote models if no local model with the same name exists
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

  private func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      print("Model is already loading. Please wait.")
      return
    }
    isLoadingModel = true
    yoloView.resetLayers()
    if !firstLoad {
      showLoadingOverlay()
      yoloView.setInferenceFlag(ok: false)
    } else {
      firstLoad = false
    }

    self.activityIndicator.startAnimating()
    self.downloadProgressView.progress = 0.0
    self.downloadProgressView.isHidden = true
    self.downloadProgressLabel.isHidden = true
    self.view.isUserInteractionEnabled = false
    self.modelTableView.isUserInteractionEnabled = false

    print("Start loading model: \(entry.displayName)")
    print("  - displayName: \(entry.displayName)")
    print("  - identifier: \(entry.identifier)")
    print("  - isLocalBundle: \(entry.isLocalBundle)")
    print("  - task: \(task)")
    
    // Store current entry for external display notification
    currentLoadingEntry = entry

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }
        let yoloTask = self.convertTaskNameToYOLOTask(task)

        guard let folderURL = self.tasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          DispatchQueue.main.async {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        DispatchQueue.main.async {
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"
          self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
            switch result {
            case .success():
              self.finishLoadingModel(success: true, modelName: entry.displayName)
            case .failure(let error):
              print(error)
              self.finishLoadingModel(success: false, modelName: entry.displayName)
            }
          }
        }
      }
    } else {
      let yoloTask = self.convertTaskNameToYOLOTask(task)

      let key = entry.identifier  // "yolov8n", "yolov8m-seg", etc.

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

        let localZipFileName = remoteURL.lastPathComponent  // ex. "yolov8n.mlpackage.zip"

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

  private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String) {
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let localModelURL = documentsDirectory.appendingPathComponent(key).appendingPathExtension("mlmodelc")

    DispatchQueue.main.async {
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"
      self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) { result in
        switch result {
        case .success():
          self.finishLoadingModel(success: true, modelName: displayName)
        case .failure(let error):
          print(error)
          self.finishLoadingModel(success: false, modelName: displayName)
        }
      }
    }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async {
      self.activityIndicator.stopAnimating()
      self.downloadProgressView.isHidden = true

      self.downloadProgressLabel.isHidden = true
      //            self.downloadProgressLabel.isHidden = false
      //            self.downloadProgressLabel.text = "Loading \(modelName)"

      self.view.isUserInteractionEnabled = true
      self.modelTableView.isUserInteractionEnabled = true
      self.isLoadingModel = false

      self.modelTableView.reloadData()
      
      // Notify external display of model change
      if success {
        // Update currentModelName
        self.currentModelName = processString(modelName)
        
        let yoloTask = self.convertTaskNameToYOLOTask(self.currentTask)
        
        // Determine the correct model path for external display
        var fullModelPath = ""
        
        // Use the stored entry from loadModel
        if let entry = self.currentLoadingEntry {
            if entry.isLocalBundle {
                // For local bundle models
                if let folderURL = self.tasks.first(where: { $0.name == self.currentTask })?.folder,
                   let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) {
                    let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
                    fullModelPath = modelURL.path
                    print("📦 External display local model path: \(fullModelPath)")
                }
            } else {
                // For remote models, use the cached path
                let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let localModelURL = documentsDirectory
                  .appendingPathComponent(entry.identifier)
                  .appendingPathExtension("mlmodelc")
                fullModelPath = localModelURL.path
                print("☁️ External display cached model path: \(fullModelPath)")
                
                // Verify the cached model exists
                if !FileManager.default.fileExists(atPath: fullModelPath) {
                    print("❌ Cached model not found at: \(fullModelPath)")
                    return
                }
            }
        }
        
        // Only notify if we have a valid path
        if !fullModelPath.isEmpty {
            ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
            print("✅ Model loaded successfully and notified to external display: \(modelName)")
            
            // Also check if external display is waiting for initial model
            self.checkAndNotifyExternalDisplayIfReady()
        } else {
            print("❌ Could not determine model path for external display")
        }
      }

      if let ip = self.selectedIndexPath {
        self.modelTableView.selectRow(at: ip, animated: false, scrollPosition: .none)
      }
      if !self.firstLoad {
        self.hideLoadingOverlay()
      }
      self.yoloView.setInferenceFlag(ok: true)

      if success {
        print("Finished loading model: \(modelName)")
        self.currentModelName = modelName
        DispatchQueue.main.async {
          self.labelName.text = processString(modelName)
          // Force text color to white
          self.labelName.textColor = .white
        }

        self.downloadProgressLabel.text = "Finished loading model \(modelName)"
        self.downloadProgressLabel.isHidden = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.downloadProgressLabel.isHidden = true
          self.downloadProgressLabel.text = ""
        }

      } else {
        print("Failed to load model: \(modelName)")
      }
    }
  }

  private func convertTaskNameToYOLOTask(_ task: String) -> YOLOTask {
    switch task {
    case "Detect": return .detect
    case "Segment": return .segment
    case "Classify": return .classify
    case "Pose": return .pose
    case "OBB": return .obb
    default: return .detect
    }
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  @IBAction func indexChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    let index = sender.selectedSegmentIndex
    guard tasks.indices.contains(index) else { return }

    let newTask = tasks[index].name

    if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
      let alert = UIAlertController(
        title: "\(newTask) Models not found",
        message: "Please add or define models for \(newTask).",
        preferredStyle: .alert
      )
      alert.addAction(
        UIAlertAction(
          title: "OK", style: .cancel,
          handler: { _ in
            alert.dismiss(animated: true)
          }))
      self.present(alert, animated: true)

      if let oldIndex = tasks.firstIndex(where: { $0.name == currentTask }) {
        sender.selectedSegmentIndex = oldIndex
      }
      return
    }

    currentTask = newTask
    selectedIndexPath = nil
    
    // Notify external display of task change immediately
    NotificationCenter.default.post(
      name: .taskDidChange,
      object: nil,
      userInfo: ["task": newTask]
    )

    reloadModelEntriesAndLoadFirst(for: currentTask)

    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * 30 + 2)
    )
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  private func setupTableView() {
    modelTableView.delegate = self
    modelTableView.dataSource = self
    modelTableView.register(
      ModelTableViewCell.self, forCellReuseIdentifier: ModelTableViewCell.identifier)

    modelTableView.backgroundColor = .clear
    modelTableView.separatorStyle = .none
    modelTableView.isScrollEnabled = false

    tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
    tableViewBGView.layer.cornerRadius = 5  // 選択時の枠のcorner radiusに合わせる
    tableViewBGView.clipsToBounds = true

    view.addSubview(tableViewBGView)
    view.addSubview(modelTableView)

    modelTableView.translatesAutoresizingMaskIntoConstraints = false
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * 30 + 2)
    )
  }

  private func setupButtons() {
    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
    shareButton.setImage(
      UIImage(systemName: "square.and.arrow.up", withConfiguration: config), for: .normal)
    shareButton.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(shareButtonTapped)))
    view.addSubview(shareButton)

    recordButton.setImage(UIImage(systemName: "video", withConfiguration: config), for: .normal)
    recordButton.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(recordScreen)))
    view.addSubview(recordButton)
    
    // External display UI toggle button
    externalUIToggleButton.setImage(UIImage(systemName: "eye", withConfiguration: config), for: .normal)
    externalUIToggleButton.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(toggleExternalDisplayUI)))
    view.addSubview(externalUIToggleButton)
    
    // Initially hide the toggle button (only show when external display is connected)
    externalUIToggleButton.isHidden = true

    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    if view.bounds.width > view.bounds.height {
      shareButton.tintColor = .darkGray
      recordButton.tintColor = .darkGray
      externalUIToggleButton.tintColor = .darkGray
      let tableViewWidth = view.bounds.width * 0.2
      modelTableView.frame = CGRect(
        x: segmentedControl.frame.maxX + 20, y: 20, width: tableViewWidth, height: 200)
    } else {
      shareButton.tintColor = .systemGray
      recordButton.tintColor = .systemGray
      externalUIToggleButton.tintColor = .systemGray
      let tableViewWidth = view.bounds.width * 0.4
      modelTableView.frame = CGRect(
        x: view.bounds.width - tableViewWidth - 8,
        y: segmentedControl.frame.maxY + 25,
        width: tableViewWidth,
        height: 200)
    }

    shareButton.frame = CGRect(
      x: view.bounds.maxX - 49.5,
      y: view.bounds.maxY - 66,
      width: 49.5,
      height: 49.5
    )
    recordButton.frame = CGRect(
      x: shareButton.frame.minX - 49.5,
      y: view.bounds.maxY - 66,
      width: 49.5,
      height: 49.5
    )
    externalUIToggleButton.frame = CGRect(
      x: recordButton.frame.minX - 49.5,
      y: view.bounds.maxY - 66,
      width: 49.5,
      height: 49.5
    )

    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * 30 + 2)
    )
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] captured in
      guard let self = self else { return }
      if let image = captured {
        DispatchQueue.main.async {
          let activityViewController = UIActivityViewController(
            activityItems: [image], applicationActivities: nil
          )
          activityViewController.popoverPresentationController?.sourceView = self.View0
          self.present(activityViewController, animated: true, completion: nil)
        }
      } else {
        print("error capturing photo")
      }
    }
  }

  @objc func recordScreen() {
    let recorder = RPScreenRecorder.shared()
    recorder.isMicrophoneEnabled = true

    if !recorder.isRecording {
      AudioServicesPlaySystemSound(1117)
      recordButton.tintColor = .red
      recorder.startRecording { error in
        if let error = error {
          print("Screen recording start error: \(error)")
        } else {
          print("Started screen recording.")
        }
      }
    } else {
      AudioServicesPlaySystemSound(1118)
      if view.bounds.width > view.bounds.height {
        recordButton.tintColor = .darkGray
      } else {
        recordButton.tintColor = .systemGray
      }
      recorder.stopRecording { previewVC, error in
        if let error = error {
          print("Stop recording error: \(error)")
        }
        if let previewVC = previewVC {
          previewVC.previewControllerDelegate = self
          self.present(previewVC, animated: true, completion: nil)
        }
      }
    }
  }
  
  @objc func toggleExternalDisplayUI() {
    selection.selectionChanged()
    
    isExternalUIVisible.toggle()
    
    // Update button icon
    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
    let iconName = isExternalUIVisible ? "eye" : "eye.slash"
    externalUIToggleButton.setImage(UIImage(systemName: iconName, withConfiguration: config), for: .normal)
    
    // Send notification to external display
    NotificationCenter.default.post(
      name: .externalDisplayUIToggle,
      object: nil,
      userInfo: ["visible": isExternalUIVisible]
    )
    
    print("🟢 External display UI visibility: \(isExternalUIVisible)")
  }
  
  @objc func sliderValueChanged(_ sender: UISlider) {
    // Send threshold values to external display
    let conf = Double(round(100 * yoloView.sliderConf.value)) / 100
    let iou = Double(round(100 * yoloView.sliderIoU.value)) / 100
    let maxItems = Int(yoloView.sliderNumItems.value)
    
    NotificationCenter.default.post(
      name: .thresholdDidChange,
      object: nil,
      userInfo: [
        "conf": conf,
        "iou": iou,
        "maxItems": maxItems
      ]
    )
    
    print("📊 Threshold changed - Conf: \(conf), IoU: \(iou), Max items: \(maxItems)")
  }
  
  deinit {
    NotificationCenter.default.removeObserver(self)
  }
  
  private func debugCheckModelFolders() {
    print("\n🔍 DEBUG: Checking model folders...")
    let folders = ["DetectModels", "SegmentModels", "ClassifyModels", "PoseModels", "OBBModels"]
    
    for folder in folders {
      if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
        print("✅ \(folder) found at: \(folderURL.path)")
        
        do {
          let files = try FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
          let models = files.filter { $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage" }
          print("   📦 Models: \(models.map { $0.lastPathComponent })")
        } catch {
          print("   ❌ Error reading folder: \(error)")
        }
      } else {
        print("❌ \(folder) NOT FOUND in bundle")
      }
    }
    print("\n")
  }
}

// MARK: - UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return currentModels.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    return 30
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    // Get custom cell
    let cell =
      tableView.dequeueReusableCell(withIdentifier: ModelTableViewCell.identifier, for: indexPath)
      as! ModelTableViewCell
    let entry = currentModels[indexPath.row]

    // Check if the model is remote and not yet downloaded
    let isDownloaded =
      entry.isRemote ? ModelCacheManager.shared.isModelDownloaded(key: entry.identifier) : true

    // Format model name using the processString function
    let formattedName = processString(entry.displayName)

    // Configure the cell
    cell.configure(with: formattedName, isRemote: entry.isRemote, isDownloaded: isDownloaded)

    return cell
  }

  func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
    selection.selectionChanged()

    selectedIndexPath = indexPath
    let selectedEntry = currentModels[indexPath.row]

    loadModel(entry: selectedEntry, forTask: currentTask)
  }

  // This method is no longer needed as we adjust cell background layout in layoutSubviews

}

extension ViewController: RPPreviewViewControllerDelegate {
  func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
    previewController.dismiss(animated: true)
  }
}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
      self.labelFPS.textColor = .white
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async {
      // Share results with external display
      ExternalDisplayManager.shared.shareResults(result)
      
      // Also send via notification for direct communication
      NotificationCenter.default.post(
        name: .yoloResultsAvailable,
        object: nil,
        userInfo: ["result": result]
      )
    }
  }

}

// MARK: - External Display Support
extension ViewController {
  
  private func setupExternalDisplayNotifications() {
    // Listen for external display connection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayConnected(_:)),
      name: .externalDisplayConnected,
      object: nil
    )
    
    // Listen for external display disconnection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayDisconnected(_:)),
      name: .externalDisplayDisconnected,
      object: nil
    )
    
    // Listen for when external display is ready
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayReady(_:)),
      name: .externalDisplayReady,
      object: nil
    )
  }
  
  @objc private func handleExternalDisplayConnected(_ notification: Notification) {
    print("External display connected")
    
    // Stop camera but keep UI visible on iPhone when external display is connected
    DispatchQueue.main.async {
      // Stop the video capture to release camera for external display
      self.yoloView.stop()
      
      // Don't hide the YOLOView - just keep it visible with controls
      // The stop() method should have already stopped the camera feed
      
      self.showExternalDisplayStatus()
      
      // Show external display UI toggle button
      self.externalUIToggleButton.isHidden = false
      
      // Make sure sliders remain visible and functional
      self.yoloView.sliderConf.isHidden = false
      self.yoloView.labelSliderConf.isHidden = false
      self.yoloView.sliderIoU.isHidden = false
      self.yoloView.labelSliderIoU.isHidden = false
      self.yoloView.sliderNumItems.isHidden = false
      self.yoloView.labelSliderNumItems.isHidden = false
      
      // Also ensure buttons are visible
      self.yoloView.playButton.isHidden = false
      self.yoloView.pauseButton.isHidden = false
      self.yoloView.switchCameraButton.isHidden = false
      
      // Keep model table view visible for model selection
      self.modelTableView.isHidden = false
      self.tableViewBGView.isHidden = false
      
      print("🔴 Stopped main YOLOView camera but kept controls visible")
      
      // Force orientation update when external display connects
      if let windowScene = self.view.window?.windowScene {
        if #available(iOS 16.0, *) {
          windowScene.requestGeometryUpdate(.iOS(interfaceOrientations: [.landscapeLeft, .landscapeRight]))
        } else {
          UIViewController.attemptRotationToDeviceOrientation()
        }
      }
      
      // Wait a bit before sending model info to ensure external display is ready
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        // Send current model to external display
        if let entry = self.currentLoadingEntry ?? self.currentModels.first {
          self.notifyExternalDisplayOfCurrentModel()
        }
        
        // Send current threshold values
        self.sliderValueChanged(self.yoloView.sliderConf)
        
        // Send current task
        NotificationCenter.default.post(
          name: .taskDidChange,
          object: nil,
          userInfo: ["task": self.currentTask]
        )
      }
    }
  }
  
  private func notifyExternalDisplayOfCurrentModel() {
    // Get current model info and send to external display
    let yoloTask = convertTaskNameToYOLOTask(currentTask)
    
    var fullModelPath = currentModelName
    if let entry = currentLoadingEntry ?? currentModels.first(where: { processString($0.displayName) == currentModelName }),
       entry.isLocalBundle,
       let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
       let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) {
        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        fullModelPath = modelURL.path
    }
    
    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }
  
  @objc private func handleExternalDisplayDisconnected(_ notification: Notification) {
    print("External display disconnected")
    
    // Show and restart YOLOView on main display when external display is disconnected
    DispatchQueue.main.async {
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()
      
      // Hide external display UI toggle button
      self.externalUIToggleButton.isHidden = true
      
      // Show model table view again
      self.modelTableView.isHidden = false
      self.tableViewBGView.isHidden = false
      
      // Restart the main YOLOView video capture
      self.yoloView.resume()
      print("🟢 Restarted main YOLOView video capture")
      
      // Force orientation update when external display disconnects
      if let windowScene = self.view.window?.windowScene {
        if #available(iOS 16.0, *) {
          windowScene.requestGeometryUpdate(.iOS(interfaceOrientations: [.portrait, .landscapeLeft, .landscapeRight]))
          // Force immediate orientation update
          self.setNeedsUpdateOfSupportedInterfaceOrientations()
          
          // If currently in landscape, try to rotate to portrait
          if UIDevice.current.orientation.isLandscape {
            let value = UIInterfaceOrientation.portrait.rawValue
            UIDevice.current.setValue(value, forKey: "orientation")
          }
        } else {
          UIViewController.attemptRotationToDeviceOrientation()
          
          // For older iOS versions, force rotation
          if UIDevice.current.orientation.isLandscape {
            let value = UIInterfaceOrientation.portrait.rawValue
            UIDevice.current.setValue(value, forKey: "orientation")
          }
        }
      }
    }
  }
  
  @objc private func handleExternalDisplayReady(_ notification: Notification) {
    print("External display ready to receive content")
    print("  - currentTask: \(currentTask)")
    print("  - currentModelName: \(currentModelName)")
    print("  - currentModels count: \(currentModels.count)")
    
    // Only send model if we have one loaded
    guard !currentTask.isEmpty && !currentModels.isEmpty else {
        print("  - No task or models loaded yet, skipping notification")
        return
    }
    
    // Share current task/model info
    let yoloTask = convertTaskNameToYOLOTask(currentTask)
    
    // Get the full model path for external display
    var fullModelPath = ""
    
    // Use the first model if no specific model is selected yet
    if let currentEntry = currentModels.first(where: { processString($0.displayName) == currentModelName }) ?? currentModels.first {
        print("  - Found model entry: \(currentEntry.displayName)")
        
        if currentEntry.isLocalBundle,
           let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
           let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) {
            let modelURL = folderPathURL.appendingPathComponent(currentEntry.identifier)
            fullModelPath = modelURL.path
            print("  - Full model path: \(fullModelPath)")
        }
    } else {
        print("  - No model entry found!")
        return
    }
    
    // Only notify if we have a valid path
    guard !fullModelPath.isEmpty else {
        print("  - Empty model path, skipping notification")
        return
    }
    
    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }
  
  private func checkAndNotifyExternalDisplayIfReady() {
    // Check if external display is connected and ready
    let hasExternalDisplay = UIApplication.shared.connectedScenes
      .compactMap({ $0 as? UIWindowScene })
      .contains(where: { $0.screen != UIScreen.main })
    
    if hasExternalDisplay {
      print("External display detected, sending current model")
      // Wait a bit to ensure external display is fully initialized
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        self.handleExternalDisplayReady(Notification(name: .externalDisplayReady))
      }
    }
  }
  
  private func checkForExternalDisplays() {
    print("Checking for external displays...")
    print("Number of screens: \(UIScreen.screens.count)")
    
    if UIScreen.screens.count > 1 {
      print("External display detected!")
      for (index, screen) in UIScreen.screens.enumerated() {
        print("Screen \(index): \(screen)")
        print("  Bounds: \(screen.bounds)")
        print("  Scale: \(screen.scale)")
        print("  Available modes: \(screen.availableModes.count)")
        for mode in screen.availableModes {
          print("    Mode: \(mode.size)")
        }
      }
      
      // Check if external display scene is active
      if let windowScene = UIApplication.shared.connectedScenes
        .compactMap({ $0 as? UIWindowScene })
        .first(where: { $0.screen != UIScreen.main }) {
        print("External display window scene found: \(windowScene)")
      } else {
        print("No external display window scene found")
      }
    } else {
      print("No external display connected")
    }
  }
  
  private func showExternalDisplayStatus() {
    // Create a label to show external display status
    let statusLabel = UILabel()
    statusLabel.text = "📱 Camera is shown on external display\n🔄 Please use landscape orientation"
    statusLabel.textColor = .white
    statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.8)
    statusLabel.textAlignment = .center
    statusLabel.font = .systemFont(ofSize: 18, weight: .medium)
    statusLabel.numberOfLines = 0  // Allow unlimited lines
    statusLabel.adjustsFontSizeToFitWidth = true
    statusLabel.minimumScaleFactor = 0.8
    statusLabel.layer.cornerRadius = 10
    statusLabel.layer.masksToBounds = true
    statusLabel.tag = 9999 // Tag for later removal
    
    view.addSubview(statusLabel)
    
    statusLabel.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      statusLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
      statusLabel.widthAnchor.constraint(equalToConstant: 280),
      statusLabel.heightAnchor.constraint(equalToConstant: 140)
    ])
  }
  
  private func hideExternalDisplayStatus() {
    // Remove the status label
    view.subviews.first(where: { $0.tag == 9999 })?.removeFromSuperview()
  }
}
