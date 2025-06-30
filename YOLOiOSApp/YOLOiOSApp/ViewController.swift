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
import Photos
import PhotosUI
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
    let labelFrame = modelNameLabel.frame
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
class ViewController: UIViewController, YOLOViewDelegate, ModelDropdownViewDelegate, PHPickerViewControllerDelegate {

  var yoloView: YOLOView!
  
  // New UI Components
  private let statusMetricBar = StatusMetricBar()
  private let cameraPreviewContainer = UIView()
  private let taskTabStrip = TaskTabStrip()
  private let shutterBar = ShutterBar()
  private let rightSideToolBar = RightSideToolBar()
  private let parameterEditView = ParameterEditView()
  private let thresholdSlider = ThresholdSliderView()
  private let modelDropdown = ModelDropdownView()
  private let modelSizeFilterBar = ModelSizeFilterBar()
  
  // Photo library inference model cache
  private var photoInferenceModel: YOLO?
  private var photoInferenceModelKey: String?
  
  // Watermark
  private let watermarkImageView = UIImageView()
  
  // UI State
  private var isNewUIActive = true // Toggle for new/old UI
  private var currentThresholds: [String: Float] = [
    "confidence": 0.5,
    "iou": 0.5,
    "itemsMax": 30,
    "lineThickness": 3.0
  ]
  private var currentSizeFilter: ModelSizeFilterBar.ModelSize = .nano
  private var isSizeFilterShowing = false
  
  // Constraint management for orientation
  private var portraitConstraints: [NSLayoutConstraint] = []
  private var landscapeConstraints: [NSLayoutConstraint] = []
  private var commonConstraints: [NSLayoutConstraint] = []

  let selection = UISelectionFeedbackGenerator()
  var firstLoad = true

  private let downloadProgressView: UIProgressView = {
    let pv = UIProgressView(progressViewStyle: .default)
    pv.progress = 0.0
    pv.isHidden = true
    pv.progressTintColor = .ultralyticsLime
    pv.trackTintColor = UIColor.white.withAlphaComponent(0.3)
    return pv
  }()

  private let downloadProgressLabel: UILabel = {
    let label = UILabel()
    label.text = ""
    label.textAlignment = .center
    label.textColor = .white
    label.font = UIFont.systemFont(ofSize: 14, weight: .medium)
    label.isHidden = true
    return label
  }()

  private var loadingOverlayView: UIView?

  func showLoadingOverlay() {
    guard loadingOverlayView == nil else { return }
    
    // Create overlay
    let overlay = UIView(frame: view.bounds)
    overlay.backgroundColor = UIColor.black.withAlphaComponent(0.7)
    overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    
    // Create container for loading indicator
    let containerView = UIView()
    containerView.backgroundColor = UIColor.ultralyticsSurfaceDark.withAlphaComponent(0.95)
    containerView.layer.cornerRadius = 12
    containerView.translatesAutoresizingMaskIntoConstraints = false
    
    // Create activity indicator
    let loadingIndicator = UIActivityIndicatorView(style: .large)
    loadingIndicator.color = .ultralyticsLime
    loadingIndicator.translatesAutoresizingMaskIntoConstraints = false
    loadingIndicator.startAnimating()
    
    // Create loading label
    let loadingLabel = UILabel()
    loadingLabel.text = "Loading Model..."
    loadingLabel.textColor = .white
    loadingLabel.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    loadingLabel.translatesAutoresizingMaskIntoConstraints = false
    
    // Add views
    overlay.addSubview(containerView)
    containerView.addSubview(loadingIndicator)
    containerView.addSubview(loadingLabel)
    
    view.addSubview(overlay)
    loadingOverlayView = overlay
    
    // Layout
    NSLayoutConstraint.activate([
      containerView.centerXAnchor.constraint(equalTo: overlay.centerXAnchor),
      containerView.centerYAnchor.constraint(equalTo: overlay.centerYAnchor),
      containerView.widthAnchor.constraint(equalToConstant: 200),
      containerView.heightAnchor.constraint(equalToConstant: 120),
      
      loadingIndicator.centerXAnchor.constraint(equalTo: containerView.centerXAnchor),
      loadingIndicator.centerYAnchor.constraint(equalTo: containerView.centerYAnchor, constant: -15),
      
      loadingLabel.topAnchor.constraint(equalTo: loadingIndicator.bottomAnchor, constant: 12),
      loadingLabel.centerXAnchor.constraint(equalTo: containerView.centerXAnchor)
    ])
    
    // Bring progress views to front
    view.bringSubviewToFront(downloadProgressView)
    view.bringSubviewToFront(downloadProgressLabel)
    
    // Animate in
    overlay.alpha = 0
    UIView.animate(withDuration: 0.2) {
      overlay.alpha = 1
    }
    
    view.isUserInteractionEnabled = false
  }

  func hideLoadingOverlay() {
    guard let overlay = loadingOverlayView else { return }
    
    UIView.animate(withDuration: 0.2, animations: {
      overlay.alpha = 0
    }) { _ in
      overlay.removeFromSuperview()
      self.loadingOverlayView = nil
    }
    
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


  private var selectedIndexPath: IndexPath?

  override func viewDidLoad() {
    super.viewDidLoad()

    yoloView = YOLOView(frame: view.bounds, modelPathOrName: "", task: .detect)
      

    setupTaskSegmentedControl()
    loadModelsForAllTasks()

    if tasks.indices.contains(2) {
      // segmentedControl no longer exists in new UI
      currentTask = tasks[2].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    // Old UI setup methods - no longer needed with new UI
    // setupTableView()
    // setupButtons()

    yoloView.delegate = self
    yoloView.labelName.isHidden = true
    yoloView.labelFPS.isHidden = true
    
    // Hide all default YOLOView UI elements
    yoloView.toolbar.isHidden = true
    yoloView.sliderNumItems.isHidden = true
    yoloView.sliderConf.isHidden = true
    yoloView.sliderIoU.isHidden = true
    yoloView.labelSliderNumItems.isHidden = true
    yoloView.labelSliderConf.isHidden = true
    yoloView.labelSliderIoU.isHidden = true
    yoloView.labelZoom.isHidden = true
    yoloView.playButton.isHidden = true
    yoloView.pauseButton.isHidden = true
    yoloView.switchCameraButton.isHidden = true
    
    // Setup new UI if active
    if isNewUIActive {
      setupNewUI()
    }

    downloadProgressView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressView)

    downloadProgressLabel.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressLabel)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: 20),
      downloadProgressView.widthAnchor.constraint(equalToConstant: 200),
      downloadProgressView.heightAnchor.constraint(equalToConstant: 4),

      downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      downloadProgressLabel.topAnchor.constraint(
        equalTo: downloadProgressView.bottomAnchor, constant: 12),
    ])

    ModelDownloadManager.shared.progressHandler = { [weak self] progress in
      guard let self = self else { return }
      DispatchQueue.main.async {
        self.downloadProgressView.progress = Float(progress)
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false
        let percentage = Int(progress * 100)
        self.downloadProgressLabel.text = "Downloading \(percentage)%"
        
        // Update loading label if it exists
        if let overlay = self.loadingOverlayView,
           let container = overlay.subviews.first,
           let label = container.subviews.first(where: { $0 is UILabel && ($0 as? UILabel)?.text?.contains("Loading") == true }) as? UILabel {
          label.text = "Downloading Model..."
        }
      }
    }
  }
  
  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    
    // Load latest photo for thumbnail
    loadLatestPhotoThumbnail()
    
    // Add observer for when app becomes active
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(appDidBecomeActive),
      name: UIApplication.didBecomeActiveNotification,
      object: nil)
  }
  
  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    
    // Remove observer
    NotificationCenter.default.removeObserver(self, name: UIApplication.didBecomeActiveNotification, object: nil)
  }
  
  @objc private func appDidBecomeActive() {
    // Refresh thumbnail when app becomes active
    loadLatestPhotoThumbnail()
  }
  
  private func loadLatestPhotoThumbnail() {
    // Check photo library authorization
    let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    
    switch status {
    case .authorized, .limited:
      fetchLatestPhoto()
    case .notDetermined:
      PHPhotoLibrary.requestAuthorization(for: .readWrite) { [weak self] newStatus in
        if newStatus == .authorized || newStatus == .limited {
          DispatchQueue.main.async {
            self?.fetchLatestPhoto()
          }
        }
      }
    default:
      // No access, show default thumbnail
      break
    }
  }
  
  private func fetchLatestPhoto() {
    let fetchOptions = PHFetchOptions()
    fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    fetchOptions.fetchLimit = 1
    
    let fetchResult = PHAsset.fetchAssets(with: .image, options: fetchOptions)
    
    guard let latestAsset = fetchResult.firstObject else { return }
    
    let options = PHImageRequestOptions()
    options.version = .current
    options.deliveryMode = .opportunistic
    options.resizeMode = .exact
    
    let targetSize = CGSize(width: 96, height: 96) // 2x the button size for retina
    
    PHImageManager.default().requestImage(
      for: latestAsset,
      targetSize: targetSize,
      contentMode: .aspectFill,
      options: options) { [weak self] image, _ in
        if let image = image {
          DispatchQueue.main.async {
            self?.shutterBar.updateThumbnail(image)
          }
        }
    }
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)

    // Old UI text color enforcement - no longer needed

    // Override system appearance mode setting to ensure consistent styling
    view.overrideUserInterfaceStyle = .dark
  }

  // Called when trait collection changes (dark mode/light mode)
  override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
    super.traitCollectionDidChange(previousTraitCollection)

    // Old UI text color enforcement - no longer needed
  }
  
  // Called when orientation changes
  override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
    super.viewWillTransition(to: size, with: coordinator)
    
    // Only handle orientation for new UI
    guard isNewUIActive else { return }
    
    let isLandscape = size.width > size.height
    
    coordinator.animate(alongsideTransition: { _ in
      // Deactivate all orientation-specific constraints
      NSLayoutConstraint.deactivate(self.portraitConstraints)
      NSLayoutConstraint.deactivate(self.landscapeConstraints)
      
      // Activate appropriate constraints
      if isLandscape {
        NSLayoutConstraint.activate(self.landscapeConstraints)
      } else {
        NSLayoutConstraint.activate(self.portraitConstraints)
      }
      
      // Update ShutterBar layout
      self.shutterBar.updateLayoutForOrientation(isLandscape: isLandscape)
      
      // Update ModelDropdown layout
      self.modelDropdown.updateLayoutForOrientation(isLandscape: isLandscape)
      
      // Force layout update
      self.view.layoutIfNeeded()
      
      // Debug: Print current orientation
      print("Orientation changed to: \(isLandscape ? "Landscape" : "Portrait")")
    }, completion: nil)
  }
  

  private func setupTaskSegmentedControl() {
    // Old UI segmented control setup - no longer needed
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
      // Old UI table view code removed
      DispatchQueue.main.async {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.selectedIndexPath = firstIndex
        let firstModel = self.currentModels[0]
        self.loadModel(entry: firstModel, forTask: taskName)
      }
    } else {
      print("No models found for task: \(taskName)")
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

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.map { (modelName, url) -> ModelEntry in
      ModelEntry(
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
    // Always show loading overlay
    showLoadingOverlay()
    yoloView.setInferenceFlag(ok: false)
    
    if firstLoad {
      firstLoad = false
    }
    self.downloadProgressView.progress = 0.0
    self.downloadProgressView.isHidden = true
    self.downloadProgressLabel.isHidden = true
    self.view.isUserInteractionEnabled = false

    print("Start loading model: \(entry.displayName)")

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
            case .success(let loadResult):
              // Cache metadata for bundle models
              if let metadata = loadResult.metadata {
                ModelCacheManager.shared.cacheMetadata(for: entry.identifier, metadata: metadata)
              }
              
              self.finishLoadingModel(success: true, modelName: entry.displayName, metadata: loadResult.metadata)
            case .failure(let error):
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

  private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String)
  {
    let localModelURL = ModelCacheManager.shared.getDocumentsDirectory()
      .appendingPathComponent(key)
      .appendingPathExtension("mlmodelc")

    DispatchQueue.main.async {
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"
      self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) { result in
        switch result {
        case .success(let loadResult):
          self.finishLoadingModel(success: true, modelName: displayName, metadata: loadResult.metadata)
        case .failure(let error):
          self.finishLoadingModel(success: false, modelName: displayName)
        }
      }
    }
  }

  private func finishLoadingModel(success: Bool, modelName: String, metadata: [String: String]? = nil) {
    DispatchQueue.main.async {
      self.downloadProgressView.isHidden = true

      self.downloadProgressLabel.isHidden = true
      //            self.downloadProgressLabel.isHidden = false
      //            self.downloadProgressLabel.text = "Loading \(modelName)"

      self.view.isUserInteractionEnabled = true
      self.isLoadingModel = false
      // Always hide loading overlay
      self.hideLoadingOverlay()
      self.yoloView.setInferenceFlag(ok: true)

      if success {
        self.currentModelName = modelName
        DispatchQueue.main.async {
          // Old UI label update - no longer needed
          
          // Update new UI with metadata
          self.updateUIAfterModelLoad(success: true, modelName: modelName, metadata: metadata)
        }

        self.downloadProgressLabel.text = "Finished loading model \(modelName)"
        self.downloadProgressLabel.isHidden = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.downloadProgressLabel.isHidden = true
          self.downloadProgressLabel.text = ""
        }

      } else {
        // Failed to load model
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

    reloadModelEntriesAndLoadFirst(for: currentTask)

    // Old UI table view background update - removed
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }


  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    // Old UI layout code removed - now using Auto Layout constraints
    /*
      if view.bounds.width > view.bounds.height {
        shareButton.tintColor = .darkGray
        recordButton.tintColor = .darkGray
        let tableViewWidth = view.bounds.width * 0.2
        modelTableView.frame = CGRect(
          x: segmentedControl.frame.maxX + 20, y: 20, width: tableViewWidth, height: 200)
      } else {
        shareButton.tintColor = .systemGray
        recordButton.tintColor = .systemGray
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

      tableViewBGView.frame = CGRect(
        x: modelTableView.frame.minX - 1,
        y: modelTableView.frame.minY - 1,
        width: modelTableView.frame.width + 2,
        height: CGFloat(currentModels.count * 30 + 2)
      )
    */
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
          activityViewController.popoverPresentationController?.sourceView = self.view
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
      // recordButton.tintColor = .red
      recorder.startRecording { error in
        if let error = error {
          print("Screen recording start error: \(error)")
        } else {
          print("Started screen recording.")
        }
      }
    } else {
      AudioServicesPlaySystemSound(1118)
      // Old UI button color update - removed
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
    // Old UI FPS label update - no longer needed
    // FPS is now handled by statusMetricBar in new UI
    
    // Update new UI metrics
    if isNewUIActive {
      statusMetricBar.updateMetrics(fps: fps, latency: inferenceTime)
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async {
    }
  }

}

// MARK: - New UI Setup
extension ViewController {
  private func setupNewUI() {
    view.backgroundColor = .ultralyticsSurfaceDark
    
    // Camera Preview Container
    cameraPreviewContainer.backgroundColor = .black
    cameraPreviewContainer.layer.cornerRadius = 18
    cameraPreviewContainer.clipsToBounds = true
    
    // Add components to view (order matters for z-index)
    [cameraPreviewContainer, taskTabStrip, shutterBar, rightSideToolBar, thresholdSlider].forEach {
      view.addSubview($0)
      $0.translatesAutoresizingMaskIntoConstraints = false
    }
    
    // Add model dropdown before status bar
    view.addSubview(modelDropdown)
    modelDropdown.translatesAutoresizingMaskIntoConstraints = false
    
    // Add model size filter bar
    view.addSubview(modelSizeFilterBar)
    modelSizeFilterBar.translatesAutoresizingMaskIntoConstraints = false
    modelSizeFilterBar.alpha = 0 // Start hidden
    
    // Add status bar last so it stays on top
    view.addSubview(statusMetricBar)
    statusMetricBar.translatesAutoresizingMaskIntoConstraints = false
    
    // Move YOLOView to camera preview container
    if let yoloView = yoloView {
      yoloView.removeFromSuperview()
      cameraPreviewContainer.addSubview(yoloView)
      yoloView.translatesAutoresizingMaskIntoConstraints = false
      NSLayoutConstraint.activate([
        yoloView.topAnchor.constraint(equalTo: cameraPreviewContainer.topAnchor),
        yoloView.leadingAnchor.constraint(equalTo: cameraPreviewContainer.leadingAnchor),
        yoloView.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor),
        yoloView.bottomAnchor.constraint(equalTo: cameraPreviewContainer.bottomAnchor)
      ])
      
      // Ensure user interaction is enabled
      cameraPreviewContainer.isUserInteractionEnabled = true
      yoloView.isUserInteractionEnabled = true
    }
    
    // Add watermark AFTER YOLOView so it appears on top
    setupWatermark()
    
    setupNewUIConstraints()
    setupNewUIActions()
    
    // Load models for the new UI
    loadModelsForAllTasks()
    
    // Initial task setup - this will trigger model loading
    taskTabStrip.selectedTask = .detect
    handleTaskChange(to: .detect)
    
    // Listen for hidden info notification
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(showHiddenInfo),
      name: .showHiddenInfo,
      object: nil
    )
  }
  
  private func setupNewUIConstraints() {
    // Common constraints (always active regardless of orientation)
    commonConstraints = [
      // Status Bar
      statusMetricBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: -8),
      statusMetricBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      
      // Camera Preview (top and leading are common)
      cameraPreviewContainer.topAnchor.constraint(equalTo: statusMetricBar.bottomAnchor, constant: 2),
      cameraPreviewContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 6),
      
      // Threshold Slider (overlay)
      thresholdSlider.topAnchor.constraint(equalTo: view.topAnchor),
      thresholdSlider.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      thresholdSlider.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      thresholdSlider.bottomAnchor.constraint(equalTo: view.bottomAnchor),
      
      // Model Dropdown
      modelDropdown.topAnchor.constraint(equalTo: view.topAnchor),
      modelDropdown.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      modelDropdown.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      modelDropdown.bottomAnchor.constraint(equalTo: view.bottomAnchor),
      
      // Model Size Filter Bar
      modelSizeFilterBar.topAnchor.constraint(equalTo: statusMetricBar.bottomAnchor),
      modelSizeFilterBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      modelSizeFilterBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      modelSizeFilterBar.heightAnchor.constraint(equalToConstant: 36)
    ]
    
    // Portrait-specific constraints
    portraitConstraints = [
      // Status Bar
      statusMetricBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      
      // Shutter Bar at bottom
      shutterBar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
      shutterBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      shutterBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      shutterBar.heightAnchor.constraint(equalToConstant: 96),
      
      // Task Tab Strip above shutter bar
      taskTabStrip.bottomAnchor.constraint(equalTo: shutterBar.topAnchor),
      taskTabStrip.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      taskTabStrip.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      
      // Camera Preview
      cameraPreviewContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -6),
      cameraPreviewContainer.bottomAnchor.constraint(equalTo: taskTabStrip.topAnchor, constant: -2),
      
      // Right Tool Bar
      rightSideToolBar.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor, constant: -12),
      rightSideToolBar.bottomAnchor.constraint(equalTo: cameraPreviewContainer.bottomAnchor, constant: -20)
    ]
    
    // Landscape-specific constraints
    landscapeConstraints = [
      // Status Bar - make room for shutter bar on right
      statusMetricBar.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -96),
      
      // Shutter Bar on right side
      shutterBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      shutterBar.topAnchor.constraint(equalTo: view.topAnchor),
      shutterBar.bottomAnchor.constraint(equalTo: view.bottomAnchor),
      shutterBar.widthAnchor.constraint(equalToConstant: 96),
      
      // Task Tab Strip at bottom with margin for shutter bar
      taskTabStrip.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
      taskTabStrip.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      taskTabStrip.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -96),
      
      // Camera Preview adjusted for shutter bar
      cameraPreviewContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -102),
      cameraPreviewContainer.bottomAnchor.constraint(equalTo: taskTabStrip.topAnchor, constant: -2),
      
      // Right Tool Bar adjusted position
      rightSideToolBar.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor, constant: -12),
      rightSideToolBar.bottomAnchor.constraint(equalTo: cameraPreviewContainer.bottomAnchor, constant: -20)
    ]
    
    // Activate common constraints
    NSLayoutConstraint.activate(commonConstraints)
    
    // Activate portrait constraints by default
    NSLayoutConstraint.activate(portraitConstraints)
  }
  
  private func setupNewUIActions() {
    print("ViewController: setupNewUIActions called")
    
    // Status bar actions
    statusMetricBar.onModelTap = { [weak self] in
      print("ViewController: onModelTap closure called")
      self?.showModelSelector()
    }
    
    statusMetricBar.onSizeTap = { [weak self] in
      print("ViewController: onSizeTap closure called")
      self?.toggleSizeFilter()
    }
    
    print("ViewController: onModelTap and onSizeTap closures set")
    
    // Model dropdown delegate
    modelDropdown.delegate = self
    
    // Task tab actions
    taskTabStrip.onTaskChange = { [weak self] task in
      self?.handleTaskChange(to: task)
    }
    
    // Shutter bar actions
    shutterBar.onShutterTap = { [weak self] in
      self?.capturePhoto()
    }
    
    shutterBar.onShutterLongPress = { [weak self] in
      self?.toggleRecording()
    }
    
    shutterBar.onFlipCamera = { [weak self] in
      self?.flipCamera()
    }
    
    shutterBar.onThumbnailTap = { [weak self] in
      self?.showLastCapture()
    }
    
    // Right toolbar actions
    rightSideToolBar.onZoomChanged = { [weak self] zoomLevel in
      self?.handleZoomChange(to: zoomLevel)
    }
    
    rightSideToolBar.onToolSelected = { [weak self] tool in
      self?.handleParameterTool(tool)
    }
    
    // Threshold slider actions
    thresholdSlider.onValueChange = { [weak self] value in
      self?.handleSliderValueChange(value)
    }
    
    thresholdSlider.onHide = { [weak self] in
      // Show task tab strip when slider hides
      UIView.animate(withDuration: 0.2) {
        self?.taskTabStrip.alpha = 1
      }
    }
    
    // Model size filter actions
    modelSizeFilterBar.onSizeSelected = { [weak self] size in
      self?.handleSizeFilterChange(to: size)
    }
  }
  
  
  // MARK: - New UI Actions
  
  private func showModelSelector() {
    print("showModelSelector called")
    print("Current models count: \(currentModels.count)")
    print("Current model name: \(currentModelName)")
    print("isNewUIActive: \(isNewUIActive)")
    
    // Debug: Check if models are loaded
    if currentModels.isEmpty {
      print("WARNING: No models loaded!")
      return
    }
    
    // Ensure dropdown is on top
    view.bringSubviewToFront(modelDropdown)
    
    // Find the current model's identifier
    let currentModelIdentifier = currentModels.first { model in
      model.displayName == currentModelName
    }?.identifier
    
    print("Current model identifier: \(currentModelIdentifier ?? "nil")")
    print("Model dropdown frame: \(modelDropdown.frame)")
    print("Model dropdown superview: \(modelDropdown.superview != nil)")
    
    // Filter models based on current size filter
    let filteredModels = currentModels.filter { model in
      // Always show custom models
      if model.modelVersion == "Custom" {
        return true
      }
      // Show models matching the current size filter
      return model.modelSize == currentSizeFilter.rawValue
    }
    
    // Configure and toggle dropdown with filtered models
    modelDropdown.configure(with: filteredModels, currentModel: currentModelIdentifier)
    modelDropdown.toggle()
    
    // Ensure status bar stays on top
    view.bringSubviewToFront(statusMetricBar)
  }
  
  private func handleTaskChange(to task: TaskTabStrip.Task) {
    // Hide dropdown if it's showing
    if modelDropdown.isShowing {
      modelDropdown.hide()
    }
    
    // Clear cached photo inference model when switching tasks
    photoInferenceModel = nil
    photoInferenceModelKey = nil
    
    let taskName: String
    switch task {
    case .detect:
      taskName = "Detect"
    case .segment:
      taskName = "Segment"
    case .classify:
      taskName = "Classify"
    case .pose:
      taskName = "Pose"
    case .obb:
      taskName = "OBB"
    }
    
    currentTask = taskName
    reloadModelEntriesAndLoadFirst(for: taskName)
  }
  
  private func toggleSizeFilter() {
    if isSizeFilterShowing {
      // Hide the size filter
      modelSizeFilterBar.hide { [weak self] in
        self?.isSizeFilterShowing = false
      }
    } else {
      // Show the size filter
      isSizeFilterShowing = true
      modelSizeFilterBar.show()
      
      // Hide model dropdown if it's showing
      if modelDropdown.isShowing {
        modelDropdown.hide()
      }
    }
  }
  
  private func handleSizeFilterChange(to size: ModelSizeFilterBar.ModelSize) {
    currentSizeFilter = size
    
    // Update status bar to reflect new size
    let sizeString = size.displayName
    statusMetricBar.updateModel(name: extractModelVersion(from: currentModelName), size: sizeString)
    
    // Hide size filter
    modelSizeFilterBar.hide { [weak self] in
      self?.isSizeFilterShowing = false
    }
    
    // Find and load a model matching the new size
    loadModelForCurrentSizeFilter()
  }
  
  private func loadModelForCurrentSizeFilter() {
    // Get current model version
    let currentVersion = extractModelVersion(from: currentModelName)
    
    // Find a model matching the current version and new size
    if let matchingModel = currentModels.first(where: { model in
      let modelVersion = model.modelVersion
      let modelSize = model.modelSize
      return modelVersion == currentVersion && modelSize == currentSizeFilter.rawValue
    }) {
      // Load the matching model
      loadModel(entry: matchingModel, forTask: currentTask)
    } else {
      // No exact match found - try to find any model with the selected size
      if let sizeMatchingModel = currentModels.first(where: { model in
        return model.modelSize == currentSizeFilter.rawValue
      }) {
        loadModel(entry: sizeMatchingModel, forTask: currentTask)
      }
    }
  }
  
  private func extractModelVersion(from modelName: String) -> String {
    let name = modelName.lowercased()
    if name.contains("yolo11") {
      return "YOLO11"
    } else if name.contains("yolov8") {
      return "YOLOv8"
    } else if name.contains("yolov5") {
      return "YOLOv5"
    } else {
      return "Custom"
    }
  }
  
  private func capturePhoto() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] captured in
      guard let self = self else { return }
      if let image = captured {
        // Update thumbnail
        self.shutterBar.updateThumbnail(image)
        
        // Save to photo library
        UIImageWriteToSavedPhotosAlbum(image, self, #selector(self.image(_:didFinishSavingWithError:contextInfo:)), nil)
      } else {
        print("error capturing photo")
      }
    }
  }
  
  @objc private func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
    DispatchQueue.main.async {
      if let error = error {
        // Show error alert
        let alert = UIAlertController(title: "Save Error", message: error.localizedDescription, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        self.present(alert, animated: true)
      } else {
        // Show success feedback
        let feedback = UINotificationFeedbackGenerator()
        feedback.notificationOccurred(.success)
        
        // Don't play sound - AVCapturePhotoOutput already plays shutter sound
        // AudioServicesPlaySystemSound(1108)
        
        // Update thumbnail with newly saved image
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
          self?.loadLatestPhotoThumbnail()
        }
        
        // Optionally show a brief success message
        let successView = UIView(frame: CGRect(x: 0, y: 0, width: 200, height: 50))
        successView.backgroundColor = UIColor.black.withAlphaComponent(0.8)
        successView.layer.cornerRadius = 25
        successView.center = self.view.center
        
        let label = UILabel(frame: successView.bounds)
        label.text = "Saved to Photos"
        label.textColor = .white
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        successView.addSubview(label)
        
        self.view.addSubview(successView)
        successView.alpha = 0
        
        UIView.animate(withDuration: 0.3, animations: {
          successView.alpha = 1
        }) { _ in
          UIView.animate(withDuration: 0.3, delay: 1.0, options: [], animations: {
            successView.alpha = 0
          }) { _ in
            successView.removeFromSuperview()
          }
        }
      }
    }
  }
  
  private func toggleRecording() {
    let recorder = RPScreenRecorder.shared()
    recorder.isMicrophoneEnabled = true
    
    if !recorder.isRecording {
      AudioServicesPlaySystemSound(1117)
      shutterBar.setRecording(true)
      recorder.startRecording { error in
        if let error = error {
          print("Screen recording start error: \(error)")
        } else {
          print("Started screen recording.")
        }
      }
    } else {
      AudioServicesPlaySystemSound(1118)
      shutterBar.setRecording(false)
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
  
  private func flipCamera() {
    // Use YOLOView's switch camera button tap action
    yoloView.switchCameraButton.sendActions(for: .touchUpInside)
  }
  
  private func showLastCapture() {
    // Show photo picker
    showPhotoPicker()
  }
  
  private func showPhotoPicker() {
    var configuration = PHPickerConfiguration()
    configuration.selectionLimit = 1
    configuration.filter = .images
    
    let picker = PHPickerViewController(configuration: configuration)
    picker.delegate = self
    present(picker, animated: true)
  }
  
  // MARK: - PHPickerViewControllerDelegate
  
  func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
    picker.dismiss(animated: true)
    
    guard let result = results.first else { return }
    
    result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
      if let image = object as? UIImage {
        DispatchQueue.main.async {
          self?.processSelectedImage(image)
        }
      }
    }
  }
  
  private func processSelectedImage(_ image: UIImage) {
    // Fix image orientation if needed
    let orientedImage = image.fixedOrientation()
    
    // Show loading indicator
    let loadingView = UIView(frame: view.bounds)
    loadingView.backgroundColor = UIColor.black.withAlphaComponent(0.5)
    let spinner = UIActivityIndicatorView(style: .large)
    spinner.color = .white
    spinner.center = loadingView.center
    spinner.startAnimating()
    loadingView.addSubview(spinner)
    view.addSubview(loadingView)
    
    // Get current model info
    guard let currentModel = currentModels.first(where: { $0.displayName == currentModelName }) else {
      loadingView.removeFromSuperview()
      showResultPopup(image: orientedImage)
      return
    }
    
    // Create a unique key for this model and task combination
    let modelKey = "\(currentTask)_\(currentModel.identifier)"
    
    // Check if we already have a cached model for this combination
    if let cachedModel = photoInferenceModel, photoInferenceModelKey == modelKey {
      // Use cached model for faster inference
      print("Using cached model for inference")
      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        let result = cachedModel(orientedImage)
        
        DispatchQueue.main.async {
          loadingView.removeFromSuperview()
          
          // Debug: Print inference results
          print("YOLO Inference Results (cached model):")
          print("- Number of detections: \(result.boxes.count)")
          print("- Annotated image exists: \(result.annotatedImage != nil)")
          
          if result.boxes.isEmpty {
            print("No objects detected in the image")
          } else {
            for (index, box) in result.boxes.prefix(5).enumerated() {
              print("  Box \(index): \(box.cls) (conf: \(String(format: "%.2f", box.conf)))")
            }
          }
          
          // Check if we have an annotated image
          if let annotatedImage = result.annotatedImage {
            self?.showResultPopup(image: annotatedImage)
          } else {
            // If no annotated image, show original
            print("Warning: No annotated image generated")
            self?.showResultPopup(image: orientedImage)
          }
        }
      }
      return
    }
    
    // No cached model, need to load it
    print("Loading new model for inference")
    
    // Get model path
    let modelPath: String
    if currentModel.isLocalBundle {
      // For local bundle models, construct the full path like YOLOView expects
      guard let taskInfo = tasks.first(where: { $0.name == currentTask }),
            let folderURL = Bundle.main.url(forResource: taskInfo.folder, withExtension: nil) else {
        print("Error: Could not find task folder for \(currentTask)")
        loadingView.removeFromSuperview()
        showResultPopup(image: orientedImage)
        return
      }
      let modelURL = folderURL.appendingPathComponent(currentModel.identifier)
      modelPath = modelURL.path
    } else {
      // For remote models, provide the full path to the downloaded model
      modelPath = getDocumentsDirectory().appendingPathComponent(currentModel.identifier)
                                        .appendingPathExtension("mlmodelc").path
    }
    
    // Determine task type
    let taskType: YOLOTask
    switch currentTask {
    case "Classify":
      taskType = .classify
    case "Segment":
      taskType = .segment
    case "Pose":
      taskType = .pose
    case "OBB":
      taskType = .obb
    default:
      taskType = .detect
    }
    
    // Debug: Print model path
    print("Model path: \(modelPath)")
    print("File exists: \(FileManager.default.fileExists(atPath: modelPath))")
    
    // Create YOLO model and run inference
    let yolo = YOLO(modelPath, task: taskType) { [weak self] result in
      DispatchQueue.main.async {
        switch result {
        case .success(let model):
          // Cache the model for future use
          self?.photoInferenceModel = model
          self?.photoInferenceModelKey = modelKey
          print("Model cached for future use")
          
          // Run inference on the image
          DispatchQueue.global(qos: .userInitiated).async {
            let result = model(orientedImage)
            
            DispatchQueue.main.async {
              loadingView.removeFromSuperview()
              
              // Debug: Print inference results
              print("YOLO Inference Results:")
              print("- Number of detections: \(result.boxes.count)")
              print("- Task type: \(taskType)")
              print("- Annotated image exists: \(result.annotatedImage != nil)")
              
              if result.boxes.isEmpty {
                print("No objects detected in the image")
              } else {
                for (index, box) in result.boxes.prefix(5).enumerated() {
                  print("  Box \(index): \(box.cls) (conf: \(String(format: "%.2f", box.conf)))")
                }
              }
              
              // Check if we have an annotated image
              if let annotatedImage = result.annotatedImage {
                self?.showResultPopup(image: annotatedImage)
              } else {
                // If no annotated image, show original
                print("Warning: No annotated image generated")
                self?.showResultPopup(image: orientedImage)
              }
            }
          }
          
        case .failure(let error):
          loadingView.removeFromSuperview()
          print("Error loading model: \(error)")
          self?.showResultPopup(image: orientedImage)
        }
      }
    }
  }
  
  private func showResultPopup(image: UIImage) {
    // Create popup view controller
    let popupVC = UIViewController()
    let imageView = UIImageView(image: image)
    imageView.contentMode = .scaleAspectFit
    imageView.translatesAutoresizingMaskIntoConstraints = false
    
    popupVC.view.backgroundColor = UIColor.black.withAlphaComponent(0.9)
    popupVC.view.addSubview(imageView)
    
    // Add close button
    let closeButton = UIButton(type: .system)
    closeButton.setImage(UIImage(systemName: "xmark.circle.fill"), for: .normal)
    closeButton.tintColor = .white
    closeButton.translatesAutoresizingMaskIntoConstraints = false
    closeButton.addAction(UIAction { _ in
      popupVC.dismiss(animated: true)
    }, for: .touchUpInside)
    popupVC.view.addSubview(closeButton)
    
    // Add save button
    let saveButton = UIButton(type: .system)
    saveButton.setImage(UIImage(systemName: "square.and.arrow.down"), for: .normal)
    saveButton.tintColor = .white
    saveButton.translatesAutoresizingMaskIntoConstraints = false
    saveButton.addAction(UIAction { [weak self] _ in
      self?.saveImageToPhotos(image)
    }, for: .touchUpInside)
    popupVC.view.addSubview(saveButton)
    
    NSLayoutConstraint.activate([
      imageView.leadingAnchor.constraint(equalTo: popupVC.view.leadingAnchor, constant: 20),
      imageView.trailingAnchor.constraint(equalTo: popupVC.view.trailingAnchor, constant: -20),
      imageView.topAnchor.constraint(equalTo: popupVC.view.safeAreaLayoutGuide.topAnchor, constant: 60),
      imageView.bottomAnchor.constraint(equalTo: popupVC.view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
      
      closeButton.topAnchor.constraint(equalTo: popupVC.view.safeAreaLayoutGuide.topAnchor, constant: 20),
      closeButton.trailingAnchor.constraint(equalTo: popupVC.view.trailingAnchor, constant: -20),
      closeButton.widthAnchor.constraint(equalToConstant: 40),
      closeButton.heightAnchor.constraint(equalToConstant: 40),
      
      saveButton.topAnchor.constraint(equalTo: popupVC.view.safeAreaLayoutGuide.topAnchor, constant: 20),
      saveButton.leadingAnchor.constraint(equalTo: popupVC.view.leadingAnchor, constant: 20),
      saveButton.widthAnchor.constraint(equalToConstant: 40),
      saveButton.heightAnchor.constraint(equalToConstant: 40)
    ])
    
    popupVC.modalPresentationStyle = .fullScreen
    popupVC.modalTransitionStyle = .crossDissolve
    present(popupVC, animated: true)
  }
  
  private func saveImageToPhotos(_ image: UIImage) {
    UIImageWriteToSavedPhotosAlbum(image, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
  }
  
  
  private func handleZoomChange(to zoomLevel: Float) {
    // Apply zoom using the new public method
    yoloView?.setZoomLevel(zoomLevel)
    
    // Get the actual applied zoom level (may be clamped to device limits)
    if let actualZoom = yoloView?.getZoomLevel() {
      rightSideToolBar.updateZoomLevel(actualZoom)
    }
  }
  
  private func handleParameterTool(_ tool: RightSideToolBar.Tool) {
    // Hide task tab strip when showing slider
    UIView.animate(withDuration: 0.2) {
      self.taskTabStrip.alpha = 0
    }
    
    switch tool {
    case .itemsMax:
      let current = Int(currentThresholds["itemsMax"] ?? 30)
      thresholdSlider.showParameter(.itemsMax(current))
    case .confidence:
      let current = currentThresholds["confidence"] ?? 0.5
      thresholdSlider.showParameter(.confidence(current))
    case .iou:
      let current = currentThresholds["iou"] ?? 0.5
      thresholdSlider.showParameter(.iou(current))
    case .lineThickness:
      let current = currentThresholds["lineThickness"] ?? 3.0
      thresholdSlider.showParameter(.lineThickness(current))
    default:
      break
    }
    
    // Deactivate the toolbar after selection
    rightSideToolBar.deactivateAll()
  }
  
  private func getDocumentsDirectory() -> URL {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }
  
  private func handleSliderValueChange(_ normalizedValue: Float) {
    guard let parameter = thresholdSlider.parameter else { return }
    
    // Convert normalized value (0-1) to parameter range
    let range = parameter.range
    let actualValue = range.lowerBound + normalizedValue * (range.upperBound - range.lowerBound)
    
    switch parameter {
    case .itemsMax:
      let intValue = Int(actualValue)
      currentThresholds["itemsMax"] = Float(intValue)
      yoloView.sliderNumItems.value = Float(intValue)
      yoloView.sliderNumItems.sendActions(for: .valueChanged)
    case .confidence:
      currentThresholds["confidence"] = actualValue
      yoloView.sliderConf.value = actualValue
      yoloView.sliderConf.sendActions(for: .valueChanged)
    case .iou:
      currentThresholds["iou"] = actualValue
      yoloView.sliderIoU.value = actualValue
      yoloView.sliderIoU.sendActions(for: .valueChanged)
    case .lineThickness:
      currentThresholds["lineThickness"] = actualValue
      // Apply line thickness to YOLOView
      yoloView.setLineWidth(actualValue)
      print("Line thickness set to: \(actualValue)")
    }
  }
  
  private func handleParameterChange(_ parameter: ParameterEditView.Parameter) {
    switch parameter {
    case .itemsMax(let value):
      currentThresholds["itemsMax"] = Float(value)
      // Update via slider
      yoloView.sliderNumItems.value = Float(value)
      yoloView.sliderNumItems.sendActions(for: .valueChanged)
    case .confidence(let value):
      currentThresholds["confidence"] = value
      // Update via slider
      yoloView.sliderConf.value = value
      yoloView.sliderConf.sendActions(for: .valueChanged)
    case .iou(let value):
      currentThresholds["iou"] = value
      // Update via slider
      yoloView.sliderIoU.value = value
      yoloView.sliderIoU.sendActions(for: .valueChanged)
    case .lineThickness(let value):
      currentThresholds["lineThickness"] = value
      // Apply line thickness to YOLOView
      yoloView.setLineWidth(value)
      print("Line thickness set to: \(value)")
    }
    
    // Save to UserDefaults
    UserDefaults.standard.set(currentThresholds, forKey: "thresholds")
  }
  
  @objc private func showHiddenInfo() {
    let hiddenInfoVC = HiddenInfoViewController()
    hiddenInfoVC.modalPresentationStyle = .pageSheet
    present(hiddenInfoVC, animated: true)
  }
  
  private func setupWatermark() {
    watermarkImageView.image = UIImage(named: "ultralytics_yolo_white")
    watermarkImageView.contentMode = .scaleAspectFit
    watermarkImageView.alpha = 0.4
    watermarkImageView.isUserInteractionEnabled = false  // Allow gestures to pass through
    watermarkImageView.translatesAutoresizingMaskIntoConstraints = false
    
    // Add to camera preview container, above YOLOView
    cameraPreviewContainer.addSubview(watermarkImageView)
    
    // Constraints to center it
    NSLayoutConstraint.activate([
      watermarkImageView.centerXAnchor.constraint(equalTo: cameraPreviewContainer.centerXAnchor),
      watermarkImageView.centerYAnchor.constraint(equalTo: cameraPreviewContainer.centerYAnchor),
      watermarkImageView.widthAnchor.constraint(equalTo: cameraPreviewContainer.widthAnchor, multiplier: 0.7),
      watermarkImageView.heightAnchor.constraint(equalTo: cameraPreviewContainer.heightAnchor, multiplier: 0.35)
    ])
  }
  
  private func updateUIAfterModelLoad(success: Bool, modelName: String, metadata: [String: String]? = nil) {
    
    if success && isNewUIActive {
      // Don't update status bar here - wait until we determine the actual size
      
      // Update current size filter based on loaded model
      if let loadedModel = currentModels.first(where: { $0.displayName == modelName }) {
        if let modelSizeRaw = loadedModel.modelSize,
           let size = ModelSizeFilterBar.ModelSize(rawValue: modelSizeRaw) {
          // Standard model with known size
          currentSizeFilter = size
          modelSizeFilterBar.setSelectedSize(size, animated: false)
        } else {
          // For custom models, first check the metadata from model loading
          var detectedSize: ModelSizeFilterBar.ModelSize? = nil
          
          // Try to extract size from metadata
          if let metadata = metadata {
            detectedSize = ModelMetadataHelper.extractModelSizeFromMetadata(metadata)
          }
          
          if let size = detectedSize {
            currentSizeFilter = size
            modelSizeFilterBar.setSelectedSize(size, animated: false)
          } else if let cachedSize = ModelCacheManager.shared.getCachedModelSize(for: loadedModel.identifier),
             let size = ModelSizeFilterBar.ModelSize(rawValue: cachedSize) {
            // Fallback to cached metadata
            currentSizeFilter = size
            modelSizeFilterBar.setSelectedSize(size, animated: false)
          } else {
            // Default to nano if metadata doesn't provide size info
            currentSizeFilter = .nano
            modelSizeFilterBar.setSelectedSize(.nano, animated: false)
          }
        }
        
        // Update status bar with the determined size
        let sizeString = currentSizeFilter.displayName
        statusMetricBar.updateModel(name: processString(modelName), size: sizeString)
      }
    }
  }
}

// MARK: - ModelDropdownViewDelegate
extension ViewController {
  func modelDropdown(_ dropdown: ModelDropdownView, didSelectModel model: ModelEntry) {
    // Clear cached photo inference model when switching models
    photoInferenceModel = nil
    photoInferenceModelKey = nil
    loadModel(entry: model, forTask: currentTask)
  }
  
  func modelDropdownDidDismiss(_ dropdown: ModelDropdownView) {
    // Handle dropdown dismissal if needed
  }
  
  func modelDropdownDidRequestCustomModelGuide(_ dropdown: ModelDropdownView) {
    let guideVC = CustomModelGuideViewController()
    guideVC.modalPresentationStyle = .fullScreen
    guideVC.modalTransitionStyle = .crossDissolve
    present(guideVC, animated: true)
  }
}

// MARK: - UIImage Extension for Orientation Fix
extension UIImage {
  func fixedOrientation() -> UIImage {
    // If image orientation is already correct, return as is
    if imageOrientation == .up {
      return self
    }
    
    // We need to redraw the image in the correct orientation
    UIGraphicsBeginImageContextWithOptions(size, false, scale)
    draw(in: CGRect(origin: .zero, size: size))
    let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    
    return normalizedImage ?? self
  }
}
