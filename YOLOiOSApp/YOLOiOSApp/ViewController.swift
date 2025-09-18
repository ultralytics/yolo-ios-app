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

// MARK: - Model Selection Helpers

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  @IBOutlet weak var yoloView: YOLOView!, View0: UIView!, segmentedControl: UISegmentedControl!, 
    labelName: UILabel!, labelFPS: UILabel!, labelVersion: UILabel!,
                     activityIndicator: UIActivityIndicatorView!, logoImage: UIImageView!, modelSegmentedControl: UISegmentedControl!

  let selection = UISelectionFeedbackGenerator()

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()

  private var loadingOverlayView: UIView?

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

  private let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
    ("Classify", "ClassifyModels", .classify),
    ("Segment", "SegmentModels", .segment),
    ("Detect", "DetectModels", .detect),
    ("Pose", "PoseModels", .pose),
    ("OBB", "OBBModels", .obb),
  ]

  private var modelsForTask: [String: [String]] = [:]

  private var currentModels: [(name: String, url: URL?, isLocal: Bool)] = []
  private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] = [:]
  private var customModels: [ModelSelectionManager.ModelInfo] = []

  private var currentTask: String = ""
  private var currentModelName: String = ""

  private var isLoadingModel = false

  // MARK: - Constants
  private struct Constants {
    static let defaultTaskIndex = 2  // Detect
    static let logoURL = "https://www.ultralytics.com"
    static let progressViewWidth: CGFloat = 200
  }


  override func viewDidLoad() {
    super.viewDidLoad()

    // Setup segmented control and load models
    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    setupModelSegmentedControl()

    // Setup gestures and delegates
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    [yoloView.labelName, yoloView.labelFPS].forEach { $0?.isHidden = true }

    // Setup labels and version
    [labelName, labelFPS, labelVersion].forEach {
      $0?.textColor = .white
      $0?.overrideUserInterfaceStyle = .dark
    }
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    {
      labelVersion.text = "v\(version) (\(build))"
    }

    // Setup progress views
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
    let categorized = ModelSelectionManager.categorizeModels(from: currentModels)
    standardModels = categorized.standard
    customModels = categorized.custom

    let yoloTask = tasks.first(where: { $0.name == taskName })?.yoloTask ?? .detect
    ModelSelectionManager.setupSegmentedControl(modelSegmentedControl, standardModels: standardModels, hasCustomModels: !customModels.isEmpty, currentTask: yoloTask)

    if let firstSize = ModelSelectionManager.ModelSize.allCases.first,
       let model = standardModels[firstSize] {
      loadModel(entry: (model.name, model.url, model.isLocal), forTask: taskName)
    } else if !customModels.isEmpty {
      showCustomModelPicker()
    }
  }

  private func makeModelEntries(for taskName: String) -> [(name: String, url: URL?, isLocal: Bool)]
  {
    let localModels = (modelsForTask[taskName] ?? []).map {
      (name: ($0 as NSString).deletingPathExtension, url: nil as URL?, isLocal: true)
    }
    let localModelNames = Set(localModels.map { $0.name.lowercased() })
    let remoteModels = (remoteModelsInfo[taskName] ?? []).compactMap { modelName, url in
      localModelNames.contains(modelName.lowercased())
        ? nil : (name: modelName, url: url, isLocal: false)
    }
    return localModels + remoteModels
  }

  private func loadModel(entry: (name: String, url: URL?, isLocal: Bool), forTask task: String) {
    guard !isLoadingModel else { return }
    isLoadingModel = true
    yoloView.resetLayers()
    yoloView.setInferenceFlag(ok: false)
    setLoadingState(true, showOverlay: true)
    resetDownloadProgress()

    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
    let loadWithPath = { [weak self] (path: String) in
      self?.downloadProgressLabel.text = "Loading \(entry.name)"
      self?.downloadProgressLabel.isHidden = false
      self?.yoloView.setModel(modelPathOrName: path, task: yoloTask) { result in
        self?.finishLoadingModel(success: result.isSuccess, modelName: entry.name)
      }
    }

    if entry.isLocal {
      guard let folderPath = tasks.first(where: { $0.name == task })?.folder,
        let url = Bundle.main.url(forResource: folderPath, withExtension: nil)
      else {
        return finishLoadingModel(success: false, modelName: entry.name)
      }
      loadWithPath(url.appendingPathComponent(entry.name + ".mlpackage").path)
    } else if let remoteURL = entry.url {
      [downloadProgressView, downloadProgressLabel].forEach { $0.isHidden = false }
      YOLOModelDownloader().download(
        from: remoteURL, task: yoloTask,
        progress: { [weak self] progress in
          DispatchQueue.main.async {
            self?.downloadProgressView.progress = Float(progress)
            self?.downloadProgressLabel.text = "Downloading \(Int(progress * 100))%"
          }
        },
        completion: { [weak self] result in
          switch result {
          case .success(let path): DispatchQueue.main.async { loadWithPath(path.path) }
          case .failure: self?.finishLoadingModel(success: false, modelName: entry.name)
          }
        })
    } else {
      finishLoadingModel(success: false, modelName: entry.name)
    }
  }

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    [downloadProgressView, downloadProgressLabel].forEach { $0.isHidden = true }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async {
      self.setLoadingState(false)
      self.isLoadingModel = false
      self.resetDownloadProgress()

      if success {
        let yoloTask = self.tasks.first(where: { $0.name == self.currentTask })?.yoloTask ?? .detect
        ModelSelectionManager.setupSegmentedControl(
          self.modelSegmentedControl,
          standardModels: self.standardModels,
          hasCustomModels: !self.customModels.isEmpty,
          currentTask: yoloTask,
          preserveSelection: true
        )
      }

      self.yoloView.setInferenceFlag(ok: success)

      if success {
        self.currentModelName = modelName
        self.labelName.text = processString(modelName)
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
    reloadModelEntriesAndLoadFirst(for: currentTask)
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
    modelSegmentedControl.addTarget(self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)
  }

  func updateModelSegmentedControlAppearance() {
    guard modelSegmentedControl != nil else { return }

    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.backgroundColor = .clear

    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect
    ModelSelectionManager.updateSegmentAppearance(modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)
  }

  @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    if sender.selectedSegmentIndex < ModelSelectionManager.ModelSize.allCases.count {
      let size = ModelSelectionManager.ModelSize.allCases[sender.selectedSegmentIndex]
      if let model = standardModels[size] {
        loadModel(entry: (model.name, model.url, model.isLocal), forTask: currentTask)
      }
    } else {
      showCustomModelPicker()
    }
  }

  private func showCustomModelPicker() {
    let alert = UIAlertController(title: "Select Custom Model", message: nil, preferredStyle: .actionSheet)

    for model in customModels {
      alert.addAction(UIAlertAction(title: processString(model.name), style: .default) { [weak self] _ in
        self?.loadModel(entry: (model.name, model.url, model.isLocal), forTask: self?.currentTask ?? "")
      })
    }

    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { [weak self] _ in
      self?.modelSegmentedControl.selectedSegmentIndex = 0
      self?.modelSizeChanged(self?.modelSegmentedControl ?? UISegmentedControl())
    })

    if let popover = alert.popoverPresentationController {
      popover.sourceView = modelSegmentedControl
      popover.sourceRect = modelSegmentedControl.bounds
    }

    present(alert, animated: true)
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
  }


  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] image in
      guard let self = self, let image = image else { return print("error capturing photo") }
      DispatchQueue.main.async {
        let vc = UIActivityViewController(activityItems: [image], applicationActivities: nil)
        vc.popoverPresentationController?.sourceView = self.View0
        self.present(vc, animated: true)
      }
    }
  }

}


// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
    labelFPS.textColor = .white
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
  }
}
