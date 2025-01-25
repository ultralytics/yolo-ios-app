import AVFoundation
import CoreML
import CoreMedia
import UIKit
import YOLO
import ReplayKit
import AudioToolbox

class ViewController: UIViewController {

    @IBOutlet weak var yoloView: YOLOView!
    @IBOutlet var View0: UIView!
    @IBOutlet var segmentedControl: UISegmentedControl!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var focus: UIImageView!
    @IBOutlet weak var logoImage: UIImageView!
    
    var shareButton = UIButton()
    var recordButton = UIButton()
    let selection = UISelectionFeedbackGenerator()
    
    private let tasks: [(name: String, folder: String)] = [
        ("Detect",   "DetectModels"),
        ("Segment",  "SegmentModels"),
        ("Classify", "ClassifyModels"),
        ("Pose",     "PoseModels"),
        ("Obb",      "ObbModels"),
    ]
    
    private var modelsForTask: [String: [String]] = [:]
    
    private var currentTask: String = ""
    private var currentModelName: String = ""
    
    private var currentModels: [String] = []
    private var isLoadingModel = false

    private let modelTableView: UITableView = {
        let table = UITableView()
        table.isHidden = true
        table.layer.cornerRadius = 8
        table.clipsToBounds = true
        return table
    }()
    
    private let tableViewBGView = UIView()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupTaskSegmentedControl()
        loadModelsForAllTasks()
        
        if tasks.indices.contains(0) {
            segmentedControl.selectedSegmentIndex = 0
            currentTask = tasks[0].name
            loadFirstModel(for: currentTask)
            currentModels = modelsForTask[currentTask] ?? []
            
            if !currentModels.isEmpty {
                modelTableView.isHidden = false
                modelTableView.reloadData()
                
                DispatchQueue.main.async {
                    let firstIndex = IndexPath(row: 0, section: 0)
                    self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
                }
            }
        }
        
        setupTableView()
        setupButtons()
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
        var result: [String] = []
        
        if let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil) {
            do {
                let fileURLs = try FileManager.default.contentsOfDirectory(
                    at: folderURL,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles]
                )
                for fileURL in fileURLs {
                    if fileURL.pathExtension == "mlmodel" || fileURL.pathExtension == "mlpackage" {
                        let fileName = fileURL.lastPathComponent
                        result.append(fileName)
                    }
                }
            } catch {
                print("Error reading contents of folder \(folderName): \(error)")
            }
        }
        
        return result.sorted()
    }
    
    private func loadFirstModel(for task: String) {
        guard let models = modelsForTask[task], !models.isEmpty else {
            print("No models found for task: \(task)")
            return
        }
        let firstModel = models[0]
        loadModel(named: firstModel, forTask: task)
        
        let firstIndex = IndexPath(row: 0, section: 0)
        modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
    }
    
    private func loadModel(named modelName: String, forTask task: String) {
        guard !isLoadingModel else {
            print("Model is already loading. Please wait.")
            return
        }
        isLoadingModel = true
        
        activityIndicator.startAnimating()
        self.view.isUserInteractionEnabled = false
        modelTableView.isUserInteractionEnabled = false
        
        print("Start loading model: \(modelName)")
        
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            let yoloTask = self.convertTaskNameToYOLOTask(task)

            guard let modelFolder = self.tasks.first(where: { $0.name == task })?.folder,
                  let folderURL = Bundle.main.url(forResource: modelFolder, withExtension: nil) else {
                DispatchQueue.main.async {
                    self.finishLoadingModel(success: false, modelName: modelName)
                }
                return
            }
            
            let modelURL = folderURL.appendingPathComponent(modelName)
            
            DispatchQueue.main.async {
                self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
                    switch result {
                    case .success():
                        self.finishLoadingModel(success: true, modelName: modelName)
                    case .failure(let error):
                        print(error)
                        self.finishLoadingModel(success: false, modelName: modelName)
                    }
                }
            }
        }
    }
    
    private func finishLoadingModel(success: Bool, modelName: String) {
        self.activityIndicator.stopAnimating()
        self.view.isUserInteractionEnabled = true
        self.modelTableView.isUserInteractionEnabled = true
        self.isLoadingModel = false
        
        if success {
            print("Finished loading model: \(modelName)")
            self.currentModelName = modelName
        } else {
            print("Failed to load model: \(modelName)")
        }
    }
    
    private func convertTaskNameToYOLOTask(_ task: String) -> YOLOTask {
        switch task {
        case "Detect":   return .detect
        case "Segment":  return .segment
        case "Classify": return .classify
        case "Pose":     return .pose
        case "Obb":      return .obb
        default:         return .detect
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
        
        if modelsForTask[newTask]!.isEmpty {
            let alert = UIAlertController(title: "\(newTask)Models not found", message: "Please add coreml models for \(newTask) to the \(newTask)Models directory in the Xcode project", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { _ in
                alert.dismiss(animated: true)
            }))
            self.present(alert, animated: true)
            sender.selectedSegmentIndex = tasks.firstIndex(where: { name, _ in
                name == currentTask
            })!
            return
        }
        
        currentTask = newTask
        currentModels = modelsForTask[currentTask] ?? []
        
        if !currentModels.isEmpty {
            loadModel(named: currentModels[0], forTask: currentTask)
            modelTableView.isHidden = false
            modelTableView.reloadData()
            
            DispatchQueue.main.async {
                let firstIndex = IndexPath(row: 0, section: 0)
                self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
            }
        } else {
            print("No models available for task: \(currentTask)")
            modelTableView.isHidden = true
        }
        tableViewBGView.frame = CGRect(x: modelTableView.frame.minX-1, y: modelTableView.frame.minY-1, width: modelTableView.frame.width+2, height: CGFloat(modelsForTask[currentTask]!.count*30+2))
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
        modelTableView.register(UITableViewCell.self, forCellReuseIdentifier: "ModelCell")
        modelTableView.backgroundColor = .clear
        modelTableView.separatorStyle = .none
        modelTableView.isScrollEnabled = false
        tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
        tableViewBGView.layer.cornerRadius = 8
        tableViewBGView.clipsToBounds = true
        view.addSubview(tableViewBGView)

        view.addSubview(modelTableView)
        modelTableView.translatesAutoresizingMaskIntoConstraints = false
        
        tableViewBGView.frame = CGRect(x: modelTableView.frame.minX-1, y: modelTableView.frame.minY-1, width: modelTableView.frame.width+2, height: CGFloat(modelsForTask[currentTask]!.count*30+2))

    }
    
    private func setupButtons() {
        let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
        shareButton.setImage(UIImage(systemName: "square.and.arrow.up", withConfiguration: config), for: .normal)
        shareButton.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(shareButtonTapped)))
        view.addSubview(shareButton)
        
        recordButton.setImage(UIImage(systemName: "video", withConfiguration: config), for: .normal)
        recordButton.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(recordScreen)))
        view.addSubview(recordButton)

        logoImage.isUserInteractionEnabled = true
        logoImage.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        if view.bounds.width > view.bounds.height {
            shareButton.tintColor = .darkGray
            recordButton.tintColor = .darkGray
            let tableViewWidth = view.bounds.width * 0.2
            modelTableView.frame = CGRect(x: segmentedControl.frame.maxX + 20, y: 20, width: tableViewWidth, height: 200)
        } else {
            shareButton.tintColor = .systemGray
            recordButton.tintColor = .systemGray
            let tableViewWidth = view.bounds.width * 0.4
            modelTableView.frame = CGRect(x: view.bounds.width - tableViewWidth - 8, y: segmentedControl.frame.maxY + 25, width: tableViewWidth, height: 200)

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
        tableViewBGView.frame = CGRect(x: modelTableView.frame.minX-1, y: modelTableView.frame.minY-1, width: modelTableView.frame.width+2, height: CGFloat(modelsForTask[currentTask]!.count*30+2))
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
            recorder.startRecording() { error in
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

        let cell = tableView.dequeueReusableCell(withIdentifier: "ModelCell", for: indexPath)
        
        let fileName = currentModels[indexPath.row]
        let displayName = (fileName as NSString).deletingPathExtension
        
        cell.textLabel?.textAlignment = .center
        cell.textLabel?.text = displayName
        cell.textLabel?.font = UIFont.systemFont(ofSize: 14, weight: .medium)
        
        cell.backgroundColor = .clear
        
        let selectedBGView = UIView()
        selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
        selectedBGView.layer.cornerRadius = 8
        selectedBGView.layer.masksToBounds = true
        selectedBGView.frame = CGRect(x: 2, y:2, width: cell.contentView.bounds.width - 4, height: cell.contentView.bounds.height - 4)
        cell.selectedBackgroundView = selectedBGView
        
        cell.selectionStyle = .default
        
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let selectedModel = currentModels[indexPath.row]
        
        selection.selectionChanged()
        
        loadModel(named: selectedModel, forTask: currentTask)
    }
    
    func tableView(_ tableView: UITableView,
                   willDisplay cell: UITableViewCell,
                   forRowAt indexPath: IndexPath) {
        if let selectedBGView = cell.selectedBackgroundView {
            let insetRect = cell.bounds.insetBy(dx: 4, dy: 4)
            selectedBGView.frame = insetRect
        }
    }
}

extension ViewController:RPPreviewViewControllerDelegate {
    func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
        previewController.dismiss(animated: true)
    }
}
