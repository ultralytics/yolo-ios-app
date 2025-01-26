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
    
    // 進捗表示用のプログレスバー
    private let downloadProgressView: UIProgressView = {
        let pv = UIProgressView(progressViewStyle: .default)
        pv.progress = 0.0
        pv.isHidden = true
        return pv
    }()
    
    private let tasks: [(name: String, folder: String)] = [
        ("Detect",   "DetectModels"),
        ("Segment",  "SegmentModels"),
        ("Classify", "ClassifyModels"),
        ("Pose",     "PoseModels"),
        ("Obb",      "ObbModels"),
    ]
    
    /// バンドルから読み込んだファイル一覧
    private var modelsForTask: [String: [String]] = [:]
    
    /// テーブル表示用: バンドルモデル + リモートモデル
    private var currentModels: [ModelEntry] = []
    
    private var currentTask: String = ""
    private var currentModelName: String = ""
    
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
        
        // 初期タスクをDetect(0番目)に
        if tasks.indices.contains(0) {
            segmentedControl.selectedSegmentIndex = 0
            currentTask = tasks[0].name
            
            // テーブル用のデータを作り直し、最初のモデルをロード
            reloadModelEntriesAndLoadFirst(for: currentTask)
        }
        
        setupTableView()
        setupButtons()
        
        // プログレスバーを画面に追加
        downloadProgressView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(downloadProgressView)
        
        NSLayoutConstraint.activate([
            downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            downloadProgressView.topAnchor.constraint(equalTo: activityIndicator.bottomAnchor, constant: 8),
            downloadProgressView.widthAnchor.constraint(equalToConstant: 200),
            downloadProgressView.heightAnchor.constraint(equalToConstant: 2)
        ])
        
        // ModelDownloadManager のダウンロード進捗コールバックを設定
        ModelDownloadManager.shared.progressHandler = { [weak self] progress in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.downloadProgressView.progress = Float(progress)
            }
        }
    }
    
    /// タスク用のセグメントコントロール初期化
    private func setupTaskSegmentedControl() {
        segmentedControl.removeAllSegments()
        for (index, taskInfo) in tasks.enumerated() {
            segmentedControl.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
        }
    }
    
    /// 全タスクのバンドル内モデルをあらかじめ読み込む
    private func loadModelsForAllTasks() {
        for taskInfo in tasks {
            let taskName = taskInfo.name
            let folderName = taskInfo.folder
            let modelFiles = getModelFiles(in: folderName)
            modelsForTask[taskName] = modelFiles
        }
    }
    
    /// バンドル内の {folderName} ディレクトリをスキャンし、mlmodel / mlpackage だけリスト化
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
    
    /// 現在のタスクに応じて、バンドルモデル & リモートモデルをまとめて currentModels を組み立て、
    /// 最初のモデルをロードする
    private func reloadModelEntriesAndLoadFirst(for taskName: String) {
        currentModels = makeModelEntries(for: taskName)
        
        if !currentModels.isEmpty {
            modelTableView.isHidden = false
            modelTableView.reloadData()
            
            // とりあえず先頭のモデルをロード
            DispatchQueue.main.async {
                let firstIndex = IndexPath(row: 0, section: 0)
                self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
                let firstModel = self.currentModels[0]
                self.loadModel(entry: firstModel, forTask: taskName)
            }
        } else {
            print("No models found for task: \(taskName)")
            modelTableView.isHidden = true
        }
    }
    
    /// タスクに応じて、バンドル内モデル一覧 + リモートモデル一覧 を `[ModelEntry]` に変換
    private func makeModelEntries(for taskName: String) -> [ModelEntry] {
        // 1) バンドル内モデル
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
        
        // 2) リモートモデル一覧
        // 例: remoteModelsInfo は [String: [(String, URL)]] で定義してある想定
        let remoteList = remoteModelsInfo[taskName] ?? []
        let remoteEntries = remoteList.map { (modelName, url) -> ModelEntry in
            ModelEntry(
                displayName: modelName,
                identifier: modelName, // ダウンロード時のキーに使う (拡張子は付けない)
                isLocalBundle: false,
                isRemote: true,
                remoteURL: url
            )
        }
        
        // バンドルモデルを先頭に、リモートモデルを後に連結
        return localEntries + remoteEntries
    }
    
    /// バンドル or リモートモデルをロード (リモートの場合、未ダウンロードならダウンロード→ロード)
    private func loadModel(entry: ModelEntry, forTask task: String) {
        guard !isLoadingModel else {
            print("Model is already loading. Please wait.")
            return
        }
        isLoadingModel = true
        
        // UIロック＆インジケータ開始
        self.activityIndicator.startAnimating()
        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = true
        self.view.isUserInteractionEnabled = false
        self.modelTableView.isUserInteractionEnabled = false
        
        print("Start loading model: \(entry.displayName)")
        
        if entry.isLocalBundle {
            // -------------------------------
            // バンドル内モデルをそのままロード
            // -------------------------------
            DispatchQueue.global().async { [weak self] in
                guard let self = self else { return }
                let yoloTask = self.convertTaskNameToYOLOTask(task)
                
                guard let folderURL = self.tasks.first(where: { $0.name == task })?.folder,
                      let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) else {
                    DispatchQueue.main.async {
                        self.finishLoadingModel(success: false, modelName: entry.displayName)
                    }
                    return
                }
                
                let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
                DispatchQueue.main.async {
                    // YOLOViewへのセット
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
            // -------------------------------
            // リモートモデルの場合
            // -------------------------------
            let yoloTask = self.convertTaskNameToYOLOTask(task)
            
            // `key` はキャッシュ用の「識別子」(拡張子なし)
            let key = entry.identifier  // "yolov8n", "yolov8m-seg", etc.

            if ModelCacheManager.shared.isModelDownloaded(key: key) {
                // 既にダウンロード済みならそのままロード
                loadCachedModelAndSetToYOLOView(key: key, yoloTask: yoloTask, displayName: entry.displayName)
            } else {
                // ダウンロード開始
                guard let remoteURL = entry.remoteURL else {
                    self.finishLoadingModel(success: false, modelName: entry.displayName)
                    return
                }
                
                // プログレスバーを表示
                self.downloadProgressView.progress = 0.0
                self.downloadProgressView.isHidden = false
                
                // ここで fileName に `remoteURL.lastPathComponent` を用いることで
                // 例:  remoteURL.lastPathComponent -> "yolov8n.mlpackage.zip"
                // ローカル保存先も "yolov8n.mlpackage.zip" となり ZIP 解凍可能
                let localZipFileName = remoteURL.lastPathComponent  // ex. "yolov8n.mlpackage.zip"
                
                ModelCacheManager.shared.loadModel(
                    from: localZipFileName,   // ダウンロード先ファイル名
                    remoteURL: remoteURL,
                    key: key
                ) { [weak self] mlModel, loadedKey in
                    guard let self = self else { return }
                    if mlModel == nil {
                        // ダウンロード or コンパイル失敗
                        self.finishLoadingModel(success: false, modelName: entry.displayName)
                        return
                    }
                    // ダウンロード成功 → YOLOViewへセット
                    self.loadCachedModelAndSetToYOLOView(key: loadedKey,
                                                         yoloTask: yoloTask,
                                                         displayName: entry.displayName)
                }
            }
        }
    }
    
    /// 既にダウンロード＆コンパイル済みのモデル (Documents/*.mlmodelc) を読み込み、YOLOViewへセット
    private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String) {
        // Documents/key.mlmodelc が存在しているはず
        let localModelURL = ModelCacheManager.shared.getDocumentsDirectory()
            .appendingPathComponent(key)
            .appendingPathExtension("mlmodelc")
        
        DispatchQueue.main.async {
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
    
    /// モデルロード完了後の処理 (UIの再有効化等)
    private func finishLoadingModel(success: Bool, modelName: String) {
        DispatchQueue.main.async {
            self.activityIndicator.stopAnimating()
            self.downloadProgressView.isHidden = true
            self.view.isUserInteractionEnabled = true
            self.modelTableView.isUserInteractionEnabled = true
            self.isLoadingModel = false
            self.modelTableView.reloadData()
        }
        
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
    
    /// タスクのセグメントを切り替えた
    @IBAction func indexChanged(_ sender: UISegmentedControl) {
        selection.selectionChanged()
        
        let index = sender.selectedSegmentIndex
        guard tasks.indices.contains(index) else { return }
        
        let newTask = tasks[index].name
        
        // バンドルモデルが存在しないタスクの場合はアラート
        if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
            let alert = UIAlertController(
                title: "\(newTask) Models not found",
                message: "Please add or define models for \(newTask).",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { _ in
                alert.dismiss(animated: true)
            }))
            self.present(alert, animated: true)
            
            // セグメントを元に戻す
            if let oldIndex = tasks.firstIndex(where: { $0.name == currentTask }) {
                sender.selectedSegmentIndex = oldIndex
            }
            return
        }
        
        currentTask = newTask
        
        // 表示内容を更新して先頭モデルをロード
        reloadModelEntriesAndLoadFirst(for: currentTask)
        
        // 背景枠のサイズ更新
        tableViewBGView.frame = CGRect(
            x: modelTableView.frame.minX-1,
            y: modelTableView.frame.minY-1,
            width: modelTableView.frame.width+2,
            height: CGFloat(currentModels.count*30+2)
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
        tableViewBGView.frame = CGRect(
            x: modelTableView.frame.minX-1,
            y: modelTableView.frame.minY-1,
            width: modelTableView.frame.width+2,
            height: CGFloat(currentModels.count*30+2)
        )
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
            modelTableView.frame = CGRect(x: view.bounds.width - tableViewWidth - 8,
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
            x: modelTableView.frame.minX-1,
            y: modelTableView.frame.minY-1,
            width: modelTableView.frame.width+2,
            height: CGFloat(currentModels.count*30+2)
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
        let entry = currentModels[indexPath.row]
        
        cell.textLabel?.textAlignment = .center
        cell.textLabel?.text = entry.displayName
        cell.textLabel?.font = UIFont.systemFont(ofSize: 14, weight: .medium)
        cell.backgroundColor = .clear
        
        // ダウンロードアイコン（icloud.and.arrow.down）をアクセサリに表示
        // ただし isRemote かつ 未ダウンロード時のみ
        if entry.isRemote {
            let isDownloaded = ModelCacheManager.shared.isModelDownloaded(key: entry.identifier)
            if !isDownloaded {
                cell.accessoryView = UIImageView(image: UIImage(systemName: "icloud.and.arrow.down"))
            } else {
                cell.accessoryView = nil
            }
        } else {
            cell.accessoryView = nil
        }
        
        // セル選択時の背景
        let selectedBGView = UIView()
        selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
        selectedBGView.layer.cornerRadius = 8
        selectedBGView.layer.masksToBounds = true
        selectedBGView.frame = CGRect(
            x: 2, y: 2,
            width: cell.contentView.bounds.width - 4,
            height: cell.contentView.bounds.height - 4
        )
        cell.selectedBackgroundView = selectedBGView
        
        cell.selectionStyle = .default
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        selection.selectionChanged()
        let selectedEntry = currentModels[indexPath.row]
        
        // バンドル or リモートモデルをロード
        loadModel(entry: selectedEntry, forTask: currentTask)
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

extension ViewController: RPPreviewViewControllerDelegate {
    func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
        previewController.dismiss(animated: true)
    }
}
