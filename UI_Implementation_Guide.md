# Ultralytics YOLO iOS App - UI実装ガイド

## 1. デザインシステムの実装

### 1.1 カラーパレット (Colors.swift)
```swift
import UIKit

extension UIColor {
    // Primary Colors
    static let ultralyticsLime = UIColor(red: 207/255, green: 255/255, blue: 26/255, alpha: 1.0) // #CFFF1A
    static let ultralyticsBrown = UIColor(red: 106/255, green: 85/255, blue: 69/255, alpha: 1.0) // #6A5545
    
    // Surface Colors
    static let ultralyticsSurfaceDark = UIColor.black // #000000
    
    // Text Colors
    static let ultralyticsTextPrimary = UIColor.white // #FFFFFF
    static let ultralyticsTextSubtle = UIColor(red: 125/255, green: 125/255, blue: 125/255, alpha: 1.0) // #7D7D7D
    
    // Convenience initializer
    convenience init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            red: CGFloat(r) / 255,
            green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255,
            alpha: CGFloat(a) / 255
        )
    }
}
```

### 1.2 タイポグラフィ (Typography.swift)
```swift
import UIKit

struct Typography {
    // Status Bar
    static let statusBarFont = UIFont.systemFont(ofSize: 10, weight: .bold).rounded()
    
    // Task Tabs
    static let tabLabelFont = UIFont.systemFont(ofSize: 11, weight: .semibold).rounded()
    
    // Labels
    static let labelFont = UIFont.systemFont(ofSize: 8, weight: .bold).rounded()
    
    // Toast
    static let toastFont = UIFont.systemFont(ofSize: 10, weight: .bold).rounded()
}

extension UIFont {
    func rounded() -> UIFont {
        guard let descriptor = fontDescriptor.withDesign(.rounded) else { return self }
        return UIFont(descriptor: descriptor, size: pointSize)
    }
}
```

## 2. UIコンポーネントの実装

### 2.1 Status & Metric Bar (StatusMetricBar.swift)
```swift
import UIKit

class StatusMetricBar: UIView {
    // UI Elements
    private let logoImageView = UIImageView()
    private let modelButton = UIButton(type: .system)
    private let sizeLabel = UILabel()
    private let fpsLabel = UILabel()
    private let latencyLabel = UILabel()
    private let stackView = UIStackView()
    
    // Properties
    var onModelTap: (() -> Void)?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        backgroundColor = .ultralyticsSurfaceDark
        
        // Logo
        logoImageView.image = UIImage(named: "ultralytics_logo")
        logoImageView.contentMode = .scaleAspectFit
        logoImageView.widthAnchor.constraint(equalToConstant: 20).isActive = true
        logoImageView.heightAnchor.constraint(equalToConstant: 20).isActive = true
        
        // Model Button
        modelButton.setTitle("YOLO11 ▼", for: .normal)
        modelButton.setTitleColor(.ultralyticsTextPrimary, for: .normal)
        modelButton.titleLabel?.font = Typography.statusBarFont
        modelButton.addTarget(self, action: #selector(modelButtonTapped), for: .touchUpInside)
        
        // Labels
        [sizeLabel, fpsLabel, latencyLabel].forEach { label in
            label.textColor = .ultralyticsTextPrimary
            label.font = Typography.statusBarFont
        }
        
        sizeLabel.text = "SMALL"
        fpsLabel.text = "0.0 FPS"
        latencyLabel.text = "0.0 ms"
        
        // Stack View
        stackView.axis = .horizontal
        stackView.spacing = 12
        stackView.alignment = .center
        stackView.distribution = .fill
        
        [logoImageView, modelButton, sizeLabel, fpsLabel, latencyLabel].forEach {
            stackView.addArrangedSubview($0)
        }
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor),
            heightAnchor.constraint(equalToConstant: 36)
        ])
        
        // Long press gesture for hidden info
        let longPress = UILongPressGestureRecognizer(target: self, action: #selector(logoLongPressed))
        longPress.minimumPressDuration = 1.0
        logoImageView.isUserInteractionEnabled = true
        logoImageView.addGestureRecognizer(longPress)
    }
    
    @objc private func modelButtonTapped() {
        onModelTap?()
    }
    
    @objc private func logoLongPressed(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            // Show hidden info page
            NotificationCenter.default.post(name: .showHiddenInfo, object: nil)
        }
    }
    
    func updateMetrics(fps: Double, latency: Double) {
        fpsLabel.text = String(format: "%.1f FPS", fps)
        latencyLabel.text = String(format: "%.1f ms", latency)
    }
    
    func updateModel(name: String, size: String) {
        modelButton.setTitle("\(name) ▼", for: .normal)
        sizeLabel.text = size.uppercased()
    }
}

extension Notification.Name {
    static let showHiddenInfo = Notification.Name("showHiddenInfo")
}
```

### 2.2 Task Tab Strip (TaskTabStrip.swift)
```swift
import UIKit

class TaskTabStrip: UIView {
    enum Task: String, CaseIterable {
        case detect = "DETECT"
        case segment = "SEGMENT"
        case classify = "CLASS"
        
        var yoloTask: YOLO.Task {
            switch self {
            case .detect: return .detect
            case .segment: return .segment
            case .classify: return .classify
            }
        }
    }
    
    private var buttons: [UIButton] = []
    private let underlineView = UIView()
    private var underlineLeadingConstraint: NSLayoutConstraint?
    private var underlineWidthConstraint: NSLayoutConstraint?
    
    var selectedTask: Task = .detect {
        didSet {
            updateSelection()
        }
    }
    
    var onTaskChange: ((Task) -> Void)?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        backgroundColor = .ultralyticsSurfaceDark
        
        // Create buttons
        let stackView = UIStackView()
        stackView.axis = .horizontal
        stackView.distribution = .fillEqually
        stackView.spacing = 0
        
        for task in Task.allCases {
            let button = UIButton(type: .system)
            button.setTitle(task.rawValue, for: .normal)
            button.titleLabel?.font = Typography.tabLabelFont
            button.tag = Task.allCases.firstIndex(of: task)!
            button.addTarget(self, action: #selector(taskButtonTapped), for: .touchUpInside)
            buttons.append(button)
            stackView.addArrangedSubview(button)
        }
        
        // Underline
        underlineView.backgroundColor = .ultralyticsLime
        
        addSubview(stackView)
        addSubview(underlineView)
        
        stackView.translatesAutoresizingMaskIntoConstraints = false
        underlineView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor),
            stackView.topAnchor.constraint(equalTo: topAnchor),
            stackView.bottomAnchor.constraint(equalTo: bottomAnchor),
            heightAnchor.constraint(equalToConstant: 28),
            
            underlineView.heightAnchor.constraint(equalToConstant: 2),
            underlineView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
        
        updateSelection()
    }
    
    @objc private func taskButtonTapped(_ sender: UIButton) {
        guard let task = Task.allCases[safe: sender.tag] else { return }
        selectedTask = task
        onTaskChange?(task)
    }
    
    private func updateSelection() {
        // Update button colors
        for (index, button) in buttons.enumerated() {
            let isSelected = index == Task.allCases.firstIndex(of: selectedTask)
            button.setTitleColor(isSelected ? .ultralyticsLime : .ultralyticsTextSubtle, for: .normal)
        }
        
        // Animate underline
        guard let selectedIndex = Task.allCases.firstIndex(of: selectedTask),
              let selectedButton = buttons[safe: selectedIndex] else { return }
        
        UIView.animate(withDuration: 0.15, delay: 0, options: .curveEaseOut) {
            self.underlineView.frame = CGRect(
                x: selectedButton.frame.minX,
                y: self.bounds.height - 2,
                width: selectedButton.frame.width,
                height: 2
            )
        }
    }
}

extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}
```

### 2.3 Shutter Bar (ShutterBar.swift)
```swift
import UIKit

class ShutterBar: UIView {
    // UI Elements
    private let thumbnailButton = UIButton(type: .custom)
    private let shutterButton = UIButton(type: .custom)
    private let flipCameraButton = UIButton(type: .custom)
    
    // Callbacks
    var onThumbnailTap: (() -> Void)?
    var onShutterTap: (() -> Void)?
    var onShutterLongPress: (() -> Void)?
    var onFlipCamera: (() -> Void)?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        backgroundColor = .ultralyticsSurfaceDark
        
        // Thumbnail Button
        thumbnailButton.backgroundColor = .ultralyticsBrown
        thumbnailButton.layer.cornerRadius = 8
        thumbnailButton.clipsToBounds = true
        thumbnailButton.addTarget(self, action: #selector(thumbnailTapped), for: .touchUpInside)
        
        // Shutter Button
        shutterButton.backgroundColor = .white
        shutterButton.layer.cornerRadius = 34
        shutterButton.layer.borderWidth = 4
        shutterButton.layer.borderColor = UIColor.black.cgColor
        
        let shutterTap = UITapGestureRecognizer(target: self, action: #selector(shutterTapped))
        let shutterLongPress = UILongPressGestureRecognizer(target: self, action: #selector(shutterLongPressed))
        shutterLongPress.minimumPressDuration = 0.7
        
        shutterButton.addGestureRecognizer(shutterTap)
        shutterButton.addGestureRecognizer(shutterLongPress)
        
        // Flip Camera Button
        flipCameraButton.backgroundColor = .clear
        flipCameraButton.setImage(UIImage(systemName: "camera.rotate"), for: .normal)
        flipCameraButton.tintColor = .white
        flipCameraButton.addTarget(self, action: #selector(flipCameraTapped), for: .touchUpInside)
        
        // Layout
        [thumbnailButton, shutterButton, flipCameraButton].forEach {
            addSubview($0)
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
        
        NSLayoutConstraint.activate([
            // Container height
            heightAnchor.constraint(equalToConstant: 96),
            
            // Thumbnail
            thumbnailButton.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 24),
            thumbnailButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            thumbnailButton.widthAnchor.constraint(equalToConstant: 48),
            thumbnailButton.heightAnchor.constraint(equalToConstant: 48),
            
            // Shutter
            shutterButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            shutterButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            shutterButton.widthAnchor.constraint(equalToConstant: 68),
            shutterButton.heightAnchor.constraint(equalToConstant: 68),
            
            // Flip Camera
            flipCameraButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -24),
            flipCameraButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            flipCameraButton.widthAnchor.constraint(equalToConstant: 44),
            flipCameraButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    @objc private func thumbnailTapped() {
        onThumbnailTap?()
    }
    
    @objc private func shutterTapped() {
        // Flash animation
        let flashView = UIView(frame: UIScreen.main.bounds)
        flashView.backgroundColor = .white
        flashView.alpha = 0
        
        if let window = UIApplication.shared.windows.first {
            window.addSubview(flashView)
            
            UIView.animate(withDuration: 0.08, animations: {
                flashView.alpha = 0.8
            }) { _ in
                flashView.removeFromSuperview()
            }
        }
        
        onShutterTap?()
    }
    
    @objc private func shutterLongPressed(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            onShutterLongPress?()
            // Start recording animation
            UIView.animate(withDuration: 0.3) {
                self.shutterButton.backgroundColor = .red
            }
        }
    }
    
    @objc private func flipCameraTapped() {
        onFlipCamera?()
    }
    
    func updateThumbnail(_ image: UIImage?) {
        thumbnailButton.setImage(image, for: .normal)
    }
    
    func setRecording(_ isRecording: Bool) {
        shutterButton.backgroundColor = isRecording ? .red : .white
    }
}
```

### 2.4 Parameter Edit View (ParameterEditView.swift)
```swift
import UIKit

class ParameterEditView: UIView {
    enum Parameter {
        case itemsMax(Int)
        case confidence(Float)
        case iou(Float)
        case lineThickness(Float)
        
        var title: String {
            switch self {
            case .itemsMax: return "ITEMS MAX"
            case .confidence: return "CONFIDENCE THRESHOLD"
            case .iou: return "IoU THRESHOLD"
            case .lineThickness: return "LINE THICKNESS"
            }
        }
        
        var range: ClosedRange<Float> {
            switch self {
            case .itemsMax: return 1...30
            case .confidence, .iou: return 0...1
            case .lineThickness: return 0.5...3.0
            }
        }
        
        var step: Float {
            switch self {
            case .itemsMax: return 1
            case .confidence, .iou: return 0.02
            case .lineThickness: return 0.1
            }
        }
        
        var value: Float {
            switch self {
            case .itemsMax(let v): return Float(v)
            case .confidence(let v), .iou(let v), .lineThickness(let v): return v
            }
        }
    }
    
    private let toastView = UIView()
    private let toastLabel = UILabel()
    private let slider = UISlider()
    private var hideTimer: Timer?
    
    var onValueChange: ((Parameter) -> Void)?
    private var currentParameter: Parameter?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        // Toast View
        toastView.backgroundColor = UIColor.ultralyticsBrown.withAlphaComponent(0.95)
        toastView.layer.cornerRadius = 14
        toastView.alpha = 0
        
        toastLabel.textColor = .ultralyticsTextPrimary
        toastLabel.font = Typography.toastFont
        toastLabel.textAlignment = .center
        
        toastView.addSubview(toastLabel)
        addSubview(toastView)
        
        // Slider
        slider.minimumTrackTintColor = .ultralyticsLime
        slider.maximumTrackTintColor = .ultralyticsSurfaceDark
        slider.thumbTintColor = .ultralyticsLime
        slider.alpha = 0
        slider.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        
        // Custom thumb
        let thumbView = UIView(frame: CGRect(x: 0, y: 0, width: 4, height: 20))
        thumbView.backgroundColor = .ultralyticsLime
        thumbView.layer.cornerRadius = 2
        
        UIGraphicsBeginImageContextWithOptions(thumbView.bounds.size, false, 0)
        thumbView.layer.render(in: UIGraphicsGetCurrentContext()!)
        let thumbImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        slider.setThumbImage(thumbImage, for: .normal)
        
        addSubview(slider)
        
        // Layout
        [toastView, toastLabel, slider].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
        
        NSLayoutConstraint.activate([
            // Toast
            toastView.topAnchor.constraint(equalTo: topAnchor, constant: 20),
            toastView.centerXAnchor.constraint(equalTo: centerXAnchor),
            toastView.widthAnchor.constraint(equalToConstant: 120),
            toastView.heightAnchor.constraint(equalToConstant: 28),
            
            toastLabel.centerXAnchor.constraint(equalTo: toastView.centerXAnchor),
            toastLabel.centerYAnchor.constraint(equalTo: toastView.centerYAnchor),
            
            // Slider
            slider.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
            slider.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
            slider.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -20),
            slider.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    func showParameter(_ parameter: Parameter) {
        currentParameter = parameter
        
        // Configure slider
        slider.minimumValue = parameter.range.lowerBound
        slider.maximumValue = parameter.range.upperBound
        slider.value = parameter.value
        
        // Update toast
        updateToastLabel()
        
        // Show with animation
        UIView.animate(withDuration: 0.15) {
            self.toastView.alpha = 1
            self.slider.alpha = 1
        }
        
        resetHideTimer()
    }
    
    @objc private func sliderValueChanged() {
        guard let parameter = currentParameter else { return }
        
        // Snap to step
        let step = parameter.step
        let roundedValue = round(slider.value / step) * step
        slider.value = roundedValue
        
        // Update parameter
        let newParameter: Parameter
        switch parameter {
        case .itemsMax:
            newParameter = .itemsMax(Int(roundedValue))
        case .confidence:
            newParameter = .confidence(roundedValue)
        case .iou:
            newParameter = .iou(roundedValue)
        case .lineThickness:
            newParameter = .lineThickness(roundedValue)
        }
        
        currentParameter = newParameter
        updateToastLabel()
        onValueChange?(newParameter)
        
        resetHideTimer()
    }
    
    private func updateToastLabel() {
        guard let parameter = currentParameter else { return }
        
        let valueText: String
        switch parameter {
        case .itemsMax(let v):
            valueText = "\(parameter.title): \(v)"
        case .confidence(let v), .iou(let v):
            valueText = String(format: "%@: %.2f", parameter.title, v)
        case .lineThickness(let v):
            valueText = String(format: "%@: %.1f", parameter.title, v)
        }
        
        toastLabel.text = valueText
    }
    
    private func resetHideTimer() {
        hideTimer?.invalidate()
        hideTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { _ in
            self.hide()
        }
    }
    
    func hide() {
        hideTimer?.invalidate()
        
        UIView.animate(withDuration: 0.3) {
            self.toastView.alpha = 0
            self.slider.alpha = 0
        }
    }
}
```

## 3. ViewControllerの統合

### 3.1 新しいViewController構造
```swift
class NewYOLOViewController: UIViewController {
    // UI Components
    private let statusBar = StatusMetricBar()
    private let cameraPreviewContainer = UIView()
    private var yoloView: YOLOView!
    private let rightToolBar = RightSideToolBar()
    private let taskTabStrip = TaskTabStrip()
    private let shutterBar = ShutterBar()
    private let parameterEditView = ParameterEditView()
    
    // Model Management
    private var currentModel: YOLO?
    private let modelManager = ModelManager()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        setupActions()
        loadInitialModel()
    }
    
    private func setupUI() {
        view.backgroundColor = .ultralyticsSurfaceDark
        
        // Camera Preview Setup
        cameraPreviewContainer.backgroundColor = .black
        cameraPreviewContainer.layer.cornerRadius = 18
        cameraPreviewContainer.clipsToBounds = true
        
        // Initialize YOLO View
        yoloView = YOLOView(frame: cameraPreviewContainer.bounds)
        yoloView.videoCapture.cameraPosition = .back
        cameraPreviewContainer.addSubview(yoloView)
        
        // Add all components
        [statusBar, cameraPreviewContainer, taskTabStrip, shutterBar, parameterEditView, rightToolBar].forEach {
            view.addSubview($0)
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
    }
    
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            // Status Bar
            statusBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            statusBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            statusBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            // Camera Preview (16:9 aspect ratio)
            cameraPreviewContainer.topAnchor.constraint(equalTo: statusBar.bottomAnchor, constant: 8),
            cameraPreviewContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 6),
            cameraPreviewContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -6),
            cameraPreviewContainer.heightAnchor.constraint(equalTo: cameraPreviewContainer.widthAnchor, multiplier: 9.0/16.0),
            
            // Task Tab Strip
            taskTabStrip.topAnchor.constraint(equalTo: cameraPreviewContainer.bottomAnchor),
            taskTabStrip.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            taskTabStrip.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            // Shutter Bar
            shutterBar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            shutterBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            shutterBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            // Right Tool Bar
            rightToolBar.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor, constant: -12),
            rightToolBar.centerYAnchor.constraint(equalTo: cameraPreviewContainer.centerYAnchor),
            
            // Parameter Edit View (overlay)
            parameterEditView.topAnchor.constraint(equalTo: cameraPreviewContainer.topAnchor),
            parameterEditView.leadingAnchor.constraint(equalTo: cameraPreviewContainer.leadingAnchor),
            parameterEditView.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor),
            parameterEditView.bottomAnchor.constraint(equalTo: taskTabStrip.topAnchor)
        ])
    }
}
```

## 4. 移行時の注意点

### 4.1 既存機能の保持
- YOLOViewのデリゲートメソッドは引き続き使用
- モデルのロード/切り替えロジックは再利用
- パフォーマンス計測ロジックは維持

### 4.2 段階的な実装
1. まず新しいViewControllerを別ファイルで作成
2. 設定で新旧UIを切り替え可能にする
3. 十分なテスト後に完全移行

### 4.3 パフォーマンス最適化
- レイヤーのラスタライズを適切に設定
- 不要な再描画を避ける
- アニメーションはCALayerで実装

これらのコードサンプルを基に、段階的に新しいUIを実装していくことができます。