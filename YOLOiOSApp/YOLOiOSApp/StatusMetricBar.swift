// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class StatusMetricBar: UIView, UIGestureRecognizerDelegate {
    // UI Elements
    private let logoImageView = UIImageView()
    private let modelButton = UIButton(type: .system)
    private let sizeLabel = UILabel()
    private let fpsLabel = UILabel()
    private let latencyLabel = UILabel()
    private let stackView = UIStackView()
    
    // Subtitle labels
    private let modelSubtitleLabel = UILabel()
    private let modelDropdownIcon = UILabel()
    private let sizeSubtitleLabel = UILabel()
    private let sizeDropdownIcon = UILabel()
    private let fpsSubtitleLabel = UILabel()
    private let latencySubtitleLabel = UILabel()
    
    // Container views
    private let logoContainer = UIView()
    private let modelContainer = UIView()
    private let sizeContainer = UIView()
    private let fpsContainer = UIView()
    private let latencyContainer = UIView()
    
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
        logoImageView.image = UIImage(named: "ultralytics_icon") // Using existing asset
        logoImageView.contentMode = .scaleAspectFit
        
        // Model Button
        modelButton.setTitle("YOLO11", for: .normal)
        modelButton.setTitleColor(.ultralyticsTextPrimary, for: .normal)
        modelButton.titleLabel?.font = Typography.statusBarFont
        modelButton.contentHorizontalAlignment = .center
        modelButton.isUserInteractionEnabled = false  // Disable button to let container handle tap
        
        // Labels
        [sizeLabel, fpsLabel, latencyLabel].forEach { label in
            label.textColor = .ultralyticsTextPrimary
            label.font = Typography.statusBarFont
            label.textAlignment = .center
        }
        
        sizeLabel.text = "SMALL"
        fpsLabel.text = "0.0"
        latencyLabel.text = "0.0"
        
        // Configure subtitle labels
        [modelSubtitleLabel, sizeSubtitleLabel, fpsSubtitleLabel, latencySubtitleLabel].forEach { label in
            label.textColor = .ultralyticsTextSubtle
            label.font = UIFont.systemFont(ofSize: 9, weight: .regular)
            label.textAlignment = .center
        }
        
        modelSubtitleLabel.text = "MODEL"
        sizeSubtitleLabel.text = "SIZE"
        fpsSubtitleLabel.text = "FPS"
        latencySubtitleLabel.text = "MS"
        
        // Configure dropdown icons
        [modelDropdownIcon, sizeDropdownIcon].forEach { label in
            label.text = "⌄"
            label.textColor = .ultralyticsTextSubtle
            label.font = UIFont.systemFont(ofSize: 8, weight: .regular)  // Smaller font for dropdown icon
            label.textAlignment = .center
        }
        
        // Stack View
        stackView.axis = .horizontal
        stackView.spacing = 8  // Small spacing between elements
        stackView.alignment = .center
        stackView.distribution = .fillEqually
        stackView.isUserInteractionEnabled = true
        
        // Use the instance container views (already declared as properties)
        
        // Ensure containers don't block touch events
        [logoContainer, modelContainer, sizeContainer, fpsContainer, latencyContainer].forEach {
            $0.isUserInteractionEnabled = true
            stackView.addArrangedSubview($0)
        }
        
        // Create vertical stacks for model and size
        let modelStack = UIStackView()
        modelStack.axis = .vertical
        modelStack.spacing = -2  // Negative spacing for tighter layout
        modelStack.alignment = .center
        modelStack.distribution = .fill
        modelStack.isUserInteractionEnabled = true
        
        let modelSubtitleStack = UIStackView()
        modelSubtitleStack.axis = .vertical
        modelSubtitleStack.spacing = -2  // Negative spacing to bring elements closer
        modelSubtitleStack.alignment = .center
        modelSubtitleStack.addArrangedSubview(modelSubtitleLabel)
        modelSubtitleStack.addArrangedSubview(modelDropdownIcon)
        
        modelStack.addArrangedSubview(modelButton)
        modelStack.addArrangedSubview(modelSubtitleStack)
        
        // Remove the tap gesture from model stack (we'll add it to the container instead)
        
        let sizeStack = UIStackView()
        sizeStack.axis = .vertical
        sizeStack.spacing = -2  // Negative spacing for tighter layout
        sizeStack.alignment = .center
        sizeStack.distribution = .fill
        
        sizeStack.addArrangedSubview(sizeLabel)
        sizeStack.addArrangedSubview(sizeSubtitleLabel)
        
        let fpsStack = UIStackView()
        fpsStack.axis = .vertical
        fpsStack.spacing = -2  // Negative spacing for tighter layout
        fpsStack.alignment = .center
        fpsStack.distribution = .fill
        fpsStack.addArrangedSubview(fpsLabel)
        fpsStack.addArrangedSubview(fpsSubtitleLabel)
        
        let latencyStack = UIStackView()
        latencyStack.axis = .vertical
        latencyStack.spacing = -2  // Negative spacing for tighter layout
        latencyStack.alignment = .center
        latencyStack.distribution = .fill
        latencyStack.addArrangedSubview(latencyLabel)
        latencyStack.addArrangedSubview(latencySubtitleLabel)
        
        // Add elements to their containers
        logoContainer.addSubview(logoImageView)
        modelContainer.addSubview(modelStack)
        sizeContainer.addSubview(sizeStack)
        fpsContainer.addSubview(fpsStack)
        latencyContainer.addSubview(latencyStack)
        
        // Disable user interaction on child views to ensure container gets the tap
        modelStack.isUserInteractionEnabled = false
        modelButton.isUserInteractionEnabled = false
        modelSubtitleStack.isUserInteractionEnabled = false
        modelSubtitleLabel.isUserInteractionEnabled = false
        modelDropdownIcon.isUserInteractionEnabled = false
        print("StatusMetricBar: Disabled user interaction on child views to ensure container gets tap")
        
        // Add tap gesture to entire model container
        let modelTapGesture = UITapGestureRecognizer(target: self, action: #selector(modelButtonTapped))
        modelTapGesture.delegate = self
        modelContainer.addGestureRecognizer(modelTapGesture)
        modelContainer.isUserInteractionEnabled = true
        
        print("StatusMetricBar: Added tap gesture to modelContainer")
        print("StatusMetricBar: modelContainer.isUserInteractionEnabled = \(modelContainer.isUserInteractionEnabled)")
        
        // Center elements in their containers
        logoImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            logoImageView.centerXAnchor.constraint(equalTo: logoContainer.centerXAnchor),
            logoImageView.centerYAnchor.constraint(equalTo: logoContainer.centerYAnchor),
            logoImageView.widthAnchor.constraint(equalToConstant: 20),
            logoImageView.heightAnchor.constraint(equalToConstant: 20)
        ])
        
        modelStack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            modelStack.centerXAnchor.constraint(equalTo: modelContainer.centerXAnchor),
            // Remove centerY constraint to allow manual positioning
            modelStack.topAnchor.constraint(equalTo: modelContainer.topAnchor),
            // Align model button baseline with size label baseline
            modelButton.firstBaselineAnchor.constraint(equalTo: sizeLabel.firstBaselineAnchor)
        ])
        
        sizeStack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            sizeStack.centerXAnchor.constraint(equalTo: sizeContainer.centerXAnchor),
            sizeStack.centerYAnchor.constraint(equalTo: sizeContainer.centerYAnchor)
        ])
        
        fpsStack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            fpsStack.centerXAnchor.constraint(equalTo: fpsContainer.centerXAnchor),
            fpsStack.centerYAnchor.constraint(equalTo: fpsContainer.centerYAnchor)
        ])
        
        latencyStack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            latencyStack.centerXAnchor.constraint(equalTo: latencyContainer.centerXAnchor),
            latencyStack.centerYAnchor.constraint(equalTo: latencyContainer.centerYAnchor)
        ])
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor),
            heightAnchor.constraint(equalToConstant: 44),
            
            // Set fixed heights for main labels to ensure alignment
            modelButton.heightAnchor.constraint(equalToConstant: 16),
            sizeLabel.heightAnchor.constraint(equalToConstant: 16),
            fpsLabel.heightAnchor.constraint(equalToConstant: 16),
            latencyLabel.heightAnchor.constraint(equalToConstant: 16),
            
            // Set fixed heights for subtitle labels
            modelSubtitleLabel.heightAnchor.constraint(equalToConstant: 10),
            sizeSubtitleLabel.heightAnchor.constraint(equalToConstant: 10),
            fpsSubtitleLabel.heightAnchor.constraint(equalToConstant: 10),
            latencySubtitleLabel.heightAnchor.constraint(equalToConstant: 10),
            
            // Set fixed heights for dropdown icons
            modelDropdownIcon.heightAnchor.constraint(equalToConstant: 8),
            sizeDropdownIcon.heightAnchor.constraint(equalToConstant: 8)
        ])
        
        // Long press gesture for hidden info
        let longPress = UILongPressGestureRecognizer(target: self, action: #selector(logoLongPressed))
        longPress.minimumPressDuration = 1.0
        logoImageView.isUserInteractionEnabled = true
        logoImageView.addGestureRecognizer(longPress)
        
        // Add a test tap to the entire StatusMetricBar
        let testTap = UITapGestureRecognizer(target: self, action: #selector(testTapped))
        self.addGestureRecognizer(testTap)
        self.isUserInteractionEnabled = true
        print("StatusMetricBar: Added test tap gesture to self")
    }
    
    @objc private func modelButtonTapped() {
        print("StatusMetricBar: modelButtonTapped called")
        print("StatusMetricBar: onModelTap is \(onModelTap != nil ? "set" : "nil")")
        onModelTap?()
    }
    
    @objc private func logoLongPressed(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            // Show hidden info page
            NotificationCenter.default.post(name: .showHiddenInfo, object: nil)
        }
    }
    
    @objc private func testTapped(_ gesture: UITapGestureRecognizer) {
        let location = gesture.location(in: self)
        print("StatusMetricBar: TEST TAP detected at \(location)")
        print("StatusMetricBar: Frame = \(frame)")
        print("StatusMetricBar: Bounds = \(bounds)")
        print("StatusMetricBar: modelContainer frame = \(modelContainer.frame)")
        
        // Check if tap is in the horizontal range of modelContainer
        // Since height is 0, we check X position and use parent's height
        let modelContainerX = modelContainer.frame.minX
        let modelContainerMaxX = modelContainer.frame.maxX
        
        print("StatusMetricBar: Checking if tap X \(location.x) is between \(modelContainerX) and \(modelContainerMaxX)")
        
        if location.x >= modelContainerX && location.x <= modelContainerMaxX {
            print("StatusMetricBar: Tap is in modelContainer X range - triggering model tap")
            modelButtonTapped()
        }
    }
    
    func updateMetrics(fps: Double, latency: Double) {
        fpsLabel.text = String(format: "%.1f", fps)
        latencyLabel.text = String(format: "%.1f", latency)
    }
    
    func updateModel(name: String, size: String) {
        modelButton.setTitle(name, for: .normal)
        sizeLabel.text = size.uppercased()
    }
    
    // MARK: - Layout
    
    override func layoutSubviews() {
        super.layoutSubviews()
        print("StatusMetricBar: layoutSubviews called")
        print("StatusMetricBar: self.frame = \(frame)")
        print("StatusMetricBar: modelContainer.frame = \(modelContainer.frame)")
        print("StatusMetricBar: isUserInteractionEnabled = \(isUserInteractionEnabled)")
        print("StatusMetricBar: modelContainer.isUserInteractionEnabled = \(modelContainer.isUserInteractionEnabled)")
    }
    
    // MARK: - Hit Testing
    
    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        let view = super.hitTest(point, with: event)
        if modelContainer.frame.contains(point) {
            print("StatusMetricBar: Hit test - point \(point) is in modelContainer")
        }
        return view
    }
    
    // MARK: - UIGestureRecognizerDelegate
    
    override func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
        print("StatusMetricBar: gestureRecognizerShouldBegin called")
        let location = gestureRecognizer.location(in: self)
        print("StatusMetricBar: Tap location: \(location)")
        print("StatusMetricBar: modelContainer frame: \(modelContainer.frame)")
        return true
    }
    
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        print("StatusMetricBar: shouldReceive touch called")
        let location = touch.location(in: self)
        print("StatusMetricBar: Touch location in StatusMetricBar: \(location)")
        print("StatusMetricBar: Touch view: \(touch.view)")
        return true
    }
}

extension Notification.Name {
    static let showHiddenInfo = Notification.Name("showHiddenInfo")
}
