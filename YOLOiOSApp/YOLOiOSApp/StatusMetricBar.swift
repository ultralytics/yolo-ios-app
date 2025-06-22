// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class StatusMetricBar: UIView {
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
        modelButton.addTarget(self, action: #selector(modelButtonTapped), for: .touchUpInside)
        modelButton.isUserInteractionEnabled = true
        
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
            label.font = UIFont.systemFont(ofSize: 10, weight: .regular)
            label.textAlignment = .center
        }
        
        modelSubtitleLabel.text = "MODEL"
        sizeSubtitleLabel.text = "SIZE"
        fpsSubtitleLabel.text = "FPS"
        latencySubtitleLabel.text = "MS"
        
        // Configure dropdown icons
        [modelDropdownIcon, sizeDropdownIcon].forEach { label in
            label.text = "▼"
            label.textColor = .ultralyticsTextSubtle
            label.font = UIFont.systemFont(ofSize: 8, weight: .regular)
            label.textAlignment = .center
        }
        
        // Stack View
        stackView.axis = .horizontal
        stackView.spacing = 0
        stackView.alignment = .center
        stackView.distribution = .fillEqually
        stackView.isUserInteractionEnabled = true
        
        // Create container views for each element to ensure equal spacing
        let logoContainer = UIView()
        let modelContainer = UIView()
        let sizeContainer = UIView()
        let fpsContainer = UIView()
        let latencyContainer = UIView()
        
        // Ensure containers don't block touch events
        [logoContainer, modelContainer, sizeContainer, fpsContainer, latencyContainer].forEach {
            $0.isUserInteractionEnabled = true
            stackView.addArrangedSubview($0)
        }
        
        // Create vertical stacks for model and size
        let modelStack = UIStackView()
        modelStack.axis = .vertical
        modelStack.spacing = 2
        modelStack.alignment = .center
        modelStack.isUserInteractionEnabled = true
        
        let modelSubtitleStack = UIStackView()
        modelSubtitleStack.axis = .horizontal
        modelSubtitleStack.spacing = 2
        modelSubtitleStack.alignment = .center
        modelSubtitleStack.addArrangedSubview(modelSubtitleLabel)
        modelSubtitleStack.addArrangedSubview(modelDropdownIcon)
        
        modelStack.addArrangedSubview(modelButton)
        modelStack.addArrangedSubview(modelSubtitleStack)
        
        let sizeStack = UIStackView()
        sizeStack.axis = .vertical
        sizeStack.spacing = 2
        sizeStack.alignment = .center
        
        let sizeSubtitleStack = UIStackView()
        sizeSubtitleStack.axis = .horizontal
        sizeSubtitleStack.spacing = 2
        sizeSubtitleStack.alignment = .center
        sizeSubtitleStack.addArrangedSubview(sizeSubtitleLabel)
        sizeSubtitleStack.addArrangedSubview(sizeDropdownIcon)
        
        sizeStack.addArrangedSubview(sizeLabel)
        sizeStack.addArrangedSubview(sizeSubtitleStack)
        
        let fpsStack = UIStackView()
        fpsStack.axis = .vertical
        fpsStack.spacing = 2
        fpsStack.alignment = .center
        fpsStack.addArrangedSubview(fpsLabel)
        fpsStack.addArrangedSubview(fpsSubtitleLabel)
        
        let latencyStack = UIStackView()
        latencyStack.axis = .vertical
        latencyStack.spacing = 2
        latencyStack.alignment = .center
        latencyStack.addArrangedSubview(latencyLabel)
        latencyStack.addArrangedSubview(latencySubtitleLabel)
        
        // Add elements to their containers
        logoContainer.addSubview(logoImageView)
        modelContainer.addSubview(modelStack)
        sizeContainer.addSubview(sizeStack)
        fpsContainer.addSubview(fpsStack)
        latencyContainer.addSubview(latencyStack)
        
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
            modelStack.centerYAnchor.constraint(equalTo: modelContainer.centerYAnchor)
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
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor),
            heightAnchor.constraint(equalToConstant: 44)
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
        fpsLabel.text = String(format: "%.1f", fps)
        latencyLabel.text = String(format: "%.1f", latency)
    }
    
    func updateModel(name: String, size: String) {
        modelButton.setTitle(name, for: .normal)
        sizeLabel.text = size.uppercased()
    }
}

extension Notification.Name {
    static let showHiddenInfo = Notification.Name("showHiddenInfo")
}