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
        modelButton.setTitle("YOLO11 ▼", for: .normal)
        modelButton.setTitleColor(.ultralyticsTextPrimary, for: .normal)
        modelButton.titleLabel?.font = Typography.statusBarFont
        modelButton.addTarget(self, action: #selector(modelButtonTapped), for: .touchUpInside)
        modelButton.isUserInteractionEnabled = true
        
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
        
        // Add elements to their containers
        logoContainer.addSubview(logoImageView)
        modelContainer.addSubview(modelButton)
        sizeContainer.addSubview(sizeLabel)
        fpsContainer.addSubview(fpsLabel)
        latencyContainer.addSubview(latencyLabel)
        
        // Center elements in their containers
        logoImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            logoImageView.centerXAnchor.constraint(equalTo: logoContainer.centerXAnchor),
            logoImageView.centerYAnchor.constraint(equalTo: logoContainer.centerYAnchor),
            logoImageView.widthAnchor.constraint(equalToConstant: 20),
            logoImageView.heightAnchor.constraint(equalToConstant: 20)
        ])
        
        modelButton.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            modelButton.leadingAnchor.constraint(equalTo: modelContainer.leadingAnchor),
            modelButton.trailingAnchor.constraint(equalTo: modelContainer.trailingAnchor),
            modelButton.topAnchor.constraint(equalTo: modelContainer.topAnchor),
            modelButton.bottomAnchor.constraint(equalTo: modelContainer.bottomAnchor)
        ])
        
        sizeLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            sizeLabel.centerXAnchor.constraint(equalTo: sizeContainer.centerXAnchor),
            sizeLabel.centerYAnchor.constraint(equalTo: sizeContainer.centerYAnchor)
        ])
        
        fpsLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            fpsLabel.centerXAnchor.constraint(equalTo: fpsContainer.centerXAnchor),
            fpsLabel.centerYAnchor.constraint(equalTo: fpsContainer.centerYAnchor)
        ])
        
        latencyLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            latencyLabel.centerXAnchor.constraint(equalTo: latencyContainer.centerXAnchor),
            latencyLabel.centerYAnchor.constraint(equalTo: latencyContainer.centerYAnchor)
        ])
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor),
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