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
        logoImageView.translatesAutoresizingMaskIntoConstraints = false
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