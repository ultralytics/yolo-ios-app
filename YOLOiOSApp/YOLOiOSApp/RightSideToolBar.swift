// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class RightSideToolBar: UIView {
    enum Tool: Int, CaseIterable {
        case zoom = 0
        case itemsMax = 1
        case confidence = 2
        case iou = 3
        case lineThickness = 4
        
        var icon: String {
            switch self {
            case .zoom: return "1.0x" // Will be updated dynamically
            case .itemsMax: return "square.stack.3d.up"
            case .confidence: return "target"
            case .iou: return "square.on.square"
            case .lineThickness: return "pencil.line"
            }
        }
        
        var usesCustomIcon: Bool {
            return false // All use system icons now
        }
        
        var isTextButton: Bool {
            return self == .zoom
        }
    }
    
    private var buttons: [UIButton] = []
    private var activeTool: Tool?
    private var currentZoomLevel: Float = 1.0
    
    var onToolSelected: ((Tool) -> Void)?
    var onZoomChanged: ((Float) -> Void)?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        backgroundColor = .clear
        
        let stackView = UIStackView()
        stackView.axis = .vertical
        stackView.spacing = 12
        stackView.distribution = .fillEqually
        
        for tool in Tool.allCases {
            let button = createToolButton(for: tool)
            buttons.append(button)
            stackView.addArrangedSubview(button)
        }
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: topAnchor),
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor),
            stackView.bottomAnchor.constraint(equalTo: bottomAnchor),
            widthAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    private func createToolButton(for tool: Tool) -> UIButton {
        let button = UIButton(type: .custom)
        button.tag = tool.rawValue
        button.backgroundColor = UIColor.black.withAlphaComponent(0.3)
        button.layer.cornerRadius = 20
        button.layer.borderWidth = 0
        
        if tool.isTextButton {
            button.setTitle(tool.icon, for: .normal)
            button.titleLabel?.font = UIFont.systemFont(ofSize: 14, weight: .semibold)
            button.setTitleColor(.white, for: .normal)
        } else {
            // Use system icon
            button.setImage(UIImage(systemName: tool.icon), for: .normal)
            button.tintColor = .white
            button.imageView?.contentMode = .scaleAspectFit
        }
        
        button.addTarget(self, action: #selector(toolButtonTapped), for: .touchUpInside)
        
        NSLayoutConstraint.activate([
            button.widthAnchor.constraint(equalToConstant: 40),
            button.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        return button
    }
    
    @objc private func toolButtonTapped(_ sender: UIButton) {
        guard let tool = Tool(rawValue: sender.tag) else { return }
        
        if tool == .zoom {
            // Cycle through zoom levels: 0.5x -> 1.0x -> 3.0x -> 0.5x
            let nextZoom: Float
            if currentZoomLevel < 0.75 {
                nextZoom = 1.0
            } else if currentZoomLevel < 2.0 {
                nextZoom = 3.0
            } else {
                nextZoom = 0.5
            }
            
            currentZoomLevel = nextZoom
            sender.setTitle(String(format: "%.1fx", nextZoom), for: .normal)
            
            // Change color based on zoom level
            let isDefaultZoom = abs(nextZoom - 1.0) < 0.1
            sender.setTitleColor(isDefaultZoom ? .white : .ultralyticsLime, for: .normal)
            
            onZoomChanged?(nextZoom)
        } else {
            // Handle parameter tools
            if activeTool == tool {
                // Deactivate
                setButtonActive(sender, active: false)
                activeTool = nil
            } else {
                // Deactivate previous
                if let previousTool = activeTool,
                   let previousButton = buttons[safe: previousTool.rawValue] {
                    setButtonActive(previousButton, active: false)
                }
                
                // Activate new
                setButtonActive(sender, active: true)
                activeTool = tool
                onToolSelected?(tool)
            }
        }
    }
    
    private func setButtonActive(_ button: UIButton, active: Bool) {
        if active {
            button.backgroundColor = .ultralyticsLime
            button.layer.borderWidth = 0
            button.tintColor = .black
            button.setTitleColor(.black, for: .normal)
        } else {
            button.backgroundColor = UIColor.black.withAlphaComponent(0.3)
            button.layer.borderWidth = 0
            button.tintColor = .white
            button.setTitleColor(.white, for: .normal)
        }
    }
    
    func deactivateAll() {
        for (index, button) in buttons.enumerated() {
            if index != Tool.zoom.rawValue {
                setButtonActive(button, active: false)
            }
        }
        activeTool = nil
    }
    
    func updateZoomLevel(_ level: Float) {
        currentZoomLevel = level
        if let zoomButton = buttons[safe: Tool.zoom.rawValue] {
            zoomButton.setTitle(String(format: "%.1fx", level), for: .normal)
            let isDefaultZoom = abs(level - 1.0) < 0.1
            zoomButton.setTitleColor(isDefaultZoom ? .white : .ultralyticsLime, for: .normal)
        }
    }
}