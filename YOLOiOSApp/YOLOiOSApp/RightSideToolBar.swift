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
            case .itemsMax: return "square.stack"
            case .confidence: return "chart.dots.scatter"
            case .iou: return "intersect.circle"
            case .lineThickness: return "pencil.line"
            }
        }
        
        var isTextButton: Bool {
            return self == .zoom
        }
    }
    
    private var buttons: [UIButton] = []
    private var activeTool: Tool?
    private var isZoomed = false
    
    var onToolSelected: ((Tool) -> Void)?
    var onZoomToggle: ((Bool) -> Void)?
    
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
        button.backgroundColor = .ultralyticsBrown
        button.layer.cornerRadius = 20
        button.clipsToBounds = true
        
        if tool.isTextButton {
            button.setTitle(tool.icon, for: .normal)
            button.titleLabel?.font = Typography.labelFont
            button.setTitleColor(.white, for: .normal)
        } else {
            button.setImage(UIImage(systemName: tool.icon), for: .normal)
            button.tintColor = .white
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
            // Toggle zoom
            isZoomed.toggle()
            sender.setTitle(isZoomed ? "1.8x" : "1.0x", for: .normal)
            sender.setTitleColor(isZoomed ? .ultralyticsLime : .white, for: .normal)
            onZoomToggle?(isZoomed)
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
            button.tintColor = .black
        } else {
            button.backgroundColor = .ultralyticsBrown
            button.tintColor = .white
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
}