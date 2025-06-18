// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit
import YOLO

class TaskTabStrip: UIView {
    enum Task: String, CaseIterable {
        case detect = "DETECT"
        case segment = "SEGMENT"
        case classify = "CLASS"
        
        var yoloTask: YOLOTask {
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
    
    override func layoutSubviews() {
        super.layoutSubviews()
        updateSelection()
    }
}

extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}