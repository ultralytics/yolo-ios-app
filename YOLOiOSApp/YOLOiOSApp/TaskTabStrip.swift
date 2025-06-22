// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit
import YOLO

class TaskTabStrip: UIView {
    enum Task: String, CaseIterable {
        case detect = "DETECT"
        case segment = "SEGMENT"
        case classify = "CLASSIFY"
        case pose = "POSE"
        case obb = "OBB"
        
        var yoloTask: YOLOTask {
            switch self {
            case .detect: return .detect
            case .segment: return .segment
            case .classify: return .classify
            case .pose: return .pose
            case .obb: return .obb
            }
        }
    }
    
    private let scrollView = UIScrollView()
    private let contentView = UIView()
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
        
        // Configure scroll view
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.bounces = false
        
        // Create buttons
        let stackView = UIStackView()
        stackView.axis = .horizontal
        stackView.distribution = .fillProportionally
        stackView.spacing = 8
        
        for task in Task.allCases {
            let button = UIButton(type: .system)
            button.setTitle(task.rawValue, for: .normal)
            button.titleLabel?.font = Typography.tabLabelFont
            button.tag = Task.allCases.firstIndex(of: task)!
            button.addTarget(self, action: #selector(taskButtonTapped), for: .touchUpInside)
            button.contentEdgeInsets = UIEdgeInsets(top: 0, left: 12, bottom: 0, right: 12)
            buttons.append(button)
            stackView.addArrangedSubview(button)
        }
        
        // Underline
        underlineView.backgroundColor = .ultralyticsLime
        
        // Add views
        addSubview(scrollView)
        scrollView.addSubview(contentView)
        contentView.addSubview(stackView)
        contentView.addSubview(underlineView)
        
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        contentView.translatesAutoresizingMaskIntoConstraints = false
        stackView.translatesAutoresizingMaskIntoConstraints = false
        underlineView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Scroll view constraints
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor),
            scrollView.topAnchor.constraint(equalTo: topAnchor),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor),
            heightAnchor.constraint(equalToConstant: 28),
            
            // Content view constraints
            contentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
            contentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
            contentView.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor),
            contentView.heightAnchor.constraint(equalTo: scrollView.heightAnchor),
            
            // Stack view constraints
            stackView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            stackView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            stackView.topAnchor.constraint(equalTo: contentView.topAnchor),
            stackView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            
            // Underline constraints
            underlineView.heightAnchor.constraint(equalToConstant: 2),
            underlineView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor)
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
        
        // Convert button frame to content view coordinates
        let buttonFrame = selectedButton.convert(selectedButton.bounds, to: contentView)
        
        UIView.animate(withDuration: 0.15, delay: 0, options: .curveEaseOut) {
            self.underlineView.frame = CGRect(
                x: buttonFrame.minX,
                y: self.contentView.bounds.height - 2,
                width: buttonFrame.width,
                height: 2
            )
        }
        
        // Scroll to center the selected button
        DispatchQueue.main.async {
            // Calculate the center position of the button
            let buttonCenter = buttonFrame.midX
            
            // Calculate the target offset to center the button
            let scrollViewWidth = self.scrollView.bounds.width
            let contentWidth = self.contentView.bounds.width
            let targetOffset = buttonCenter - (scrollViewWidth / 2)
            
            // Clamp the offset to valid range
            let minOffset: CGFloat = 0
            let maxOffset = max(0, contentWidth - scrollViewWidth)
            let clampedOffset = min(max(targetOffset, minOffset), maxOffset)
            
            // Animate scroll to center
            UIView.animate(withDuration: 0.3, delay: 0, options: .curveEaseInOut) {
                self.scrollView.contentOffset = CGPoint(x: clampedOffset, y: 0)
            }
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        // Wait for layout to complete before updating selection
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.updateSelection()
        }
    }
}

extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}