// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

/// A horizontal filter bar for selecting model sizes
class ModelSizeFilterBar: UIView {
  
  // MARK: - Properties
  
  /// Available model sizes
  enum ModelSize: String, CaseIterable {
    case nano = "n"
    case small = "s"
    case medium = "m"
    case large = "l"
    case xlarge = "x"
    
    var displayName: String {
      switch self {
      case .nano: return "NANO"
      case .small: return "SMALL"
      case .medium: return "MEDIUM"
      case .large: return "LARGE"
      case .xlarge: return "XLARGE"
      }
    }
  }
  
  /// Currently selected size
  private(set) var selectedSize: ModelSize = .small
  
  /// Callback when size selection changes
  var onSizeSelected: ((ModelSize) -> Void)?
  
  /// Scroll view for horizontal scrolling
  private let scrollView = UIScrollView()
  
  /// Stack view containing size buttons
  private let stackView: UIStackView = {
    let stack = UIStackView()
    stack.axis = .horizontal
    stack.distribution = .equalSpacing
    stack.alignment = .center
    stack.spacing = 4
    stack.translatesAutoresizingMaskIntoConstraints = false
    return stack
  }()
  
  
  /// Collection of size buttons
  private var buttons: [UIButton] = []
  
  
  // MARK: - Initialization
  
  override init(frame: CGRect) {
    super.init(frame: frame)
    setupUI()
  }
  
  required init?(coder: NSCoder) {
    super.init(coder: coder)
    setupUI()
  }
  
  // MARK: - UI Setup
  
  private func setupUI() {
    // Configure self
    backgroundColor = .ultralyticsSurfaceDark
    translatesAutoresizingMaskIntoConstraints = false
    
    // Configure scroll view
    scrollView.showsHorizontalScrollIndicator = false
    scrollView.showsVerticalScrollIndicator = false
    scrollView.bounces = false
    scrollView.translatesAutoresizingMaskIntoConstraints = false
    
    // Create buttons for each size
    for size in ModelSize.allCases {
      let button = createSizeButton(for: size)
      buttons.append(button)
      stackView.addArrangedSubview(button)
    }
    
    // Add views
    addSubview(scrollView)
    scrollView.addSubview(stackView)
    
    // Setup constraints
    NSLayoutConstraint.activate([
      // Scroll view
      scrollView.topAnchor.constraint(equalTo: topAnchor),
      scrollView.leadingAnchor.constraint(equalTo: leadingAnchor),
      scrollView.trailingAnchor.constraint(equalTo: trailingAnchor),
      scrollView.bottomAnchor.constraint(equalTo: bottomAnchor),
      
      // Stack view - centered
      stackView.topAnchor.constraint(equalTo: scrollView.topAnchor),
      stackView.leadingAnchor.constraint(greaterThanOrEqualTo: scrollView.leadingAnchor, constant: 8),
      stackView.trailingAnchor.constraint(lessThanOrEqualTo: scrollView.trailingAnchor, constant: -8),
      stackView.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor),
      stackView.heightAnchor.constraint(equalTo: scrollView.heightAnchor),
      stackView.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor),
    ])
    
    // Set initial selection
    updateSelection()
  }
  
  private func createSizeButton(for size: ModelSize) -> UIButton {
    let button = UIButton(type: .system)
    button.setTitle(size.displayName, for: .normal)
    button.titleLabel?.font = Typography.tabLabelFont
    button.addTarget(self, action: #selector(sizeButtonTapped(_:)), for: .touchUpInside)
    button.tag = ModelSize.allCases.firstIndex(of: size) ?? 0
    button.contentEdgeInsets = UIEdgeInsets(top: 0, left: 8, bottom: 0, right: 8)
    return button
  }
  
  // MARK: - Actions
  
  @objc private func sizeButtonTapped(_ sender: UIButton) {
    guard sender.tag < ModelSize.allCases.count else { return }
    let size = ModelSize.allCases[sender.tag]
    
    // Update selection
    selectedSize = size
    updateSelection()
    
    // Notify delegate
    onSizeSelected?(size)
    
    // Haptic feedback
    let impact = UIImpactFeedbackGenerator(style: .light)
    impact.impactOccurred()
  }
  
  // MARK: - State Updates
  
  private func updateSelection() {
    // Update button colors
    for (index, button) in buttons.enumerated() {
      let isSelected = index == ModelSize.allCases.firstIndex(of: selectedSize)
      button.setTitleColor(isSelected ? .white : .ultralyticsTextSubtle, for: .normal)
    }
  }
  
  
  // MARK: - Public Methods
  
  /// Update the selected size programmatically
  func setSelectedSize(_ size: ModelSize, animated: Bool = true) {
    selectedSize = size
    updateSelection()
  }
  
  /// Show the filter bar with animation
  func show(completion: (() -> Void)? = nil) {
    UIView.animate(
      withDuration: 0.2,
      animations: {
        self.alpha = 1
      },
      completion: { _ in
        completion?()
      }
    )
  }
  
  /// Hide the filter bar with animation
  func hide(completion: (() -> Void)? = nil) {
    UIView.animate(
      withDuration: 0.2,
      animations: {
        self.alpha = 0
      },
      completion: { _ in
        completion?()
      }
    )
  }
}

