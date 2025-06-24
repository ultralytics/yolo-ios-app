// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class ThresholdSliderView: UIView {
    // MARK: - Properties
    
    private let overlayView = UIView()
    private let containerView = UIView()
    private let trackLayer = CALayer()
    private let ticksLayer = CAShapeLayer()
    private let selectionLineLayer = CALayer()  // Yellow center line
    private let leftGradientLayer = CAGradientLayer()
    private let rightGradientLayer = CAGradientLayer()
    private let parameterLabel = UILabel()
    
    private var value: Float = 0.5
    private var isDragging = false
    private var idleTimer: Timer?
    private let feedbackGenerator = UIImpactFeedbackGenerator(style: .light)
    
    var onValueChange: ((Float) -> Void)?
    var onHide: (() -> Void)?
    var parameter: ParameterEditView.Parameter?
    
    // Layout constants from spec
    private let containerHeight: CGFloat = 48
    private let horizontalPadding: CGFloat = 0  // Full width for drum style
    private let trackHeight: CGFloat = 48  // Full height
    private let selectionLineWidth: CGFloat = 4
    private let selectionLineHeight: CGFloat = 32
    
    // Colors from spec
    private let containerBackgroundColor = UIColor(red: 0.043, green: 0.059, blue: 0.082, alpha: 1.0) // #0B0F15
    private let trackColor = UIColor(red: 0.078, green: 0.102, blue: 0.137, alpha: 1.0) // #141A23
    private let thumbColor = UIColor(red: 0.757, green: 1.0, blue: 0.0, alpha: 1.0) // #C1FF00
    
    // MARK: - Initialization
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    // MARK: - Setup
    
    private func setupUI() {
        isUserInteractionEnabled = false
        
        // Overlay for tap to dismiss
        overlayView.backgroundColor = .clear
        overlayView.alpha = 0
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(overlayTapped))
        overlayView.addGestureRecognizer(tapGesture)
        addSubview(overlayView)
        
        // Container
        containerView.backgroundColor = containerBackgroundColor
        containerView.alpha = 0
        addSubview(containerView)
        
        // Parameter label
        parameterLabel.textColor = .white
        parameterLabel.font = UIFont.systemFont(ofSize: 16, weight: .semibold)
        parameterLabel.textAlignment = .center
        parameterLabel.alpha = 0
        addSubview(parameterLabel)
        
        // Track layer (background)
        trackLayer.backgroundColor = containerBackgroundColor.cgColor
        containerView.layer.addSublayer(trackLayer)
        
        // Ticks layer
        ticksLayer.strokeColor = UIColor.white.cgColor
        ticksLayer.fillColor = UIColor.clear.cgColor
        containerView.layer.addSublayer(ticksLayer)
        
        // Selection line (yellow center indicator)
        selectionLineLayer.backgroundColor = thumbColor.cgColor
        selectionLineLayer.cornerRadius = 2
        containerView.layer.addSublayer(selectionLineLayer)
        
        // Edge gradients for drum effect
        leftGradientLayer.colors = [
            containerBackgroundColor.cgColor,
            containerBackgroundColor.withAlphaComponent(0).cgColor
        ]
        leftGradientLayer.locations = [0.0, 1.0]
        leftGradientLayer.startPoint = CGPoint(x: 0, y: 0.5)
        leftGradientLayer.endPoint = CGPoint(x: 1, y: 0.5)
        containerView.layer.addSublayer(leftGradientLayer)
        
        rightGradientLayer.colors = [
            containerBackgroundColor.withAlphaComponent(0).cgColor,
            containerBackgroundColor.cgColor
        ]
        rightGradientLayer.locations = [0.0, 1.0]
        rightGradientLayer.startPoint = CGPoint(x: 0, y: 0.5)
        rightGradientLayer.endPoint = CGPoint(x: 1, y: 0.5)
        containerView.layer.addSublayer(rightGradientLayer)
        
        // Gesture recognizer
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        containerView.addGestureRecognizer(panGesture)
        
        // Layout
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        containerView.translatesAutoresizingMaskIntoConstraints = false
        parameterLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Overlay fills entire view
            overlayView.topAnchor.constraint(equalTo: topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: bottomAnchor),
            
            // Parameter label at top
            parameterLabel.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 60),
            parameterLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            parameterLabel.widthAnchor.constraint(equalTo: widthAnchor, constant: -48),
            
            // Container at task selection button position
            containerView.leadingAnchor.constraint(equalTo: leadingAnchor),
            containerView.trailingAnchor.constraint(equalTo: trailingAnchor),
            containerView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -140),
            containerView.heightAnchor.constraint(equalToConstant: containerHeight)
        ])
        
        feedbackGenerator.prepare()
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        
        let containerBounds = containerView.bounds
        
        // Update track (full size)
        trackLayer.frame = containerBounds
        
        // Update selection line (center position)
        selectionLineLayer.frame = CGRect(
            x: (containerBounds.width - selectionLineWidth) / 2,
            y: (containerHeight - selectionLineHeight) / 2,
            width: selectionLineWidth,
            height: selectionLineHeight
        )
        
        // Update edge gradients
        let gradientWidth: CGFloat = 60
        leftGradientLayer.frame = CGRect(
            x: 0,
            y: 0,
            width: gradientWidth,
            height: containerHeight
        )
        
        rightGradientLayer.frame = CGRect(
            x: containerBounds.width - gradientWidth,
            y: 0,
            width: gradientWidth,
            height: containerHeight
        )
        
        // Update ticks
        updateTickMarks()
    }
    
    // MARK: - Tick Marks
    
    private func updateTickMarks() {
        // Use single path for better performance
        let path = UIBezierPath()
        let containerWidth = containerView.bounds.width
        
        // Fixed spacing for 1% increments
        let tickSpacing: CGFloat = 8.0  // Space between each 1% tick
        let normalTickLength: CGFloat = 14.0  // Standard tick
        let mediumTickLength: CGFloat = 20.0  // Every 5th tick
        
        // Current position in 0-100 scale
        let currentPercent = value * 100
        
        // Calculate tick positions based on current value
        let pixelsPerPercent = tickSpacing
        let valueOffset = CGFloat(currentPercent) * pixelsPerPercent
        
        // Calculate starting position
        let centerX = containerWidth / 2
        let startOffset = fmod(valueOffset, tickSpacing)
        var x = centerX - startOffset
        var percentIndex = Int(currentPercent)
        
        // Draw ticks to the left of center
        while x >= 0 && percentIndex >= 0 {
            let isMediumTick = percentIndex % 5 == 0
            let tickLength = isMediumTick ? mediumTickLength : normalTickLength
            let tickY = (containerHeight - tickLength) / 2
            
            path.move(to: CGPoint(x: x, y: tickY))
            path.addLine(to: CGPoint(x: x, y: tickY + tickLength))
            
            x -= tickSpacing
            percentIndex -= 1
        }
        
        // Reset for right side
        x = centerX - startOffset + tickSpacing
        percentIndex = Int(currentPercent) + 1
        
        // Draw ticks to the right of center
        while x <= containerWidth && percentIndex <= 100 {
            let isMediumTick = percentIndex % 5 == 0
            let tickLength = isMediumTick ? mediumTickLength : normalTickLength
            let tickY = (containerHeight - tickLength) / 2
            
            path.move(to: CGPoint(x: x, y: tickY))
            path.addLine(to: CGPoint(x: x, y: tickY + tickLength))
            
            x += tickSpacing
            percentIndex += 1
        }
        
        // Update layer with single path
        ticksLayer.path = path.cgPath
        ticksLayer.strokeColor = UIColor.white.cgColor
        ticksLayer.lineWidth = 1
    }
    
    // MARK: - Value Update
    
    private func updateValue(_ newValue: Float) {
        value = newValue
        updateTickMarks()  // Redraw ticks for new position
        updateParameterLabel()
    }
    
    // MARK: - Gesture Handling
    
    @objc private func overlayTapped() {
        hide()
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let parameter = parameter else { return }
        
        let translation = gesture.translation(in: containerView)
        let containerWidth = containerView.bounds.width
        
        switch gesture.state {
        case .began:
            isDragging = true
            resetIdleTimer()
            
        case .changed:
            // Calculate sensitivity based on parameter type
            let sensitivity: Float
            switch parameter {
            case .itemsMax:
                sensitivity = 1.5  // Less sensitive for discrete values
            case .confidence, .iou:
                sensitivity = 2.0  // Medium sensitivity
            case .lineThickness:
                sensitivity = 1.8  // Slightly less sensitive
            }
            
            // Calculate value change based on horizontal drag
            let deltaValue = Float(-translation.x / containerWidth) * sensitivity
            let newValue = value + deltaValue
            let clampedValue = max(0, min(1, newValue))
            
            // Update without snapping for smooth movement
            let oldValue = value
            updateValue(clampedValue)
            onValueChange?(clampedValue)
            
            // Haptic feedback every 1% (every tick)
            let oldPercent = Int(oldValue * 100)
            let newPercent = Int(clampedValue * 100)
            if oldPercent != newPercent {
                feedbackGenerator.impactOccurred()
            }
            
            // Reset translation
            gesture.setTranslation(.zero, in: containerView)
            
        case .ended, .cancelled:
            isDragging = false
            // Snap based on parameter type
            snapToNearestValue()
            
        default:
            break
        }
    }
    
    private func snapToNearestValue() {
        // Snap to nearest 1%
        let snappedValue = round(value * 100) / 100
        
        if snappedValue != value {
            updateValue(snappedValue)
            onValueChange?(snappedValue)
        }
    }
    
    // MARK: - Animations
    
    private func animateFocusGlow(_ focused: Bool) {
        // No glow effect for drum style
    }
    
    private func animateAppear() {
        // Appear animation: opacity and translate Y
        containerView.alpha = 0
        containerView.transform = CGAffineTransform(translationX: 0, y: 20)
        parameterLabel.alpha = 0
        overlayView.alpha = 0
        
        UIView.animate(withDuration: 0.25, delay: 0, options: .curveEaseOut) {
            self.overlayView.alpha = 1
            self.containerView.alpha = 1
            self.containerView.transform = .identity
            self.parameterLabel.alpha = 1
        }
    }
    
    private func animateIdleDim() {
        UIView.animate(withDuration: 0.3) {
            self.containerView.alpha = 0.3
        }
    }
    
    private func restoreFromIdle() {
        UIView.animate(withDuration: 0.2) {
            self.containerView.alpha = 1.0
        }
    }
    
    // MARK: - Timer Management
    
    private func resetIdleTimer() {
        idleTimer?.invalidate()
        restoreFromIdle()
        
        // Don't start idle timer - slider should stay visible
    }
    
    // MARK: - Public Methods
    
    func showParameter(_ parameter: ParameterEditView.Parameter) {
        self.parameter = parameter
        isUserInteractionEnabled = true
        
        // Convert parameter value to 0-1 range
        let range = parameter.range
        let normalizedValue = (parameter.value - range.lowerBound) / (range.upperBound - range.lowerBound)
        updateValue(normalizedValue)
        
        animateAppear()
        resetIdleTimer()
    }
    
    func hide() {
        idleTimer?.invalidate()
        
        UIView.animate(withDuration: 0.25, delay: 0, options: .curveEaseIn) {
            self.overlayView.alpha = 0
            self.containerView.alpha = 0
            self.parameterLabel.alpha = 0
        } completion: { _ in
            self.isUserInteractionEnabled = false
            self.onHide?()
        }
    }
    
    private func updateParameterLabel() {
        guard let parameter = parameter else { return }
        
        // Convert normalized value back to parameter range
        let range = parameter.range
        let actualValue = range.lowerBound + value * (range.upperBound - range.lowerBound)
        
        let valueText: String
        switch parameter {
        case .itemsMax:
            valueText = "\(parameter.title): \(Int(actualValue))"
        case .confidence, .iou:
            valueText = String(format: "%@: %.2f", parameter.title, actualValue)
        case .lineThickness:
            valueText = String(format: "%@: %.1f", parameter.title, actualValue)
        }
        
        parameterLabel.text = valueText
    }
    
    // MARK: - Hit Testing
    
    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
        // For drum style, entire container is interactive
        return super.point(inside: point, with: event)
    }
    
    // MARK: - Accessibility
    
    override var accessibilityTraits: UIAccessibilityTraits {
        get { [.adjustable] }
        set { }
    }
    
    override var accessibilityValue: String? {
        get { "\(Int(value * 100)) percent" }
        set { }
    }
    
    override func accessibilityIncrement() {
        let newValue = min(1.0, value + 0.01)  // 1% increment
        updateValue(newValue)
        onValueChange?(value)
    }
    
    override func accessibilityDecrement() {
        let newValue = max(0.0, value - 0.01)  // 1% decrement
        updateValue(newValue)
        onValueChange?(value)
    }
}
