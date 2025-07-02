// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class ParameterEditView: UIView {
    enum Parameter {
        case itemsMax(Int)
        case confidence(Float)
        case iou(Float)
        case lineThickness(Float)
        
        var title: String {
            switch self {
            case .itemsMax: return "ITEMS MAX"
            case .confidence: return "CONFIDENCE THRESHOLD"
            case .iou: return "IoU THRESHOLD"
            case .lineThickness: return "LINE THICKNESS"
            }
        }
        
        var range: ClosedRange<Float> {
            switch self {
            case .itemsMax: return 1...30
            case .confidence, .iou: return 0...1
            case .lineThickness: return 0.5...3.0
            }
        }
        
        var step: Float {
            switch self {
            case .itemsMax: return 1
            case .confidence, .iou: return 0.02
            case .lineThickness: return 0.1
            }
        }
        
        var value: Float {
            switch self {
            case .itemsMax(let v): return Float(v)
            case .confidence(let v), .iou(let v), .lineThickness(let v): return v
            }
        }
    }
    
    private let toastView = UIView()
    private let toastLabel = UILabel()
    private let slider = UISlider()
    private var hideTimer: Timer?
    
    var onValueChange: ((Parameter) -> Void)?
    private var currentParameter: Parameter?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        // Allow touches to pass through when not showing controls
        isUserInteractionEnabled = false
        
        // Toast View
        toastView.backgroundColor = UIColor.ultralyticsBrown.withAlphaComponent(0.95)
        toastView.layer.cornerRadius = 14
        toastView.alpha = 0
        
        toastLabel.textColor = .ultralyticsTextPrimary
        toastLabel.font = Typography.toastFont
        toastLabel.textAlignment = .center
        
        toastView.addSubview(toastLabel)
        addSubview(toastView)
        
        // Slider
        slider.minimumTrackTintColor = .ultralyticsLime
        slider.maximumTrackTintColor = .ultralyticsSurfaceDark
        slider.thumbTintColor = .ultralyticsLime
        slider.alpha = 0
        slider.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        
        // Custom thumb
        let thumbView = UIView(frame: CGRect(x: 0, y: 0, width: 4, height: 20))
        thumbView.backgroundColor = .ultralyticsLime
        thumbView.layer.cornerRadius = 2
        
        UIGraphicsBeginImageContextWithOptions(thumbView.bounds.size, false, 0)
        if let context = UIGraphicsGetCurrentContext() {
            thumbView.layer.render(in: context)
            let thumbImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            
            slider.setThumbImage(thumbImage, for: .normal)
            slider.setThumbImage(thumbImage, for: .highlighted)
        }
        
        addSubview(slider)
        
        // Layout
        [toastView, toastLabel, slider].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
        
        NSLayoutConstraint.activate([
            // Toast
            toastView.topAnchor.constraint(equalTo: topAnchor, constant: 20),
            toastView.centerXAnchor.constraint(equalTo: centerXAnchor),
            toastView.widthAnchor.constraint(greaterThanOrEqualToConstant: 120),
            toastView.heightAnchor.constraint(equalToConstant: 28),
            
            toastLabel.leadingAnchor.constraint(equalTo: toastView.leadingAnchor, constant: 12),
            toastLabel.trailingAnchor.constraint(equalTo: toastView.trailingAnchor, constant: -12),
            toastLabel.centerYAnchor.constraint(equalTo: toastView.centerYAnchor),
            
            // Slider
            slider.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
            slider.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
            slider.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -20),
            slider.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    func showParameter(_ parameter: Parameter) {
        currentParameter = parameter
        
        // Enable interaction when showing controls
        isUserInteractionEnabled = true
        
        // Configure slider
        slider.minimumValue = parameter.range.lowerBound
        slider.maximumValue = parameter.range.upperBound
        slider.value = parameter.value
        
        // Update toast
        updateToastLabel()
        
        // Show with animation
        UIView.animate(withDuration: 0.15) {
            self.toastView.alpha = 1
            self.slider.alpha = 1
        }
        
        resetHideTimer()
    }
    
    @objc private func sliderValueChanged() {
        guard let parameter = currentParameter else { return }
        
        // Snap to step
        let step = parameter.step
        let roundedValue = round(slider.value / step) * step
        slider.value = roundedValue
        
        // Update parameter
        let newParameter: Parameter
        switch parameter {
        case .itemsMax:
            newParameter = .itemsMax(Int(roundedValue))
        case .confidence:
            newParameter = .confidence(roundedValue)
        case .iou:
            newParameter = .iou(roundedValue)
        case .lineThickness:
            newParameter = .lineThickness(roundedValue)
        }
        
        currentParameter = newParameter
        updateToastLabel()
        onValueChange?(newParameter)
        
        resetHideTimer()
    }
    
    private func updateToastLabel() {
        guard let parameter = currentParameter else { return }
        
        let valueText: String
        switch parameter {
        case .itemsMax(let v):
            valueText = "\(parameter.title): \(v)"
        case .confidence(let v), .iou(let v):
            valueText = String(format: "%@: %.2f", parameter.title, v)
        case .lineThickness(let v):
            valueText = String(format: "%@: %.1f", parameter.title, v)
        }
        
        toastLabel.text = valueText
    }
    
    private func resetHideTimer() {
        hideTimer?.invalidate()
        hideTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { _ in
            self.hide()
        }
    }
    
    func hide() {
        hideTimer?.invalidate()
        
        UIView.animate(withDuration: 0.3) {
            self.toastView.alpha = 0
            self.slider.alpha = 0
        } completion: { _ in
            // Disable interaction when hidden
            self.isUserInteractionEnabled = false
        }
    }
}