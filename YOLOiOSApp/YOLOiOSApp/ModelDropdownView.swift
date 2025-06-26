// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

protocol ModelDropdownViewDelegate: AnyObject {
    func modelDropdown(_ dropdown: ModelDropdownView, didSelectModel model: ModelEntry)
    func modelDropdownDidDismiss(_ dropdown: ModelDropdownView)
}

class ModelDropdownView: UIView {
    
    // MARK: - Properties
    
    weak var delegate: ModelDropdownViewDelegate?
    
    private let overlayView = UIView()
    private let containerView = UIView()
    private let tableView = UITableView()
    
    private var models: [ModelEntry] = []
    private var groupedModels: [(title: String, models: [ModelEntry])] = []
    private var currentModelIdentifier: String?
    
    private var containerHeightConstraint: NSLayoutConstraint?
    private var containerTopConstraint: NSLayoutConstraint?
    private var containerTrailingConstraint: NSLayoutConstraint?
    private var isExpanded = false
    private var isLandscape = false
    
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
        // Disable interaction initially
        isUserInteractionEnabled = false
        isHidden = false  // Ensure view is not hidden
        
        // Overlay
        overlayView.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        overlayView.alpha = 0
        overlayView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(overlayTapped)))
        
        // Container
        containerView.backgroundColor = UIColor.black.withAlphaComponent(0.98)
        containerView.layer.cornerRadius = 16
        containerView.layer.maskedCorners = [.layerMinXMaxYCorner, .layerMaxXMaxYCorner]  // Bottom corners only
        containerView.layer.shadowColor = UIColor.black.cgColor
        containerView.layer.shadowOffset = CGSize(width: 0, height: 10)
        containerView.layer.shadowRadius = 20
        containerView.layer.shadowOpacity = 0.5
        containerView.clipsToBounds = false
        
        // TableView
        tableView.backgroundColor = .clear
        tableView.separatorStyle = .none
        tableView.separatorInset = .zero
        tableView.separatorColor = .clear
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(ModelDropdownCell.self, forCellReuseIdentifier: ModelDropdownCell.identifier)
        tableView.register(ModelDropdownHeaderView.self, forHeaderFooterViewReuseIdentifier: ModelDropdownHeaderView.identifier)
        tableView.layer.cornerRadius = 16
        tableView.layer.maskedCorners = [.layerMinXMaxYCorner, .layerMaxXMaxYCorner]  // Bottom corners only
        tableView.clipsToBounds = true
        tableView.bounces = true  // Enable bounce for better scroll feedback
        tableView.showsVerticalScrollIndicator = true
        tableView.indicatorStyle = .white  // White scroll indicator for dark background
        tableView.contentInset = UIEdgeInsets(top: 0, left: 0, bottom: 20, right: 0)  // Add bottom padding for better scrolling
        
        // Add subviews
        addSubview(overlayView)
        addSubview(containerView)
        containerView.addSubview(tableView)
        
        // Add swipe gestures to dismiss
        let swipeUp = UISwipeGestureRecognizer(target: self, action: #selector(handleSwipeUp))
        swipeUp.direction = .up
        containerView.addGestureRecognizer(swipeUp)
        
        let swipeDown = UISwipeGestureRecognizer(target: self, action: #selector(handleSwipeDown))
        swipeDown.direction = .down
        tableView.addGestureRecognizer(swipeDown)
        
        // Layout
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        containerView.translatesAutoresizingMaskIntoConstraints = false
        tableView.translatesAutoresizingMaskIntoConstraints = false
        
        // Create the container top constraint separately so we can update it
        // In both portrait and landscape, position at safe area top + 36 (to account for StatusMetricBar)
        containerTopConstraint = containerView.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 36)
        
        // Create trailing constraint separately for landscape adjustment
        containerTrailingConstraint = containerView.trailingAnchor.constraint(equalTo: trailingAnchor)
        
        NSLayoutConstraint.activate([
            // Overlay - starts at safe area top
            overlayView.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: bottomAnchor),
            
            // Container - positioned below StatusMetricBar
            containerTopConstraint!,
            containerView.leadingAnchor.constraint(equalTo: leadingAnchor),
            containerTrailingConstraint!,
            
            // TableView
            tableView.topAnchor.constraint(equalTo: containerView.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)
        ])
        
        // Height constraint
        containerHeightConstraint = containerView.heightAnchor.constraint(equalToConstant: 0)
        containerHeightConstraint?.isActive = true
        
        // Ensure the dropdown is initially hidden properly
        containerView.alpha = 1.0
        overlayView.alpha = 0
        isExpanded = false
    }
    
    // MARK: - Public Methods
    
    func configure(with models: [ModelEntry], currentModel: String?) {
        self.models = models
        self.currentModelIdentifier = currentModel
        groupModels()
        tableView.reloadData()
    }
    
    func show() {
        guard !isExpanded else { 
            print("ModelDropdownView: Already expanded, returning")
            return 
        }
        isExpanded = true
        
        print("ModelDropdownView: Showing dropdown")
        print("ModelDropdownView: isHidden = \(isHidden)")
        print("ModelDropdownView: superview = \(superview != nil ? "exists" : "nil")")
        
        // Ensure view is visible
        isHidden = false
        
        // Enable interaction when showing
        isUserInteractionEnabled = true
        
        // Calculate height more accurately
        let calculatedHeight = calculateContentHeight()
        var finalHeight = calculatedHeight
        
        print("ModelDropdownView: Calculated height = \(finalHeight)")
        
        // In landscape, limit height to 85% of screen height to show more items while leaving space for interaction
        if isLandscape {
            let maxLandscapeHeight = UIScreen.main.bounds.height * 0.85
            finalHeight = min(finalHeight, maxLandscapeHeight)
            print("ModelDropdownView: Landscape mode - limiting height to \(finalHeight)")
        }
        
        // Check if content fits on screen
        let statusBarOffset: CGFloat = 54 // Status bar height + padding
        let safeAreaBottom = window?.safeAreaInsets.bottom ?? 0
        let bottomPadding: CGFloat = safeAreaBottom // Just safe area, no extra padding
        let availableHeight = UIScreen.main.bounds.height - statusBarOffset - bottomPadding
        
        print("ModelDropdownView: Screen height = \(UIScreen.main.bounds.height)")
        print("ModelDropdownView: Available height = \(availableHeight)")
        print("ModelDropdownView: Content height = \(finalHeight)")
        
        // Use calculated height (already limited for landscape)
        if finalHeight <= availableHeight {
            tableView.isScrollEnabled = false
            containerHeightConstraint?.constant = finalHeight
        } else {
            tableView.isScrollEnabled = true
            containerHeightConstraint?.constant = availableHeight
        }
        
        UIView.animate(withDuration: 0.2, delay: 0, options: .curveEaseOut) {
            self.overlayView.alpha = 1
            self.layoutIfNeeded()
        } completion: { _ in
            print("ModelDropdownView: Animation completed")
            print("ModelDropdownView: Final container frame = \(self.containerView.frame)")
            print("ModelDropdownView: Final tableView contentSize = \(self.tableView.contentSize)")
        }
    }
    
    func toggle() {
        if isExpanded {
            hide()
        } else {
            show()
        }
    }
    
    func updateLayoutForOrientation(isLandscape: Bool) {
        // StatusMetricBar is at -8 from safe area top and has height of 44
        // So the bottom of StatusMetricBar is at: -8 + 44 = 36 from safe area top
        // This is the same for both orientations since we're using safe area
        
        // The dropdown should always be positioned at 36 points from safe area top
        // to align perfectly with the bottom of StatusMetricBar
        containerTopConstraint?.constant = 36
        
        // Store orientation state for use in height calculation
        self.isLandscape = isLandscape
        
        // In landscape, leave space for ShutterBar on the right (96 points)
        containerTrailingConstraint?.constant = isLandscape ? -96 : 0
        
        // If already showing, update the height
        if isExpanded {
            // Recalculate and apply height limit for landscape
            let calculatedHeight = calculateContentHeight()
            var finalHeight = calculatedHeight
            
            if isLandscape {
                let maxLandscapeHeight = UIScreen.main.bounds.height * 0.85
                finalHeight = min(finalHeight, maxLandscapeHeight)
            }
            
            containerHeightConstraint?.constant = finalHeight
            tableView.isScrollEnabled = finalHeight < calculatedHeight
        }
        
        layoutIfNeeded()
    }
    
    var isShowing: Bool {
        return isExpanded
    }
    
    func hide() {
        guard isExpanded else { return }
        isExpanded = false
        
        containerHeightConstraint?.constant = 0
        
        UIView.animate(withDuration: 0.4, delay: 0, options: .curveEaseInOut, animations: {
            self.overlayView.alpha = 0
            self.layoutIfNeeded()
        }) { _ in
            // Disable interaction when hidden
            self.isUserInteractionEnabled = false
            self.delegate?.modelDropdownDidDismiss(self)
        }
    }
    
    // MARK: - Private Methods
    
    private func groupModels() {
        var selected: [ModelEntry] = []
        var downloaded: [ModelEntry] = []
        var available: [ModelEntry] = []
        
        for model in models {
            if model.identifier == currentModelIdentifier {
                selected.append(model)
            } else if model.isLocalBundle || (model.isRemote && ModelCacheManager.shared.isModelDownloaded(key: model.identifier)) {
                downloaded.append(model)
            } else {
                available.append(model)
            }
        }
        
        groupedModels = []
        
        if !selected.isEmpty {
            groupedModels.append(("", selected))  // Empty string for no header
        }
        if !downloaded.isEmpty {
            groupedModels.append(("DOWNLOADED", downloaded))
        }
        if !available.isEmpty {
            groupedModels.append(("AVAILABLE", available))
        }
    }
    
    private func calculateContentHeight() -> CGFloat {
        var height: CGFloat = 0
        
        print("ModelDropdownView: Calculating content height")
        print("ModelDropdownView: Number of sections: \(groupedModels.count)")
        
        // Top padding
        height += 20
        
        for (index, section) in groupedModels.enumerated() {
            print("ModelDropdownView: Section '\(section.title)' has \(section.models.count) models")
            
            // Header height (including padding)
            if !(index == 0 && section.title.isEmpty) {
                height += 30 // Reduced header height
            }
            
            // Cell heights - each cell is 52pt + separator
            height += CGFloat(section.models.count) * 53 // 52pt cell + 1pt for separators
            
            // Footer height for selected section divider
            if index == 0 && section.title.isEmpty {
                height += 6 // Minimal footer height
            }
            
            // Section bottom padding (except for first section which has footer)
            if !(index == 0 && section.title.isEmpty) {
                height += 2  // Minimal padding
            }
        }
        
        // Bottom padding for corner radius and safe area
        height += 30
        
        print("ModelDropdownView: Total calculated height: \(height)")
        
        return height
    }
    
    @objc private func overlayTapped() {
        hide()
    }
    
    @objc private func handleSwipeUp() {
        hide()
    }
    
    @objc private func handleSwipeDown() {
        hide()
    }
}

// MARK: - UITableViewDataSource

extension ModelDropdownView: UITableViewDataSource {
    
    func numberOfSections(in tableView: UITableView) -> Int {
        return groupedModels.count
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return groupedModels[section].models.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: ModelDropdownCell.identifier, for: indexPath) as! ModelDropdownCell
        
        let model = groupedModels[indexPath.section].models[indexPath.row]
        let isSelected = model.identifier == currentModelIdentifier
        let isDownloaded = model.isLocalBundle || (model.isRemote && ModelCacheManager.shared.isModelDownloaded(key: model.identifier))
        let isFirstInSection = indexPath.row == 0
        let isLastInSection = indexPath.row == groupedModels[indexPath.section].models.count - 1
        
        // For selected items in the first section, always treat as last to hide border
        // Also hide border for last item in each section
        let hideBottomBorder = isLastInSection || (isSelected && indexPath.section == 0)
        
        cell.configure(with: model, 
                      status: isSelected ? .selected : (isDownloaded ? .downloaded : .notDownloaded),
                      isFirstInSection: isFirstInSection,
                      isLastInSection: hideBottomBorder)
        
        return cell
    }
}

// MARK: - UITableViewDelegate

extension ModelDropdownView: UITableViewDelegate {
    
    func tableView(_ tableView: UITableView, viewForHeaderInSection section: Int) -> UIView? {
        let header = tableView.dequeueReusableHeaderFooterView(withIdentifier: ModelDropdownHeaderView.identifier) as! ModelDropdownHeaderView
        header.configure(title: groupedModels[section].title, showDivider: section > 0)
        return header
    }
    
    func tableView(_ tableView: UITableView, heightForHeaderInSection section: Int) -> CGFloat {
        // No header for first section (selected)
        if section == 0 && groupedModels[section].title.isEmpty {
            return 0
        }
        return 30  // Reduced header height
    }
    
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 52
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        
        let model = groupedModels[indexPath.section].models[indexPath.row]
        delegate?.modelDropdown(self, didSelectModel: model)
        hide()
    }
    
    func tableView(_ tableView: UITableView, viewForFooterInSection section: Int) -> UIView? {
        // Add thick divider after selected section
        if section == 0 && groupedModels[section].title.isEmpty {
            let footerView = UIView()
            footerView.backgroundColor = .clear
            
            let divider = UIView()
            divider.backgroundColor = UIColor.systemGray.withAlphaComponent(0.5)
            footerView.addSubview(divider)
            divider.translatesAutoresizingMaskIntoConstraints = false
            
            NSLayoutConstraint.activate([
                divider.leadingAnchor.constraint(equalTo: footerView.leadingAnchor, constant: 16),
                divider.trailingAnchor.constraint(equalTo: footerView.trailingAnchor, constant: -16),
                divider.topAnchor.constraint(equalTo: footerView.topAnchor, constant: 1),  // Minimal padding
                divider.heightAnchor.constraint(equalToConstant: 4)  // Thicker divider
            ])
            
            return footerView
        }
        return nil
    }
    
    func tableView(_ tableView: UITableView, heightForFooterInSection section: Int) -> CGFloat {
        // Height for thick divider after selected section
        if section == 0 && groupedModels[section].title.isEmpty {
            return 6  // Minimal space for divider
        }
        return 0
    }
}

// MARK: - ModelDropdownCell

class ModelDropdownCell: UITableViewCell {
    
    static let identifier = "ModelDropdownCell"
    
    enum Status {
        case selected
        case downloaded
        case notDownloaded
        case downloading(progress: Float)
    }
    
    private let statusImageView = UIImageView()
    private let nameLabel = UILabel()
    private let sizeLabel = UILabel()
    private let progressView = UIProgressView()
    private let borderView = UIView()
    
    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func prepareForReuse() {
        super.prepareForReuse()
        borderView.isHidden = false
        borderView.backgroundColor = UIColor.systemGray.withAlphaComponent(0.2)
    }
    
    private func setupUI() {
        backgroundColor = .clear
        selectionStyle = .none
        
        // Remove any default separators
        separatorInset = UIEdgeInsets(top: 0, left: UIScreen.main.bounds.width, bottom: 0, right: 0)
        
        // Status Image
        statusImageView.contentMode = .scaleAspectFit
        statusImageView.tintColor = .white
        
        // Name Label
        nameLabel.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        nameLabel.textColor = .white
        
        // Size Label - hidden since we don't need it
        sizeLabel.font = UIFont.systemFont(ofSize: 12, weight: .regular)
        sizeLabel.textColor = .systemGray
        sizeLabel.isHidden = true
        
        // Progress View
        progressView.progressTintColor = .ultralyticsLime
        progressView.trackTintColor = UIColor.systemGray.withAlphaComponent(0.3)
        progressView.isHidden = true
        
        // Add subviews
        [statusImageView, nameLabel, sizeLabel, progressView].forEach {
            contentView.addSubview($0)
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
        
        // Layout
        NSLayoutConstraint.activate([
            nameLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            nameLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),  // Center without offset
            
            sizeLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
            sizeLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
            
            statusImageView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            statusImageView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            statusImageView.widthAnchor.constraint(equalToConstant: 24),
            statusImageView.heightAnchor.constraint(equalToConstant: 24),
            
            progressView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            progressView.trailingAnchor.constraint(equalTo: statusImageView.leadingAnchor, constant: -12),
            progressView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            progressView.heightAnchor.constraint(equalToConstant: 2)
        ])
        
        // Bottom border setup
        borderView.backgroundColor = UIColor.systemGray.withAlphaComponent(0.2)
        contentView.addSubview(borderView)
        borderView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            borderView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            borderView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            borderView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            borderView.heightAnchor.constraint(equalToConstant: 0.5)
        ])
    }
    
    func configure(with model: ModelEntry, status: Status, isFirstInSection: Bool = false, isLastInSection: Bool = false) {
        nameLabel.text = model.displayName.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ").uppercased()
        sizeLabel.text = ModelSizeHelper.getModelSize(from: model.displayName)
        
        switch status {
        case .selected:
            statusImageView.image = UIImage(systemName: "checkmark")
            statusImageView.tintColor = .ultralyticsLime
            nameLabel.textColor = .ultralyticsLime
            sizeLabel.textColor = .ultralyticsLime
            contentView.backgroundColor = .clear
            progressView.isHidden = true
            borderView.isHidden = true  // Hide border for selected item
            
        case .downloaded:
            statusImageView.image = nil  // No icon for downloaded models
            statusImageView.tintColor = .white
            nameLabel.textColor = .white
            sizeLabel.textColor = .systemGray
            contentView.backgroundColor = .clear
            progressView.isHidden = true
            borderView.isHidden = isLastInSection  // Hide border for last item in section
            
        case .notDownloaded:
            statusImageView.image = UIImage(systemName: "arrow.down.circle.dotted")
            statusImageView.tintColor = .white
            nameLabel.textColor = .white
            sizeLabel.textColor = .systemGray
            contentView.backgroundColor = .clear
            progressView.isHidden = true
            borderView.isHidden = isLastInSection  // Hide border for last item in section
            
        case .downloading(let progress):
            statusImageView.image = UIImage(systemName: "circle.dotted")
            statusImageView.tintColor = .ultralyticsLime
            nameLabel.textColor = .white
            sizeLabel.textColor = .systemGray
            contentView.backgroundColor = .clear
            progressView.isHidden = false
            progressView.progress = progress
            borderView.isHidden = isLastInSection  // Hide border for last item in section
        }
    }
}

// MARK: - ModelDropdownHeaderView

class ModelDropdownHeaderView: UITableViewHeaderFooterView {
    
    static let identifier = "ModelDropdownHeaderView"
    
    private let titleLabel = UILabel()
    private let dividerView = UIView()
    
    override init(reuseIdentifier: String?) {
        super.init(reuseIdentifier: reuseIdentifier)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupUI() {
        contentView.backgroundColor = UIColor.black.withAlphaComponent(0.98)
        
        titleLabel.font = UIFont.systemFont(ofSize: 14, weight: .semibold)
        titleLabel.textColor = .systemGray
        
        dividerView.backgroundColor = UIColor.systemGray.withAlphaComponent(0.3)
        
        contentView.addSubview(titleLabel)
        // Don't add divider view at all
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            titleLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor)
        ])
    }
    
    func configure(title: String, showDivider: Bool = false) {
        titleLabel.text = title
        // Divider is no longer used
    }
}