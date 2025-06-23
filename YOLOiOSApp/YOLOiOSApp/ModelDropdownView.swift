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
    private var isExpanded = false
    
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
        containerView.layer.maskedCorners = [.layerMinXMaxYCorner, .layerMaxXMaxYCorner]
        containerView.layer.shadowColor = UIColor.black.cgColor
        containerView.layer.shadowOffset = CGSize(width: 0, height: 10)
        containerView.layer.shadowRadius = 20
        containerView.layer.shadowOpacity = 0.5
        containerView.clipsToBounds = false
        
        // TableView
        tableView.backgroundColor = .clear
        tableView.separatorStyle = .none
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(ModelDropdownCell.self, forCellReuseIdentifier: ModelDropdownCell.identifier)
        tableView.register(ModelDropdownHeaderView.self, forHeaderFooterViewReuseIdentifier: ModelDropdownHeaderView.identifier)
        tableView.layer.cornerRadius = 16
        tableView.layer.maskedCorners = [.layerMinXMaxYCorner, .layerMaxXMaxYCorner]
        tableView.clipsToBounds = true
        
        // Add subviews
        addSubview(overlayView)
        addSubview(containerView)
        containerView.addSubview(tableView)
        
        // Layout
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        containerView.translatesAutoresizingMaskIntoConstraints = false
        tableView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Overlay
            overlayView.topAnchor.constraint(equalTo: topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: bottomAnchor),
            
            // Container
            containerView.topAnchor.constraint(equalTo: topAnchor),
            containerView.leadingAnchor.constraint(equalTo: leadingAnchor),
            containerView.trailingAnchor.constraint(equalTo: trailingAnchor),
            
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
        
        // Ensure view is visible and on top
        isHidden = false
        superview?.bringSubviewToFront(self)
        
        // Enable interaction when showing
        isUserInteractionEnabled = true
        
        // Calculate height
        let maxHeight = UIScreen.main.bounds.height * 0.6
        let contentHeight = calculateContentHeight()
        let finalHeight = min(contentHeight, maxHeight)
        
        print("ModelDropdownView: Calculated height = \(finalHeight)")
        print("ModelDropdownView: Frame = \(frame)")
        
        containerHeightConstraint?.constant = finalHeight
        
        UIView.animate(withDuration: 0.2, delay: 0, options: .curveEaseOut) {
            self.overlayView.alpha = 1
            self.layoutIfNeeded()
        } completion: { _ in
            print("ModelDropdownView: Animation completed")
            print("ModelDropdownView: Final frame = \(self.frame)")
            print("ModelDropdownView: Container frame = \(self.containerView.frame)")
        }
    }
    
    func toggle() {
        if isExpanded {
            hide()
        } else {
            show()
        }
    }
    
    var isShowing: Bool {
        return isExpanded
    }
    
    func hide() {
        guard isExpanded else { return }
        isExpanded = false
        
        containerHeightConstraint?.constant = 0
        
        UIView.animate(withDuration: 0.2, delay: 0, options: .curveEaseOut, animations: {
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
            groupedModels.append(("SELECTED", selected))
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
        
        for section in groupedModels {
            print("ModelDropdownView: Section '\(section.title)' has \(section.models.count) models")
            height += 44 // Header height
            height += CGFloat(section.models.count) * 52 // Row height
        }
        
        let totalHeight = height + 20 // Extra padding
        print("ModelDropdownView: Total calculated height: \(totalHeight)")
        
        return totalHeight
    }
    
    @objc private func overlayTapped() {
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
        
        cell.configure(with: model, status: isSelected ? .selected : (isDownloaded ? .downloaded : .notDownloaded))
        
        return cell
    }
}

// MARK: - UITableViewDelegate

extension ModelDropdownView: UITableViewDelegate {
    
    func tableView(_ tableView: UITableView, viewForHeaderInSection section: Int) -> UIView? {
        let header = tableView.dequeueReusableHeaderFooterView(withIdentifier: ModelDropdownHeaderView.identifier) as! ModelDropdownHeaderView
        header.configure(title: groupedModels[section].title)
        return header
    }
    
    func tableView(_ tableView: UITableView, heightForHeaderInSection section: Int) -> CGFloat {
        return 44
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
    
    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupUI() {
        backgroundColor = .clear
        selectionStyle = .none
        
        // Status Image
        statusImageView.contentMode = .scaleAspectFit
        statusImageView.tintColor = .white
        
        // Name Label
        nameLabel.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        nameLabel.textColor = .white
        
        // Size Label
        sizeLabel.font = UIFont.systemFont(ofSize: 12, weight: .regular)
        sizeLabel.textColor = .systemGray
        
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
            statusImageView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            statusImageView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            statusImageView.widthAnchor.constraint(equalToConstant: 24),
            statusImageView.heightAnchor.constraint(equalToConstant: 24),
            
            nameLabel.leadingAnchor.constraint(equalTo: statusImageView.trailingAnchor, constant: 12),
            nameLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor, constant: -8),
            
            sizeLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
            sizeLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
            
            progressView.leadingAnchor.constraint(equalTo: statusImageView.trailingAnchor, constant: 12),
            progressView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            progressView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            progressView.heightAnchor.constraint(equalToConstant: 2)
        ])
        
        // Bottom border
        let borderView = UIView()
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
    
    func configure(with model: ModelEntry, status: Status) {
        nameLabel.text = model.displayName.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ").uppercased()
        sizeLabel.text = ModelSizeHelper.getModelSize(from: model.displayName)
        
        switch status {
        case .selected:
            statusImageView.image = UIImage(systemName: "checkmark.circle.fill")
            statusImageView.tintColor = .ultralyticsLime
            contentView.backgroundColor = UIColor.ultralyticsLime.withAlphaComponent(0.1)
            progressView.isHidden = true
            
        case .downloaded:
            statusImageView.image = UIImage(systemName: "circle")
            statusImageView.tintColor = .white
            contentView.backgroundColor = .clear
            progressView.isHidden = true
            
        case .notDownloaded:
            statusImageView.image = UIImage(systemName: "arrow.down.circle")
            statusImageView.tintColor = .white
            contentView.backgroundColor = .clear
            progressView.isHidden = true
            
        case .downloading(let progress):
            statusImageView.image = UIImage(systemName: "circle")
            statusImageView.tintColor = .ultralyticsLime
            contentView.backgroundColor = .clear
            progressView.isHidden = false
            progressView.progress = progress
        }
    }
}

// MARK: - ModelDropdownHeaderView

class ModelDropdownHeaderView: UITableViewHeaderFooterView {
    
    static let identifier = "ModelDropdownHeaderView"
    
    private let titleLabel = UILabel()
    
    override init(reuseIdentifier: String?) {
        super.init(reuseIdentifier: reuseIdentifier)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupUI() {
        contentView.backgroundColor = UIColor.black.withAlphaComponent(0.98)
        
        titleLabel.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
        titleLabel.textColor = .systemGray
        
        contentView.addSubview(titleLabel)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            titleLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor)
        ])
    }
    
    func configure(title: String) {
        titleLabel.text = title
    }
}