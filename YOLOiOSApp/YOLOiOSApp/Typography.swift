// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

struct Typography {
    // Status Bar
    static let statusBarFont = UIFont.systemFont(ofSize: 12, weight: .bold).rounded()
    
    // Task Tabs
    static let tabLabelFont = UIFont.systemFont(ofSize: 14, weight: .semibold).rounded()
    
    // Labels
    static let labelFont = UIFont.systemFont(ofSize: 8, weight: .bold).rounded()
    
    // Toast
    static let toastFont = UIFont.systemFont(ofSize: 10, weight: .bold).rounded()
}

extension UIFont {
    func rounded() -> UIFont {
        guard let descriptor = fontDescriptor.withDesign(.rounded) else { return self }
        return UIFont(descriptor: descriptor, size: pointSize)
    }
}