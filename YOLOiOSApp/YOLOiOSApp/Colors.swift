// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

extension UIColor {
    // Primary Colors
    static let ultralyticsLime = UIColor(red: 207/255, green: 255/255, blue: 26/255, alpha: 1.0) // #CFFF1A
    static let ultralyticsBrown = UIColor(red: 106/255, green: 85/255, blue: 69/255, alpha: 1.0) // #6A5545
    
    // Surface Colors
    static let ultralyticsSurfaceDark = UIColor.black // #000000
    
    // Text Colors
    static let ultralyticsTextPrimary = UIColor.white // #FFFFFF
    static let ultralyticsTextSubtle = UIColor(red: 125/255, green: 125/255, blue: 125/255, alpha: 1.0) // #7D7D7D
    
    // Convenience initializer for hex colors
    convenience init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            red: CGFloat(r) / 255,
            green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255,
            alpha: CGFloat(a) / 255
        )
    }
}