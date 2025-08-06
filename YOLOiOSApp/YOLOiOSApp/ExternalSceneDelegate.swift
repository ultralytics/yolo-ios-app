// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

@objc(ExternalSceneDelegate)
class ExternalSceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        
        guard let windowScene = scene as? UIWindowScene else { 
            return 
        }
        
        let externalScreen = windowScene.screen
        // Select the highest resolution available
        
        if let bestMode = externalScreen.availableModes.max(by: { 
            $0.size.width * $0.size.height < $1.size.width * $1.size.height 
        }) {
            externalScreen.currentMode = bestMode
        }
        
        // Create window for external display
        window = UIWindow(windowScene: windowScene)
        window?.frame = externalScreen.bounds
        
        // Set overscan compensation for proper edge handling (on the screen, not window)
        externalScreen.overscanCompensation = .scale
        
        // Create and set the external view controller
        let externalVC = ExternalViewController()
        
        // Configure for full screen without safe area
        externalVC.modalPresentationStyle = .fullScreen
        window?.rootViewController = externalVC
        
        // Ignore safe area for full screen display
        if #available(iOS 11.0, *) {
            window?.insetsLayoutMarginsFromSafeArea = false
            externalVC.additionalSafeAreaInsets = UIEdgeInsets(top: 0, left: 0, bottom: 0, right: 0)
        }
        
        window?.isHidden = false
        window?.makeKeyAndVisible()
        
        // Force layout update
        window?.setNeedsLayout()
        window?.layoutIfNeeded()
        
        // Notify main app that external display is connected
        NotificationCenter.default.post(
            name: .externalDisplayConnected,
            object: nil,
            userInfo: ["screen": externalScreen]
        )
    }
    
    func sceneDidDisconnect(_ scene: UIScene) {
        // Notify main app that external display is disconnected
        NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
        window = nil
    }
}