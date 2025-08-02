// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

@objc(ExternalSceneDelegate)
class ExternalSceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        print("🔵 ExternalSceneDelegate: scene willConnectTo called")
        print("🔵 Session role: \(session.role.rawValue)")
        
        guard let windowScene = scene as? UIWindowScene else { 
            print("🔴 ExternalSceneDelegate: Failed to cast to UIWindowScene")
            return 
        }
        
        let externalScreen = windowScene.screen
        print("🔵 External screen: \(externalScreen)")
        print("🔵 External screen bounds: \(externalScreen.bounds)")
        
        // Select the highest resolution available
        print("Available display modes:")
        for mode in externalScreen.availableModes {
            print("  - \(mode.size.width) x \(mode.size.height)")
        }
        
        if let bestMode = externalScreen.availableModes.max(by: { 
            $0.size.width * $0.size.height < $1.size.width * $1.size.height 
        }) {
            externalScreen.currentMode = bestMode
            print("External display set to resolution: \(bestMode.size)")
            print("Display name: \(externalScreen.description)")
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
        
        print("🔵 External window created and made visible")
        print("🔵 Window frame: \(window?.frame ?? .zero)")
        print("🔵 Window isHidden: \(window?.isHidden ?? true)")
        print("🔵 Window alpha: \(window?.alpha ?? 0)")
        print("🔵 Root VC: \(window?.rootViewController)")
        print("🔵 Root VC view: \(window?.rootViewController?.view)")
        
        // Force layout update
        window?.setNeedsLayout()
        window?.layoutIfNeeded()
        
        // Debug window hierarchy
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            print("🔵 Window check after 1 second:")
            print("🔵 Window frame: \(self.window?.frame ?? .zero)")
            print("🔵 Window isKeyWindow: \(self.window?.isKeyWindow ?? false)")
            print("🔵 Window screen: \(self.window?.screen)")
        }
        
        // Notify main app that external display is connected
        NotificationCenter.default.post(
            name: .externalDisplayConnected,
            object: nil,
            userInfo: ["screen": externalScreen]
        )
        
        print("🔵 ExternalSceneDelegate setup complete")
    }
    
    func sceneDidDisconnect(_ scene: UIScene) {
        // Notify main app that external display is disconnected
        NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
        window = nil
    }
}