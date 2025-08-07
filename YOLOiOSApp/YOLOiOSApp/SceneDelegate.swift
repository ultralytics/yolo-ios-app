// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    
    var window: UIWindow?
    
    // Track external display connection state
    static var hasExternalDisplay = false
    
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = (scene as? UIWindowScene) else { return }
        
        // Create window with the scene
        window = UIWindow(windowScene: windowScene)
        
        // Load the main storyboard and instantiate the initial view controller
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        if let initialVC = storyboard.instantiateInitialViewController() {
            window?.rootViewController = initialVC
        }
        
        window?.makeKeyAndVisible()
        
        // Listen for external display connection/disconnection
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleExternalDisplayConnected),
            name: .externalDisplayConnected,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleExternalDisplayDisconnected),
            name: .externalDisplayDisconnected,
            object: nil
        )
    }
    
    @objc private func handleExternalDisplayConnected() {
        SceneDelegate.hasExternalDisplay = true
        updateOrientationLock()
    }
    
    @objc private func handleExternalDisplayDisconnected() {
        SceneDelegate.hasExternalDisplay = false
        updateOrientationLock()
    }
    
    private func updateOrientationLock() {
        // Force orientation update
        if let windowScene = window?.windowScene {
            if #available(iOS 16.0, *) {
                windowScene.requestGeometryUpdate(.iOS(interfaceOrientations: supportedInterfaceOrientations))
                
                // Also force the view controller to update its orientation
                if let rootVC = window?.rootViewController {
                    rootVC.setNeedsUpdateOfSupportedInterfaceOrientations()
                }
            } else {
                // For older iOS versions, we need to trigger orientation update
                UIViewController.attemptRotationToDeviceOrientation()
                
                // Force the view to layout again
                if let rootVC = window?.rootViewController {
                    rootVC.view.setNeedsLayout()
                    rootVC.view.layoutIfNeeded()
                }
            }
        }
    }
    
    var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        // Only support landscape when external display is connected
        if SceneDelegate.hasExternalDisplay {
            return [.landscapeLeft, .landscapeRight]
        } else {
            // Support all orientations when no external display
            return [.portrait, .landscapeLeft, .landscapeRight]
        }
    }
    
    func sceneDidDisconnect(_ scene: UIScene) {
        // Called as the scene is being released by the system.
    }
    
    func sceneDidBecomeActive(_ scene: UIScene) {
        // Called when the scene has moved from an inactive state to an active state.
    }
    
    func sceneWillResignActive(_ scene: UIScene) {
        // Called when the scene will move from an active state to an inactive state.
    }
    
    func sceneWillEnterForeground(_ scene: UIScene) {
        // Called as the scene transitions from the background to the foreground.
    }
    
    func sceneDidEnterBackground(_ scene: UIScene) {
        // Called as the scene transitions from the foreground to the background.
    }
}