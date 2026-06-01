// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

  var window: UIWindow?

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = (scene as? UIWindowScene) else { return }

    window = UIWindow(windowScene: windowScene)

    let storyboard = UIStoryboard(name: "Main", bundle: nil)
    if let initialVC = storyboard.instantiateInitialViewController() {
      window?.rootViewController = initialVC
    }

    window?.makeKeyAndVisible()
  }

  func sceneDidBecomeActive(_ scene: UIScene) {
    ExternalDisplayManager.refreshModeIfNeeded()
  }
}
