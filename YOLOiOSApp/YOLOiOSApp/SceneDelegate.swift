// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

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
      (initialVC as? ViewController)?.initialImportURL =
        connectionOptions.userActivities
        .first(where: { $0.activityType == NSUserActivityTypeBrowsingWeb })?.webpageURL
      window?.rootViewController = initialVC
    }

    window?.makeKeyAndVisible()
  }

  func scene(_ scene: UIScene, continue userActivity: NSUserActivity) {
    guard userActivity.activityType == NSUserActivityTypeBrowsingWeb,
      let url = userActivity.webpageURL
    else { return }
    (window?.rootViewController as? ViewController)?.importModel(from: url)
  }

  func sceneDidBecomeActive(_ scene: UIScene) {
    ExternalDisplayManager.refreshModeIfNeeded()
  }
}
