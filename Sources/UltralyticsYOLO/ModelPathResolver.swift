// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Resolves a user-supplied model string to an on-disk URL.
///
/// Accepts either an absolute filesystem path to `.aimodel`/`.mlmodel`/`.mlpackage`/`.mlmodelc`, or a bundle resource
/// name (searched for `.aimodel` when Core AI is available, then `.mlmodelc` and `.mlpackage`, in the main bundle).
enum ModelPathResolver {
  static func resolve(_ modelPathOrName: String) -> URL? {
    let lowercased = modelPathOrName.lowercased()
    if [".aimodel", ".mlmodel", ".mlpackage", ".mlmodelc"].contains(where: lowercased.hasSuffix) {
      let url = URL(fileURLWithPath: modelPathOrName)
      return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
    let extensions =
      (BasePredictor.isCoreAIAvailable ? ["aimodel"] : []) + ["mlmodelc", "mlpackage"]
    return extensions.lazy.compactMap {
      Bundle.main.url(forResource: modelPathOrName, withExtension: $0)
    }
    .first
  }
}
