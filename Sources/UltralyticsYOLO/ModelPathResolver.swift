// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Resolves a user-supplied model string to an on-disk URL.
///
/// Accepts either an absolute filesystem path to `.mlmodel`/`.mlpackage`/`.mlmodelc`/`.aimodel`, or a bundle resource
/// name (searched for `.mlmodelc` then `.mlpackage` in the main bundle). Core AI is opt-in by extension: a bundled
/// Core AI model resolves only from a name that carries `.aimodel`.
enum ModelPathResolver {
  static func resolve(_ modelPathOrName: String) -> URL? {
    let lowercased = modelPathOrName.lowercased()
    if [".mlmodel", ".mlpackage", ".mlmodelc", ".aimodel"].contains(where: lowercased.hasSuffix) {
      let url = URL(fileURLWithPath: modelPathOrName)
      if FileManager.default.fileExists(atPath: url.path) { return url }
      return lowercased.hasSuffix(".aimodel")
        ? Bundle.main.url(forResource: modelPathOrName, withExtension: nil) : nil
    }
    return Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      ?? Bundle.main.url(forResource: modelPathOrName, withExtension: "mlpackage")
  }
}
