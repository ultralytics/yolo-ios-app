// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Resolves a user-supplied model string to an on-disk URL.
///
/// Accepts either an absolute filesystem path to `.mlmodel`/`.mlpackage`/`.mlmodelc`, or a
/// bundle resource name (searched for `.mlmodelc` then `.mlpackage` in the main bundle).
enum ModelPathResolver {
  static func resolve(_ modelPathOrName: String) -> URL? {
    let lowercased = modelPathOrName.lowercased()
    if lowercased.hasSuffix(".mlmodel") || lowercased.hasSuffix(".mlpackage")
      || lowercased.hasSuffix(".mlmodelc")
    {
      let url = URL(fileURLWithPath: modelPathOrName)
      return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
    return Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      ?? Bundle.main.url(forResource: modelPathOrName, withExtension: "mlpackage")
  }
}
