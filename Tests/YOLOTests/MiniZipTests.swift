// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation
import XCTest

@testable import YOLO

/// Verifies `MiniZip`, the dependency-free ZIP extractor used for unpacking downloaded model archives.
///
/// The fixture is a real archive produced by Python's `zipfile` (an independent PKZIP implementation), embedded
/// as base64 so the test is self-contained and runs on the iOS simulator with no on-disk resources. It mixes a
/// DEFLATE entry, a STORED entry in a subdirectory, and macOS resource-fork junk (`__MACOSX/` and `._` sidecars)
/// so a single round-trip exercises inflation, verbatim copy, nested-directory creation, and the skip filter the
/// downloader relies on.
final class MiniZipTests: XCTestCase {

  /// A `model.mlpackage`-shaped archive: `labels.txt` (DEFLATE), `data/blob.bin` (STORED), plus macOS metadata.
  private let fixtureBase64 = """
    UEsDBBQAAAAIAAAAIQCkHSt4DwAAAAgBAAAaAAAAbW9kZWwubWxwYWNrYWdlL2xhYmVscy50eHTL\
    SM3JyVeozAcSGSOZCQBQSwMEFAAAAAAAAAAhAOjjWSUIAAAACAAAAB0AAABtb2RlbC5tbHBhY2th\
    Z2UvZGF0YS9ibG9iLmJpbgcLDQD/gEAgUEsDBBQAAAAAAL2Vv1w5nPsGBAAAAAQAAAAlAAAAX19N\
    QUNPU1gvbW9kZWwubWxwYWNrYWdlLy5fbGFiZWxzLnR4dGp1bmtQSwMEFAAAAAAAvZW/XDmc+wYE\
    AAAABAAAABwAAABtb2RlbC5tbHBhY2thZ2UvLl9sYWJlbHMudHh0anVua1BLAQIUAxQAAAAIAAAA\
    IQCkHSt4DwAAAAgBAAAaAAAAAAAAAAAAAACAAQAAAABtb2RlbC5tbHBhY2thZ2UvbGFiZWxzLnR4\
    dFBLAQIUAxQAAAAAAAAAIQDo41klCAAAAAgAAAAdAAAAAAAAAAAAAACAAUcAAABtb2RlbC5tbHBh\
    Y2thZ2UvZGF0YS9ibG9iLmJpblBLAQIUAxQAAAAAAL2Vv1w5nPsGBAAAAAQAAAAlAAAAAAAAAAAA\
    AACAAYoAAABfX01BQ09TWC9tb2RlbC5tbHBhY2thZ2UvLl9sYWJlbHMudHh0UEsBAhQDFAAAAAAA\
    vZW/XDmc+wYEAAAABAAAABwAAAAAAAAAAAAAAIAB0QAAAG1vZGVsLm1scGFja2FnZS8uX2xhYmVs\
    cy50eHRQSwUGAAAAAAQABAAwAQAADwEAAAAA
    """

  /// Writes an arbitrary base64-encoded archive to a temporary `.zip` and returns its URL.
  private func writeZip(_ base64: String) throws -> URL {
    let data = try XCTUnwrap(Data(base64Encoded: base64), "fixture must be valid base64")
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString).appendingPathExtension("zip")
    try data.write(to: url)
    return url
  }

  /// Writes the embedded fixture to a temporary `.zip` and returns its URL.
  private func writeFixture() throws -> URL {
    let data = try XCTUnwrap(Data(base64Encoded: fixtureBase64), "fixture must be valid base64")
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension("zip")
    try data.write(to: url)
    return url
  }

  /// Extraction inflates DEFLATE/STORED entries, recreates the directory tree, and skips macOS metadata.
  func testExtractRoundTripAndSkip() throws {
    let archive = try writeFixture()
    defer { try? FileManager.default.removeItem(at: archive) }

    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    try MiniZip.extract(at: archive, to: destination) { path in
      path.hasPrefix("__MACOSX") || path.contains("._")
    }

    let fileManager = FileManager.default
    let labels = destination.appendingPathComponent("model.mlpackage/labels.txt")
    let blob = destination.appendingPathComponent("model.mlpackage/data/blob.bin")

    // DEFLATE entry inflates to its exact original bytes.
    XCTAssertEqual(
      try String(contentsOf: labels, encoding: .utf8),
      String(repeating: "hello yolo ", count: 24),
      "DEFLATE entry should inflate to original text")

    // STORED entry in a nested subdirectory is copied verbatim.
    XCTAssertEqual(
      try Data(contentsOf: blob), Data([7, 11, 13, 0, 255, 128, 64, 32]),
      "STORED entry should be copied byte-for-byte")

    // macOS resource-fork metadata is filtered out by the skip predicate.
    XCTAssertFalse(
      fileManager.fileExists(atPath: destination.appendingPathComponent("__MACOSX").path),
      "__MACOSX tree must be skipped")
    XCTAssertFalse(
      fileManager.fileExists(
        atPath: destination.appendingPathComponent("model.mlpackage/._labels.txt").path),
      "._ AppleDouble sidecar must be skipped")
  }

  /// Without a skip predicate, every entry — including the macOS junk — is written.
  func testExtractWithoutSkipWritesEverything() throws {
    let archive = try writeFixture()
    defer { try? FileManager.default.removeItem(at: archive) }

    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    try MiniZip.extract(at: archive, to: destination)

    XCTAssertTrue(
      FileManager.default.fileExists(
        atPath: destination.appendingPathComponent("model.mlpackage/._labels.txt").path),
      "default extraction should write all entries")
  }

  /// Non-archive bytes are reported as an error rather than crashing the parser.
  func testGarbageInputThrows() throws {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString).appendingPathExtension("zip")
    try Data(repeating: 0xAB, count: 4096).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }

    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    XCTAssertThrowsError(try MiniZip.extract(at: url, to: destination)) { error in
      XCTAssertTrue(error is MiniZip.MiniZipError, "garbage input should throw a MiniZipError")
    }
  }

  /// A truncated archive (missing its central directory) is rejected, protecting against partial downloads.
  func testTruncatedArchiveThrows() throws {
    let full = try XCTUnwrap(Data(base64Encoded: fixtureBase64))
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString).appendingPathExtension("zip")
    try full.prefix(full.count - 120).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }

    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    XCTAssertThrowsError(try MiniZip.extract(at: url, to: destination))
  }

  /// A valid archive whose trailing comment contains the EOCD signature bytes is still read correctly —
  /// the end-of-central-directory scan must validate the comment length, not stop at the first match.
  func testCommentContainingSignatureIsHandled() throws {
    let url = try writeZip(
      "UEsDBBQAAAAIAAOav1y2t1DJDwAAAFgAAAAaAAAAbW9kZWwubWxwYWNrYWdlL2xhYmVscy50eHTLSM3JyVeozAcSGdRk"
        + "AgBQSwECFAMUAAAACAADmr9ctrdQyQ8AAABYAAAAGgAAAAAAAAAAAAAAgAEAAAAAbW9kZWwubWxwYWNrYWdlL2xhYmVs"
        + "cy50eHRQSwUGAAAAAAEAAQBIAAAARwAAABkAUEsFBkNPTU1FTlRQQURESU5HUEFERElORw==")
    defer { try? FileManager.default.removeItem(at: url) }
    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    try MiniZip.extract(at: url, to: destination)
    XCTAssertEqual(
      try String(
        contentsOf: destination.appendingPathComponent("model.mlpackage/labels.txt"),
        encoding: .utf8),
      String(repeating: "hello yolo ", count: 8),
      "EOCD scan must skip a signature embedded in the archive comment")
  }

  /// A zero-byte entry has a legitimate CRC-32 of 0; it must extract instead of being flagged corrupt.
  func testEmptyFileEntryExtracts() throws {
    let url = try writeZip(
      "UEsDBBQAAAAIAAOav1wAAAAAAgAAAAAAAAAZAAAAbW9kZWwubWxwYWNrYWdlL2VtcHR5LmJpbgMAUEsDBBQAAAAIAAOav1xH"
        + "3dx5BAAAAAIAAAAYAAAAbW9kZWwubWxwYWNrYWdlL25vdGUudHh0y88GAFBLAQIUAxQAAAAIAAOav1wAAAAAAgAAAAAAAAAZ"
        + "AAAAAAAAAAAAAACAAQAAAABtb2RlbC5tbHBhY2thZ2UvZW1wdHkuYmluUEsBAhQDFAAAAAgAA5q/XEfd3HkEAAAAAgAAABgA"
        + "AAAAAAAAAAAAAIABOQAAAG1vZGVsLm1scGFja2FnZS9ub3RlLnR4dFBLBQYAAAAAAgACAI0AAABzAAAAAAA=")
    defer { try? FileManager.default.removeItem(at: url) }
    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    try MiniZip.extract(at: url, to: destination)
    XCTAssertEqual(
      try Data(contentsOf: destination.appendingPathComponent("model.mlpackage/empty.bin")).count,
      0,
      "empty entry should extract to a zero-byte file")
    XCTAssertEqual(
      try String(
        contentsOf: destination.appendingPathComponent("model.mlpackage/note.txt"), encoding: .utf8),
      "ok", "non-empty sibling should still extract")
  }

  /// An entry advertising an implausibly large uncompressed size is rejected before any allocation,
  /// guarding against decompression-bomb out-of-memory attacks from untrusted download URLs.
  func testDecompressionBombRejected() throws {
    // A 4-byte payload whose central-directory uncompressed size has been patched to ~500 MB.
    let url = try writeZip(
      "UEsDBBQAAAAIAAOav1zJl7hQBgAAAAQAAAAZAAAAbW9kZWwubWxwYWNrYWdlL3NtYWxsLnR4dCvJzKsEAFBLAQIUAxQAAAAI"
        + "AAOav1zJl7hQBgAAAABlzR0ZAAAAAAAAAAAAAACAAQAAAABtb2RlbC5tbHBhY2thZ2Uvc21hbGwudHh0UEsFBgAAAAABAAEA"
        + "RwAAAD0AAAAAAA==")
    defer { try? FileManager.default.removeItem(at: url) }
    let destination = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: destination) }

    XCTAssertThrowsError(try MiniZip.extract(at: url, to: destination)) { error in
      guard case MiniZip.MiniZipError.entryTooLarge = error else {
        return XCTFail("expected .entryTooLarge, got \(error)")
      }
    }
  }
}
