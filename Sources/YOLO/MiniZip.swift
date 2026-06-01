// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing dependency-free ZIP extraction.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  MiniZip is a minimal, self-contained reader for standard PKZIP (.zip) archives built only on Apple's
//  Foundation and Compression frameworks — no third-party dependencies. It parses the central directory and
//  inflates STORED (method 0) and DEFLATE (method 8) entries, which together cover the model archives the SDK
//  downloads. Apple's `COMPRESSION_ZLIB` decodes the raw DEFLATE streams that ZIP method 8 stores. Unsupported
//  features (encryption, ZIP64, other compression methods) are rejected rather than silently mis-extracted.

import Compression
import Foundation

/// A minimal, dependency-free extractor for standard PKZIP `.zip` archives.
public enum MiniZip {

  /// Errors thrown while parsing or extracting a ZIP archive.
  public enum MiniZipError: LocalizedError {
    case notAZipFile
    case corruptArchive
    case unsupportedFeature(String)
    case inflateFailed(String)
    case unsafePath(String)
    case entryTooLarge(String)

    public var errorDescription: String? {
      switch self {
      case .notAZipFile:
        return "Not a valid ZIP archive (end-of-central-directory record not found)"
      case .corruptArchive:
        return "Corrupt ZIP archive (unexpected record signature or truncated data)"
      case .unsupportedFeature(let f): return "Unsupported ZIP feature: \(f)"
      case .inflateFailed(let path): return "Failed to inflate entry: \(path)"
      case .unsafePath(let path): return "Refusing to extract entry outside destination: \(path)"
      case .entryTooLarge(let path):
        return "Entry advertises an implausible decompressed size (possible ZIP bomb): \(path)"
      }
    }
  }

  /// A single file or directory record parsed from the archive's central directory.
  private struct Entry {
    let path: String
    let compressionMethod: UInt16
    let compressedSize: Int
    let uncompressedSize: Int
    let crc32: UInt32
    let dataOffset: Int  // Absolute byte offset of the entry's file data within the archive.
  }

  // ZIP record signatures (little-endian).
  private static let endOfCentralDirectorySignature: UInt32 = 0x0605_4b50
  private static let centralDirectorySignature: UInt32 = 0x0201_4b50
  private static let localFileHeaderSignature: UInt32 = 0x0403_4b50
  private static let zip64SizeSentinel: UInt32 = 0xFFFF_FFFF

  // A single raw-DEFLATE stream cannot expand by more than ~1032:1; entries claiming more than this for
  // their compressed size are rejected as decompression bombs before any allocation.
  private static let maxDeflateExpansionRatio = 1032

  /// Extracts every entry of the ZIP at `archiveURL` into `destinationURL`.
  ///
  /// Parent directories are created as needed. Entries for which `skip` returns `true` (matched on their
  /// archive-relative path) are ignored — the downloader uses this to drop macOS resource-fork metadata.
  ///
  /// - Parameters:
  ///   - archiveURL: The `.zip` file to read.
  ///   - destinationURL: The directory to extract into (created if absent).
  ///   - skip: Predicate evaluated against each entry's path; matching entries are not written.
  public static func extract(
    at archiveURL: URL, to destinationURL: URL, skip: (String) -> Bool = { _ in false }
  ) throws {
    // Memory-map when safe so large model archives are not loaded wholesale into RAM.
    let data = try Data(contentsOf: archiveURL, options: .mappedIfSafe)
    let entries = try parseCentralDirectory(data)

    let fileManager = FileManager.default
    try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
    let root = destinationURL.standardizedFileURL.path

    for entry in entries {
      if skip(entry.path) { continue }

      let destination = destinationURL.appendingPathComponent(entry.path)

      // Guard against Zip Slip: a crafted entry path must not escape the destination directory.
      let resolved = destination.standardizedFileURL.path
      guard resolved == root || resolved.hasPrefix(root + "/") else {
        throw MiniZipError.unsafePath(entry.path)
      }

      // Directory entries carry a trailing slash and no data.
      if entry.path.hasSuffix("/") {
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
        continue
      }

      try fileManager.createDirectory(
        at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

      let payload = try contents(of: entry, in: data)
      try payload.write(to: destination)
    }
  }

  /// Parses the archive's central directory into a list of entries.
  private static func parseCentralDirectory(_ data: Data) throws -> [Entry] {
    let count = data.count
    guard count >= 22 else { throw MiniZipError.notAZipFile }

    // Locate the end-of-central-directory record by scanning backwards. The trailing comment may be up to
    // 0xFFFF bytes, so search at most that far plus the 22-byte record itself.
    let searchLimit = min(count, 22 + 0xFFFF)
    var eocd = -1
    var i = count - 22
    while i >= count - searchLimit {
      // A real EOCD's comment-length field must point exactly at the end of the file. Verifying this
      // rejects stray signature bytes that happen to appear inside a trailing archive comment.
      if u32(data, i) == endOfCentralDirectorySignature, i + 22 + Int(u16(data, i + 20)) == count {
        eocd = i
        break
      }
      i -= 1
    }
    guard eocd >= 0 else { throw MiniZipError.notAZipFile }

    let entryCount = Int(u16(data, eocd + 10))
    let centralDirectoryOffset = Int(u32(data, eocd + 16))
    guard centralDirectoryOffset <= count else { throw MiniZipError.corruptArchive }

    var entries: [Entry] = []
    entries.reserveCapacity(entryCount)

    var p = centralDirectoryOffset
    for _ in 0..<entryCount {
      guard p + 46 <= count, u32(data, p) == centralDirectorySignature else {
        throw MiniZipError.corruptArchive
      }

      let flags = u16(data, p + 8)
      let method = u16(data, p + 10)
      let crc = u32(data, p + 16)
      let compressedSize = u32(data, p + 20)
      let uncompressedSize = u32(data, p + 24)
      let nameLength = Int(u16(data, p + 28))
      let extraLength = Int(u16(data, p + 30))
      let commentLength = Int(u16(data, p + 32))
      let localHeaderOffset = Int(u32(data, p + 42))

      if flags & 0x1 != 0 { throw MiniZipError.unsupportedFeature("encryption") }
      if compressedSize == zip64SizeSentinel || uncompressedSize == zip64SizeSentinel
        || UInt32(localHeaderOffset) == zip64SizeSentinel
      {
        throw MiniZipError.unsupportedFeature("ZIP64")
      }

      let nameStart = p + 46
      guard nameStart + nameLength <= count else { throw MiniZipError.corruptArchive }
      let name = String(
        decoding: data[(data.startIndex + nameStart)..<(data.startIndex + nameStart + nameLength)],
        as: UTF8.self)

      // The local header repeats the name/extra fields, whose lengths can differ from the central record,
      // so the data offset must be computed from the local header, not the central directory.
      guard localHeaderOffset + 30 <= count,
        u32(data, localHeaderOffset) == localFileHeaderSignature
      else { throw MiniZipError.corruptArchive }
      let localNameLength = Int(u16(data, localHeaderOffset + 26))
      let localExtraLength = Int(u16(data, localHeaderOffset + 28))
      let dataOffset = localHeaderOffset + 30 + localNameLength + localExtraLength
      guard dataOffset + Int(compressedSize) <= count else { throw MiniZipError.corruptArchive }

      entries.append(
        Entry(
          path: name, compressionMethod: method, compressedSize: Int(compressedSize),
          uncompressedSize: Int(uncompressedSize), crc32: crc, dataOffset: dataOffset))

      p = nameStart + nameLength + extraLength + commentLength
    }

    return entries
  }

  /// Returns the decompressed bytes for `entry`, verifying its CRC-32.
  private static func contents(of entry: Entry, in data: Data) throws -> Data {
    let start = data.startIndex + entry.dataOffset
    let raw = data.subdata(in: start..<(start + entry.compressedSize))

    let output: Data
    switch entry.compressionMethod {
    case 0:  // STORED — data is already uncompressed.
      output = raw
    case 8:  // DEFLATE — ZIP stores a raw DEFLATE stream, which COMPRESSION_ZLIB decodes.
      output = try inflate(raw, expectedSize: entry.uncompressedSize, path: entry.path)
    default:
      throw MiniZipError.unsupportedFeature("compression method \(entry.compressionMethod)")
    }

    // The central directory always carries the real CRC-32, including the legitimate value 0 (e.g. empty
    // files), so verify unconditionally rather than treating 0 as "absent".
    if crc32(output) != entry.crc32 {
      throw MiniZipError.corruptArchive
    }
    return output
  }

  /// Inflates a raw DEFLATE stream to `expectedSize` bytes using the Compression framework.
  private static func inflate(_ input: Data, expectedSize: Int, path: String) throws -> Data {
    guard expectedSize > 0 else { return Data() }

    // Reject decompression bombs before allocating: a crafted header can advertise a huge uncompressed
    // size to force an out-of-memory allocation. Bound the claim by DEFLATE's maximum expansion ratio.
    guard expectedSize <= input.count * maxDeflateExpansionRatio else {
      throw MiniZipError.entryTooLarge(path)
    }

    var output = Data(count: expectedSize)
    let written = output.withUnsafeMutableBytes { destination -> Int in
      input.withUnsafeBytes { source -> Int in
        guard let destinationBase = destination.bindMemory(to: UInt8.self).baseAddress,
          let sourceBase = source.bindMemory(to: UInt8.self).baseAddress
        else { return 0 }
        return compression_decode_buffer(
          destinationBase, expectedSize, sourceBase, input.count, nil, COMPRESSION_ZLIB)
      }
    }
    guard written == expectedSize else { throw MiniZipError.inflateFailed(path) }
    return output
  }

  // MARK: - Little-endian readers

  /// Reads a little-endian `UInt16` at byte offset `offset`.
  private static func u16(_ data: Data, _ offset: Int) -> UInt16 {
    let base = data.startIndex + offset
    return UInt16(data[base]) | (UInt16(data[base + 1]) << 8)
  }

  /// Reads a little-endian `UInt32` at byte offset `offset`.
  private static func u32(_ data: Data, _ offset: Int) -> UInt32 {
    let base = data.startIndex + offset
    return UInt32(data[base]) | (UInt32(data[base + 1]) << 8) | (UInt32(data[base + 2]) << 16)
      | (UInt32(data[base + 3]) << 24)
  }

  // MARK: - CRC-32 (IEEE 802.3 / PKZIP)

  /// Precomputed CRC-32 lookup table using the standard reversed polynomial `0xEDB88320`.
  private static let crcTable: [UInt32] = (0..<256).map { index in
    var value = UInt32(index)
    for _ in 0..<8 {
      value = (value & 1) != 0 ? (0xEDB8_8320 ^ (value >> 1)) : (value >> 1)
    }
    return value
  }

  /// Computes the CRC-32 of `data` to validate extracted entries against the archive's stored checksum.
  private static func crc32(_ data: Data) -> UInt32 {
    var crc: UInt32 = 0xFFFF_FFFF
    data.withUnsafeBytes { buffer in
      for byte in buffer.bindMemory(to: UInt8.self) {
        crc = crcTable[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
      }
    }
    return crc ^ 0xFFFF_FFFF
  }
}
