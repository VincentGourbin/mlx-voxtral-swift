/**
 * MalformedQuantizationLoadingTests - K-88 (suite of K-27 and K-8)
 *
 * A quantization block that cannot be read stops each live loader (STT, TTS, Realtime) with an error instead of a
 * forced cast or a silently unquantized model: TTS with `invalidConfiguration` naming the block, STT with the
 * configuration's decoding error at `quantization`, Realtime with its configuration error (the message is logged). The folder links a real
 * quantized pack and replaces only its `config.json` (skipped when the pack is not downloaded; no GPU work: the error
 * comes before any weight is applied).
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class MalformedQuantizationLoadingTests: XCTestCase {

    private let fm = FileManager.default

    /// The malformed blocks: not an object, `bits` missing, `group_size` of the wrong type, empty
    private let blocks: [(name: String, value: Any)] = [
        ("not an object", "8bit"),
        ("without bits", ["group_size": 64]),
        ("group_size as text", ["group_size": "64", "bits": 4]),
        ("empty object", [String: Any]()),
    ]

    /// A copy of `source` whose `config.json` has `block` as its quantization (the original keys otherwise)
    private func folder(from source: URL, block: Any) throws -> URL {
        let folder = fm.temporaryDirectory.appendingPathComponent("voxtral-badquant-\(UUID().uuidString)")
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        for name in try fm.contentsOfDirectory(atPath: source.path) where !name.hasPrefix(".") && name != "config.json" {
            try fm.createSymbolicLink(at: folder.appendingPathComponent(name),
                                      withDestinationURL: source.appendingPathComponent(name))
        }
        var config = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json")))
                as? [String: Any])
        config.removeValue(forKey: "quantization_config")
        config["quantization"] = block
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        return folder
    }

    private func check(_ pack: String, _ source: URL?, load: (URL) throws -> Void, expected: (Error) -> Bool) throws {
        guard let source else { throw XCTSkip("\(pack) is not downloaded") }
        for block in blocks {
            let folder = try folder(from: source, block: block.value)
            XCTAssertThrowsError(try load(folder), "\(pack), \(block.name)") { error in
                print("[k88] BADQUANT \(pack), \(block.name): \(String("\(error)".prefix(200)))")
                XCTAssertTrue(expected(error), "\(pack), \(block.name): \(error)")
            }
        }
    }

    func testSTTLoaderRefusesAMalformedQuantization() throws {
        let info = try XCTUnwrap(VoxtralModelRegistry.model(withId: "mini-3b-8bit"))
        try check("mini-3b-8bit", VoxtralModelDownloader.findModelPath(for: info),
                  load: { _ = try loadVoxtralStandardModel(modelPath: $0.path) },
                  expected: { "\($0)".contains("quantization") })
    }

    func testTTSLoaderRefusesAMalformedQuantization() throws {
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-6bit"))
        try check("tts-4b-6bit", VoxtralModelDownloader.findTTSModelPath(for: info),
                  load: { _ = try loadVoxtralTTSModel(from: $0) },
                  expected: {
                      guard case VoxtralTTSError.invalidConfiguration(let message) = $0 else { return false }
                      return message.contains("quantization")
                  })
    }

    func testRealtimeLoaderRefusesAMalformedQuantization() throws {
        let info = try XCTUnwrap(VoxtralRealtimeRegistry.model(withId: "realtime-4b-4bit"))
        try check("realtime-4b-4bit", VoxtralModelDownloader.findRealtimeModelPath(for: info),
                  load: { _ = try loadVoxtralRealtimeModel(from: $0) },
                  expected: { $0 is DecodingError || "\($0)".contains("quantization") })
    }
}
