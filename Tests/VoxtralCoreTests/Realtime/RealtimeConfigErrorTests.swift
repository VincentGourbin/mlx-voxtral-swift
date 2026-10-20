/**
 * RealtimeConfigErrorTests - K-87
 *
 * Next to a params.json, an unreadable config.json (invalid JSON, or the mlx schema with a field missing) is reported
 * with `invalidConfiguration` naming the file, instead of being replaced by params.json in silence. The transformers
 * config.json of the original Mistral repository still gives way to params.json (K-9, `RealtimeOriginalRepoTests`).
 * Fixtures: the realtime-mlx config.json and the realtime-original params.json, copied and damaged.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class RealtimeConfigErrorTests: XCTestCase {

    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }

    /// A folder with the original params.json and `config` as config.json
    private func folder(config: Data) throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-rtconfig-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("realtime-original/params.json"),
                                         to: folder.appendingPathComponent("params.json"))
        try config.write(to: folder.appendingPathComponent("config.json"))
        return folder
    }

    private func assertNamesConfigJSON(_ folder: URL, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try loadRealtimeConfig(from: folder), what, file: file, line: line) { error in
            guard case VoxtralRealtimeError.invalidConfiguration(let message) = error else {
                return XCTFail("\(what): \(error)", file: file, line: line)
            }
            XCTAssertTrue(message.contains(folder.appendingPathComponent("config.json").path), "\(what): \(message)",
                          file: file, line: line)
            print("[k87] RTCONFIG \(what) : \(message.replacingOccurrences(of: folder.path, with: "<dir>"))")
        }
    }

    func testInvalidJSONNextToParamsIsReported() throws {
        assertNamesConfigJSON(try folder(config: Data("{ \"decoder\": ".utf8)), "invalid JSON")
    }

    func testIncompleteMLXSchemaNextToParamsIsReported() throws {
        var mlx = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(contentsOf: fixtures.appendingPathComponent("realtime-mlx/config.json"))) as? [String: Any])
        var decoder = try XCTUnwrap(mlx["decoder"] as? [String: Any])
        decoder.removeValue(forKey: "dim")
        mlx["decoder"] = decoder
        assertNamesConfigJSON(try folder(config: JSONSerialization.data(withJSONObject: mlx)), "mlx schema without decoder.dim")
    }

    func testTransformersConfigStillGivesWayToParams() throws {
        let config = try loadRealtimeConfig(from: try folder(
            config: Data(contentsOf: fixtures.appendingPathComponent("realtime-original/config.json"))))
        XCTAssertEqual(config.decoder.dim, 3072)
        XCTAssertNil(config.quantization)
    }
}
