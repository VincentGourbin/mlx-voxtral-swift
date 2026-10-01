/**
 * RegistrySizesTests - K-24 (M-03)
 *
 * Every registry entry (STT, TTS, Realtime) declares the exact bytes of its weight files and a displayed size within
 * ±5 % of the Hub bytes recorded in docs/Weights.md §1, and its real precision (the Mistral STT packs are bf16, not
 * "float16"). Before K-24 the sizes were rough guesses ("~6 GB" for 9.36 GB).
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class RegistrySizesTests: XCTestCase {

    /// id → bytes, from the table of docs/Weights.md §1 ("| STT · `mini-3b` ★ | … | 9 356 474 312 | …")
    private func hubBytes() throws -> [String: Int64] {
        let doc = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("docs/Weights.md")
        var bytes: [String: Int64] = [:]
        for line in try String(contentsOf: doc, encoding: .utf8).components(separatedBy: "\n") where line.hasPrefix("| ") {
            let cells = line.split(separator: "|", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
            guard cells.count > 4, cells[1].contains(" · `"),
                  let id = cells[1].split(separator: "`").dropFirst().first else { continue }
            let digits = cells[4].prefix { $0.isNumber || $0 == " " }.filter(\.isNumber)
            if let value = Int64(digits) { bytes[String(id)] = value }
        }
        return bytes
    }

    /// "9.36 GB" → bytes (GB = 10⁹)
    private func displayedBytes(_ size: String) -> Double? {
        Double(size.replacingOccurrences(of: " GB", with: "")).map { $0 * 1e9 }
    }

    func testEveryEntryMatchesTheHubBytes() throws {
        let hub = try hubBytes()
        XCTAssertEqual(hub.count, 13, "13 registry ids in docs/Weights.md §1")
        let entries: [(String, String, Int64?)] =
            ModelRegistry.models.map { ($0.id, $0.size, $0.approximateBytes) }
            + VoxtralTTSRegistry.models.map { ($0.id, $0.size, $0.approximateBytes) }
            + VoxtralRealtimeRegistry.models.map { ($0.id, $0.size, $0.approximateBytes) }
        for (id, size, bytes) in entries {
            let expected = try XCTUnwrap(hub[id], "\(id) missing from docs/Weights.md")
            let declared = try XCTUnwrap(bytes, "\(id): no approximateBytes")
            XCTAssertEqual(Double(declared), Double(expected), accuracy: 0.05 * Double(expected), id)
            let shown = try XCTUnwrap(displayedBytes(size), "\(id): size \"\(size)\" is not \"<x> GB\"")
            XCTAssertEqual(shown, Double(expected), accuracy: 0.05 * Double(expected), "\(id): \(size)")
        }
        print("[registry] \(entries.count) entries within ±5 % of docs/Weights.md")
    }

    func testMistralSTTPacksAreBF16() throws {
        for id in ["mini-3b", "small-24b"] {
            XCTAssertEqual(try XCTUnwrap(ModelRegistry.model(withId: id)).quantization, "bfloat16", id)
        }
    }
}
