/**
 * TekkenReferenceIdsTests - K-88 (suite of K-7: the real-file comparison left with the legacy initializer, K-31)
 *
 * On a model's real `tekken.json`, the Swift tokenizer gives the ids of Mistral's own `mistral_common` for 20 FR/EN
 * sentences (`Fixtures/tekken/reference-ids.json`). Run on every STT, TTS and Realtime pack on disk; skipped when none is.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class TekkenReferenceIdsTests: XCTestCase {

    private struct Reference: Decodable {
        struct Case: Decodable { let text: String; let ids: [Int] }
        let cases: [Case]
    }

    private func folders() -> [(String, URL)] {
        var found: [(String, URL)] = []
        if let info = VoxtralModelRegistry.model(withId: "mini-3b-8bit"),
           let url = VoxtralModelDownloader.findModelPath(for: info) { found.append((info.id, url)) }
        if let info = VoxtralTTSRegistry.model(withId: "tts-4b-6bit"),
           let url = VoxtralModelDownloader.findTTSModelPath(for: info) { found.append((info.id, url)) }
        if let info = VoxtralRealtimeRegistry.model(withId: "realtime-4b-4bit"),
           let url = VoxtralModelDownloader.findRealtimeModelPath(for: info) { found.append((info.id, url)) }
        return found
    }

    func testIdsMatchMistralCommonOnTheRealTekken() throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/tekken/reference-ids.json")
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixture))
        XCTAssertEqual(reference.cases.count, 20)
        let folders = folders()
        guard !folders.isEmpty else { throw XCTSkip("no pack with a tekken.json is downloaded") }
        for (id, folder) in folders {
            let tokenizer = try TekkenTokenizer.load(modelPath: folder.path)
            var identical = 0
            for item in reference.cases {
                let ids = tokenizer.encode(item.text)
                XCTAssertEqual(ids, item.ids, "\(id): \(item.text)")
                if ids == item.ids { identical += 1 }
            }
            print("[k88] TEKKEN \(id) : \(identical)/\(reference.cases.count) identical to mistral_common")
        }
    }
}
