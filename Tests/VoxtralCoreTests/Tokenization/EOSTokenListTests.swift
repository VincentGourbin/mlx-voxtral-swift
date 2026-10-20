/**
 * EOSTokenListTests - K-88 (suite of K-27)
 *
 * `generation_config.json` may give `eos_token_id` as one id or as a list (Hugging Face writes both): for every STT
 * pack of the registry on disk, the tokenizer loads with either form and keeps the first id, and a value that is
 * neither still fails with `invalidTokenizer` (K-27). The folder links the pack and replaces only that file.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class EOSTokenListTests: XCTestCase {

    private let fm = FileManager.default

    private func folder(from source: URL, eos: Any) throws -> URL {
        let folder = fm.temporaryDirectory.appendingPathComponent("voxtral-eos-\(UUID().uuidString)")
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        for name in try fm.contentsOfDirectory(atPath: source.path)
        where !name.hasPrefix(".") && name != "generation_config.json" {
            try fm.createSymbolicLink(at: folder.appendingPathComponent(name),
                                      withDestinationURL: source.appendingPathComponent(name))
        }
        try JSONSerialization.data(withJSONObject: ["bos_token_id": 1, "eos_token_id": eos, "pad_token_id": 11])
            .write(to: folder.appendingPathComponent("generation_config.json"))
        return folder
    }

    func testEOSAsAnIdOrAListForEveryPack() throws {
        let packs = VoxtralModelRegistry.models.compactMap { info in
            VoxtralModelDownloader.findModelPath(for: info).map { (info.id, $0) }
        }
        guard !packs.isEmpty else { throw XCTSkip("no STT pack is downloaded") }
        for (id, source) in packs {
            for (eos, expected) in [(2 as Any, 2), ([2] as Any, 2), ([7, 2] as Any, 7)] {
                let tokenizer = try TekkenTokenizer.load(modelPath: folder(from: source, eos: eos).path)
                XCTAssertEqual(tokenizer.eosToken, expected, "\(id), eos_token_id \(eos)")
            }
            XCTAssertThrowsError(try TekkenTokenizer.load(modelPath: folder(from: source, eos: "2").path), id) { error in
                guard case VoxtralError.invalidTokenizer = error else { return XCTFail("\(id): \(error)") }
            }
            print("[k88] EOS \(id) : id, [id], [a, b] read ; text refused")
        }
    }
}
