/**
 * VerifiedLoadingTests - K-7 (S-04, MLX-018)
 *
 * A model folder missing a shard must fail to load and name the missing keys, instead of
 * leaving those weights at their random initialization. Uses the local mini-3b-8bit pack
 * (skipped when it is not downloaded): the test folder links every file but the first shard.
 *
 * Written against APIs that existed before K-7 so it compiles, and fails, without the fix.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class VerifiedLoadingTests: XCTestCase {

    func testFolderMissingAShardNamesTheMissingKeys() throws {
        guard let info = VoxtralModelRegistry.model(withId: "mini-3b-8bit"),
              let source = VoxtralModelDownloader.findModelPath(for: info) else {
            throw XCTSkip("mini-3b-8bit is not downloaded")
        }
        let fm = FileManager.default
        let folder = fm.temporaryDirectory.appendingPathComponent("voxtral-missing-shard-\(UUID().uuidString)")
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: folder) }
        for name in try fm.contentsOfDirectory(atPath: source.path)
        where !name.hasPrefix(".") && name != "model-00001-of-00002.safetensors" {
            try fm.createSymbolicLink(at: folder.appendingPathComponent(name),
                                      withDestinationURL: source.appendingPathComponent(name))
        }

        XCTAssertThrowsError(try loadVoxtralStandardModel(modelPath: folder.path)) { error in
            let text = "\(error)"
            let keys = text.split(separator: "\"").filter { $0.contains(".layers.") }
            print("[verified-loading] \(keys.count) missing layer keys, e.g. \(keys.first ?? "none")")
            XCTAssertTrue(text.contains("missingWeights"), "expected a missing-weights error, got \(text.prefix(200))")
            XCTAssertFalse(keys.isEmpty, "the error must name ≥ 1 missing layer key")
        }
    }
}
