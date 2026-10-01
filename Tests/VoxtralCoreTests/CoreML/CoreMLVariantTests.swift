/**
 * CoreMLVariantTests - K-24
 *
 * The Core ML encoder variant follows the model's `text_config.hidden_size` (5120 → Small, 3072 → Mini), not its
 * name: a Small folder called `x/model` used to get the Mini encoder (wrong output width).
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class CoreMLVariantTests: XCTestCase {

    private var sandbox: URL!

    override func setUpWithError() throws {
        sandbox = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-variant-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: sandbox)
    }

    private func folder(_ name: String, hiddenSize: Int) throws -> URL {
        let url = sandbox.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let config = #"{"model_type": "voxtral", "text_config": {"hidden_size": \#(hiddenSize)}}"#
        try Data(config.utf8).write(to: url.appendingPathComponent("config.json"))
        return url
    }

    func testSmallFolderWithAnUninformativeName() throws {
        let url = try folder("x/model", hiddenSize: 5120)
        XCTAssertEqual(VoxtralCoreMLVariant.fromMLXModelRepoId(url.path), .small)
        XCTAssertEqual(VoxtralCoreMLVariant.variant(forConfigAt: url), .small)
    }

    func testMiniFolderNamedSmall() throws {
        let url = try folder("small-looking/name", hiddenSize: 3072)
        XCTAssertEqual(VoxtralCoreMLVariant.fromMLXModelRepoId(url.path), .mini)
    }

    func testRepoIdWithoutLocalFolderFallsBackToTheName() {
        XCTAssertEqual(VoxtralCoreMLVariant.fromMLXModelRepoId("VincentGOURBIN/voxtral-small-4bit-mixed"), .small)
        XCTAssertEqual(VoxtralCoreMLVariant.fromMLXModelRepoId("mzbac/voxtral-mini-3b-8bit"), .mini)
    }
}
