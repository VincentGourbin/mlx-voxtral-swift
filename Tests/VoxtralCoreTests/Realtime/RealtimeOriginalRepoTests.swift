/**
 * RealtimeOriginalRepoTests - K-9 (M-01, ASK-17)
 *
 * The original Mistral repository (mistralai/Voxtral-Mini-4B-Realtime-2602) also ships a transformers config.json and
 * model.safetensors: the loader skips that config.json for params.json (Mistral format, no quantization), the
 * mlx-community config.json is still read, and the pipeline refuses an unknown model id instead of silently loading
 * the default. Fixtures: the two config files of each repository (Tests/VoxtralCoreTests/Fixtures).
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class RealtimeOriginalRepoTests: XCTestCase {

    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }

    func testTransformersConfigIsSkippedForParamsJSON() throws {
        let config = try loadRealtimeConfig(from: fixtures.appendingPathComponent("realtime-original"))
        XCTAssertNil(config.quantization, "the original checkpoint is not quantized")
        XCTAssertEqual(config.decoder.dim, 3072)
        XCTAssertEqual(config.decoder.nLayers, 26)
        XCTAssertEqual(config.encoderArgs.slidingWindow, 750)
    }

    func testMLXCommunityConfigIsStillRead() throws {
        let config = try loadRealtimeConfig(from: fixtures.appendingPathComponent("realtime-mlx"))
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.decoder.dim, 3072)
    }

    func testUnknownModelIdThrows() {
        XCTAssertThrowsError(try VoxtralRealtimePipeline.modelInfo(for: "realtime-9b")) { error in
            guard case VoxtralRealtimeError.invalidConfiguration = error else { return XCTFail("\(error)") }
        }
    }

    func testNilModelIdIsTheDefault() throws {
        XCTAssertEqual(try VoxtralRealtimePipeline.modelInfo(for: nil).id, VoxtralRealtimeRegistry.defaultModel.id)
    }

    /// Real checkpoint (VOXTRAL_RT_ORIGINAL_DIR = a downloaded realtime-4b folder): every model key is loaded (the
    /// verified update checks missing keys and shapes) and no checkpoint key is left unused
    func testOriginalCheckpointKeysMatchTheModel() throws {
        guard let path = ProcessInfo.processInfo.environment["VOXTRAL_RT_ORIGINAL_DIR"] else {
            throw XCTSkip("Set VOXTRAL_RT_ORIGINAL_DIR to a downloaded realtime-4b folder")
        }
        let dir = URL(fileURLWithPath: path)
        let model = try loadVoxtralRealtimeModel(from: dir)
        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let checkpointKeys = Set(sanitizeRealtimeWeights(try loadAllRealtimeWeights(from: dir)).keys)
        let missing = modelKeys.subtracting(checkpointKeys), unused = checkpointKeys.subtracting(modelKeys)
        print("[realtime-4b] LOAD verify: \(missing.count) missing, \(unused.count) unused (\(checkpointKeys.count) keys) \(unused.sorted().prefix(5))")
        XCTAssertTrue(missing.isEmpty, "\(missing.sorted().prefix(10))")
        XCTAssertTrue(unused.isEmpty, "\(unused.sorted().prefix(10))")
    }
}
