/**
 * ConsolidatedExclusionTests - K-24 (S-07, M-01)
 *
 * Mistral repositories ship their weights twice (transformers shards and `consolidated.safetensors`). The STT and
 * Realtime downloads no longer fetch the copy their loaders never read (9.36 instead of 18.7 GB for Mini 3B); the
 * official TTS pack, whose only weights are `consolidated.safetensors`, still downloads it.
 */

import XCTest
@testable import VoxtralCore

final class ConsolidatedExclusionTests: XCTestCase {

    private let sttRepo = ["config.json", "consolidated.safetensors", "generation_config.json",
                           "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                           "model.safetensors.index.json", "params.json", "preprocessor_config.json", "tekken.json"]
    private let ttsRepo = ["consolidated.safetensors", "params.json", "tekken.json",
                           "voice_embedding/neutral_female.pt", "README.md"]
    private let realtimeRepo = ["config.json", "consolidated.safetensors", "model.safetensors", "params.json", "tekken.json"]

    func testSTTSkipsConsolidated() {
        let kept = ModelDownloader.selectFiles(sttRepo, matching: ModelDownloader.sttDownloadGlobs,
                                               excluding: ModelDownloader.unusedConsolidatedWeights)
        XCTAssertFalse(kept.contains("consolidated.safetensors"))
        XCTAssertTrue(kept.contains("model-00001-of-00002.safetensors"))
        XCTAssertTrue(kept.contains("model-00002-of-00002.safetensors"))
        XCTAssertTrue(kept.contains("tekken.json"))
    }

    func testRealtimeSkipsConsolidated() {
        let kept = ModelDownloader.selectFiles(realtimeRepo, matching: ModelDownloader.realtimeDownloadGlobs,
                                               excluding: ModelDownloader.unusedConsolidatedWeights)
        XCTAssertEqual(Set(kept), ["config.json", "model.safetensors", "params.json", "tekken.json"])
    }

    func testTTSKeepsConsolidated() {
        let kept = ModelDownloader.selectFiles(ttsRepo, matching: ModelDownloader.ttsDownloadGlobs, excluding: [])
        XCTAssertEqual(Set(kept), ["consolidated.safetensors", "params.json", "tekken.json",
                                   "voice_embedding/neutral_female.pt"])
    }
}
