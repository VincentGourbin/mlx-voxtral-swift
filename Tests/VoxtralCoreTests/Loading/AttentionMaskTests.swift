/**
 * AttentionMaskTests - K-3 (P-02, P-17, MLX-019)
 *
 * The STT decoder's attention masks come from the cache (boolean, shaped like the keys it presents), not from a
 * hand-made additive fp32 mask: a bf16 model no longer fails with "Mask type must promote to output type", a
 * wrapped RotatingKVCache prefilled in two chunks gets the right shape, and the legacy decoder no longer stops at
 * its second prefill chunk (its [T, T] mask did not match offset + T keys).
 * The legacy test builds the legacy decoder as loadVoxtralModel(modelPath:dtype:lazy:) does, reduced.
 */

import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import VoxtralCore

final class AttentionMaskTests: XCTestCase {

    func testBF16PrefillDoesNotFailMaskPromotion() throws {
        let model = try makeReducedVoxtralModel()
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let inputIds = MLXArray((0 ..< 16).map { Int32(30 + $0) }).reshaped([1, 16])
        XCTAssertNoThrow(try model.generateStream(inputIds: inputIds, maxNewTokens: 1, memoryOptimization: .disabled))
    }

    func testMaskIsBooleanAndShapedLikeTheKeys() {
        let cache = KVCacheSimple()
        _ = cache.update(keys: MLXArray.zeros([1, 2, 5, 16]), values: MLXArray.zeros([1, 2, 5, 16]))
        let mask = LlamaStandardModel.causalMask(n: 3, cache: cache)
        XCTAssertEqual(mask?.dtype, .bool)
        XCTAssertEqual(mask?.shape, [3, 8])  // offset 5 + 3 new positions
        XCTAssertNil(LlamaStandardModel.causalMask(n: 1, cache: cache))
    }

    func testWrappedRotatingCacheTwoChunkPrefill() throws {
        let model = try makeReducedVoxtralModel()
        let language = try XCTUnwrap(model.language_model as? LlamaStandardModel)
        let caches: [any KVCache] = (0 ..< 2).map { _ in RotatingKVCache(maxSize: 8, keep: 0) }
        let embeddings = MLXRandom.normal([1, 12, 64], key: MLXRandom.key(3))
        var keysShapes: [[Int]] = []
        try withMLXErrors { errors in
            for chunk in [embeddings[0..., 0 ..< 6, 0...], embeddings[0..., 6 ..< 12, 0...]] {
                let mask = try XCTUnwrap(LlamaStandardModel.causalMask(n: chunk.dim(1), cache: caches[0]))
                XCTAssertEqual(mask.dtype, .bool)
                let out = language(inputs: nil, mask: nil, cache: caches, inputsEmbeds: chunk)
                MLX.eval(out)
                try errors.check()
                keysShapes.append([mask.dim(0), mask.dim(1), caches[0].state[0].dim(2)])
            }
        }
        print("[mask] rotating chunks (mask rows, mask cols, keys) = \(keysShapes)")
        XCTAssertEqual(keysShapes.count, 2)
        XCTAssertEqual(keysShapes[1][1], keysShapes[1][2], "the mask spans the keys of the wrapped cache")
    }

    /// The legacy decoder (LlamaModel), built the way loadVoxtralModel(modelPath:dtype:lazy:) builds it
    /// (VoxtralForConditionalGeneration(config:) from the config dictionaries), reduced and with random weights:
    /// that loader cannot load the only bf16 folder on the machine (keyNotFound audio_tower.conv2.weight after its
    /// sanitize, a separate legacy-loader defect), so the decoder path is exercised directly.
    func testLegacyPrompt600PositionsDoesNotStop() throws {
        let text: [String: Any] = [
            "vocab_size": 2048, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5, "rope_theta": 1_000_000.0,
        ]
        let audio: [String: Any] = [
            "hidden_size": 32, "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 2,
            "head_dim": 16, "max_source_positions": 1500, "num_mel_bins": 128,
        ]
        let config = PythonVoxtralConfig(
            audio_config: VoxtralEncoderConfig.fromDictionary(audio), text_config: VoxtralTextConfig.fromDictionary(text),
            audio_token_id: 24, projector_hidden_act: "gelu")
        MLXRandom.seed(5)
        let model = VoxtralForConditionalGeneration(config: config)
        XCTAssertTrue(model.language_model is LlamaModel, "the legacy initializer builds the legacy decoder")
        let inputIds = MLXArray((0 ..< 600).map { Int32(1000 + $0 % 500) }).reshaped([1, 600])
        XCTAssertNoThrow(try model.generateStream(inputIds: inputIds, maxNewTokens: 1, memoryOptimization: .disabled))
        print("[mask] LEGACY prompt 600 positions (legacy decoder, 2 prefill chunks) : OK (pas d'arrêt)")
    }
}
