/**
 * STTComputeDTypeTests - K-40 (P-01, P-05)
 *
 * The STT forward pass computes in the decoder's dtype: an fp32 mel, fp32 encoder positions (the bf16 checkpoint
 * stores them in fp32) or bf16 embedding scales next to fp16 weights (the quantized packs) no longer promote the
 * encoder, the KV cache and the logits to fp32.
 */

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest
@testable import VoxtralCore

final class STTComputeDTypeTests: XCTestCase {

    /// Reduced model in `dtype`, with the checkpoint's odd tensor (`override`) kept in its own dtype
    private func model(_ dtype: DType, override: (suffix: String, dtype: DType)) throws -> VoxtralForConditionalGeneration {
        let model = try makeReducedVoxtralModel()
        let weights = try XCTUnwrap(model.standardModel)  // owns the encoder, projector and decoder actually run
        var overridden = 0
        let params = weights.parameters().flattened().map { key, value in
            let keep = key.hasSuffix(override.suffix)
            if keep { overridden += 1 }
            return (key, value.asType(keep ? override.dtype : dtype))
        }
        XCTAssertEqual(overridden, 1, "one \(override.suffix) tensor")
        weights.update(parameters: ModuleParameters.unflattened(params))
        return model
    }

    /// Prefill of 4 text tokens around the 375 audio positions of one 30 s window (reduced audio tower)
    private func prefill(_ model: VoxtralForConditionalGeneration) throws -> (logits: MLXArray, cache: [any KVCache]) {
        let audioTokenId = Int32(model.config.audio_token_id)
        let ids = [Int32(30), 31] + Array(repeating: audioTokenId, count: 375) + [Int32(32), 33]
        let inputIds = MLXArray(ids).reshaped([1, ids.count])
        let mel = MLXRandom.normal([1, 128, 3000], key: MLXRandom.key(7))  // fp32, as the feature extractor gives it
        let cache: [any KVCache] = (0 ..< model.getLanguageModelLayerCount()).map { _ in KVCacheSimple() }
        var logits = MLXArray(0)
        try withMLXErrors { errors in
            logits = model(inputIds: inputIds, inputFeatures: mel, pastKeyValues: cache).logits
            MLX.eval(logits)
            try errors.check()
        }
        return (logits, cache)
    }

    func testBF16CheckpointWithFP32PositionsComputesInBF16() throws {
        let (logits, cache) = try prefill(try model(.bfloat16, override: ("embedPositions.weight", .float32)))
        print("[dtype] bf16 checkpoint: kv=\(cache[0].state[0].dtype) logits=\(logits.dtype)")
        XCTAssertEqual(cache[0].state[0].dtype, .bfloat16, "KV cache after prefill")
        XCTAssertEqual(logits.dtype, .bfloat16)
    }

    func testFP16WeightsWithBF16EmbeddingsComputeInFP16() throws {
        let (logits, cache) = try prefill(try model(.float16, override: ("embedTokens.weight", .bfloat16)))
        print("[dtype] fp16 weights, bf16 embeddings: kv=\(cache[0].state[0].dtype) logits=\(logits.dtype)")
        XCTAssertEqual(cache[0].state[0].dtype, .float16, "KV cache after prefill")
        XCTAssertEqual(logits.dtype, .float16)
    }

    func testEncoderOutputTakesTheConvDType() throws {
        let model = try model(.bfloat16, override: ("embedPositions.weight", .float32))
        let encoder = try XCTUnwrap(model.standardModel?.audioTower)
        let out = encoder(MLXRandom.normal([1, 128, 3000], key: MLXRandom.key(5)))
        XCTAssertEqual(out.dtype, .bfloat16)
    }
}
