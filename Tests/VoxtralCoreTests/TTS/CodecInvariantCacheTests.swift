/**
 * CodecInvariantCacheTests — K-58 (P-39): the TTS codec computes its weight norm and its semantic centroids once per
 * set of parameters instead of at every decode. The bf16 compute measured with them was retired (stream gain −0.6 %,
 * SNR 36 dB against float32, `BENCHMARKS.md` §« 2026-10-19 — K-58 »): the codec still runs in float32.
 *
 * Random codec (the pack's layout, small dims): the caches stay out of `parameters()`, a cached decode equals the
 * uncached formula bit for bit, and the caches follow `update(parameters:)`.
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class CodecInvariantCacheTests: XCTestCase {

    private func codecConfig() throws -> VoxtralTTSConfiguration.AudioTokenizerConfiguration {
        let json = """
        {
            "channels": 1, "sampling_rate": 24000,
            "pretransform_patch_size": 240, "patch_proj_kernel_size": 7,
            "semantic_codebook_size": 64, "semantic_dim": 16,
            "acoustic_codebook_size": 21, "acoustic_dim": 36,
            "conv_weight_norm": true, "causal": true,
            "attn_sliding_window_size": 16, "half_attn_window_upon_downsampling": true,
            "dim": 64, "hidden_dim": 128, "head_dim": 16,
            "n_heads": 4, "n_kv_heads": 2,
            "qk_norm_eps": 1e-6, "qk_norm": true,
            "use_biases": false, "norm_eps": 1e-5,
            "layer_scale": true, "layer_scale_init": 0.01,
            "decoder_transformer_lengths_str": "2,2,2,2",
            "decoder_convs_kernels_str": "3,4,4,4",
            "decoder_convs_strides_str": "1,2,2,2",
            "voice": {}
        }
        """
        return try JSONDecoder().decode(VoxtralTTSConfiguration.AudioTokenizerConfiguration.self, from: Data(json.utf8))
    }

    /// Random parameters (×0.05, seed 58) in `dtype`, like the packs' bf16 codec.
    private func codec(_ dtype: DType) throws -> VoxtralCodecDecoder {
        let decoder = VoxtralCodecDecoder(config: try codecConfig())
        MLXRandom.seed(58)
        decoder.update(parameters: decoder.parameters().mapValues { (MLXRandom.normal($0.shape) * 0.05).asType(dtype) })
        return decoder
    }

    private func randomCodes(frames: Int) -> MLXArray {
        let semantic = MLXRandom.randInt(low: 2, high: 66, [1, frames, 1], key: MLXRandom.key(1))
        let acoustic = MLXRandom.randInt(low: 2, high: 23, [1, frames, 36], key: MLXRandom.key(2))
        return MLX.concatenated([semantic, acoustic], axis: -1).asType(.int32)
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { MLX.max(MLX.abs(a - b)).item(Float.self) }

    /// The cached weights and centroids are the uncached formulas, and the waveform stays float32 on bf16 weights.
    func testCachedValuesEqualTheFormulas() throws {
        let decoder = try codec(.bfloat16)
        let codes = randomCodes(frames: 20)
        XCTAssertEqual(decoder.decode(codes).dtype, .float32, "waveform")
        let p = decoder.outputProj.conv.parametrizations.weight
        let v = p.original1
        let formula = p.original0 * v / MLX.sqrt((v * v).sum(axes: [1, 2], keepDims: true) + 1e-12)
        XCTAssertEqual(p.weight.dtype, formula.dtype)
        XCTAssertEqual(maxAbs(p.weight.asType(.float32), formula.asType(.float32)), 0, "weight norm")
        let semantic = decoder.quantizer.semanticCodebook
        let centroids = semantic.embedding_sum.asType(.float32) / MLX.maximum(
            MLX.expandedDimensions(semantic.cluster_usage.asType(.float32), axis: -1), MLXArray(Float(1e-8)))
        XCTAssertEqual(maxAbs(semantic.codebook, centroids), 0, "centroids")
    }

    func testCachesStayOutOfParametersAndFollowUpdates() throws {
        let decoder = try codec(.float32)
        let keys = decoder.parameters().flattened().map(\.0)
        XCTAssertFalse(keys.contains { $0.lowercased().contains("cache") }, "\(keys.filter { $0.contains("ache") })")
        let codes = randomCodes(frames: 12)
        let before = decoder.decode(codes)
        XCTAssertEqual(maxAbs(decoder.decode(codes), before), 0, "cached = recomputed")
        // New parameters: weight norm and centroids must follow
        decoder.update(parameters: decoder.parameters().mapValues { $0 * 1.5 })
        let fresh = VoxtralCodecDecoder(config: try codecConfig())
        fresh.update(parameters: decoder.parameters())
        let after = decoder.decode(codes)
        XCTAssertGreaterThan(maxAbs(after, before), 0)
        XCTAssertEqual(maxAbs(after, fresh.decode(codes)), 0, "stale cache after update")
    }
}
