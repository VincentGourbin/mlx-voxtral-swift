/**
 * CodecDTypeTests — K-58: the TTS codec runs in the dtype of its weights (bf16 in the packs) instead of being promoted
 * to float32 by the centroids, the FSQ levels and the causal padding; weight norm and centroids are computed once.
 *
 * Random codec (the pack's layout, small dims), parameters cast to bfloat16 like the packs: every stage's output in
 * bfloat16, the waveform in float32 and close to the float32 codec; the caches stay out of `parameters()` and follow
 * `update(parameters:)`.
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class CodecDTypeTests: XCTestCase {

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

    /// Random float32 parameters (×0.05), the same for a given seed.
    private func randomParameters(_ decoder: VoxtralCodecDecoder) -> ModuleParameters {
        MLXRandom.seed(58)
        return decoder.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 }
    }

    private func codec(_ dtype: DType) throws -> VoxtralCodecDecoder {
        let decoder = VoxtralCodecDecoder(config: try codecConfig())
        decoder.update(parameters: randomParameters(decoder).mapValues { $0.asType(dtype) })
        return decoder
    }

    private func randomCodes(frames: Int) -> MLXArray {
        let semantic = MLXRandom.randInt(low: 2, high: 66, [1, frames, 1], key: MLXRandom.key(1))
        let acoustic = MLXRandom.randInt(low: 2, high: 23, [1, frames, 36], key: MLXRandom.key(2))
        return MLX.concatenated([semantic, acoustic], axis: -1).asType(.int32)
    }

    func testEveryStageRunsInTheWeightDType() throws {
        let decoder = try codec(.bfloat16)
        XCTAssertEqual(decoder.computeDType, .bfloat16)
        var x = decoder.quantizer.decode(randomCodes(frames: 20)).asType(decoder.computeDType)
        let windows = VoxtralCodecDecoder.windowSizes
        for i in stride(from: 0, to: decoder.decoderBlocks.count, by: 2) {
            let stage = i / 2, s = decoder.strides[stage]
            x = try XCTUnwrap(decoder.decoderBlocks[i] as? ConvBlock).conv(x, stride: s, transpose: s > 1)
            XCTAssertEqual(x.dtype, .bfloat16, "conv \(stage)")
            x = try XCTUnwrap(decoder.decoderBlocks[i + 1] as? CodecTransformerBlock)(
                x, alibiSlopes: decoder._alibiSlopes, windowSize: windows[stage])
            XCTAssertEqual(x.dtype, .bfloat16, "transformer \(stage)")
        }
        XCTAssertEqual(decoder.outputProj.conv(x, stride: 1, transpose: false).dtype, .bfloat16, "output projection")
        // Banded and sliced inference path (K-41): long enough for time slices
        let long = decoder.forwardEmbeddings(
            decoder.quantizer.decode(randomCodes(frames: 300)).asType(.bfloat16),
            timeChunk: VoxtralCodecDecoder.inferenceTimeChunk)
        XCTAssertEqual(long.dtype, .bfloat16, "sliced path")
        XCTAssertEqual(decoder.decode(randomCodes(frames: 20)).dtype, .float32, "waveform")
    }

    /// The bf16 codec stays close to the float32 one on the same (bf16-representable) weights.
    func testBFloat16WaveformIsCloseToFloat32() throws {
        let bf16 = try codec(.bfloat16)
        let fp32 = VoxtralCodecDecoder(config: try codecConfig())
        fp32.update(parameters: bf16.parameters().mapValues { $0.asType(.float32) })
        XCTAssertEqual(fp32.computeDType, .float32)
        let codes = randomCodes(frames: 60)
        let a = bf16.decode(codes), b = fp32.decode(codes)
        let noise = MLX.sum(MLX.square(a - b)).item(Float.self)
        let signal = MLX.sum(MLX.square(b)).item(Float.self)
        let snr = 10 * log10(signal / max(noise, 1e-30))
        print("[k58] random codec bf16 vs fp32 SNR \(snr) dB")
        XCTAssertGreaterThan(snr, 20)
    }

    func testCachesStayOutOfParametersAndFollowUpdates() throws {
        let decoder = try codec(.float32)
        let keys = decoder.parameters().flattened().map(\.0)
        XCTAssertFalse(keys.contains { $0.lowercased().contains("cache") }, "\(keys.filter { $0.contains("ache") })")
        let codes = randomCodes(frames: 12)
        let before = decoder.decode(codes)
        XCTAssertEqual(MLX.max(MLX.abs(decoder.decode(codes) - before)).item(Float.self), 0, "cached = recomputed")
        // New parameters: weight norm and centroids must follow
        decoder.update(parameters: decoder.parameters().mapValues { $0 * 1.5 })
        let fresh = VoxtralCodecDecoder(config: try codecConfig())
        fresh.update(parameters: decoder.parameters())
        let after = decoder.decode(codes)
        XCTAssertGreaterThan(MLX.max(MLX.abs(after - before)).item(Float.self), 0)
        XCTAssertEqual(MLX.max(MLX.abs(after - fresh.decode(codes))).item(Float.self), 0, "stale cache after update")
    }
}
