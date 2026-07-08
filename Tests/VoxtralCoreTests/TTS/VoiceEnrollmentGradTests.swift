/**
 * VoiceEnrollmentGradTests - Differentiability smoke tests for the
 * voice-enrollment optimization path (gradient descent on codec codes
 * through the frozen decoder).
 *
 * These tests verify that every building block of the enrollment loop
 * supports gradients in mlx-swift: FFT-based spectral losses,
 * (transposed) weight-norm convolutions, the manual codec attention,
 * and the straight-through-estimator pattern.
 */

import XCTest
import MLX
import MLXFFT
@testable import VoxtralCore

@available(macOS 14.0, *)
final class VoiceEnrollmentGradTests: XCTestCase {

    override func setUp() {
        super.setUp()
        MLXRandom.seed(42)
    }

    /// Gradient must flow through rfft magnitude (basis of all STFT losses).
    func testFFTLossGradient() {
        let x = MLXRandom.normal([1024])
        let target = MLXRandom.normal([513])

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            let spec = MLXFFT.rfft(inputs[0])
            let mag = MLX.abs(spec)
            return [MLX.mean(MLX.abs(mag - target))]
        }

        let grads = MLX.grad(loss)([x])
        MLX.eval(grads)

        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite, "FFT gradient is not finite")
        XCTAssertGreaterThan(gradNorm, 0, "FFT gradient is zero — no flow through rfft")
    }

    /// Gradient must flow through the causal transposed convolution
    /// (the decoder's upsampling stages).
    func testConvTransposeGradient() {
        // In transpose mode the checkpoint weight layout (out_ch, in_ch, K)
        // is reinterpreted: the input must carry `outChannels` channels.
        let conv = WeightNormConv(outChannels: 8, inChannels: 4, kernelSize: 3)
        // A fresh module has zero direction weights (they normally come from
        // the checkpoint) — randomize so the output actually depends on input.
        conv.parametrizations.weight.original1 = MLXRandom.normal([8, 4, 3])
        let x = MLXRandom.normal([1, 16, 8])

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            [MLX.mean(MLX.square(conv(inputs[0], stride: 2, transpose: true)))]
        }

        let grads = MLX.grad(loss)([x])
        MLX.eval(grads)

        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite)
        XCTAssertGreaterThan(gradNorm, 0, "no gradient through convTransposed1d")
    }

    /// Small codec config for fast module-level gradient tests.
    private func makeTinyCodecConfig() throws -> VoxtralTTSConfiguration.AudioTokenizerConfiguration {
        let json = """
        {
            "channels": 1, "sampling_rate": 24000,
            "pretransform_patch_size": 240, "patch_proj_kernel_size": 3,
            "semantic_codebook_size": 64, "semantic_dim": 16,
            "acoustic_codebook_size": 21, "acoustic_dim": 8,
            "conv_weight_norm": true, "causal": true,
            "attn_sliding_window_size": 4, "half_attn_window_upon_downsampling": true,
            "dim": 32, "hidden_dim": 64, "head_dim": 8,
            "n_heads": 4, "n_kv_heads": 4,
            "qk_norm_eps": 1e-6, "qk_norm": true,
            "use_biases": false, "norm_eps": 1e-5,
            "layer_scale": true, "layer_scale_init": 0.01,
            "decoder_transformer_lengths_str": "1,1,1,1",
            "decoder_convs_kernels_str": "3,4,4,4",
            "decoder_convs_strides_str": "1,2,2,2",
            "voice": {}
        }
        """
        return try JSONDecoder().decode(
            VoxtralTTSConfiguration.AudioTokenizerConfiguration.self,
            from: Data(json.utf8)
        )
    }

    /// Isolate the raw op: gradient of MLX.convTransposed1d w.r.t. its input.
    func testRawConvTransposedGradient() {
        let x = MLXRandom.normal([1, 16, 8])
        let w = MLXRandom.normal([4, 3, 8])  // (C_out, K, C_in)

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            [MLX.sum(MLX.square(MLX.convTransposed1d(inputs[0], w, stride: 2, padding: 0)))]
        }

        let grads = MLX.grad(loss)([x])
        MLX.eval(grads)
        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite)
        XCTAssertGreaterThan(gradNorm, 0, "raw convTransposed1d has zero input gradient")
    }

    /// Gradient must flow through the manual codec attention
    /// (matmul + softmax + ALiBi/window masks).
    func testCodecAttentionGradient() throws {
        let config = try makeTinyCodecConfig()
        let attention = CodecAttention(config: config)
        let slopes = getAlibiSlopes(nHeads: config.nHeads)
        let x = MLXRandom.normal([1, 12, config.dim])

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            [MLX.mean(MLX.square(attention(inputs[0], alibiSlopes: slopes, windowSize: 4)))]
        }

        let grads = MLX.grad(loss)([x])
        MLX.eval(grads)

        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite)
        XCTAssertGreaterThan(gradNorm, 0, "no gradient through CodecAttention")
    }

    /// The straight-through estimator: forward returns the hard value,
    /// backward passes the soft gradient unchanged.
    func testStraightThroughEstimator() {
        let logits = MLXRandom.normal([5, 16])
        let target = MLXRandom.normal([5, 16])

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            let soft = MLX.softmax(inputs[0], axis: -1)
            let hard = MLX.round(soft)
            let ste = soft + MLX.stopGradient(hard - soft)
            // Linear loss: gradient = target ⊙ ∂soft/∂logits, nonzero even
            // when the hard forward value is all zeros.
            return [MLX.sum(ste * target)]
        }

        let grads = MLX.grad(loss)([logits])
        MLX.eval(grads)

        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite)
        XCTAssertGreaterThan(gradNorm, 0, "STE blocked the gradient")
    }
}
