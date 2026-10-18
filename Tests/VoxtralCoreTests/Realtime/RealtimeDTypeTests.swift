/**
 * RealtimeDTypeTests - K-38 (P-60)
 *
 * The Realtime path computes in the model's 16-bit dtype. The mlx-community packs store their matrices in fp16 but
 * norms, biases and the Ada MLP in fp32 (300 tensors); with the fp32 mel, the fp32 RoPE tables and the fp32 time
 * embedding, each of them promoted the encoder, the decoder, the KV cache and the logits to fp32. Reduced model with
 * that layout: after the load-time cast, encoder output, KV cache and logits are fp16 (red before: float32).
 */

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest
@testable import VoxtralCore

final class RealtimeDTypeTests: XCTestCase {

    private func tinyModel() throws -> VoxtralRealtimeModel {
        let json = """
        {"decoder": {"dim": 64, "n_layers": 2, "head_dim": 16, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 2,
                     "vocab_size": 256, "norm_eps": 1e-5, "rope_theta": 1000000.0, "sliding_window": 64,
                     "tied_embeddings": true, "ada_rms_norm_t_cond": true, "ada_rms_norm_t_cond_dim": 8},
         "encoder_args": {"audio_encoding_args": {"sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 16,
                                                  "hop_length": 160, "window_size": 400, "global_log_mel_max": 1.5,
                                                  "transcription_format": "streaming"},
                          "dim": 32, "n_layers": 2, "head_dim": 8, "hidden_dim": 64, "n_heads": 4, "n_kv_heads": 4,
                          "use_biases": true, "rope_theta": 1000000.0, "norm_eps": 1e-5, "sliding_window": 16,
                          "downsample_factor": 4},
         "model_type": "voxtral_realtime"}
        """
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(38)
        let model = VoxtralRealtimeModel(config: config)
        // The packs' layout: matrices (and the tied table, the convs) in fp16; norms, biases, Ada MLP in fp32
        let params = model.parameters().flattened().map { key, value -> (String, MLXArray) in
            let small = value.ndim == 1 || key.contains("ada_rms_norm")
            return (key, value.asType(small ? .float32 : .float16))
        }
        XCTAssertTrue(params.contains { $0.1.dtype == .float32 })
        model.update(parameters: ModuleParameters.unflattened(params))
        return model
    }

    /// Encoder, prefill of 8 positions and one decode step: the dtypes of the arrays that decide the compute.
    private func run(_ model: VoxtralRealtimeModel) -> (encoder: MLXArray, kv: MLXArray, logits: MLXArray) {
        model.decoder.precomputeAdaScales(tCond: computeTimeEmbedding(tValue: 6, dim: model.config.decoder.dim))
        let mel = MLXRandom.normal([16, 64], key: MLXRandom.key(3))  // fp32, as `prepareMel` gives it
        let audio = model.encodeAudio(mel)
        let cache = model.decoder.createCache()
        let tokens = model.decoder.embedTokens(MLXArray((0 ..< Int32(audio.dim(0))).map { $0 + 3 }))
        var hidden = model.decoder.forward(embeds: audio + tokens, cache: cache)
        let step = audio[0] + model.decoder.embedToken(5)
        hidden = model.decoder.forward(embeds: step.expandedDimensions(axis: 0), cache: cache)
        let logits = model.decoder.logits(hidden[0])
        MLX.eval(audio, logits)
        return (audio, cache[0].state[0], logits)
    }

    func testPackLayoutComputesInFP16AfterTheLoadCast() throws {
        let model = try tinyModel()
        let dtype = realtimeComputeDType(model)
        XCTAssertEqual(dtype, .float16, "the tied token table's dtype")
        castRealtimeParameters(model, to: dtype)
        XCTAssertFalse(model.parameters().flattened().contains { $0.1.dtype == .float32 }, "no fp32 parameter left")
        let (encoder, kv, logits) = run(model)
        print("[dtype] realtime pack layout: encoder=\(encoder.dtype) kv=\(kv.dtype) logits=\(logits.dtype)")
        XCTAssertEqual(encoder.dtype, .float16, "encoder output (fp32 mel and RoPE tables)")
        XCTAssertEqual(kv.dtype, .float16, "KV cache, layer 0")
        XCTAssertEqual(logits.dtype, .float16, "logits")
    }

    func testFP32ModelIsLeftInFP32() throws {
        let model = try tinyModel()
        let all32 = model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        model.update(parameters: ModuleParameters.unflattened(all32))
        castRealtimeParameters(model, to: realtimeComputeDType(model))
        let (encoder, kv, logits) = run(model)
        XCTAssertEqual(encoder.dtype, .float32)
        XCTAssertEqual(kv.dtype, .float32)
        XCTAssertEqual(logits.dtype, .float32)
    }

    func testFP16PathMatchesFP32Path() throws {
        // Same values (same seed, matrices rounded to fp16 in both): all in fp32, or the pack layout then the load cast
        let reference = try tinyModel()
        reference.update(parameters: ModuleParameters.unflattened(
            reference.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }))
        let half = try tinyModel()
        castRealtimeParameters(half, to: realtimeComputeDType(half))
        let a = run(reference).logits, b = run(half).logits.asType(.float32)
        let rel = (MLX.sqrt(MLX.sum(MLX.square(a - b))) / MLX.sqrt(MLX.sum(MLX.square(a)))).item(Float.self)
        print("[dtype] fp16 vs fp32 logits: rel_l2=\(rel), argmax \(MLX.argMax(a).item(Int.self)) / \(MLX.argMax(b).item(Int.self))")
        XCTAssertLessThan(rel, 2e-2)
    }
}
