/**
 * RealtimeFusedRoPETests — K-65 (P-68): the Realtime encoder applies its interleaved RoPE with the fused kernel
 * (`MLXFast.RoPE(traditional: true)`, offset = absolute position of the chunk) instead of the explicit form with
 * cos/sin tables (`interleavedRoPE` + `computeRoPEFreqs`, kept here as the reference).
 *
 * Reference = the encoder as it was before K-65, rebuilt from the same modules: same chunks, same rotating caches,
 * same masks; only the RoPE differs. Small random encoder in fp32 and fp16 (several windows, so the offset matters).
 *
 * Gated (VOXTRAL_REALTIME_ROPE=1, `TEST_RUNNER_VOXTRAL_REALTIME_ROPE=1 xcodebuild test …`; loads the packs of
 * VOXTRAL_REALTIME_ROPE_MODELS, default realtime-4b-4bit,realtime-4b-fp16): encoder output on C-moyen EN, fused
 * against reference in the model's dtype, relative L2 < 1e-2 (`PARITY` lines).
 */

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class RealtimeFusedRoPETests: XCTestCase {

    // MARK: - Reference (before K-65)

    private func referenceAttention(
        _ attention: RealtimeEncoderAttention, _ x: MLXArray, start: Int, cache: any KVCache,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let n = x.dim(0), heads = attention.nHeads, headDim = attention.headDim
        let (cos, sin) = computeRoPEFreqs(
            positions: MLXArray(Int32(start) ..< Int32(start + n)), headDim: headDim, theta: attention.ropeTheta,
            dtype: x.dtype
        )
        let q = interleavedRoPE(attention.wq(x), cos: cos, sin: sin, nHeads: heads, headDim: headDim)
        let k = interleavedRoPE(attention.wk(x), cos: cos, sin: sin, nHeads: heads, headDim: headDim)
        let v = attention.wv(x)
        let qh = q.reshaped(1, n, heads, headDim).transposed(0, 2, 1, 3)
        var kh = k.reshaped(1, n, heads, headDim).transposed(0, 2, 1, 3)
        var vh = v.reshaped(1, n, heads, headDim).transposed(0, 2, 1, 3)
        (kh, vh) = cache.update(keys: kh, values: vh)
        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: attention.scale, mask: maskMode)
        return attention.wo(out.transposed(0, 2, 1, 3).reshaped(n, heads * headDim))
    }

    /// `encodeChunked` before K-65, layer by layer
    private func referenceEncodeChunked(_ encoder: VoxtralRealtimeEncoder, _ convOut: MLXArray) -> MLXArray {
        let window = encoder.config.slidingWindow
        let caches = encoder.layers.map { _ in RotatingKVCache(maxSize: window, keep: 0) }
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: convOut.dim(0), by: window) {
            let end = min(start + window, convOut.dim(0))
            let maskMode = caches[0].makeMask(n: end - start, windowSize: window, returnArray: false)
            var x = convOut[start ..< end]
            for (layer, cache) in zip(encoder.layers, caches) {
                let h = referenceAttention(layer.attention, layer.attentionNorm(x), start: start, cache: cache,
                                           maskMode: maskMode)
                x = x + h
                let f = layer.ffnNorm(x)
                x = x + layer.w2(silu(layer.w1(f)) * layer.w3(f))
            }
            x = encoder.transformerNorm(x)
            MLX.eval(x)
            chunks.append(x)
        }
        return MLX.concatenated(chunks, axis: 0)
    }

    private func relativeL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32), b = b.asType(.float32)
        return (MLX.sqrt(MLX.sum((a - b) * (a - b))) / MLX.sqrt(MLX.sum(b * b))).item(Float.self)
    }

    // MARK: - Small random encoder

    private func tinyEncoder(dtype: DType) throws -> VoxtralRealtimeEncoder {
        let json = """
        {"dim": 64, "n_layers": 2, "head_dim": 16, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 4,
         "use_biases": true, "rope_theta": 1000000.0, "norm_eps": 1e-5, "sliding_window": 8,
         "downsample_factor": 4,
         "audio_encoding_args": {"sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 16, "hop_length": 160,
                                 "window_size": 400, "global_log_mel_max": 1.5, "transcription_format": "streaming"}}
        """
        let config = try JSONDecoder().decode(RealtimeEncoderConfig.self, from: Data(json.utf8))
        MLXRandom.seed(65)
        let encoder = VoxtralRealtimeEncoder(config: config, decoderDim: 32)
        encoder.update(parameters: encoder.parameters().mapValues { (MLXRandom.normal($0.shape) * 0.2).asType(dtype) })
        return encoder
    }

    func testFusedRoPEMatchesTheExplicitForm() throws {
        for (dtype, tolerance) in [(DType.float32, Float(1e-5)), (.float16, 2e-3)] {
            let encoder = try tinyEncoder(dtype: dtype)
            // 40 conv positions = 5 windows of 8: the reference rotates at absolute positions 0 … 39, so a wrong
            // offset in a later chunk breaks the parity
            let convOut = encoder.convStem(MLXRandom.normal([16, 80], key: MLXRandom.key(2)).asType(dtype))
            XCTAssertEqual(convOut.dim(0), 40)
            let fused = encoder.encodeChunked(convOut)
            let reference = referenceEncodeChunked(encoder, convOut)
            XCTAssertEqual(fused.shape, reference.shape)
            XCTAssertEqual(fused.dtype, dtype)
            let l2 = relativeL2(fused, reference)
            print("[k65] fused RoPE \(dtype): encoder L2 rel = \(l2)")
            XCTAssertLessThan(l2, tolerance, "\(dtype)")
        }
    }

    // MARK: - Real packs (gated)

    func testPacksOnCMoyen() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_REALTIME_ROPE"] == "1", "Set VOXTRAL_REALTIME_ROPE=1 to run the pack parity")
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let audio = try loadAudio(root.appendingPathComponent("docs/eval/clips/c_moyen_en.wav").path)
        let ids = (env["VOXTRAL_REALTIME_ROPE_MODELS"] ?? "realtime-4b-4bit,realtime-4b-fp16").split(separator: ",")
        for id in ids.map(String.init) {
            let info = try VoxtralRealtimePipeline.modelInfo(for: id)
            let dir = try await VoxtralModelDownloader.downloadRealtimeModel(info) { _, _ in }
            let model = try loadVoxtralRealtimeModel(from: dir)
            var (mel, _) = logMelSpectrogram(audio, globalMax: model.config.audioEncoding.globalLogMelMax)
            if mel.dim(1) % 2 != 0 { mel = mel[0..., 1...] }
            // The encoder casts the fp32 mel to the conv dtype at its entry (K-38)
            let convOut = model.encoder.convStem(mel.asType(model.encoder.conv1.conv.weight.dtype))
            MLX.eval(convOut)
            let fused = model.encoder.encodeChunked(convOut)
            let reference = referenceEncodeChunked(model.encoder, convOut)
            let l2 = relativeL2(fused, reference)
            let projected = relativeL2(model.encoder.downsampleAndProject(fused),
                                       model.encoder.downsampleAndProject(reference))
            print("[k65] PARITY \(id) dtype=\(fused.dtype) positions=\(convOut.dim(0)) transformer L2 rel=\(l2) "
                  + "embeddings L2 rel=\(projected)")
            XCTAssertEqual(fused.dtype, realtimeComputeDType(model), "\(id): the model's dtype")
            XCTAssertLessThan(projected, 1e-2, id)
            XCTAssertLessThan(l2, 1e-2, id)
        }
    }
}
