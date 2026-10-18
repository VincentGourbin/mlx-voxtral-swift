/**
 * IncrementalCodecDecodeTests — K-43: streaming decodes each chunk's new frames from a bounded left context instead of
 * re-decoding all the accumulated codes.
 *
 * The codec is causal with a bounded receptive field (causal convolutions, truncated transposed convolutions, sliding
 * windows 2 to 16): the samples of frame t depend on frames t − R … t only. Random weights: the receptive field of the
 * pack's configuration, incremental == full decode for chunk sizes 1, 7 and 10, and the smallest exact context
 * (`MINCONTEXT` line), which must not exceed R.
 *
 * Gated (VOXTRAL_TTS_STREAM=1, `TEST_RUNNER_VOXTRAL_TTS_STREAM=1 xcodebuild test …`; loads tts-4b-6bit):
 * streaming == batch on raw audio (`trimLeadIn` / `trimTail` off), 3 texts × 3 seeds (`PARITY` lines); with the
 * default configuration, the batch's lead-in trim equals the tail of the stream (documented gap: the stream keeps the
 * lead-in); and no chunk reaches the consumer after the pipeline is back to `.ready`.
 * Optional: VOXTRAL_TTS_STREAM_TEXTS (default short,medium,long), VOXTRAL_TTS_STREAM_SEEDS (default 1,2,3).
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class IncrementalCodecDecodeTests: XCTestCase {

    // MARK: - Random codec

    private func codecConfig(lengths: String = "2,2,2,2") throws -> VoxtralTTSConfiguration.AudioTokenizerConfiguration {
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
            "decoder_transformer_lengths_str": "\(lengths)",
            "decoder_convs_kernels_str": "3,4,4,4",
            "decoder_convs_strides_str": "1,2,2,2",
            "voice": {}
        }
        """
        return try JSONDecoder().decode(VoxtralTTSConfiguration.AudioTokenizerConfiguration.self, from: Data(json.utf8))
    }

    private func randomCodec() throws -> VoxtralCodecDecoder {
        MLXRandom.seed(43)
        let decoder = VoxtralCodecDecoder(config: try codecConfig())
        decoder.update(parameters: decoder.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        return decoder
    }

    /// Codes with the +2 offset: semantic in [2, 66), acoustic in [2, 23).
    private func randomCodes(frames: Int) -> MLXArray {
        let semantic = MLXRandom.randInt(low: 2, high: 66, [1, frames, 1], key: MLXRandom.key(1))
        let acoustic = MLXRandom.randInt(low: 2, high: 23, [1, frames, 36], key: MLXRandom.key(2))
        return MLX.concatenated([semantic, acoustic], axis: -1).asType(.int32)
    }

    /// Stream-like decode: chunks of `chunk` frames, each from `context` previous frames.
    private func incremental(_ decoder: VoxtralCodecDecoder, _ codes: MLXArray, chunk: Int, context: Int) -> MLXArray {
        var pieces: [MLXArray] = []
        var start = 0
        while start < codes.dim(1) {
            let end = min(start + chunk, codes.dim(1))
            let piece = decoder.decode(codes[0..., ..<end, 0...], from: start, context: context)
            MLX.eval(piece)
            pieces.append(piece)
            start = end
        }
        return MLX.concatenated(pieces, axis: 1)
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { MLX.max(MLX.abs(a - b)).item(Float.self) }

    func testReceptiveFieldOfThePackConfiguration() throws {
        // 6 (projection) + 2·16 → ⌈41/2⌉ = 21; + 2·8 → ⌈40/2⌉ = 20; + 2·4 → ⌈31/2⌉ = 16; + 2·2 + 2 = 22 frames
        XCTAssertEqual(VoxtralCodecDecoder(config: try codecConfig()).receptiveFieldFrames, 22)
        XCTAssertEqual(VoxtralCodecDecoder(config: try codecConfig(lengths: "1,1,1,1")).receptiveFieldFrames, 14)
        XCTAssertEqual(VoxtralCodecDecoder(config: try codecConfig()).samplesPerFrame, 1920)
    }

    func testIncrementalDecodeMatchesFullDecode() throws {
        let decoder = try randomCodec()
        let codes = randomCodes(frames: 150)
        let full = decoder.decode(codes)
        MLX.eval(full)
        let scale = MLX.max(MLX.abs(full)).item(Float.self)
        XCTAssertGreaterThan(scale, 0)
        let context = decoder.receptiveFieldFrames + 2
        for chunk in [1, 7, 10] {
            let streamed = incremental(decoder, codes, chunk: chunk, context: context)
            XCTAssertEqual(streamed.shape, full.shape)
            let d = maxAbs(streamed, full)
            print("[k43] incremental chunk=\(chunk) context=\(context): |Δ|max=\(d) (scale \(scale))")
            XCTAssertLessThanOrEqual(d, 1e-5 * max(scale, 1), "chunk \(chunk)")
        }
    }

    /// The smallest context that decodes exactly is at most the computed receptive field; no context at all does not.
    func testSmallestExactContextIsWithinTheReceptiveField() throws {
        let decoder = try randomCodec()
        let codes = randomCodes(frames: 120)
        let full = decoder.decode(codes)
        MLX.eval(full)
        let scale = max(MLX.max(MLX.abs(full)).item(Float.self), 1)
        let R = decoder.receptiveFieldFrames
        var errors: [Float] = []
        for context in 0 ... R + 2 {
            errors.append(maxAbs(incremental(decoder, codes, chunk: 10, context: context), full))
        }
        // Rounding floor: the error with the full receptive field and the margin; the smallest exact context is the
        // first from which every larger one stays within twice that floor (or 1e-6 of the scale)
        let floor = max(2 * errors[R + 2], 1e-6 * scale)
        let minimal = (0 ... R + 2).first { c in errors[c...].allSatisfy { $0 <= floor } }
        let table = errors.enumerated().map { "\($0.offset):\(String(format: "%.2e", $0.element))" }.joined(separator: " ")
        print("[k43] MINCONTEXT \(minimal.map(String.init) ?? "none") (receptive field \(R), scale \(scale), floor \(floor)) |Δ|max by context: \(table)")
        let found = try XCTUnwrap(minimal, "some context decodes exactly")
        XCTAssertLessThanOrEqual(found, R)
        XCTAssertGreaterThan(found, 0, "without context the first samples of each chunk differ")
    }

    // MARK: - Real model (gated)

    nonisolated(unsafe) private static var shared: VoxtralTTSPipeline?

    private func pipeline() async throws -> VoxtralTTSPipeline {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_TTS_STREAM"] == "1",
                          "Set VOXTRAL_TTS_STREAM=1 to run the streaming parity tests")
        if let shared = Self.shared { return shared }
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-6bit")))
        Self.shared = pipeline
        return pipeline
    }

    private func text(_ name: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent("docs/eval/tts/\(name)_en.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func list(_ key: String, _ fallback: String) -> [String] {
        (ProcessInfo.processInfo.environment[key] ?? fallback).split(separator: ",").map(String.init)
    }

    func testStreamingEqualsBatch() async throws {
        let pipeline = try await pipeline()
        let saved = pipeline.configuration
        defer { pipeline.configuration = saved }
        var worst: Float = 0, cases = 0
        for name in list("VOXTRAL_TTS_STREAM_TEXTS", "short,medium,long") {
            let text = try text(name)
            for seed in list("VOXTRAL_TTS_STREAM_SEEDS", "1,2,3").compactMap(UInt64.init) {
                pipeline.configuration.trimLeadIn = false
                pipeline.configuration.trimTail = false
                let batch = try await pipeline.synthesize(text: text, voice: .neutralFemale, seed: seed).waveform
                var pieces: [MLXArray] = []
                var readyDuringStream = false
                for try await chunk in pipeline.synthesizeStreaming(text: text, voice: .neutralFemale, seed: seed) {
                    readyDuringStream = readyDuringStream || pipeline.isReady
                    pieces.append(chunk.waveform)
                }
                let stream = MLX.concatenated(pieces, axis: 0)
                XCTAssertFalse(readyDuringStream, "\(name) seed \(seed): a chunk arrived after .ready")
                XCTAssertEqual(stream.shape, batch.shape, "\(name) seed \(seed)")
                guard stream.shape == batch.shape else { continue }
                let d = maxAbs(stream.asType(.float32), batch.asType(.float32))
                // Default configuration: the batch trims the lead-in, the stream does not (it emits before knowing it)
                let trimmed = trimLeadInSilence(batch, sampleRate: pipeline.sampleRate)
                let lead = batch.dim(0) - trimmed.dim(0)
                let tail = maxAbs(stream[lead...].asType(.float32), trimmed.asType(.float32))
                print("[k43] PARITY \(name) seed=\(seed) frames=\(batch.dim(0) / 1920) chunks=\(pieces.count) |Δ|max=\(d) "
                      + "default: lead-in kept by the stream \(lead) samples, tail |Δ|max=\(tail)")
                XCTAssertLessThanOrEqual(d, 1e-4, "\(name) seed \(seed)")
                XCTAssertLessThanOrEqual(tail, 1e-4, "\(name) seed \(seed), default configuration")
                worst = max(worst, d)
                cases += 1
            }
        }
        print("[k43] PARITY stream == batch |Δ|max \(worst), \(cases) cases")
    }
}
