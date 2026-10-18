/**
 * CodecBandedAttentionTests — the codec's sliding-window attention by blocks of queries (K-41) against the full
 * (T, T) path it replaces.
 *
 * Random weights, fixed seed: for windows 2, 4, 8 and 16 (the decoder's stages) and lengths around the block size
 * (1 to 300 positions, one or several blocks, a partial last block), the banded output and its input gradient (voice
 * enrollment differentiates through the decoder) must match the full path to fp32 rounding.
 *
 * testDecodeParityOnModelCodes (gated) decodes real codes both ways: per text (short, medium, long) it generates the
 * codes once with a fixed seed, decodes them with the full and the banded attention and prints
 * `PARITY <text> frames=<n> |Δ|max=<…> rel_l2=<…>`; it asserts |Δ|max ≤ 1e-4.
 *
 * Env: VOXTRAL_TTS_CODEC_BANDED=1 (from xcodebuild: TEST_RUNNER_VOXTRAL_TTS_CODEC_BANDED=1), optional
 *   VOXTRAL_TTS_CODEC_BANDED_MODEL (default tts-4b-6bit), VOXTRAL_TTS_CODEC_BANDED_TEXTS (default short,medium,long),
 *   VOXTRAL_TTS_CODEC_BANDED_EMB (enrolled voice .safetensors; default: the neutral_female preset).
 */

import XCTest
import MLX
import MLXNN
@testable import VoxtralCore

@available(macOS 14.0, *)
final class CodecBandedAttentionTests: XCTestCase {

    override func setUp() {
        super.setUp()
        MLXRandom.seed(41)
    }

    private func makeCodecConfig() throws -> VoxtralTTSConfiguration.AudioTokenizerConfiguration {
        let json = """
        {
            "channels": 1, "sampling_rate": 24000,
            "pretransform_patch_size": 240, "patch_proj_kernel_size": 3,
            "semantic_codebook_size": 64, "semantic_dim": 16,
            "acoustic_codebook_size": 21, "acoustic_dim": 8,
            "conv_weight_norm": true, "causal": true,
            "attn_sliding_window_size": 16, "half_attn_window_upon_downsampling": true,
            "dim": 64, "hidden_dim": 128, "head_dim": 16,
            "n_heads": 4, "n_kv_heads": 2,
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

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(MLX.abs(a - b)).item(Float.self)
    }

    private func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = MLX.sqrt(MLX.sum(MLX.square(a - b))).item(Float.self)
        let n = MLX.sqrt(MLX.sum(MLX.square(b))).item(Float.self)
        return d / max(n, 1e-12)
    }

    /// Forward: banded == full for every window and length, including T below, at and around the block size.
    func testBandedMatchesFullAttention() throws {
        let config = try makeCodecConfig()
        let attention = CodecAttention(config: config)
        let slopes = getAlibiSlopes(nHeads: config.nHeads)
        let block = CodecAttention.bandBlockSize
        for window in [2, 4, 8, 16] {
            for T in [1, 5, window, window + 1, block - 1, block, block + 1, 2 * block + 7, 300] {
                // Large activations so that every score row has a clear structure (ALiBi alone would dominate).
                let x = MLXRandom.normal([2, T, config.dim]) * 3
                attention.bandedAttention = false
                let full = attention(x, alibiSlopes: slopes, windowSize: window)
                attention.bandedAttention = true
                let banded = attention(x, alibiSlopes: slopes, windowSize: window)
                MLX.eval(full, banded)
                XCTAssertEqual(banded.shape, full.shape)
                let d = maxAbs(banded, full)
                XCTAssertLessThanOrEqual(d, 1e-5, "window \(window) T \(T): |Δ|max \(d)")
                XCTAssertLessThanOrEqual(relL2(banded, full), 1e-5, "window \(window) T \(T)")
            }
        }
    }

    /// Backward: enrollment differentiates through the decoder; the banded input gradient must equal the full one.
    func testBandedGradientMatchesFull() throws {
        let config = try makeCodecConfig()
        let attention = CodecAttention(config: config)
        let slopes = getAlibiSlopes(nHeads: config.nHeads)
        let x = MLXRandom.normal([1, 2 * CodecAttention.bandBlockSize + 9, config.dim])
        func gradient(banded: Bool) -> MLXArray {
            attention.bandedAttention = banded
            let g = MLX.grad { (inputs: [MLXArray]) -> [MLXArray] in
                [MLX.mean(MLX.square(attention(inputs[0], alibiSlopes: slopes, windowSize: 8)))]
            }([x])[0]
            MLX.eval(g)
            return g
        }
        let full = gradient(banded: false)
        let banded = gradient(banded: true)
        XCTAssertGreaterThan(MLX.max(MLX.abs(full)).item(Float.self), 0)
        XCTAssertLessThanOrEqual(relL2(banded, full), 1e-5)
    }

    /// The full path is kept for windowSize 0 (no window) whatever the switch.
    func testNoWindowUsesFullPath() throws {
        let config = try makeCodecConfig()
        let attention = CodecAttention(config: config)
        let slopes = getAlibiSlopes(nHeads: config.nHeads)
        let x = MLXRandom.normal([1, 100, config.dim])
        attention.bandedAttention = false
        let full = attention(x, alibiSlopes: slopes, windowSize: 0)
        attention.bandedAttention = true
        let other = attention(x, alibiSlopes: slopes, windowSize: 0)
        MLX.eval(full, other)
        XCTAssertEqual(maxAbs(other, full), 0)
    }

    /// A layer by time slices (the inference path) equals the whole-sequence layer, for slices shorter and longer
    /// than the window and a partial last slice.
    func testLayerByTimeSlicesMatchesWhole() throws {
        let config = try makeCodecConfig()
        let layer = CodecTransformerLayer(config: config)
        // Layer scale 0.01 would hide the attention and feed-forward behind the residual.
        layer.attention_scale = MLX.ones([config.dim])
        layer.ffn_scale = MLX.ones([config.dim])
        let slopes = getAlibiSlopes(nHeads: config.nHeads)
        let x = MLXRandom.normal([1, 300, config.dim])
        for window in [2, 16] {
            let whole = layer(x, alibiSlopes: slopes, windowSize: window)
            for chunk in [7, 64, 100, 299] {
                let sliced = layer(x, alibiSlopes: slopes, windowSize: window, timeChunk: chunk)
                MLX.eval(whole, sliced)
                XCTAssertEqual(sliced.shape, whole.shape)
                let d = maxAbs(sliced, whole)
                XCTAssertLessThanOrEqual(d, 1e-5, "window \(window) chunk \(chunk): |Δ|max \(d)")
            }
        }
    }

    /// The whole decoder (random weights): the inference path (banded attention, time slices, eval by layer) equals
    /// the lazy full path.
    func testDecoderInferencePathMatchesFull() throws {
        let config = try makeCodecConfig()
        let decoder = VoxtralCodecDecoder(config: config)
        let random = decoder.parameters().mapValues { MLXRandom.normal($0.shape) * 0.2 }
        decoder.update(parameters: random)
        let embeddings = MLXRandom.normal([1, 40, config.semanticDim + config.acousticDim])
        let full = decoder.forwardEmbeddings(embeddings)
        setBanded(decoder, true)
        let sliced = decoder.forwardEmbeddings(embeddings, timeChunk: 48)  // last stage: 320 positions, 7 slices
        setBanded(decoder, false)
        let lazyFull = decoder.forwardEmbeddings(embeddings)
        MLX.eval(full, sliced, lazyFull)
        XCTAssertEqual(sliced.shape, lazyFull.shape)
        XCTAssertGreaterThan(MLX.max(MLX.abs(lazyFull)).item(Float.self), 0)
        XCTAssertLessThanOrEqual(relL2(sliced, lazyFull), 1e-5)
    }

    private func setBanded(_ module: Module, _ banded: Bool) {
        for case let (_, attention as CodecAttention) in module.namedModules() {
            attention.bandedAttention = banded
        }
        (module as? VoxtralCodecDecoder)?.bandedInference = banded
    }

    func testDecodeParityOnModelCodes() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_CODEC_BANDED"] == "1", "Set VOXTRAL_TTS_CODEC_BANDED=1 to run")
        let id = env["VOXTRAL_TTS_CODEC_BANDED_MODEL"] ?? "tts-4b-6bit"
        let texts = (env["VOXTRAL_TTS_CODEC_BANDED_TEXTS"] ?? "short,medium,long").split(separator: ",").map(String.init)
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: id)))
        defer { pipeline.unload() }
        let model = try XCTUnwrap(pipeline.ttsModel)
        let tokenizer = try XCTUnwrap(pipeline.tokenizer)
        let voice: MLXArray
        if let emb = env["VOXTRAL_TTS_CODEC_BANDED_EMB"] {
            voice = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: emb))["embedding"])
        } else {
            voice = try XCTUnwrap(pipeline.voiceEmbeddings["neutral_female"])
        }
        XCTAssertGreaterThan(model.audioTokenizer.namedModules().filter { $0.1 is CodecAttention }.count, 0)

        for name in texts {
            let text = try String(contentsOf: root.appendingPathComponent("docs/eval/tts/\(name)_en.txt"), encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let (codes, numFrames, _) = model.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                                       maxTokens: 2500, seed: 1)
            setBanded(model.audioTokenizer, false)
            let full = model.decodeToWaveform(codes)
            MLX.eval(full)
            let peakFull = MLX.max(MLX.abs(full)).item(Float.self)
            setBanded(model.audioTokenizer, true)
            let banded = model.decodeToWaveform(codes)
            MLX.eval(banded)
            XCTAssertEqual(banded.shape, full.shape)
            let d = maxAbs(banded.asType(.float32), full.asType(.float32))
            let r = relL2(banded.asType(.float32), full.asType(.float32))
            print("PARITY \(name) frames=\(numFrames) samples=\(full.dim(0)) |Δ|max=\(d) rel_l2=\(r) peak=\(peakFull)")
            XCTAssertLessThanOrEqual(d, 1e-4, "\(name): |Δ|max \(d)")
        }
    }
}
