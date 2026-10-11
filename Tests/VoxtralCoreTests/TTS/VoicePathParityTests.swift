/**
 * VoicePathParityTests — why `synthesize(voice:)` and `synthesize(voiceEmbedding:)` give different audio at the same
 * seed in bf16 only (K-94, constat de K-35).
 *
 * The `voice:` path prefills the voice prefix (BOS, BEGIN_AUDIO, the voice frames) once, caches its KV, and prefills
 * only the text suffix on each call; the `voiceEmbedding:` path prefills the whole prompt in one block. Same tokens,
 * same causal attention: the two must agree up to rounding. The test locates where they stop agreeing:
 *  - PROJ: the first layer's k_proj on the prefix rows, alone (M = prefix length) and inside the whole prompt
 *    (M = prompt length): a dense bf16 matmul may change its reduction with M, a quantized one works row by row;
 *  - KV: the cached keys of the prefix positions, last layer, after both prefills;
 *  - HIDDEN: the hidden state that drives the first frame;
 *  - CODES: both paths generated at the same seed, frame by frame (first divergent frame, semantic or acoustic).
 *
 * Env: VOXTRAL_TTS_PARITY=1 (from xcodebuild: TEST_RUNNER_VOXTRAL_TTS_PARITY=1), optional
 *   VOXTRAL_TTS_PARITY_MODELS (default "tts-4b-mlx,tts-4b-6bit,tts-4b-4bit").
 */

import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import VoxtralCore

@available(macOS 14.0, *)
final class VoicePathParityTests: XCTestCase {

    func testVoicePrefixCacheAgainstOneBlockPrefill() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_PARITY"] == "1", "Set VOXTRAL_TTS_PARITY=1 to run")
        let models = (env["VOXTRAL_TTS_PARITY_MODELS"] ?? "tts-4b-mlx,tts-4b-6bit,tts-4b-4bit")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let texts = try ["short_en", "medium_en", "long_en"].map { name in
            (name, try String(contentsOf: root.appendingPathComponent("docs/eval/tts/\(name).txt"), encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines))
        }

        func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        }

        for id in models {
            let pipeline = VoxtralTTSPipeline()
            try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: id)))
            defer { pipeline.unload() }
            let model = try XCTUnwrap(pipeline.ttsModel)
            let tokenizer = try XCTUnwrap(pipeline.tokenizer)
            let voice = try XCTUnwrap(pipeline.voiceEmbeddings[VoxtralVoice.neutralFemale.rawValue])
            let (prefixCache, prefixLen) = model.precomputeVoicePrefixCache(voiceEmbedding: voice)
            let kProj = model.layers[0].selfAttn.kProj
            print("VOICEPATH \(id) k_proj \(type(of: kProj)) voice frames \(voice.dim(0)) prefix \(prefixLen)")

            for (name, text) in texts {
                let ids = model.encodeText(text, voiceFrameCount: voice.dim(0), tokenizer: tokenizer, sanitize: true)
                let emb = model.buildInputEmbeddings(inputIds: MLXArray(ids).reshaped(1, ids.count), voiceEmbedding: voice)

                // PROJ: the same rows through the first projection, alone and inside the whole prompt
                let x = model.layers[0].inputLayernorm(emb)
                let alone = kProj(x[0..., ..<prefixLen, 0...])
                let inside = kProj(x)[0..., ..<prefixLen, 0...]
                let proj = maxDiff(alone, inside)

                // KV and HIDDEN: one-block prefill against prefix cache + suffix
                let full = model.createCache()
                var hFull = model.llmForward(inputEmbeds: emb, cache: full)
                let split = model.cloneCache(prefixCache)
                var hSplit = model.llmForward(inputEmbeds: emb[0..., prefixLen..., 0...], cache: split)
                let audioTok = model.embedTokens(
                    MLXArray([Int32(model.config.multimodal.audioModelArgs.audioTokenId)]).reshaped(1, 1))
                hFull = model.llmForward(inputEmbeds: audioTok, cache: full)
                hSplit = model.llmForward(inputEmbeds: audioTok, cache: split)
                let kvFull = full.last!.state[0][0..., 0..., ..<prefixLen, 0...]
                let kvSplit = split.last!.state[0][0..., 0..., ..<prefixLen, 0...]
                let kv = maxDiff(kvFull, kvSplit)
                let hidden = maxDiff(hFull, hSplit)

                // CODES: both paths at the same seed
                let byEmbedding = model.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                                 maxTokens: 2_400, seed: 1)
                let byVoice = model.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                             maxTokens: 2_400, seed: 1, prefixCache: prefixCache, prefixLen: prefixLen)
                let a = byEmbedding.codes[0].asArray(Int32.self), b = byVoice.codes[0].asArray(Int32.self)
                let width = byEmbedding.codes.dim(2)
                var firstFrame = -1, where_ = "none"
                for f in 0 ..< min(byEmbedding.numFrames, byVoice.numFrames) {
                    let ra = a[(f * width) ..< ((f + 1) * width)], rb = b[(f * width) ..< ((f + 1) * width)]
                    if ra != rb {
                        firstFrame = f
                        where_ = ra.first == rb.first ? "acoustic" : "semantic"
                        break
                    }
                }
                if firstFrame < 0, byEmbedding.numFrames != byVoice.numFrames {
                    firstFrame = min(byEmbedding.numFrames, byVoice.numFrames); where_ = "length"
                }
                print(String(format: "VOICEPATH %@ %@ PROJ max|Δ| %.3g · KV(last layer, prefix) %.3g · HIDDEN %.3g · ",
                             id, name, proj, kv, hidden)
                      + "frames \(byEmbedding.numFrames) / \(byVoice.numFrames) · first divergent frame \(firstFrame) (\(where_))")
            }
        }
    }
}
