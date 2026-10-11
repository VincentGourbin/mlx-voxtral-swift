/**
 * CloneOverrunTests — why tts-4b-4bit with a cloned voice produces many more frames than tts-4b-6bit (K-97,
 * constat de K-35: medium 1 806 frames against 858, long 2 149 against 1 868, ASR coverage 0.731 and 0.237).
 *
 * 1. FREE: both packs, the cloned voice and a preset voice of the same length and language (`fr_female`), the medium
 *    text, seeds 1 to 3: frames and whether the end of audio (EOA) was reached before the cap. Says whether the
 *    overrun is systematic and tied to the cloned voice.
 * 2. FORCED: the 6-bit trajectory (its codes fed back) drives both packs; at every frame each pack's semantic head
 *    reads its own LLM's hidden state on the SAME history. Reports the share of frames where the 4-bit argmax equals
 *    the 6-bit one, the first frame where they part, and the EOA margin (end_audio logit minus the best other
 *    code) of each pack at the frame where the 6-bit trajectory ends. A 4-bit pack that no longer sees the end of
 *    the text there keeps talking past it.
 *
 * Env: VOXTRAL_TTS_CLONE_OVERRUN=1 (from xcodebuild: TEST_RUNNER_VOXTRAL_TTS_CLONE_OVERRUN=1) and
 *   VOXTRAL_TTS_CLONE_OVERRUN_EMB (the cloned voice .safetensors, key "embedding").
 */

import XCTest
import MLX
import MLXLMCommon
@testable import VoxtralCore

@available(macOS 14.0, *)
final class CloneOverrunTests: XCTestCase {

    /// end_audio margin and semantic argmax (end_audio included) of one hidden state
    private func semanticRead(_ model: VoxtralTTSModel, _ h: MLXArray) -> (argmax: Int32, eoaMargin: Float) {
        let fm = model.acousticTransformer
        let logits = fm.semanticCodebookOutput(h.asType(.float32)).asType(.float32)[0]
        let valid = logits[1 ..< (fm.semanticCodebookSize + 2)]  // end_audio (1) and the semantic codes
        let best = MLX.argMax(valid).item(Int32.self) + 1
        let others = valid[1...].max().item(Float.self)
        return (best, valid[0].item(Float.self) - others)
    }

    func testFourBitCloneOverrun() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_CLONE_OVERRUN"] == "1", "Set VOXTRAL_TTS_CLONE_OVERRUN=1 to run")
        let clonePath = try XCTUnwrap(env["VOXTRAL_TTS_CLONE_OVERRUN_EMB"], "VOXTRAL_TTS_CLONE_OVERRUN_EMB required")
        let clone = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: clonePath))["embedding"])
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let text = try String(contentsOf: root.appendingPathComponent("docs/eval/tts/medium_en.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let cap = 2_379  // frameCap of the medium text (K-35 BENCH `frame_cap`)

        // reference trajectories (6-bit, seed 1) per voice, for the forced part
        var reference: [String: [[Int32]]] = [:]
        var forced: [String: [String: (agree: Int, frames: Int, firstSplit: Int, eoaMargin: Float)]] = [:]

        for id in ["tts-4b-6bit", "tts-4b-4bit"] {
            let pipeline = VoxtralTTSPipeline()
            try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: id)))
            defer { pipeline.unload() }
            let model = try XCTUnwrap(pipeline.ttsModel)
            let tokenizer = try XCTUnwrap(pipeline.tokenizer)
            let voices: [(String, MLXArray)] = [("clone_fr", clone),
                                                ("fr_female", try XCTUnwrap(pipeline.voiceEmbeddings["fr_female"]))]

            // 1. FREE
            for (name, voice) in voices {
                for seed: UInt64 in 1 ... 3 {
                    let out = model.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                             maxTokens: cap, seed: seed)
                    print("OVERRUN FREE \(id) \(name) seed \(seed): frames \(out.numFrames) · EOA \(out.numFrames < cap ? "yes" : "no (cap)")")
                    if id == "tts-4b-6bit", seed == 1 {
                        let codes = out.codes[0].asArray(Int32.self), width = out.codes.dim(2)
                        reference[name] = (0 ..< out.numFrames).map { Array(codes[($0 * width) ..< (($0 + 1) * width)]) }
                    }
                }
            }

            // 2. FORCED on the 6-bit seed-1 trajectory
            for (name, voice) in voices {
                let ref = try XCTUnwrap(reference[name])
                let ids = model.encodeText(text, voiceFrameCount: voice.dim(0), tokenizer: tokenizer, sanitize: true)
                let cache = model.createCache()
                _ = model.llmForward(inputEmbeds: model.buildInputEmbeddings(
                    inputIds: MLXArray(ids).reshaped(1, ids.count), voiceEmbedding: voice), cache: cache)
                let audioTok = Int32(model.config.multimodal.audioModelArgs.audioTokenId)
                var hidden = model.llmForward(inputEmbeds: model.embedTokens(MLXArray([audioTok]).reshaped(1, 1)),
                                              cache: cache)
                var agree = 0, firstSplit = -1, margin: Float = 0
                for f in 0 ... ref.count {  // frame ref.count is where the 6-bit trajectory read EOA
                    let read = semanticRead(model, hidden[0..., -1, 0...])
                    if f == ref.count { margin = read.eoaMargin; break }
                    if read.argmax == ref[f][0] { agree += 1 } else if firstSplit < 0 { firstSplit = f }
                    let next = model.mmAudioEmbeddings
                        .audioCodebookEmbeddings(model.codesToGlobalIndices(MLXArray(ref[f]).reshaped(1, ref[f].count)))
                        .sum(axis: 1, keepDims: true)
                    hidden = model.llmForward(inputEmbeds: next, cache: cache)
                    if f % 64 == 0 { MLX.eval(hidden) }
                }
                forced[id, default: [:]][name] = (agree, ref.count, firstSplit, margin)
                print(String(format: "OVERRUN FORCED %@ %@ on 6-bit seed 1: semantic argmax = 6-bit %d / %d (%.1f %%), "
                             + "first split %d, EOA margin at frame %d: %.2f", id, name, agree, ref.count,
                             Double(agree) / Double(ref.count) * 100, firstSplit, ref.count, margin))
            }
        }
        // the 6-bit pack must read its own trajectory back (sanity of the forced loop)
        for name in ["clone_fr", "fr_female"] {
            let six = try XCTUnwrap(forced["tts-4b-6bit"]?[name])
            XCTAssertEqual(six.agree, six.frames, "6-bit forced on its own codes must give its own semantic codes (\(name))")
            XCTAssertGreaterThan(six.eoaMargin, 0, "6-bit must read EOA where its trajectory ended (\(name))")
        }
    }

    /// 3. SWAP (K-97 verification): which module of the 4-bit pack parts from the others? The three packs' LLMs run
    /// on the same history (the 6-bit seed-1 trajectory); every LLM's hidden state is read by every pack's semantic
    /// head. Each (LLM, head) pair is compared with the bf16 pair (bf16 LLM, bf16 head), the scale for the others.
    func testSemanticAgreementBySwappedModules() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_CLONE_OVERRUN"] == "1", "Set VOXTRAL_TTS_CLONE_OVERRUN=1 to run")
        let clonePath = try XCTUnwrap(env["VOXTRAL_TTS_CLONE_OVERRUN_EMB"], "VOXTRAL_TTS_CLONE_OVERRUN_EMB required")
        let clone = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: clonePath))["embedding"])
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let text = try String(contentsOf: root.appendingPathComponent("docs/eval/tts/medium_en.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let ids = ["tts-4b-mlx", "tts-4b-6bit", "tts-4b-4bit"], short = ["bf16", "6bit", "4bit"]
        var pipelines: [VoxtralTTSPipeline] = []
        for id in ids {
            let p = VoxtralTTSPipeline()
            try await p.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: id)))
            pipelines.append(p)
        }
        defer { pipelines.forEach { $0.unload() } }
        let models = try pipelines.map { try XCTUnwrap($0.ttsModel) }
        let tokenizer = try XCTUnwrap(pipelines[1].tokenizer)
        let six = models[1]

        for (name, voice) in [("clone_fr", clone), ("fr_female", try XCTUnwrap(pipelines[1].voiceEmbeddings["fr_female"]))] {
            let out = six.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer, maxTokens: 2_379, seed: 1)
            let codes = out.codes[0].asArray(Int32.self), width = out.codes.dim(2)
            let ref = (0 ..< out.numFrames).map { Array(codes[($0 * width) ..< (($0 + 1) * width)]) }
            let tokenIds = six.encodeText(text, voiceFrameCount: voice.dim(0), tokenizer: tokenizer, sanitize: true)
            let audioTok = Int32(six.config.multimodal.audioModelArgs.audioTokenId)
            var caches = models.map { $0.createCache() }
            var hidden: [MLXArray] = []
            for (m, model) in models.enumerated() {
                _ = model.llmForward(inputEmbeds: model.buildInputEmbeddings(
                    inputIds: MLXArray(tokenIds).reshaped(1, tokenIds.count), voiceEmbedding: voice), cache: caches[m])
                hidden.append(model.llmForward(inputEmbeds: model.embedTokens(MLXArray([audioTok]).reshaped(1, 1)),
                                               cache: caches[m]))
            }
            var agree = [[Int]](repeating: [Int](repeating: 0, count: 3), count: 3)  // [llm][head] vs (bf16, bf16)
            for f in 0 ..< ref.count {
                let h = hidden.map { $0[0..., -1, 0...] }
                let truth = semanticRead(models[0], h[0]).argmax
                for llm in 0 ..< 3 { for head in 0 ..< 3 where semanticRead(models[head], h[llm]).argmax == truth {
                    agree[llm][head] += 1 } }
                for (m, model) in models.enumerated() {
                    let next = model.mmAudioEmbeddings
                        .audioCodebookEmbeddings(model.codesToGlobalIndices(MLXArray(ref[f]).reshaped(1, ref[f].count)))
                        .sum(axis: 1, keepDims: true)
                    hidden[m] = model.llmForward(inputEmbeds: next, cache: caches[m])
                }
                if f % 64 == 0 { MLX.eval(hidden) }
            }
            caches.removeAll()
            for llm in 0 ..< 3 {
                print("SWAP \(name) LLM \(short[llm]): " + (0 ..< 3).map { head in
                    String(format: "head %@ %.1f %%", short[head], Double(agree[llm][head]) / Double(ref.count) * 100)
                }.joined(separator: " · ") + " (vs bf16 LLM + bf16 head, \(ref.count) frames of the 6-bit seed-1 trajectory)")
            }
            XCTAssertEqual(agree[0][0], ref.count, "bf16 pair against itself")
        }
    }
}
