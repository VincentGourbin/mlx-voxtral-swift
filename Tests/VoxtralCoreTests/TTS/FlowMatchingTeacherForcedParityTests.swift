/**
 * FlowMatchingTeacherForcedParityTests — the flow-matching head in the weights' dtype (K-39) against the
 * float32 head it replaces, frame by frame on the SAME LLM hidden states.
 *
 * Teacher forcing: the float32 head drives the trajectory (its codes are fed back to the LLM), and at every
 * frame both variants decode the same hidden state with the same noise (the RNG is re-seeded per frame and
 * per variant). Free-running comparison would diverge at the first differing code and say nothing.
 *
 * Gate (fiche K-39): semantic codes identical 100 %; acoustic codes identical ≥ 99 % with |Δ| ≤ 1 level.
 *
 * Env: VOXTRAL_TTS_PARITY=1 (from xcodebuild: TEST_RUNNER_VOXTRAL_TTS_PARITY=1), optional
 *   VOXTRAL_TTS_PARITY_MODELS (comma-separated, default "tts-4b-mlx,tts-4b-4bit,tts-4b-6bit").
 */

import XCTest
import MLX
import MLXLMCommon
@testable import VoxtralCore

@available(macOS 14.0, *)
final class FlowMatchingTeacherForcedParityTests: XCTestCase {

    func testWeightDTypeHeadMatchesFloat32Head() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_PARITY"] == "1", "Set VOXTRAL_TTS_PARITY=1 to run")
        let models = (env["VOXTRAL_TTS_PARITY_MODELS"] ?? "tts-4b-mlx,tts-4b-4bit,tts-4b-6bit")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let texts = try ["short_en", "medium_en"].map { name in
            (name, try String(contentsOf: root.appendingPathComponent("docs/eval/tts/\(name).txt"), encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines))
        }

        for id in models {
            let pipeline = VoxtralTTSPipeline()
            let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: id))
            try await pipeline.loadModel(modelInfo: info)
            defer { pipeline.unload() }
            let model = try XCTUnwrap(pipeline.ttsModel)
            let tokenizer = try XCTUnwrap(pipeline.tokenizer)
            let voice = try XCTUnwrap(pipeline.voiceEmbeddings[VoxtralVoice.neutralFemale.rawValue])
            let fm = model.acousticTransformer
            defer { fm.computeInWeightDType = true }

            var frames = 0, semanticSame = 0, acousticSame = 0, acousticTotal = 0, maxDelta: Int32 = 0
            for (name, text) in texts {
                let ids = model.encodeText(text, voiceFrameCount: voice.dim(0), tokenizer: tokenizer, sanitize: true)
                let cache = model.createCache()
                _ = model.llmForward(
                    inputEmbeds: model.buildInputEmbeddings(inputIds: MLXArray(ids).reshaped(1, ids.count),
                                                            voiceEmbedding: voice), cache: cache)
                let audioTok = Int32(model.config.multimodal.audioModelArgs.audioTokenId)
                var hidden = model.llmForward(inputEmbeds: model.embedTokens(MLXArray([audioTok]).reshaped(1, 1)),
                                              cache: cache)
                for frame in 0 ..< 1_000 {
                    let h = hidden[0..., -1, 0...]
                    MLX.eval(h)
                    fm.computeInWeightDType = false
                    MLXRandom.seed(UInt64(frame + 1))
                    let reference = fm.decodeOneFrame(h)
                    fm.computeInWeightDType = true
                    MLXRandom.seed(UInt64(frame + 1))
                    let candidate = fm.decodeOneFrame(h)
                    MLX.eval(reference, candidate)
                    let ref = reference.asArray(Int32.self), cand = candidate.asArray(Int32.self)
                    frames += 1
                    if ref[0] == cand[0] { semanticSame += 1 }
                    for k in 1 ..< ref.count {
                        acousticTotal += 1
                        if ref[k] == cand[k] { acousticSame += 1 }
                        maxDelta = max(maxDelta, abs(ref[k] - cand[k]))
                    }
                    if ref[0] <= 1 { break }  // EOA of the reference trajectory
                    let next = model.mmAudioEmbeddings
                        .audioCodebookEmbeddings(model.codesToGlobalIndices(reference))
                        .sum(axis: 1, keepDims: true)
                    hidden = model.llmForward(inputEmbeds: next, cache: cache)
                }
                print("[parity] \(id) \(name): \(frames) frames so far")
            }
            let semanticPct = Double(semanticSame) / Double(frames) * 100
            let acousticPct = Double(acousticSame) / Double(acousticTotal) * 100
            print(String(format: "PARITY %@ dtype %@ frames %d semantic %.2f %% acoustic %.2f %% max|Δ| %d",
                         id, "\(fm.layerComputeDType)", frames, semanticPct, acousticPct, maxDelta))
            XCTAssertEqual(semanticSame, frames, "\(id): semantic codes must be identical")
            XCTAssertGreaterThanOrEqual(acousticPct, 99, "\(id): acoustic codes identical ≥ 99 %")
            XCTAssertLessThanOrEqual(maxDelta, 1, "\(id): acoustic codes within one level")
        }
    }
}
