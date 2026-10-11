/**
 * TTSCarrierEarlyCutTests — can the streaming path cut the warm-up carrier before 3 s of audio have accumulated
 * (K-68)?
 *
 * Streaming decodes the accumulated codes at every chunk and, today, waits for the 3 s scan window before calling
 * `trimLeadingCarrierAdaptive`, for fear of a cut decided on a partial waveform. The cut only reads 80 ms frame
 * levels from the start of the waveform, so a decision on a prefix equals the decision on the whole waveform when
 * (1) the codec decodes a prefix's frames like the full decode does, except for its last few frames, and (2) the
 * pause and the frame where speech resumes both lie before that edge.
 *
 * Per seed, the test generates codes in batch with the recommended carrier, decodes the whole waveform (the batch
 * decision, the reference) and every streaming prefix (3, 13, 23… frames), and reports:
 *  - EDGE: the largest per-frame level difference between a prefix decode and the full decode, by distance from
 *    the prefix end (in frames): how many tail frames a prefix decision must ignore;
 *  - CUT: per margin m, the first prefix where the cut on (prefix − m frames) is non-zero, whether it equals the
 *    reference cut, and the frames that had to be generated.
 *
 * Measured 2026-10-11 (tts-4b-6bit, clone_fr, 10 seeds): EDGE 0.00 dB at every distance, 0 mismatches at margin 0,
 * cut decided after 23 frames (median) instead of 43: the streaming path now decides on the prefix (no margin).
 * Asserts: no prefix frame differs from the full decode, and margin 0 never disagrees with the reference.
 *
 * Env: VOXTRAL_TTS_CARRIER_EARLY=1 (from xcodebuild: TEST_RUNNER_VOXTRAL_TTS_CARRIER_EARLY=1),
 *   VOXTRAL_TTS_CARRIER_EARLY_EMB (enrolled voice .safetensors, required), optional
 *   VOXTRAL_TTS_CARRIER_EARLY_MODEL (default tts-4b-6bit), VOXTRAL_TTS_CARRIER_EARLY_SEEDS (default 10).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSCarrierEarlyCutTests: XCTestCase {

    func testPrefixCutMatchesFullCut() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_CARRIER_EARLY"] == "1", "Set VOXTRAL_TTS_CARRIER_EARLY=1 to run")
        let embPath = try XCTUnwrap(env["VOXTRAL_TTS_CARRIER_EARLY_EMB"], "VOXTRAL_TTS_CARRIER_EARLY_EMB is required")
        let id = env["VOXTRAL_TTS_CARRIER_EARLY_MODEL"] ?? "tts-4b-6bit"
        let seeds = Int(env["VOXTRAL_TTS_CARRIER_EARLY_SEEDS"] ?? "10") ?? 10
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let text = try String(contentsOf: root.appendingPathComponent("docs/eval/tts/short_en.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let genText = VoxtralTTSPipeline.recommendedWarmUpVocalise + " " + text

        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: id)))
        defer { pipeline.unload() }
        let model = try XCTUnwrap(pipeline.ttsModel)
        let tokenizer = try XCTUnwrap(pipeline.tokenizer)
        let voice = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: embPath))["embedding"])
        let sampleRate = pipeline.sampleRate
        let frameSize = sampleRate * 2 / 25
        let margins = [0, 1, 2, 3, 4]

        func frameDB(_ w: MLXArray, _ i: Int) -> Float {
            let s = w[(i * frameSize) ..< ((i + 1) * frameSize)].asType(.float32)
            let r = MLX.sqrt(MLX.mean(s * s)).item(Float.self)
            return 20 * Foundation.log10(max(r, 1e-9))
        }

        var edgeWorst = [Float](repeating: 0, count: 8)  // by distance from the prefix end, 1 = last frame
        var deepestIdentical = Float(0)                  // worst difference farther than 8 frames from the end
        var mismatches = [Int](repeating: 0, count: margins.count)
        var framesNeeded = [[Int]](repeating: [], count: margins.count)
        for seed in 1 ... seeds {
            let (codes, numFrames, _) = model.generate(text: genText, voiceEmbedding: voice, tokenizer: tokenizer,
                                                       maxTokens: 216, seed: UInt64(seed))
            let full = model.decodeToWaveform(codes)
            MLX.eval(full)
            let (_, reference) = trimLeadingCarrierAdaptive(full, sampleRate: sampleRate)
            var decided = [Int?](repeating: nil, count: margins.count)
            var prefixes = [3]
            while prefixes.last! + 10 < numFrames { prefixes.append(prefixes.last! + 10) }
            for n in prefixes {
                let part = model.decodeToWaveform(codes[0..., ..<n, 0...])
                MLX.eval(part)
                let partFrames = part.dim(0) / frameSize
                for i in 0 ..< partFrames {
                    let d = abs(frameDB(part, i) - frameDB(full, i))
                    let fromEnd = partFrames - i
                    if fromEnd <= edgeWorst.count { edgeWorst[fromEnd - 1] = max(edgeWorst[fromEnd - 1], d) }
                    else { deepestIdentical = max(deepestIdentical, d) }
                }
                for (k, m) in margins.enumerated() where decided[k] == nil {
                    let keep = partFrames - m
                    guard keep > 0 else { continue }
                    let (_, cut) = trimLeadingCarrierAdaptive(part[..<(keep * frameSize)], sampleRate: sampleRate)
                    if cut > 0 {
                        decided[k] = cut
                        framesNeeded[k].append(n)
                        if cut != reference { mismatches[k] += 1 }
                        print("CUT seed \(seed) margin \(m) prefix \(n) cut \(cut) reference \(reference)")
                    }
                }
            }
            for k in margins.indices where decided[k] == nil {
                framesNeeded[k].append(numFrames)
                if reference != 0 { mismatches[k] += 1 }
                print("CUT seed \(seed) margin \(margins[k]) undecided before the end, reference \(reference)")
            }
        }
        print("EDGE \(id) max |ΔdB| by frame from the prefix end (1 = last): "
              + edgeWorst.map { String(format: "%.2f", $0) }.joined(separator: " ")
              + String(format: " ; farther: %.3f", deepestIdentical))
        for (k, m) in margins.enumerated() {
            let sorted = framesNeeded[k].sorted()
            print("CUTSUM \(id) margin \(m): mismatches \(mismatches[k])/\(seeds), frames generated before the cut "
                  + "median \(sorted[sorted.count / 2]) max \(sorted.last!)")
        }
        XCTAssertEqual(max(edgeWorst.max() ?? 0, deepestIdentical), 0, accuracy: 0.01, "the codec is no longer causal")
        XCTAssertEqual(mismatches[0], 0, "a prefix cut disagrees with the full-waveform cut")
    }
}
