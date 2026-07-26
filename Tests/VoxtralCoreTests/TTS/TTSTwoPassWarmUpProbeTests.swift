/**
 * TTSTwoPassWarmUpProbeTests — probe an EXACT warm-up carrier boundary by
 * generating the carrier and the content as two passes over ONE KV cache.
 *
 * Why: the carrier boundary cannot be recovered after the fact. Text tokens
 * carry no audio-frame alignment (the whole prompt is prefilled at once and the
 * model paces frames itself), the per-frame semantic code that marks the
 * boundary differs by seed (3648 / 3579 / 8032 observed), and the terminal
 * pause's depth ranges from −43 dB to −126 dB — so no fixed threshold, and no
 * single code, can find it. Measured consequence: the vocalise leaked into the
 * output on 2 of 3 seeds, and on another the trim ate a whole sentence.
 *
 * Design: prefill [voice ⟨next⟩ carrier ⟨repeat⟩ BEGIN_AUDIO], generate frames
 * until EOA — those are the carrier, and they are DISCARDED — then append
 * [⟨next⟩ content ⟨repeat⟩ BEGIN_AUDIO] to the SAME cache and keep generating.
 * Pass-2 frames are the content by construction, so no heuristic is involved,
 * while the carrier stays in the attention context (the point of the warm-up).
 *
 * Risk this probe exists to measure: a second utterance after EOA is likely
 * out of distribution for a model trained on single utterances. The probe
 * writes the pass-2 audio so it can be transcribed and judged.
 *
 * Env: VOXTRAL_TTS_TWOPASS=1, VOXTRAL_TTS_REPRO_EMB, VOXTRAL_TWOPASS_OUT,
 *   optional VOXTRAL_TWOPASS_SEEDS (comma-separated, default "7,42,777").
 */

import XCTest
import MLX
import MLXLMCommon
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSTwoPassWarmUpProbeTests: XCTestCase {

    func testTwoPassCarrierBoundary() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_TWOPASS"] == "1", "Set VOXTRAL_TTS_TWOPASS=1 to run")
        guard let embPath = env["VOXTRAL_TTS_REPRO_EMB"], let outPath = env["VOXTRAL_TWOPASS_OUT"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB and VOXTRAL_TWOPASS_OUT")
        }
        let outDir = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let seeds = (env["VOXTRAL_TWOPASS_SEEDS"] ?? "7,42,777")
            .split(separator: ",").compactMap { UInt64($0.trimmingCharacters(in: .whitespaces)) }

        guard let modelInfo = VoxtralTTSRegistry.model(withId: "tts-4b-mlx") else {
            throw XCTSkip("tts-4b-mlx not in registry")
        }
        let modelDir = try await ModelDownloader.downloadTTSModel(modelInfo)
        let model = try loadVoxtralTTSModel(from: modelDir)
        let tokenizer = TekkenTokenizer(modelPath: modelDir.path)
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let emb = arrays["embedding"] ?? arrays.values.first else {
            throw XCTSkip("No embedding in \(embPath)")
        }

        let carrier = VoxtralTTSPipeline.recommendedWarmUpVocalise
        let content = "Fluxforge Studio transforme votre Mac en un studio de création IA complet. Tout tourne en local, sans cloud ni abonnement."

        // Token layout mirrors encodeText: the segment appended for pass 2 is
        // everything after the voice block.
        let NEXT: Int32 = 36, REPEAT: Int32 = 35
        let beginAudio = Int32(model.config.multimodal.audioModelArgs.beginAudioTokenId)
        let audioTok = Int32(model.config.multimodal.audioModelArgs.audioTokenId)

        /// Generate frames from `hidden` until EOA; returns the codes.
        func generateFrames(_ hidden0: MLXArray, cache: [any KVCache], maxFrames: Int) -> [MLXArray] {
            var hidden = hidden0
            var frames: [MLXArray] = []
            for _ in 0 ..< maxFrames {
                let h = hidden[0..., -1, 0...]
                let codes = model.acousticTransformer.decodeOneFrame(h)
                MLX.eval(codes)
                if codes[0, 0].item(Int32.self) <= 1 { break }   // EOA
                frames.append(codes)
                let next = model.mmAudioEmbeddings
                    .audioCodebookEmbeddings(model.codesToGlobalIndices(codes))
                    .sum(axis: 1, keepDims: true)
                hidden = model.llmForward(inputEmbeds: next, cache: cache)
            }
            return frames
        }

        for seed in seeds {
            MLXRandom.seed(seed)

            // ---- Pass 1: voice + carrier, generated and DISCARDED ----
            let carrierIds = model.encodeText(carrier, voiceFrameCount: emb.dim(0),
                                              tokenizer: tokenizer, sanitize: true)
            let carrierEmb = model.buildInputEmbeddings(
                inputIds: MLXArray(carrierIds).reshaped(1, carrierIds.count), voiceEmbedding: emb)
            let cache = model.createCache()
            var hidden = model.llmForward(inputEmbeds: carrierEmb, cache: cache)
            hidden = model.llmForward(
                inputEmbeds: model.embedTokens(MLXArray([audioTok]).reshaped(1, 1)), cache: cache)
            MLX.eval(hidden)
            let carrierFrames = generateFrames(hidden, cache: cache, maxFrames: 200)

            // ---- Pass 2: append the content segment to the SAME cache ----
            var ids: [Int32] = [NEXT]
            ids.append(contentsOf: tokenizer.encode(VoxtralTTSModel.sanitizeTextForTTS(content)).map { Int32($0) })
            ids.append(REPEAT)
            ids.append(beginAudio)
            hidden = model.llmForward(
                inputEmbeds: model.embedTokens(MLXArray(ids).reshaped(1, ids.count)), cache: cache)
            hidden = model.llmForward(
                inputEmbeds: model.embedTokens(MLXArray([audioTok]).reshaped(1, 1)), cache: cache)
            MLX.eval(hidden)
            let contentFrames = generateFrames(hidden, cache: cache, maxFrames: 600)

            print("[twopass] seed \(seed): carrier \(carrierFrames.count) frames (discarded), content \(contentFrames.count) frames")
            guard !contentFrames.isEmpty else {
                print("[twopass] seed \(seed): pass 2 produced NOTHING — out of distribution")
                continue
            }
            let codes = MLX.stacked(contentFrames, axis: 1)
            let wav = model.decodeToWaveform(codes)
            MLX.eval(wav)
            let url = outDir.appendingPathComponent("twopass_s\(seed).wav")
            try WAVWriter.write(waveform: wav, to: url)
            print(String(format: "[twopass] seed %d: %.2f s -> %@",
                         seed, Double(wav.dim(0)) / 24000.0, url.path))
        }
    }
}
