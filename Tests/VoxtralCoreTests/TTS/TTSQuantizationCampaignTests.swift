/**
 * TTSQuantizationCampaignTests — settle whether a quantized model is safe for
 * ENROLLED voices, by scoring transcriptions of the same texts and seeds
 * against several TTS models.
 *
 * Issue #45 item 3 recommends bf16 for cloned voices, on the basis that q6
 * "dropped words" — a single observation, made before the enrollment high-pass
 * was fixed. A thin embedding (the broken filter cost a male fundamental
 * 24–27 dB) is plausibly more fragile under quantization, so the conclusion may
 * not survive a correct enrollment. bf16 costs ~2.5x the wall clock and 8 GB
 * against 3.5 GB, so it is worth settling rather than assuming.
 *
 * Each TTS model is loaded ONCE and drives every text × seed; the ASR is then
 * loaded once to transcribe everything. Running this through the CLI instead
 * would reload a model per generation and dominate the runtime.
 *
 * Scoring is deliberately crude and reported raw: a normalized word-overlap
 * against the input text, plus flags for a leaked warm-up vocalise and for a
 * missing tail. Word-level ASR noise is real, so compare models against each
 * other on identical inputs — not against an absolute bar.
 *
 * Env: VOXTRAL_TTS_CAMPAIGN=1, VOXTRAL_TTS_REPRO_EMB (enrolled voice),
 *   optional VOXTRAL_CAMPAIGN_MODELS (default "tts-4b-mlx,tts-4b-6bit"),
 *   VOXTRAL_CAMPAIGN_SEEDS (default "7,42,99,123,777"),
 *   VOXTRAL_CAMPAIGN_OUT (WAV dir; default a temp dir).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSQuantizationCampaignTests: XCTestCase {

    private struct Sample {
        let model: String, seed: UInt64, textIndex: Int
        let url: URL, duration: Double, genTime: Double
    }

    /// Words in `s`, lowercased and stripped of punctuation.
    private func words(_ s: String) -> [String] {
        var cleaned = ""
        for ch in s.lowercased() {
            cleaned.append(ch.isLetter || ch.isNumber ? ch : " ")
        }
        return cleaned.split(separator: " ").map(String.init)
    }

    /// Fraction of the reference's words present in the hypothesis (multiset).
    private func coverage(reference: String, hypothesis: String) -> Double {
        let ref = words(reference)
        guard !ref.isEmpty else { return 0 }
        var pool: [String: Int] = [:]
        for w in words(hypothesis) { pool[w, default: 0] += 1 }
        var hit = 0
        for w in ref where (pool[w] ?? 0) > 0 { pool[w]! -= 1; hit += 1 }
        return Double(hit) / Double(ref.count)
    }

    func testQuantizationCampaign() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_CAMPAIGN"] == "1", "Set VOXTRAL_TTS_CAMPAIGN=1 to run")
        guard let embPath = env["VOXTRAL_TTS_REPRO_EMB"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB to an enrolled-voice .safetensors")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let emb = arrays["embedding"] ?? arrays.values.first else {
            throw XCTSkip("No embedding in \(embPath)")
        }

        let modelIds = (env["VOXTRAL_CAMPAIGN_MODELS"] ?? "tts-4b-mlx,tts-4b-6bit")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let seeds = (env["VOXTRAL_CAMPAIGN_SEEDS"] ?? "7,42,99,123,777")
            .split(separator: ",").compactMap { UInt64($0.trimmingCharacters(in: .whitespaces)) }
        let useWarmUp = env["VOXTRAL_CAMPAIGN_WARMUP"] != "0"
        let outDir = URL(fileURLWithPath: env["VOXTRAL_CAMPAIGN_OUT"]
            ?? NSTemporaryDirectory()).appendingPathComponent("tts_campaign")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        // Deliberately ordinary sentences: a made-up brand name is mangled by
        // every voice (the shipped presets included) and would swamp the signal.
        let texts = [
            "Bonjour, ceci est un test de synthèse vocale avec une voix personnalisée pour vérifier la qualité du début.",
            "Le petit chat dort sur le canapé du salon pendant que la pluie tombe doucement sur les carreaux de la fenêtre.",
            "Nous avons terminé la migration du serveur hier soir, et toutes les données ont été transférées sans aucune perte.",
        ]

        var samples: [Sample] = []
        for modelId in modelIds {
            guard let info = VoxtralTTSRegistry.model(withId: modelId) else {
                print("[campaign] unknown model \(modelId) — skipped"); continue
            }
            let tts = VoxtralTTSPipeline()
            try await tts.loadModel(modelInfo: info)
            for (ti, text) in texts.enumerated() {
                for seed in seeds {
                    let t0 = Date()
                    let r = try await tts.synthesize(
                        text: text, voiceEmbedding: emb, seed: seed,
                        warmUpText: useWarmUp ? VoxtralTTSPipeline.recommendedWarmUpVocalise : nil)
                    let gen = Date().timeIntervalSince(t0)
                    let url = outDir.appendingPathComponent("\(modelId)_t\(ti)_s\(seed).wav")
                    try WAVWriter.write(waveform: r.waveform, to: url)
                    samples.append(Sample(model: modelId, seed: seed, textIndex: ti, url: url,
                                          duration: r.duration, genTime: gen))
                }
            }
            tts.unload()
        }

        // One ASR load for every sample.
        let asr = VoxtralPipeline(model: .mini3b8bit)
        try await asr.loadModel()

        print("[campaign] model / text / seed / duration / RTF / coverage / flags / transcript")
        var byModel: [String: (cov: [Double], rtf: [Double], bad: Int)] = [:]
        for s in samples {
            let hyp = (try? await asr.transcribe(audio: s.url, language: "fr")) ?? ""
            let cov = coverage(reference: texts[s.textIndex], hypothesis: hyp)
            let rtf = s.duration > 0 ? s.genTime / s.duration : 0
            // Detect a leaked vocalise on the letters alone: the ASR renders it
            // as "la la", "lalala", "là, là" or "ah là là" depending on the take,
            // so punctuation- and space-sensitive matching misses most of them.
            let head = words(hyp).prefix(8).joined()
            let leaked = head.hasPrefix("lala") || head.hasPrefix("ahlala")
                || head.hasPrefix("lalala") || head.contains("lalala")
            let short = cov < 0.8
            var flags = ""
            if leaked { flags += "VOCALISE " }
            if short { flags += "INCOMPLET " }
            let dur = String(format: "%.1f", s.duration)
            let rtfs = String(format: "%.2f", rtf)
            let covs = String(format: "%.1f", cov * 100)
            print("[campaign] \(s.model) t\(s.textIndex) s\(s.seed) \(dur)s RTF \(rtfs) cover \(covs)% \(flags)| \(hyp.prefix(52))")
            var e = byModel[s.model] ?? ([], [], 0)
            e.cov.append(cov); e.rtf.append(rtf); if leaked || short { e.bad += 1 }
            byModel[s.model] = e
        }

        print("[campaign] ---- summary ----")
        for (m, e) in byModel.sorted(by: { $0.key < $1.key }) {
            let mc = e.cov.reduce(0, +) / Double(e.cov.count)
            let mr = e.rtf.reduce(0, +) / Double(e.rtf.count)
            let mcs = String(format: "%.1f", mc * 100)
            let mrs = String(format: "%.2f", mr)
            print("[campaign] \(m): coverage \(mcs)%  RTF \(mrs)  bad \(e.bad)/\(e.cov.count)")
        }
        print("[campaign] WAVs in \(outDir.path)")
    }
}
