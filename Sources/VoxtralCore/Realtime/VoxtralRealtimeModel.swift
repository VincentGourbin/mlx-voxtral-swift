/**
 * VoxtralRealtimeModel - Complete Voxtral Realtime model assembling encoder + decoder
 *
 * Inference pipeline:
 * 1. Compute mel spectrogram from audio
 * 2. Causal encoder → 4x downsample → adapter embeddings
 * 3. Construct prompt: [BOS] + [STREAMING_PAD] * (n_left + n_delay)
 * 4. For each position: input = audio_embed + tok_embed(token_id)
 * 5. Prefill decoder, then autoregressive generation until EOS
 *
 * Reference: mlx-audio/stt/models/voxtral_realtime/voxtral_realtime.py
 */

import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon
import MLXProfiler

// MARK: - VoxtralRealtimeModel

class VoxtralRealtimeModel: Module {

    let config: VoxtralRealtimeConfiguration

    @ModuleInfo var encoder: VoxtralRealtimeEncoder
    @ModuleInfo var decoder: VoxtralRealtimeDecoder

    init(config: VoxtralRealtimeConfiguration) {
        self.config = config
        self._encoder.wrappedValue = VoxtralRealtimeEncoder(
            config: config.encoderArgs, decoderDim: config.decoder.dim
        )
        self._decoder.wrappedValue = VoxtralRealtimeDecoder(config: config.decoder)
        super.init()
    }

    // MARK: - Audio Encoding

    /// Encode mel spectrogram to audio embeddings.
    /// mel: [mel_bins, frames] → [n_tokens, decoder_dim]
    func encodeAudio(_ mel: MLXArray) -> MLXArray {
        encoder(mel)
    }

    /// Extract audio embeddings (public API for embedding extraction).
    /// mel: [mel_bins, frames] → [1, n_tokens, decoder_dim]
    func extractAudioEmbeddings(_ mel: MLXArray) -> MLXArray {
        encoder(mel).expandedDimensions(axis: 0)
    }

    // MARK: - Generation

    /// Transcribe audio from mel spectrogram.
    /// Returns (text_tokens, audio_embeddings).
    func generate(
        mel: MLXArray,
        tokenizer: TekkenTokenizer,
        maxTokens: Int = 4096,
        temperature: Float = 0.0,
        delayMs: Int = VoxtralRealtimeConfiguration.defaultTranscriptionDelayMs
    ) -> (tokens: [Int], audioEmbeddings: MLXArray) {
        let nDelay = config.numDelayTokens(delayMs: delayMs)
        let nLeft = config.nLeftPadTokens
        let session = MLXProfiler.shared.activeSession

        // Precompute Ada RMS-Norm scales for chosen delay
        let tCond = computeTimeEmbedding(tValue: Float(nDelay), dim: config.decoder.dim)
        decoder.precomputeAdaScales(tCond: tCond)

        // Encode audio
        session?.beginPhase("Audio Encoding", category: .audioEncode)
        let adapterOut = encodeAudio(mel)  // [n_audio_total, decoder_dim]
        // A caught MLX error leaves empty arrays: stop before reading shapes or items;
        // the public boundary (withMLXErrors) then throws VoxtralError.mlx (K-1)
        if MLXErrorScope.hasError { return ([], adapterOut) }
        // Cancelled during the encoding: stop before reading shapes; the pipeline throws CancellationError (K-15)
        if VoxtralCancellation.isCancelled { return ([], adapterOut) }
        DTypeAudit.report("realtime", "mel", mel)
        DTypeAudit.report("realtime", "encoder_out", adapterOut)
        let nAudioTotal = adapterOut.dim(0)
        MLX.eval(adapterOut)
        session?.endPhase("Audio Encoding", category: .audioEncode)

        let promptLen = 1 + nLeft + nDelay  // BOS + padding + delay

        // Build prompt embeddings
        var promptIds: [Int32] = [Int32(config.bosTokenId)]
        promptIds.append(contentsOf: Array(repeating: Int32(config.streamingPadTokenId), count: nLeft + nDelay))

        let promptIdsMx = MLXArray(promptIds)
        let promptTextEmbeds = decoder.embedTokens(promptIdsMx)  // [prompt_len, dim]

        // Sum audio embeds with text embeds for prompt positions
        let audioSlice = adapterOut[..<min(promptLen, nAudioTotal)]
        let textSlice = promptTextEmbeds[..<min(promptLen, audioSlice.dim(0))]
        var prefixEmbeds: MLXArray
        if audioSlice.dim(0) >= promptLen {
            prefixEmbeds = audioSlice[..<promptLen] + promptTextEmbeds
        } else {
            // Audio shorter than prompt — pad text embeds for remaining
            let audioPartLen = audioSlice.dim(0)
            let combinedPart = audioSlice + promptTextEmbeds[..<audioPartLen]
            let textOnlyPart = promptTextEmbeds[audioPartLen...]
            prefixEmbeds = MLX.concatenated([combinedPart, textOnlyPart], axis: 0)
        }

        // Create KV caches for all decoder layers
        let cache = decoder.createCache()

        // Prefill
        session?.beginPhase("Prefill", category: .prefill)
        let hidden = decoder.forward(embeds: prefixEmbeds, cache: cache)
        if MLXErrorScope.hasError { return ([], adapterOut) }
        let logits = decoder.logits(hidden[hidden.dim(0) - 1])
        MLX.eval(logits)
        DTypeAudit.report("realtime", "logits", logits)
        DTypeAudit.report("realtime", cache: cache)
        if MLXErrorScope.hasError { return ([], adapterOut) }
        session?.endPhase("Prefill", category: .prefill)

        // Autoregressive decode: one step per audio frame after the prompt (K-5). Pipelined (K-47, T15): the token
        // stays an MLXArray, step N+1 is built from it and queued by asyncEval before step N's token is read, so the
        // GPU runs step N+1 while the CPU reads token N; on EOS or the text budget one queued step is dropped.
        var pending = sampleToken(logits: logits, temperature: temperature)
        var pos = promptLen
        var stepStart = CFAbsoluteTimeGetCurrent()
        let loop = Self.decodeLoop(
            promptLen: promptLen, nAudioTotal: nAudioTotal, maxTextTokens: maxTokens, eosTokenId: config.eosTokenId,
            sample: {
                let token = pending
                // Next input: audio_embed[pos] + tok_embed[token], built before the token is read
                let embed = adapterOut[pos] + decoder.embedTokens(token.reshaped([1]))
                let hidden = decoder.forward(embeds: embed, cache: cache)
                pending = sampleToken(logits: decoder.logits(hidden[0]), temperature: temperature)
                if !MLXErrorScope.hasError { MLX.asyncEval(pending) }
                pos += 1
                return token.item(Int.self)
            },
            advance: { _, _, count in
                if VoxtralCancellation.isCancelled { return false }  // the pipeline throws CancellationError (K-15)
                if MLXErrorScope.hasError { return false }

                let stepDurationUs = UInt64((CFAbsoluteTimeGetCurrent() - stepStart) * 1_000_000)
                session?.recordStep(index: count, total: nAudioTotal - promptLen, durationUs: stepDurationUs, category: .generationStep)
                stepStart = CFAbsoluteTimeGetCurrent()

                if count % 256 == 0 {
                    Memory.clearCache()
                }
                return true
            })
        var generated = loop.tokens
        lastGenerationTruncated = loop.truncated
        if MLXErrorScope.hasError { return (generated, adapterOut) }

        // Read final pending positions (audio exhausted, text-only generation): the last queued step's token
        if generated.isEmpty || (generated.last != config.eosTokenId && !loop.stopped) {
            let token = pending.item(Int.self)
            if token != config.eosTokenId {
                generated.append(token)
            }
        }

        // Strip EOS if present
        if let last = generated.last, last == config.eosTokenId {
            generated.removeLast()
        }

        return (generated, adapterOut)
    }

    /// True when the last `generate` stopped on its text-token budget rather than the audio end (K-5)
    private(set) var lastGenerationTruncated = false

    /// The decode loop's bounds, separated from the model so they can be tested: one step per
    /// audio frame after the prompt; `sample()` gives the next token, `advance(pos, token, count)`
    /// feeds it back (false stops, e.g. on an MLX error). `stopped` is true when the loop ended
    /// before the audio did.
    static func decodeLoop(
        promptLen: Int, nAudioTotal: Int, maxTextTokens: Int, eosTokenId: Int,
        sample: () -> Int, advance: (_ pos: Int, _ token: Int, _ count: Int) -> Bool
    ) -> (tokens: [Int], steps: Int, truncated: Bool, stopped: Bool) {
        var generated: [Int] = []
        var steps = 0
        var textTokens = 0
        for pos in promptLen ..< nAudioTotal {
            let token = sample()
            generated.append(token)
            steps += 1
            if token == eosTokenId { return (generated, steps, false, true) }
            // Only text tokens spend the budget: streaming pads and specials (Tekken ids < 1000) are free
            if token >= 1_000 {
                textTokens += 1
                if textTokens >= maxTextTokens { return (generated, steps, true, true) }
            }
            if !advance(pos, token, generated.count) { return (generated, steps, false, true) }
        }
        return (generated, steps, false, false)
    }

    /// Next token, lazy (K-47): read with `.item()` once its step has been queued
    private func sampleToken(logits: MLXArray, temperature: Float) -> MLXArray {
        if temperature == 0 {
            return MLX.argMax(logits)
        }
        return MLXRandom.categorical(logits * (1.0 / temperature))
    }
}
