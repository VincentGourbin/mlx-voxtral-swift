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

// MARK: - VoxtralRealtimeModel

public class VoxtralRealtimeModel: Module {

    public let config: VoxtralRealtimeConfiguration

    @ModuleInfo var encoder: VoxtralRealtimeEncoder
    @ModuleInfo var decoder: VoxtralRealtimeDecoder

    public init(config: VoxtralRealtimeConfiguration) {
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
    public func encodeAudio(_ mel: MLXArray) -> MLXArray {
        encoder(mel)
    }

    /// Extract audio embeddings (public API for embedding extraction).
    /// mel: [mel_bins, frames] → [1, n_tokens, decoder_dim]
    public func extractAudioEmbeddings(_ mel: MLXArray) -> MLXArray {
        encoder(mel).expandedDimensions(axis: 0)
    }

    // MARK: - Generation

    /// Transcribe audio from mel spectrogram.
    /// Returns (text_tokens, audio_embeddings).
    public func generate(
        mel: MLXArray,
        tokenizer: TekkenTokenizer,
        maxTokens: Int = 4096,
        temperature: Float = 0.0,
        delayMs: Int = VoxtralRealtimeConfiguration.defaultTranscriptionDelayMs
    ) -> (tokens: [Int], audioEmbeddings: MLXArray) {
        let nDelay = config.numDelayTokens(delayMs: delayMs)
        let nLeft = config.nLeftPadTokens

        // Precompute Ada RMS-Norm scales for chosen delay
        let tCond = computeTimeEmbedding(tValue: Float(nDelay), dim: config.decoder.dim)
        decoder.precomputeAdaScales(tCond: tCond)

        // Encode audio
        let adapterOut = encodeAudio(mel)  // [n_audio_total, decoder_dim]
        let nAudioTotal = adapterOut.dim(0)
        MLX.eval(adapterOut)

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
        var hidden = decoder.forward(embeds: prefixEmbeds, cache: cache)
        var logits = decoder.logits(hidden[hidden.dim(0) - 1])
        MLX.eval(logits)

        // Autoregressive decode
        var generated: [Int] = []

        for pos in promptLen..<nAudioTotal {
            let token = nextToken(logits: logits, temperature: temperature)
            generated.append(token)

            if token == config.eosTokenId || generated.count > maxTokens {
                break
            }

            // Build next input: audio_embed[pos] + tok_embed[token]
            let audioEmb = adapterOut[pos]
            let tokEmb = decoder.embedToken(token)
            let embed = audioEmb + tokEmb

            hidden = decoder.forward(embeds: embed.expandedDimensions(axis: 0), cache: cache)
            logits = decoder.logits(hidden[0])
            MLX.eval(logits)

            if generated.count % 256 == 0 {
                MLX.GPU.clearCache()
            }
        }

        // Read final pending positions (audio exhausted, text-only generation)
        if generated.isEmpty || (generated.last != config.eosTokenId && generated.count <= maxTokens) {
            let token = nextToken(logits: logits, temperature: temperature)
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

    private func nextToken(logits: MLXArray, temperature: Float) -> Int {
        if temperature == 0 {
            return MLX.argMax(logits).item(Int.self)
        }
        return MLXRandom.categorical(logits * (1.0 / temperature)).item(Int.self)
    }
}
