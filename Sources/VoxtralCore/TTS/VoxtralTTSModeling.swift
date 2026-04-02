/**
 * VoxtralTTSModeling - Main TTS model aligned with mlx-audio reference implementation
 *
 * Reference: mlx-audio PR #607, voxtral_tts.py
 *
 * Key architecture points:
 * 1. Input: [BOS, BEGIN_AUDIO, AUDIO×N, text_tokens, BEGIN_AUDIO] with voice embedding replacement
 * 2. LLM backbone = layers + norm (no lm_head)
 * 3. Semantic prediction: DIRECT from LLM hidden via acoustic_transformer.semantic_codebook_output
 * 4. After prefill: one extra step with AUDIO token to trigger first frame
 * 5. EOA: semantic_code <= 1 (0=empty, 1=end_audio)
 * 6. Codebook feedback: global indices with cumulative offsets, summed across codebooks
 */

import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon

// MARK: - mm_audio_embeddings

/// Container matching `mm_audio_embeddings.*` weight keys.
public class MMAudioEmbeddings: Module {

    @ModuleInfo(key: "audio_codebook_embeddings") var audioCodebookEmbeddings: AudioCodebookEmbeddingsContainer
    // Module type to support both Embedding and QuantizedEmbedding
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Module

    public init(config: VoxtralTTSConfiguration) {
        // 9088 = (8192+2=8194 semantic) + pad to 8320 + (21+2)*36=828 + pad to 768 → 8320+768=9088
        let semanticPadded = (config.audioModel.semanticCodebookSize / 128 + 1) * 128  // 8320
        let acousticTotal = config.audioModel.acousticCodebookSize * config.audioModel.nAcousticCodebook
        let acousticPadded = ((acousticTotal + 127) / 128) * 128  // 768
        let audioEmbeddingSize = semanticPadded + acousticPadded  // 9088

        self._audioCodebookEmbeddings.wrappedValue = AudioCodebookEmbeddingsContainer(
            totalSize: audioEmbeddingSize, dim: config.dim
        )
        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dim
        )
        super.init()
    }

    /// Call tok_embeddings, handling both Embedding and QuantizedEmbedding
    public func embedTokens(_ indices: MLXArray) -> MLXArray {
        if let qEmb = tokEmbeddings as? QuantizedEmbedding {
            return qEmb(indices)
        } else if let emb = tokEmbeddings as? Embedding {
            return emb(indices)
        }
        fatalError("Unsupported tok_embeddings type: \(type(of: tokEmbeddings))")
    }
}

public class AudioCodebookEmbeddingsContainer: Module {
    // Module type to support both Embedding and QuantizedEmbedding
    @ModuleInfo(key: "embeddings") var embeddings: Module

    public init(totalSize: Int, dim: Int) {
        self._embeddings.wrappedValue = Embedding(embeddingCount: totalSize, dimensions: dim)
        super.init()
    }

    public func callAsFunction(_ indices: MLXArray) -> MLXArray {
        if let qEmb = embeddings as? QuantizedEmbedding {
            return qEmb(indices)
        } else if let emb = embeddings as? Embedding {
            return emb(indices)
        }
        fatalError("Unsupported embeddings type: \(type(of: embeddings))")
    }
}

// MARK: - VoxtralTTSModel

/// The complete Voxtral TTS model.
public class VoxtralTTSModel: Module {

    public let config: VoxtralTTSConfiguration

    @ModuleInfo(key: "mm_audio_embeddings") var mmAudioEmbeddings: MMAudioEmbeddings
    @ModuleInfo var layers: [LlamaDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "acoustic_transformer") var acousticTransformer: FlowMatchingAudioTransformer
    @ModuleInfo(key: "audio_tokenizer") var audioTokenizer: VoxtralCodecDecoder

    // Computed constants
    let nSpecial: Int32 = 2  // empty_audio=0, end_audio=1
    // Pre-computed codebook offset array (avoids rebuilding per frame)
    let codebookOffsets: MLXArray

    public init(config: VoxtralTTSConfiguration) {
        self.config = config

        self._mmAudioEmbeddings.wrappedValue = MMAudioEmbeddings(config: config)

        let llamaConfig = config.llamaConfig
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            LlamaDecoderLayer(config: llamaConfig)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        self._acousticTransformer.wrappedValue = FlowMatchingAudioTransformer(config: config)
        self._audioTokenizer.wrappedValue = VoxtralCodecDecoder(config: config.audioTokenizer)

        // Pre-compute codebook offset array: [0, 8194, 8217, 8240, ...]
        let semanticSize = config.audioModel.semanticCodebookSize + Int(nSpecial)
        let acousticSize = config.audioModel.acousticCodebookSize + Int(nSpecial)
        var offsets: [Int32] = [0]
        for i in 0..<config.audioModel.nAcousticCodebook {
            offsets.append(Int32(semanticSize + i * acousticSize))
        }
        self.codebookOffsets = MLXArray(offsets).reshaped(1, offsets.count)

        super.init()
    }

    // MARK: - LLM Backbone

    /// Forward through LLM backbone (layers + norm only, no lm_head).
    /// Passes nil mask — LlamaAttention uses .causal mode internally for T > 1.
    /// Relies on MLX lazy evaluation to batch GPU operations across layers.
    public func llmForward(
        inputEmbeds: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray {
        var h = inputEmbeds
        for (layer, c) in zip(layers, cache) {
            h = layer(h, attentionMask: nil, cache: c)
        }
        return norm(h)
    }

    /// Create KV caches for all LLM layers.
    public func createCache() -> [any KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    /// Embed text token IDs using the shared text embedding table.
    public func embedTokens(_ tokenIds: MLXArray) -> MLXArray {
        mmAudioEmbeddings.embedTokens(tokenIds)
    }

    // MARK: - Input Construction

    /// Encode text + voice into token IDs.
    /// Format: [BOS=1, BEGIN_AUDIO=25, AUDIO=24×N, NEXT=36, text_tokens, REPEAT=35, BEGIN_AUDIO=25]
    ///
    /// From paper Section 3.1: segments are interleaved with <next> between A1 and T2,
    /// and <repeat> between T2 and A2.
    /// Verified against mistral_common.SpeechRequest output.
    public func encodeText(_ text: String, voiceFrameCount: Int, tokenizer: TekkenTokenizer) -> [Int32] {
        let textTokens = tokenizer.encode(text).map { Int32($0) }

        let NEXT_TOKEN: Int32 = 36      // <next> — separates voice reference from text
        let REPEAT_TOKEN: Int32 = 35    // <repeat> — separates text from audio generation

        var ids: [Int32] = []
        ids.append(Int32(config.bosTokenId))                  // BOS (1)
        ids.append(Int32(config.multimodal.audioModelArgs.beginAudioTokenId))  // BEGIN_AUDIO (25)
        ids.append(contentsOf: Array(repeating: Int32(config.multimodal.audioModelArgs.audioTokenId), count: voiceFrameCount))  // AUDIO×N (24)
        ids.append(NEXT_TOKEN)                                 // <next> (36)
        ids.append(contentsOf: textTokens)                     // text tokens
        ids.append(REPEAT_TOKEN)                               // <repeat> (35)
        ids.append(Int32(config.multimodal.audioModelArgs.beginAudioTokenId))  // BEGIN_AUDIO (25)
        return ids
    }

    /// Build input embeddings with voice conditioning at AUDIO token positions.
    ///
    /// Voice embeddings REPLACE the audio token embeddings at positions where token == AUDIO (24).
    /// Reference: voxtral_tts.py lines 646-670
    public func buildInputEmbeddings(inputIds: MLXArray, voiceEmbedding: MLXArray) -> MLXArray {
        // Embed all tokens via text embedding table
        var embeddings = embedTokens(inputIds)  // (1, T, dim)

        // Find AUDIO token positions
        let audioTokenId = Int32(config.multimodal.audioModelArgs.audioTokenId)
        let audioMask = inputIds[0] .== MLXArray(audioTokenId)  // (T,) bool

        // Map each audio position to a voice embedding index
        let indices = MLX.cumsum(audioMask.asType(.int32)) - 1
        let clippedIndices = MLX.clip(indices, min: MLXArray(Int32(0)), max: MLXArray(Int32(voiceEmbedding.dim(0) - 1)))

        // Replace audio token embeddings with voice embeddings
        let voiceExpanded = voiceEmbedding[clippedIndices]  // (T, dim)
        let mask3d = MLX.expandedDimensions(audioMask, axis: -1).asType(embeddings.dtype)  // (T, 1)

        // embeddings[0] = embeddings[0] * (1 - mask) + voice * mask
        let original = embeddings[0]  // (T, dim)
        let replaced = original * (1.0 - mask3d) + voiceExpanded.asType(embeddings.dtype) * mask3d
        embeddings = MLX.expandedDimensions(replaced, axis: 0)  // (1, T, dim)

        return embeddings
    }

    // MARK: - Codebook Index Conversion

    /// Convert per-codebook codes to global embedding table indices.
    ///
    /// Layout: [semantic_cb (8194 entries)] [acoustic_cb_0 (23 entries)] [acoustic_cb_1 (23 entries)] ...
    /// Codes already include +2 special token offset from decodeOneFrame.
    /// Uses pre-computed codebookOffsets array (built once at init).
    ///
    /// Reference: voxtral_tts.py lines 621-644
    public func codesToGlobalIndices(_ codes: MLXArray) -> MLXArray {
        codes + codebookOffsets
    }

    // MARK: - Generation

    /// Generate speech from text with voice conditioning.
    ///
    /// Reference: voxtral_tts.py lines 446-586
    public func generate(
        text: String,
        voiceEmbedding: MLXArray,
        tokenizer: TekkenTokenizer,
        maxTokens: Int = 4096,
        onFrame: ((Int, MLXArray) -> Void)? = nil
    ) -> (codes: MLXArray, numFrames: Int) {
        let voiceFrameCount = voiceEmbedding.dim(0)

        // 1. Encode text to token IDs
        let inputIds = encodeText(text, voiceFrameCount: voiceFrameCount, tokenizer: tokenizer)
        let inputIdsMx = MLXArray(inputIds).reshaped(1, inputIds.count)
        // 2. Build input embeddings with voice replacement
        let inputEmbeddings = buildInputEmbeddings(inputIds: inputIdsMx, voiceEmbedding: voiceEmbedding)

        // 3. Create KV cache and prefill
        let cache = createCache()

        var hidden = llmForward(inputEmbeds: inputEmbeddings, cache: cache)
        MLX.eval(hidden)

        // 4. First decode step: inject AUDIO token to trigger first frame
        let audioTokenId = config.multimodal.audioModelArgs.audioTokenId
        let audioTokEmb = embedTokens(MLXArray([Int32(audioTokenId)]).reshaped(1, 1))
        hidden = llmForward(inputEmbeds: audioTokEmb, cache: cache)
        MLX.eval(hidden)

        // 5. Autoregressive generation
        var allCodes: [MLXArray] = []

        for i in 0..<maxTokens {
            let h = hidden[0..., -1, 0...]  // (1, dim) — last position

            // Generate one frame: semantic + acoustic codes
            let codes = acousticTransformer.decodeOneFrame(h)  // (1, 37)

            // EOA check: semantic_code <= 1 means empty_audio (0) or end_audio (1)
            let semanticCode = codes[0, 0].item(Int32.self)
            if semanticCode <= 1 {
                print("  [GEN] EOA at frame \(i)")
                break
            }

            allCodes.append(codes)
            onFrame?(i, codes)

            // Embed codes back as LLM input for next step
            let globalCodes = codesToGlobalIndices(codes)  // (1, 37)
            let codeEmbeddings = mmAudioEmbeddings.audioCodebookEmbeddings(globalCodes)  // (1, 37, dim)
            let nextEmbedding = codeEmbeddings.sum(axis: 1, keepDims: true)  // (1, 1, dim)

            // Feed through LLM
            hidden = llmForward(inputEmbeds: nextEmbedding, cache: cache)
            MLX.eval(hidden)

            if i % 50 == 0 {
                // Memory management handled externally
            }
        }

        guard !allCodes.isEmpty else {
            return (MLXArray([Int32]()), 0)
        }

        // Stack all codes: (1, N_frames, 37)
        let audioCodes = MLX.stacked(allCodes, axis: 1)
        return (audioCodes, allCodes.count)
    }

    /// Decode audio codes to waveform.
    public func decodeToWaveform(_ codes: MLXArray) -> MLXArray {
        let waveform = audioTokenizer.decode(codes)
        return waveform.squeezed(axis: 0)  // (samples,)
    }
}
