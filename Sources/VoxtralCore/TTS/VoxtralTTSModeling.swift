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
import MLXProfiler

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

// MARK: - Text Sanitization

/// Normalize text for TTS input, matching Python reference (sanitize_tts_input_text_for_demo).
private let terminalPunctuation: Set<Character> = [".", "!", "?", ":", ";", "\u{2026}"]

// MARK: - VoxtralTTSModel

/// The complete Voxtral TTS model.
public class VoxtralTTSModel: Module {

    /// Sanitize text for TTS with prosody-aware structural handling.
    ///
    /// Phase 1 converts structural formatting (paragraphs, headers, bullets) into
    /// punctuation that produces natural pauses. Phase 2 normalizes characters.
    /// Voxtral TTS infers all prosody from punctuation — there are no special tokens.
    public static func sanitizeTextForTTS(_ text: String) -> String {
        var t = text

        // === Phase 1: Structural transformations (multi-line, before whitespace collapse) ===

        // 1. Strip markdown formatting artifacts
        t = stripMarkdownFormatting(t)

        // 2. Convert bullet points to standalone sentences
        t = convertBulletPoints(t)

        // 3. Convert ALL-CAPS section headers to announced topics
        t = convertSectionHeaders(t)

        // 4. Convert paragraph breaks (double newlines) to sentence boundaries
        t = convertParagraphBreaks(t)

        // 5. Convert remaining single newlines to spaces
        t = t.replacingOccurrences(of: "\\n", with: " ", options: .regularExpression)

        // === Phase 2: Character-level sanitization (single-line) ===

        // 6. Collapse whitespace
        t = t.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)

        // 7. Convert ALL-CAPS words to Capitalized
        t = convertAllCapsWords(t)

        // 8. Verbalize symbols
        t = t.replacingOccurrences(of: "&", with: " and ")
        t = t.replacingOccurrences(of: "=", with: " equals ")
        t = t.replacingOccurrences(of: "+", with: " plus ")
        t = t.replacingOccurrences(of: " / ", with: " or ")

        // 9. Normalize dashes
        t = t.replacingOccurrences(of: "\\s*[\u{2014}\u{2015}\u{2012}\u{2013}]\\s*", with: ", ", options: .regularExpression)
        t = t.replacingOccurrences(of: " - ", with: ", ")
        t = t.replacingOccurrences(of: "[\u{2010}\u{2011}\u{FE58}\u{FE63}\u{FF0D}]", with: "-", options: .regularExpression)

        // 10. Collapse repeated/artifact punctuation
        t = t.replacingOccurrences(of: "([.!?;:])\\1+", with: "$1", options: .regularExpression)
        t = t.replacingOccurrences(of: ",\\s*,", with: ",")
        t = t.replacingOccurrences(of: "\\.\\s*\\.", with: ".", options: .regularExpression)

        // 11. Final whitespace collapse
        t = t.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)

        // 12. Ensure terminal punctuation (critical for EOA detection)
        if let last = t.last, !terminalPunctuation.contains(last) {
            t += "."
        }
        if t.isEmpty { t = "." }
        return t
    }

    // MARK: - Phase 1 Helpers

    /// Strip markdown formatting: # headers, **bold**, *italic*, > blockquotes.
    private static func stripMarkdownFormatting(_ text: String) -> String {
        var t = text
        t = t.replacingOccurrences(of: "(?m)^#{1,6}\\s+", with: "", options: .regularExpression)
        t = t.replacingOccurrences(of: "\\*{1,2}([^*]+)\\*{1,2}", with: "$1", options: .regularExpression)
        t = t.replacingOccurrences(of: "_{1,2}([^_]+)_{1,2}", with: "$1", options: .regularExpression)
        t = t.replacingOccurrences(of: "(?m)^>\\s*", with: "", options: .regularExpression)
        return t
    }

    /// Convert bullet point lines to standalone sentences.
    private static func convertBulletPoints(_ text: String) -> String {
        let lines = text.components(separatedBy: "\n")
        let bulletPattern = try! NSRegularExpression(pattern: "^\\s*(?:[-*+•]|\\d+[.)]) +")
        let result = lines.map { line -> String in
            let range = NSRange(location: 0, length: line.utf16.count)
            if let match = bulletPattern.firstMatch(in: line, range: range) {
                let afterBullet = String(line[line.index(line.startIndex, offsetBy: match.range.upperBound)...])
                    .trimmingCharacters(in: .whitespaces)
                guard !afterBullet.isEmpty else { return line }
                if let last = afterBullet.last, terminalPunctuation.contains(last) {
                    return afterBullet
                }
                return afterBullet + "."
            }
            return line
        }
        return result.joined(separator: "\n")
    }

    /// Convert ALL-CAPS section header lines to announced topics with sentence boundary.
    private static func convertSectionHeaders(_ text: String) -> String {
        let lines = text.components(separatedBy: "\n")
        var result: [String] = []

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else {
                result.append(line)
                continue
            }

            let alphas = trimmed.unicodeScalars.filter { CharacterSet.letters.contains($0) }
            let isAllCaps = alphas.count >= 2
                && trimmed == trimmed.uppercased()
                && trimmed != trimmed.lowercased()
            let isShortLine = trimmed.count <= 80

            if isAllCaps && isShortLine {
                // Ensure previous non-empty line ends with terminal punctuation
                if let lastIdx = result.lastIndex(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) {
                    var prev = result[lastIdx].trimmingCharacters(in: .whitespaces)
                    if let lastChar = prev.last, !terminalPunctuation.contains(lastChar) {
                        prev += "."
                        result[lastIdx] = prev
                    }
                }
                // Header as its own sentence
                var header = trimmed
                if let last = header.last, !terminalPunctuation.contains(last) {
                    header += "."
                }
                result.append(header)
            } else {
                result.append(line)
            }
        }
        return result.joined(separator: "\n")
    }

    /// Convert paragraph breaks (double newlines) to sentence boundaries.
    private static func convertParagraphBreaks(_ text: String) -> String {
        var t = text
        // Normalize multiple blank lines to double newline
        t = t.replacingOccurrences(of: "\\n\\s*\\n+", with: "\n\n", options: .regularExpression)
        // Paragraph break after terminal punctuation → just a space
        t = t.replacingOccurrences(of: "([.!?;:\u{2026}])\\s*\\n\\n", with: "$1 ", options: .regularExpression)
        // Paragraph break without terminal punctuation → add period + space
        t = t.replacingOccurrences(of: "\\n\\n", with: ". ", options: .regularExpression)
        return t
    }

    // MARK: - Phase 2 Helpers

    /// Convert ALL-CAPS words (2+ letters) to Capitalized.
    private static func convertAllCapsWords(_ text: String) -> String {
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        let converted = words.map { word -> String in
            let s = String(word)
            let alphas = s.unicodeScalars.filter { CharacterSet.letters.contains($0) }
            if alphas.count >= 2 && s == s.uppercased() && s != s.lowercased() {
                return s.capitalized
            }
            return s
        }
        return converted.joined(separator: " ")
    }

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
    public func encodeText(_ text: String, voiceFrameCount: Int, tokenizer: TekkenTokenizer, sanitize: Bool = true) -> [Int32] {
        let processedText = sanitize ? Self.sanitizeTextForTTS(text) : text
        let textTokens = tokenizer.encode(processedText).map { Int32($0) }

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
        sanitize: Bool = true,
        onFrame: ((Int, MLXArray) -> Void)? = nil
    ) -> (codes: MLXArray, numFrames: Int, ttft: TimeInterval) {
        let genStart = Date()
        let voiceFrameCount = voiceEmbedding.dim(0)
        let session = MLXProfiler.shared.activeSession

        // 1. Encode text to token IDs
        session?.beginPhase("Text Tokenization", category: .tokenization)
        let inputIds = encodeText(text, voiceFrameCount: voiceFrameCount, tokenizer: tokenizer, sanitize: sanitize)
        let inputIdsMx = MLXArray(inputIds).reshaped(1, inputIds.count)
        session?.endPhase("Text Tokenization", category: .tokenization)

        // 2. Build input embeddings with voice replacement
        session?.beginPhase("Voice Embedding Merge", category: .voiceEmbedding)
        let inputEmbeddings = buildInputEmbeddings(inputIds: inputIdsMx, voiceEmbedding: voiceEmbedding)
        session?.endPhase("Voice Embedding Merge", category: .voiceEmbedding)

        // 3. Create KV cache and prefill
        session?.beginPhase("Prefill", category: .prefill)
        let cache = createCache()

        var hidden = llmForward(inputEmbeds: inputEmbeddings, cache: cache)
        MLX.eval(hidden)

        // 4. First decode step: inject AUDIO token to trigger first frame
        let audioTokenId = config.multimodal.audioModelArgs.audioTokenId
        let audioTokEmb = embedTokens(MLXArray([Int32(audioTokenId)]).reshaped(1, 1))
        hidden = llmForward(inputEmbeds: audioTokEmb, cache: cache)
        MLX.eval(hidden)
        session?.endPhase("Prefill", category: .prefill)

        // 5. Autoregressive generation
        // Optimization: check EOA every N frames to reduce GPU→CPU syncs.
        // Between checks, GPU pipelines LLM forward passes without blocking.
        // May generate up to (eoaCheckInterval-1) extra frames after EOA — trimmed below.
        let eoaCheckInterval = 4
        var allCodes: [MLXArray] = []
        var ttft: TimeInterval = 0
        var eoaReached = false

        for i in 0..<maxTokens {
            let stepStart = CFAbsoluteTimeGetCurrent()
            let h = hidden[0..., -1, 0...]  // (1, dim) — last position

            // Generate one frame: semantic + acoustic codes
            let codes = acousticTransformer.decodeOneFrame(h)  // (1, 37)
            if i == 0 {
                MLX.eval(codes)
                ttft = Date().timeIntervalSince(genStart)
            }

            allCodes.append(codes)
            onFrame?(i, codes)

            // Record per-frame step timing
            let stepDurationUs = UInt64((CFAbsoluteTimeGetCurrent() - stepStart) * 1_000_000)
            session?.recordStep(index: i + 1, total: maxTokens, durationUs: stepDurationUs, category: .semanticCodeGen)

            // Periodic EOA check — sync GPU only every N frames
            if (i + 1) % eoaCheckInterval == 0 || i == 0 {
                // Check the last eoaCheckInterval frames for EOA
                let checkStart = max(0, allCodes.count - eoaCheckInterval)
                for j in checkStart..<allCodes.count {
                    let semanticCode = allCodes[j][0, 0].item(Int32.self)
                    if semanticCode <= 1 {
                        // Trim codes up to (but not including) the EOA frame
                        allCodes = Array(allCodes.prefix(j))
                        print("  [GEN] EOA at frame \(j)")
                        eoaReached = true
                        break
                    }
                }
                if eoaReached { break }
            }

            // Embed codes back as LLM input for next step
            let globalCodes = codesToGlobalIndices(codes)  // (1, 37)
            let codeEmbeddings = mmAudioEmbeddings.audioCodebookEmbeddings(globalCodes)  // (1, 37, dim)
            let nextEmbedding = codeEmbeddings.sum(axis: 1, keepDims: true)  // (1, 1, dim)

            // Feed through LLM — no eval() sync between EOA checks
            hidden = llmForward(inputEmbeds: nextEmbedding, cache: cache)

            // Sync GPU periodically to prevent unbounded compute graph growth
            if (i + 1) % eoaCheckInterval == 0 {
                MLX.eval(hidden)
            }
        }

        guard !allCodes.isEmpty else {
            return (MLXArray([Int32]()), 0, ttft)
        }

        // Stack all codes: (1, N_frames, 37)
        let audioCodes = MLX.stacked(allCodes, axis: 1)
        return (audioCodes, allCodes.count, ttft)
    }

    /// Decode audio codes to waveform.
    public func decodeToWaveform(_ codes: MLXArray) -> MLXArray {
        let waveform = audioTokenizer.decode(codes)
        return waveform.squeezed(axis: 0)  // (samples,)
    }

    // MARK: - Streaming Generation

    /// Streaming chunk yielded during generation.
    public struct GenerationChunk: @unchecked Sendable {
        /// All accumulated codes so far: (1, totalFrames, 37)
        public let accumulatedCodes: MLXArray
        /// Number of new frames in this chunk
        public let newFrameCount: Int
        /// Total frames generated so far
        public let totalFrames: Int
        /// Whether this is the final chunk
        public let isFinal: Bool
    }

    /// Generate speech codes as a stream, yielding chunks of accumulated codes every `chunkSize` frames.
    ///
    /// The caller can decode each chunk incrementally using `decodeToWaveform`.
    public func generateStreaming(
        text: String,
        voiceEmbedding: MLXArray,
        tokenizer: TekkenTokenizer,
        maxTokens: Int = 4096,
        chunkSize: Int = 10,
        sanitize: Bool = true
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            let voiceFrameCount = voiceEmbedding.dim(0)

            // 1. Encode text to token IDs
            let inputIds = encodeText(text, voiceFrameCount: voiceFrameCount, tokenizer: tokenizer, sanitize: sanitize)
            let inputIdsMx = MLXArray(inputIds).reshaped(1, inputIds.count)

            // 2. Build input embeddings with voice replacement
            let inputEmbeddings = buildInputEmbeddings(inputIds: inputIdsMx, voiceEmbedding: voiceEmbedding)

            // 3. Create KV cache and prefill
            let cache = createCache()
            var hidden = llmForward(inputEmbeds: inputEmbeddings, cache: cache)
            MLX.eval(hidden)

            // 4. First decode step: inject AUDIO token
            let audioTokenId = config.multimodal.audioModelArgs.audioTokenId
            let audioTokEmb = embedTokens(MLXArray([Int32(audioTokenId)]).reshaped(1, 1))
            hidden = llmForward(inputEmbeds: audioTokEmb, cache: cache)
            MLX.eval(hidden)

            // 5. Autoregressive generation with streaming
            var allCodes: [MLXArray] = []
            var chunkFrameCount = 0

            for i in 0..<maxTokens {
                // Check for cancellation
                if Task.isCancelled {
                    continuation.finish()
                    return
                }

                let h = hidden[0..., -1, 0...]
                let codes = acousticTransformer.decodeOneFrame(h)

                // EOA check
                let semanticCode = codes[0, 0].item(Int32.self)
                if semanticCode <= 1 {
                    print("  [GEN] EOA at frame \(i)")
                    // Yield final chunk if there are pending frames
                    if !allCodes.isEmpty {
                        let audioCodes = MLX.stacked(allCodes, axis: 1)
                        continuation.yield(GenerationChunk(
                            accumulatedCodes: audioCodes,
                            newFrameCount: chunkFrameCount,
                            totalFrames: allCodes.count,
                            isFinal: true
                        ))
                    }
                    continuation.finish()
                    return
                }

                allCodes.append(codes)
                chunkFrameCount += 1

                // Yield chunk every chunkSize frames
                if chunkFrameCount >= chunkSize {
                    let audioCodes = MLX.stacked(allCodes, axis: 1)
                    continuation.yield(GenerationChunk(
                        accumulatedCodes: audioCodes,
                        newFrameCount: chunkFrameCount,
                        totalFrames: allCodes.count,
                        isFinal: false
                    ))
                    chunkFrameCount = 0
                }

                // Embed codes back as LLM input for next step
                let globalCodes = codesToGlobalIndices(codes)
                let codeEmbeddings = mmAudioEmbeddings.audioCodebookEmbeddings(globalCodes)
                let nextEmbedding = codeEmbeddings.sum(axis: 1, keepDims: true)
                hidden = llmForward(inputEmbeds: nextEmbedding, cache: cache)
                MLX.eval(hidden)
            }

            // Max tokens reached — yield final
            if !allCodes.isEmpty {
                let audioCodes = MLX.stacked(allCodes, axis: 1)
                continuation.yield(GenerationChunk(
                    accumulatedCodes: audioCodes,
                    newFrameCount: chunkFrameCount,
                    totalFrames: allCodes.count,
                    isFinal: true
                ))
            }
            continuation.finish()
        }
    }
}
