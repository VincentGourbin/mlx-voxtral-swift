/**
 * VoxtralTTSConfiguration - Configuration structs for Voxtral TTS model
 *
 * Parses the params.json format used by Voxtral-4B-TTS-2603.
 * Unlike the STT model which uses config.json (HuggingFace format),
 * the TTS model uses Mistral's native params.json format.
 */

import Foundation

// MARK: - Top-Level Configuration

/// Complete TTS model configuration parsed from params.json
struct VoxtralTTSConfiguration: Codable, Sendable {
    /// LLM backbone dimension
    let dim: Int
    /// Number of transformer layers
    let nLayers: Int
    /// Attention head dimension
    let headDim: Int
    /// MLP hidden dimension
    let hiddenDim: Int
    /// Number of attention heads
    let nHeads: Int
    /// Number of key-value heads (GQA)
    let nKVHeads: Int
    /// Whether to use biases in linear layers
    let useBiases: Bool
    /// RoPE theta for positional encoding
    let ropeTheta: Float
    /// Layer norm epsilon
    let normEps: Float
    /// Vocabulary size (text tokens)
    let vocabSize: Int
    /// Whether embeddings are tied (input/output share weights)
    let tiedEmbeddings: Bool
    /// Maximum sequence length
    let maxSeqLen: Int
    /// Maximum position embeddings
    let maxPositionEmbeddings: Int
    /// Model type identifier
    let modelType: String
    /// Multimodal (audio) configuration
    let multimodal: MultimodalConfiguration

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case useBiases = "use_biases"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case vocabSize = "vocab_size"
        case tiedEmbeddings = "tied_embeddings"
        case maxSeqLen = "max_seq_len"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
        case multimodal
    }
}

// MARK: - Multimodal Configuration

extension VoxtralTTSConfiguration {
    /// Container for audio model and tokenizer configurations
    struct MultimodalConfiguration: Codable, Sendable {
        /// BOS token ID
        let bosTokenId: Int
        /// Audio generation model configuration
        let audioModelArgs: AudioModelConfiguration
        /// Audio tokenizer (codec) configuration
        let audioTokenizerArgs: AudioTokenizerConfiguration

        enum CodingKeys: String, CodingKey {
            case bosTokenId = "bos_token_id"
            case audioModelArgs = "audio_model_args"
            case audioTokenizerArgs = "audio_tokenizer_args"
        }
    }
}

// MARK: - Audio Model Configuration

extension VoxtralTTSConfiguration {
    /// Configuration for the audio generation model (LLM + Flow Matching)
    struct AudioModelConfiguration: Codable, Sendable {
        /// Semantic codebook size (VQ vocabulary)
        let semanticCodebookSize: Int
        /// Acoustic codebook size (FSQ levels per dimension)
        let acousticCodebookSize: Int
        /// Number of acoustic codebook dimensions
        let nAcousticCodebook: Int
        /// Audio encoding parameters
        let audioEncodingArgs: AudioEncodingConfiguration
        /// Token ID for audio frames
        let audioTokenId: Int
        /// Token ID for beginning of audio
        let beginAudioTokenId: Int
        /// How to combine embeddings ("sum")
        let inputEmbeddingConcatType: String
        /// Flow matching transformer configuration
        let acousticTransformerArgs: FlowMatchingConfiguration
        /// Probability of unconditional generation during training
        let pUncond: Float
        /// Token ID used when condition is dropped (CFG unconditional)
        let conditionDroppedTokenId: Int

        enum CodingKeys: String, CodingKey {
            case semanticCodebookSize = "semantic_codebook_size"
            case acousticCodebookSize = "acoustic_codebook_size"
            case nAcousticCodebook = "n_acoustic_codebook"
            case audioEncodingArgs = "audio_encoding_args"
            case audioTokenId = "audio_token_id"
            case beginAudioTokenId = "begin_audio_token_id"
            case inputEmbeddingConcatType = "input_embedding_concat_type"
            case acousticTransformerArgs = "acoustic_transformer_args"
            case pUncond = "p_uncond"
            case conditionDroppedTokenId = "condition_dropped_token_id"
        }
    }
}

// MARK: - Audio Encoding Configuration

extension VoxtralTTSConfiguration {
    /// Parameters for audio token encoding/interleaving
    struct AudioEncodingConfiguration: Codable, Sendable {
        /// Codebook pattern type ("parallel")
        let codebookPattern: String
        /// Number of codebooks (semantic + acoustic = 1 + 36 = 37)
        let numCodebooks: Int
        /// Output audio sampling rate in Hz
        let samplingRate: Int
        /// Audio frame rate in Hz (tokens per second)
        let frameRate: Float

        enum CodingKeys: String, CodingKey {
            case codebookPattern = "codebook_pattern"
            case numCodebooks = "num_codebooks"
            case samplingRate = "sampling_rate"
            case frameRate = "frame_rate"
        }
    }
}

// MARK: - Flow Matching Configuration

extension VoxtralTTSConfiguration {
    /// Configuration for the 3-layer bidirectional flow matching transformer
    struct FlowMatchingConfiguration: Codable, Sendable {
        /// Input dimension from LLM hidden state
        let inputDim: Int
        /// Transformer hidden dimension
        let dim: Int
        /// Number of transformer layers
        let nLayers: Int
        /// Attention head dimension
        let headDim: Int
        /// MLP hidden dimension
        let hiddenDim: Int
        /// Number of attention heads
        let nHeads: Int
        /// Number of key-value heads
        let nKVHeads: Int
        /// Whether to use biases
        let useBiases: Bool
        /// RoPE theta
        let ropeTheta: Float
        /// Noise sigma (minimum noise scale)
        let sigma: Float
        /// Maximum noise scale
        let sigmaMax: Float

        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case dim
            case nLayers = "n_layers"
            case headDim = "head_dim"
            case hiddenDim = "hidden_dim"
            case nHeads = "n_heads"
            case nKVHeads = "n_kv_heads"
            case useBiases = "use_biases"
            case ropeTheta = "rope_theta"
            case sigma
            case sigmaMax = "sigma_max"
        }
    }
}

// MARK: - Audio Tokenizer (Codec) Configuration

extension VoxtralTTSConfiguration {
    /// Configuration for the Voxtral Codec decoder
    struct AudioTokenizerConfiguration: Codable, Sendable {
        /// Number of audio channels (1 = mono)
        let channels: Int
        /// Audio sampling rate
        let samplingRate: Int
        /// Waveform patch size (240 samples = 10ms at 24kHz)
        let pretransformPatchSize: Int
        /// Initial projection kernel size
        let patchProjKernelSize: Int
        /// Semantic codebook size
        let semanticCodebookSize: Int
        /// Semantic embedding dimension
        let semanticDim: Int
        /// Acoustic codebook size (FSQ levels)
        let acousticCodebookSize: Int
        /// Acoustic embedding dimension (= number of acoustic codebooks)
        let acousticDim: Int
        /// Whether to use weight normalization on convolutions
        let convWeightNorm: Bool
        /// Whether convolutions are causal
        let causal: Bool
        /// Sliding window size for attention
        let attnSlidingWindowSize: Int
        /// Whether to halve window size at each downsampling (encoder) / double at upsampling (decoder)
        let halfAttnWindowUponDownsampling: Bool
        /// Transformer hidden dimension
        let dim: Int
        /// MLP hidden dimension
        let hiddenDim: Int
        /// Attention head dimension
        let headDim: Int
        /// Number of attention heads
        let nHeads: Int
        /// Number of key-value heads
        let nKVHeads: Int
        /// QK normalization epsilon
        let qkNormEps: Float
        /// Whether to use QK normalization
        let qkNorm: Bool
        /// Whether to use biases
        let useBiases: Bool
        /// Layer norm epsilon
        let normEps: Float
        /// Whether to use LayerScale
        let layerScale: Bool
        /// LayerScale initial value
        let layerScaleInit: Float
        /// Number of transformer layers per decoder block (e.g., "2,2,2,2")
        let decoderTransformerLengthsStr: String
        /// Convolution kernel sizes per decoder block (e.g., "3,4,4,4")
        let decoderConvsKernelsStr: String
        /// Convolution strides per decoder block (e.g., "1,2,2,2")
        let decoderConvsStridesStr: String
        /// Voice preset name → index mapping
        let voice: [String: Int]

        enum CodingKeys: String, CodingKey {
            case channels
            case samplingRate = "sampling_rate"
            case pretransformPatchSize = "pretransform_patch_size"
            case patchProjKernelSize = "patch_proj_kernel_size"
            case semanticCodebookSize = "semantic_codebook_size"
            case semanticDim = "semantic_dim"
            case acousticCodebookSize = "acoustic_codebook_size"
            case acousticDim = "acoustic_dim"
            case convWeightNorm = "conv_weight_norm"
            case causal
            case attnSlidingWindowSize = "attn_sliding_window_size"
            case halfAttnWindowUponDownsampling = "half_attn_window_upon_downsampling"
            case dim
            case hiddenDim = "hidden_dim"
            case headDim = "head_dim"
            case nHeads = "n_heads"
            case nKVHeads = "n_kv_heads"
            case qkNormEps = "qk_norm_eps"
            case qkNorm = "qk_norm"
            case useBiases = "use_biases"
            case normEps = "norm_eps"
            case layerScale = "layer_scale"
            case layerScaleInit = "layer_scale_init"
            case decoderTransformerLengthsStr = "decoder_transformer_lengths_str"
            case decoderConvsKernelsStr = "decoder_convs_kernels_str"
            case decoderConvsStridesStr = "decoder_convs_strides_str"
            case voice
        }

        // MARK: - Computed Properties

        /// Parsed decoder transformer layer counts per block
        var decoderTransformerLengths: [Int] {
            decoderTransformerLengthsStr.split(separator: ",").compactMap { Int($0) }
        }

        /// Parsed decoder convolution kernel sizes per block
        var decoderConvsKernels: [Int] {
            decoderConvsKernelsStr.split(separator: ",").compactMap { Int($0) }
        }

        /// Parsed decoder convolution strides per block
        var decoderConvsStrides: [Int] {
            decoderConvsStridesStr.split(separator: ",").compactMap { Int($0) }
        }

        /// Total latent dimension (semantic + acoustic)
        var latentDim: Int { semanticDim + acousticDim }

        /// Number of decoder blocks
        var numDecoderBlocks: Int { decoderTransformerLengths.count }

        /// Total upsampling factor from codec frames to waveform patches
        var totalUpsamplingFactor: Int {
            decoderConvsStrides.reduce(1, *)
        }
    }
}

// MARK: - Convenience Accessors

extension VoxtralTTSConfiguration {
    /// Audio model configuration shortcut
    var audioModel: AudioModelConfiguration { multimodal.audioModelArgs }
    /// Audio tokenizer configuration shortcut
    var audioTokenizer: AudioTokenizerConfiguration { multimodal.audioTokenizerArgs }
    /// Flow matching configuration shortcut
    var flowMatching: FlowMatchingConfiguration { multimodal.audioModelArgs.acousticTransformerArgs }
    /// BOS token ID
    var bosTokenId: Int { multimodal.bosTokenId }

    /// Build a LlamaConfig compatible with the existing VoxtralLlama.swift
    var llamaConfig: LlamaConfig {
        LlamaConfig(
            vocabSize: vocabSize,
            hiddenSize: dim,
            intermediateSize: hiddenDim,
            numHiddenLayers: nLayers,
            numAttentionHeads: nHeads,
            numKeyValueHeads: nKVHeads,
            headDim: headDim,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeTheta: ropeTheta,
            ropeTraditional: true,  // Voxtral uses interleaved RoPE (not NeoX)
            rmsNormEps: normEps,
            attentionBias: useBiases,
            mlpBias: useBiases
        )
    }

    /// Load configuration from a params.json file
    static func load(from url: URL) throws -> VoxtralTTSConfiguration {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(VoxtralTTSConfiguration.self, from: data)
    }
}
