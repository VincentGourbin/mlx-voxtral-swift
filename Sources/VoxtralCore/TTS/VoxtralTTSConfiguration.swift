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
public struct VoxtralTTSConfiguration: Codable, Sendable {
    /// LLM backbone dimension
    public let dim: Int
    /// Number of transformer layers
    public let nLayers: Int
    /// Attention head dimension
    public let headDim: Int
    /// MLP hidden dimension
    public let hiddenDim: Int
    /// Number of attention heads
    public let nHeads: Int
    /// Number of key-value heads (GQA)
    public let nKVHeads: Int
    /// Whether to use biases in linear layers
    public let useBiases: Bool
    /// RoPE theta for positional encoding
    public let ropeTheta: Float
    /// Layer norm epsilon
    public let normEps: Float
    /// Vocabulary size (text tokens)
    public let vocabSize: Int
    /// Whether embeddings are tied (input/output share weights)
    public let tiedEmbeddings: Bool
    /// Maximum sequence length
    public let maxSeqLen: Int
    /// Maximum position embeddings
    public let maxPositionEmbeddings: Int
    /// Model type identifier
    public let modelType: String
    /// Multimodal (audio) configuration
    public let multimodal: MultimodalConfiguration

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

public extension VoxtralTTSConfiguration {
    /// Container for audio model and tokenizer configurations
    struct MultimodalConfiguration: Codable, Sendable {
        /// BOS token ID
        public let bosTokenId: Int
        /// Audio generation model configuration
        public let audioModelArgs: AudioModelConfiguration
        /// Audio tokenizer (codec) configuration
        public let audioTokenizerArgs: AudioTokenizerConfiguration

        enum CodingKeys: String, CodingKey {
            case bosTokenId = "bos_token_id"
            case audioModelArgs = "audio_model_args"
            case audioTokenizerArgs = "audio_tokenizer_args"
        }
    }
}

// MARK: - Audio Model Configuration

public extension VoxtralTTSConfiguration {
    /// Configuration for the audio generation model (LLM + Flow Matching)
    struct AudioModelConfiguration: Codable, Sendable {
        /// Semantic codebook size (VQ vocabulary)
        public let semanticCodebookSize: Int
        /// Acoustic codebook size (FSQ levels per dimension)
        public let acousticCodebookSize: Int
        /// Number of acoustic codebook dimensions
        public let nAcousticCodebook: Int
        /// Audio encoding parameters
        public let audioEncodingArgs: AudioEncodingConfiguration
        /// Token ID for audio frames
        public let audioTokenId: Int
        /// Token ID for beginning of audio
        public let beginAudioTokenId: Int
        /// How to combine embeddings ("sum")
        public let inputEmbeddingConcatType: String
        /// Flow matching transformer configuration
        public let acousticTransformerArgs: FlowMatchingConfiguration
        /// Probability of unconditional generation during training
        public let pUncond: Float
        /// Token ID used when condition is dropped (CFG unconditional)
        public let conditionDroppedTokenId: Int

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

public extension VoxtralTTSConfiguration {
    /// Parameters for audio token encoding/interleaving
    struct AudioEncodingConfiguration: Codable, Sendable {
        /// Codebook pattern type ("parallel")
        public let codebookPattern: String
        /// Number of codebooks (semantic + acoustic = 1 + 36 = 37)
        public let numCodebooks: Int
        /// Output audio sampling rate in Hz
        public let samplingRate: Int
        /// Audio frame rate in Hz (tokens per second)
        public let frameRate: Float

        enum CodingKeys: String, CodingKey {
            case codebookPattern = "codebook_pattern"
            case numCodebooks = "num_codebooks"
            case samplingRate = "sampling_rate"
            case frameRate = "frame_rate"
        }
    }
}

// MARK: - Flow Matching Configuration

public extension VoxtralTTSConfiguration {
    /// Configuration for the 3-layer bidirectional flow matching transformer
    struct FlowMatchingConfiguration: Codable, Sendable {
        /// Input dimension from LLM hidden state
        public let inputDim: Int
        /// Transformer hidden dimension
        public let dim: Int
        /// Number of transformer layers
        public let nLayers: Int
        /// Attention head dimension
        public let headDim: Int
        /// MLP hidden dimension
        public let hiddenDim: Int
        /// Number of attention heads
        public let nHeads: Int
        /// Number of key-value heads
        public let nKVHeads: Int
        /// Whether to use biases
        public let useBiases: Bool
        /// RoPE theta
        public let ropeTheta: Float
        /// Noise sigma (minimum noise scale)
        public let sigma: Float
        /// Maximum noise scale
        public let sigmaMax: Float

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

public extension VoxtralTTSConfiguration {
    /// Configuration for the Voxtral Codec decoder
    struct AudioTokenizerConfiguration: Codable, Sendable {
        /// Number of audio channels (1 = mono)
        public let channels: Int
        /// Audio sampling rate
        public let samplingRate: Int
        /// Waveform patch size (240 samples = 10ms at 24kHz)
        public let pretransformPatchSize: Int
        /// Initial projection kernel size
        public let patchProjKernelSize: Int
        /// Semantic codebook size
        public let semanticCodebookSize: Int
        /// Semantic embedding dimension
        public let semanticDim: Int
        /// Acoustic codebook size (FSQ levels)
        public let acousticCodebookSize: Int
        /// Acoustic embedding dimension (= number of acoustic codebooks)
        public let acousticDim: Int
        /// Whether to use weight normalization on convolutions
        public let convWeightNorm: Bool
        /// Whether convolutions are causal
        public let causal: Bool
        /// Sliding window size for attention
        public let attnSlidingWindowSize: Int
        /// Whether to halve window size at each downsampling (encoder) / double at upsampling (decoder)
        public let halfAttnWindowUponDownsampling: Bool
        /// Transformer hidden dimension
        public let dim: Int
        /// MLP hidden dimension
        public let hiddenDim: Int
        /// Attention head dimension
        public let headDim: Int
        /// Number of attention heads
        public let nHeads: Int
        /// Number of key-value heads
        public let nKVHeads: Int
        /// QK normalization epsilon
        public let qkNormEps: Float
        /// Whether to use QK normalization
        public let qkNorm: Bool
        /// Whether to use biases
        public let useBiases: Bool
        /// Layer norm epsilon
        public let normEps: Float
        /// Whether to use LayerScale
        public let layerScale: Bool
        /// LayerScale initial value
        public let layerScaleInit: Float
        /// Number of transformer layers per decoder block (e.g., "2,2,2,2")
        public let decoderTransformerLengthsStr: String
        /// Convolution kernel sizes per decoder block (e.g., "3,4,4,4")
        public let decoderConvsKernelsStr: String
        /// Convolution strides per decoder block (e.g., "1,2,2,2")
        public let decoderConvsStridesStr: String
        /// Voice preset name → index mapping
        public let voice: [String: Int]

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
        public var decoderTransformerLengths: [Int] {
            decoderTransformerLengthsStr.split(separator: ",").compactMap { Int($0) }
        }

        /// Parsed decoder convolution kernel sizes per block
        public var decoderConvsKernels: [Int] {
            decoderConvsKernelsStr.split(separator: ",").compactMap { Int($0) }
        }

        /// Parsed decoder convolution strides per block
        public var decoderConvsStrides: [Int] {
            decoderConvsStridesStr.split(separator: ",").compactMap { Int($0) }
        }

        /// Total latent dimension (semantic + acoustic)
        public var latentDim: Int { semanticDim + acousticDim }

        /// Number of decoder blocks
        public var numDecoderBlocks: Int { decoderTransformerLengths.count }

        /// Total upsampling factor from codec frames to waveform patches
        public var totalUpsamplingFactor: Int {
            decoderConvsStrides.reduce(1, *)
        }
    }
}

// MARK: - Convenience Accessors

public extension VoxtralTTSConfiguration {
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
