/**
 * VoxtralRealtimeConfiguration - Configuration for Voxtral-Mini-4B-Realtime-2602
 *
 * Supports two config formats:
 * - params.json (Mistral original): flat top-level with nested multimodal.whisper_model_args.encoder_args
 * - config.json (mlx-community): nested decoder/encoder_args at top level
 */

import Foundation

// MARK: - Audio Encoding Configuration

public struct RealtimeAudioEncodingConfig: Codable, Sendable {
    public let samplingRate: Int
    public let frameRate: Float
    public let numMelBins: Int
    public let hopLength: Int
    public let windowSize: Int
    public let globalLogMelMax: Float
    public let transcriptionFormat: String

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case frameRate = "frame_rate"
        case numMelBins = "num_mel_bins"
        case hopLength = "hop_length"
        case windowSize = "window_size"
        case globalLogMelMax = "global_log_mel_max"
        case transcriptionFormat = "transcription_format"
    }
}

// MARK: - Encoder Configuration

public struct RealtimeEncoderConfig: Codable, Sendable {
    public let dim: Int
    public let nLayers: Int
    public let headDim: Int
    public let hiddenDim: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let useBiases: Bool
    public let ropeTheta: Float
    public let normEps: Float
    public let slidingWindow: Int
    public let downsampleFactor: Int
    public let audioEncodingArgs: RealtimeAudioEncodingConfig

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
        case slidingWindow = "sliding_window"
        case downsampleFactor = "downsample_factor"
        case audioEncodingArgs = "audio_encoding_args"
    }
}

// MARK: - Decoder Configuration

public struct RealtimeDecoderConfig: Codable, Sendable {
    public let dim: Int
    public let nLayers: Int
    public let headDim: Int
    public let hiddenDim: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let vocabSize: Int
    public let normEps: Float
    public let ropeTheta: Float
    public let slidingWindow: Int
    public let tiedEmbeddings: Bool
    public let adaRmsNormTCond: Bool
    public let adaRmsNormTCondDim: Int

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case vocabSize = "vocab_size"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case tiedEmbeddings = "tied_embeddings"
        case adaRmsNormTCond = "ada_rms_norm_t_cond"
        case adaRmsNormTCondDim = "ada_rms_norm_t_cond_dim"
    }
}

// MARK: - Top-Level Configuration (mlx-community format)

public struct VoxtralRealtimeConfiguration: Codable, Sendable {
    public let decoder: RealtimeDecoderConfig
    public let encoderArgs: RealtimeEncoderConfig
    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case decoder
        case encoderArgs = "encoder_args"
        case modelType = "model_type"
    }
}

// MARK: - Convenience Accessors

public extension VoxtralRealtimeConfiguration {

    /// Audio encoding config shortcut
    var audioEncoding: RealtimeAudioEncodingConfig {
        encoderArgs.audioEncodingArgs
    }

    /// Build a LlamaConfig for the decoder (reuses LlamaAttention + LlamaMLP)
    var decoderLlamaConfig: LlamaConfig {
        LlamaConfig(
            vocabSize: decoder.vocabSize,
            hiddenSize: decoder.dim,
            intermediateSize: decoder.hiddenDim,
            numHiddenLayers: decoder.nLayers,
            numAttentionHeads: decoder.nHeads,
            numKeyValueHeads: decoder.nKVHeads,
            headDim: decoder.headDim,
            maxPositionEmbeddings: 131072,
            ropeTheta: decoder.ropeTheta,
            ropeTraditional: true,
            rmsNormEps: decoder.normEps,
            attentionBias: false,
            mlpBias: false
        )
    }

    /// Streaming constants
    var sampleRate: Int { audioEncoding.samplingRate }
    var frameRate: Float { audioEncoding.frameRate }
    var rawAudioLengthPerToken: Int { Int(Float(sampleRate) / frameRate) }  // 1280
    var audioLengthPerToken: Int { rawAudioLengthPerToken / audioEncoding.hopLength }  // 8

    /// BOS token ID (standard Mistral)
    var bosTokenId: Int { 1 }
    /// EOS token ID
    var eosTokenId: Int { 2 }
    /// Streaming pad token ID
    var streamingPadTokenId: Int { 11 }

    /// Default number of left pad tokens
    var nLeftPadTokens: Int { 1 }

    /// Compute number of delay tokens for a given delay in milliseconds
    func numDelayTokens(delayMs: Int) -> Int {
        let delaySamples = Int(Float(delayMs) / 1000.0 * Float(sampleRate))
        return numAudioTokens(audioLength: delaySamples)
    }

    /// Compute number of audio tokens for a given audio length in samples
    func numAudioTokens(audioLength: Int) -> Int {
        var frames = audioLength
        if frames % audioEncoding.hopLength != 0 {
            frames = Int(ceil(Double(frames) / Double(audioEncoding.hopLength) - 1))
        } else {
            frames = frames / audioEncoding.hopLength
        }
        return Int(ceil(Double(frames) / Double(audioLengthPerToken)))
    }

    /// Default transcription delay in ms
    static let defaultTranscriptionDelayMs: Int = 480
}

// MARK: - Loading

public extension VoxtralRealtimeConfiguration {

    /// Load from a config.json or params.json file
    static func load(from url: URL) throws -> VoxtralRealtimeConfiguration {
        let data = try Data(contentsOf: url)

        // Try mlx-community config.json format first (has "decoder" key)
        if let _ = try? JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data) {
            return try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)
        }

        // Fall back to Mistral params.json format
        return try loadFromParamsJSON(data: data)
    }

    /// Parse Mistral params.json format into our config structure
    private static func loadFromParamsJSON(data: Data) throws -> VoxtralRealtimeConfiguration {
        let params = try JSONDecoder().decode(MistralRealtimeParams.self, from: data)

        let encoderArgs = params.multimodal.whisperModelArgs.encoderArgs
        let downsampleFactor = params.multimodal.whisperModelArgs.downsampleArgs.downsampleFactor

        let encoderConfig = RealtimeEncoderConfig(
            dim: encoderArgs.dim,
            nLayers: encoderArgs.nLayers,
            headDim: encoderArgs.headDim,
            hiddenDim: encoderArgs.hiddenDim,
            nHeads: encoderArgs.nHeads,
            nKVHeads: encoderArgs.nKVHeads,
            useBiases: encoderArgs.useBiases,
            ropeTheta: encoderArgs.ropeTheta,
            normEps: encoderArgs.normEps,
            slidingWindow: encoderArgs.slidingWindow,
            downsampleFactor: downsampleFactor,
            audioEncodingArgs: encoderArgs.audioEncodingArgs
        )

        let decoderConfig = RealtimeDecoderConfig(
            dim: params.dim,
            nLayers: params.nLayers,
            headDim: params.headDim,
            hiddenDim: params.hiddenDim,
            nHeads: params.nHeads,
            nKVHeads: params.nKVHeads,
            vocabSize: params.vocabSize,
            normEps: params.normEps,
            ropeTheta: params.ropeTheta,
            slidingWindow: params.slidingWindow,
            tiedEmbeddings: params.tiedEmbeddings,
            adaRmsNormTCond: params.adaRmsNormTCond,
            adaRmsNormTCondDim: params.adaRmsNormTCondDim
        )

        return VoxtralRealtimeConfiguration(
            decoder: decoderConfig,
            encoderArgs: encoderConfig,
            modelType: "voxtral_realtime"
        )
    }
}

// MARK: - Mistral params.json Internal Types

private struct MistralRealtimeParams: Codable {
    let dim: Int
    let nLayers: Int
    let headDim: Int
    let hiddenDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let vocabSize: Int
    let normEps: Float
    let ropeTheta: Float
    let slidingWindow: Int
    let tiedEmbeddings: Bool
    let adaRmsNormTCond: Bool
    let adaRmsNormTCondDim: Int
    let multimodal: MistralMultimodal

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case vocabSize = "vocab_size"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case tiedEmbeddings = "tied_embeddings"
        case adaRmsNormTCond = "ada_rms_norm_t_cond"
        case adaRmsNormTCondDim = "ada_rms_norm_t_cond_dim"
        case multimodal
    }
}

private struct MistralMultimodal: Codable {
    let whisperModelArgs: MistralWhisperModelArgs

    enum CodingKeys: String, CodingKey {
        case whisperModelArgs = "whisper_model_args"
    }
}

private struct MistralWhisperModelArgs: Codable {
    let encoderArgs: MistralEncoderArgs
    let downsampleArgs: MistralDownsampleArgs

    enum CodingKeys: String, CodingKey {
        case encoderArgs = "encoder_args"
        case downsampleArgs = "downsample_args"
    }
}

private struct MistralEncoderArgs: Codable {
    let dim: Int
    let nLayers: Int
    let headDim: Int
    let hiddenDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let useBiases: Bool
    let ropeTheta: Float
    let normEps: Float
    let slidingWindow: Int
    let audioEncodingArgs: RealtimeAudioEncodingConfig

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
        case slidingWindow = "sliding_window"
        case audioEncodingArgs = "audio_encoding_args"
    }
}

private struct MistralDownsampleArgs: Codable {
    let downsampleFactor: Int

    enum CodingKeys: String, CodingKey {
        case downsampleFactor = "downsample_factor"
    }
}
