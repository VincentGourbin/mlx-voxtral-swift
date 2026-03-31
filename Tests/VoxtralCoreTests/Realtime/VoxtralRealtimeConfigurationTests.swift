/**
 * VoxtralRealtimeConfigurationTests - Tests for Realtime model configuration parsing
 */

import XCTest
@testable import VoxtralCore

final class VoxtralRealtimeConfigurationTests: XCTestCase {

    // MARK: - MLX Community Config (config.json)

    func testMLXCommunityConfigDecoding() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        XCTAssertEqual(config.decoder.dim, 3072)
        XCTAssertEqual(config.decoder.nLayers, 26)
        XCTAssertEqual(config.decoder.nHeads, 32)
        XCTAssertEqual(config.decoder.nKVHeads, 8)
        XCTAssertEqual(config.decoder.headDim, 128)
        XCTAssertEqual(config.decoder.hiddenDim, 9216)
        XCTAssertEqual(config.decoder.vocabSize, 131072)
        XCTAssertEqual(config.decoder.slidingWindow, 8192)
        XCTAssertTrue(config.decoder.tiedEmbeddings)
        XCTAssertTrue(config.decoder.adaRmsNormTCond)
        XCTAssertEqual(config.decoder.adaRmsNormTCondDim, 32)
    }

    func testEncoderConfigDecoding() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        XCTAssertEqual(config.encoderArgs.dim, 1280)
        XCTAssertEqual(config.encoderArgs.nLayers, 32)
        XCTAssertEqual(config.encoderArgs.nHeads, 32)
        XCTAssertEqual(config.encoderArgs.nKVHeads, 32)
        XCTAssertEqual(config.encoderArgs.headDim, 64)
        XCTAssertEqual(config.encoderArgs.hiddenDim, 5120)
        XCTAssertEqual(config.encoderArgs.slidingWindow, 750)
        XCTAssertEqual(config.encoderArgs.downsampleFactor, 4)
        XCTAssertTrue(config.encoderArgs.useBiases)
    }

    func testAudioEncodingConfigDecoding() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        let ae = config.audioEncoding
        XCTAssertEqual(ae.samplingRate, 16000)
        XCTAssertEqual(ae.frameRate, 12.5)
        XCTAssertEqual(ae.numMelBins, 128)
        XCTAssertEqual(ae.hopLength, 160)
        XCTAssertEqual(ae.windowSize, 400)
        XCTAssertEqual(ae.globalLogMelMax, 1.5)
        XCTAssertEqual(ae.transcriptionFormat, "streaming")
    }

    func testModelType() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)
        XCTAssertEqual(config.modelType, "voxtral_realtime")
    }

    // MARK: - Computed Properties

    func testStreamingConstants() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.frameRate, 12.5)
        XCTAssertEqual(config.rawAudioLengthPerToken, 1280)
        XCTAssertEqual(config.audioLengthPerToken, 8)
    }

    func testTokenIds() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        XCTAssertEqual(config.bosTokenId, 1)
        XCTAssertEqual(config.eosTokenId, 2)
        XCTAssertEqual(config.streamingPadTokenId, 11)
    }

    func testDelayTokenCalculation() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        // 480ms = 7680 samples → ~6 tokens at 12.5 Hz
        let nDelay = config.numDelayTokens(delayMs: 480)
        XCTAssertEqual(nDelay, 6)
    }

    func testDecoderLlamaConfig() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)

        let llama = config.decoderLlamaConfig
        XCTAssertEqual(llama.vocabSize, 131072)
        XCTAssertEqual(llama.hiddenSize, 3072)
        XCTAssertEqual(llama.intermediateSize, 9216)
        XCTAssertEqual(llama.numHiddenLayers, 26)
        XCTAssertEqual(llama.numAttentionHeads, 32)
        XCTAssertEqual(llama.numKeyValueHeads, 8)
        XCTAssertFalse(llama.attentionBias)
        XCTAssertFalse(llama.mlpBias)
    }

    // MARK: - Mistral params.json Loading

    func testMistralParamsJSONLoading() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_realtime_params.json")
        try mistralParamsJSON.data(using: .utf8)!.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let config = try VoxtralRealtimeConfiguration.load(from: tempURL)
        XCTAssertEqual(config.decoder.dim, 3072)
        XCTAssertEqual(config.decoder.nLayers, 26)
        XCTAssertEqual(config.encoderArgs.dim, 1280)
        XCTAssertEqual(config.encoderArgs.nLayers, 32)
        XCTAssertEqual(config.encoderArgs.downsampleFactor, 4)
    }

    // MARK: - Load from File

    func testLoadFromNonExistentFile() {
        let url = URL(fileURLWithPath: "/tmp/nonexistent_config.json")
        XCTAssertThrowsError(try VoxtralRealtimeConfiguration.load(from: url))
    }

    // MARK: - Codable Round Trip

    func testCodableRoundTrip() throws {
        let data = mlxCommunityConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data)
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: encoded)

        XCTAssertEqual(config.decoder.dim, decoded.decoder.dim)
        XCTAssertEqual(config.encoderArgs.nLayers, decoded.encoderArgs.nLayers)
        XCTAssertEqual(config.modelType, decoded.modelType)
    }

    // MARK: - Test Data

    private var mlxCommunityConfigJSON: String {
        """
        {
            "decoder": {
                "dim": 3072, "n_layers": 26, "head_dim": 128, "hidden_dim": 9216,
                "n_heads": 32, "n_kv_heads": 8, "vocab_size": 131072,
                "norm_eps": 1e-5, "rope_theta": 1000000.0, "sliding_window": 8192,
                "tied_embeddings": true, "ada_rms_norm_t_cond": true, "ada_rms_norm_t_cond_dim": 32
            },
            "encoder_args": {
                "audio_encoding_args": {
                    "sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 128,
                    "hop_length": 160, "window_size": 400, "global_log_mel_max": 1.5,
                    "transcription_format": "streaming"
                },
                "dim": 1280, "n_layers": 32, "head_dim": 64, "hidden_dim": 5120,
                "n_heads": 32, "n_kv_heads": 32, "use_biases": true,
                "rope_theta": 1000000.0, "norm_eps": 1e-5, "sliding_window": 750,
                "downsample_factor": 4
            },
            "model_type": "voxtral_realtime"
        }
        """
    }

    private var mistralParamsJSON: String {
        """
        {
            "dim": 3072, "n_layers": 26, "head_dim": 128, "hidden_dim": 9216,
            "n_heads": 32, "n_kv_heads": 8, "use_biases": false, "causal": true,
            "rope_theta": 1000000.0, "norm_eps": 1e-5, "vocab_size": 131072,
            "model_parallel": 1, "tied_embeddings": true, "sliding_window": 8192,
            "model_max_length": 131072,
            "multimodal": {
                "whisper_model_args": {
                    "encoder_args": {
                        "audio_encoding_args": {
                            "sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 128,
                            "hop_length": 160, "window_size": 400, "chunk_length_s": null,
                            "global_log_mel_max": 1.5, "transcription_format": "streaming"
                        },
                        "dim": 1280, "n_layers": 32, "head_dim": 64, "hidden_dim": 5120,
                        "n_heads": 32, "vocab_size": 131072, "n_kv_heads": 32,
                        "use_biases": true, "use_cache": false, "rope_theta": 1000000.0,
                        "causal": true, "norm_eps": 1e-5, "pos_embed": "rope",
                        "max_source_positions": null, "ffn_type": "swiglu",
                        "norm_type": "rms_norm", "sliding_window": 750
                    },
                    "downsample_args": { "downsample_factor": 4 }
                }
            },
            "ada_rms_norm_t_cond": true, "ada_rms_norm_t_cond_dim": 32
        }
        """
    }
}
