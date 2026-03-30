/**
 * VoxtralTTSConfigurationTests - Unit tests for TTS configuration parsing
 */

import XCTest
@testable import VoxtralCore

final class VoxtralTTSConfigurationTests: XCTestCase {

    // MARK: - JSON Decoding Tests

    func testFullConfigurationDecoding() throws {
        let json = sampleParamsJSON
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        XCTAssertEqual(config.dim, 3072)
        XCTAssertEqual(config.nLayers, 28)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.hiddenDim, 8192)
        XCTAssertEqual(config.nHeads, 24)
        XCTAssertEqual(config.nKVHeads, 8)
        XCTAssertEqual(config.useBiases, false)
        XCTAssertEqual(config.ropeTheta, 1000000.0)
        XCTAssertEqual(config.normEps, 1e-5)
        XCTAssertEqual(config.vocabSize, 131072)
        XCTAssertEqual(config.tiedEmbeddings, false)
        XCTAssertEqual(config.modelType, "voxtral_tts")
    }

    func testMultimodalConfigurationDecoding() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        XCTAssertEqual(config.multimodal.bosTokenId, 1)
    }

    func testAudioModelConfigurationDecoding() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        let audioModel = config.audioModel
        XCTAssertEqual(audioModel.semanticCodebookSize, 2048)
        XCTAssertEqual(audioModel.acousticCodebookSize, 262144)
        XCTAssertEqual(audioModel.nAcousticCodebook, 36)
        XCTAssertEqual(audioModel.audioTokenId, 10)
        XCTAssertEqual(audioModel.beginAudioTokenId, 9)
        XCTAssertEqual(audioModel.inputEmbeddingConcatType, "sum")
        XCTAssertEqual(audioModel.pUncond, 0.2)
        XCTAssertEqual(audioModel.conditionDroppedTokenId, 0)
    }

    func testAudioEncodingConfigurationDecoding() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        let encoding = config.audioModel.audioEncodingArgs
        XCTAssertEqual(encoding.codebookPattern, "parallel")
        XCTAssertEqual(encoding.numCodebooks, 37)
        XCTAssertEqual(encoding.samplingRate, 24000)
        XCTAssertEqual(encoding.frameRate, 12.5)
    }

    func testFlowMatchingConfigurationDecoding() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        let fm = config.flowMatching
        XCTAssertEqual(fm.inputDim, 3072)
        XCTAssertEqual(fm.dim, 1024)
        XCTAssertEqual(fm.nLayers, 3)
        XCTAssertEqual(fm.headDim, 64)
        XCTAssertEqual(fm.hiddenDim, 4096)
        XCTAssertEqual(fm.nHeads, 16)
        XCTAssertEqual(fm.nKVHeads, 4)
        XCTAssertEqual(fm.useBiases, false)
        XCTAssertEqual(fm.sigma, 1e-5)
        XCTAssertEqual(fm.sigmaMax, 2.0)
    }

    func testAudioTokenizerConfigurationDecoding() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        let tokenizer = config.audioTokenizer
        XCTAssertEqual(tokenizer.channels, 1)
        XCTAssertEqual(tokenizer.samplingRate, 24000)
        XCTAssertEqual(tokenizer.pretransformPatchSize, 240)
        XCTAssertEqual(tokenizer.semanticCodebookSize, 2048)
        XCTAssertEqual(tokenizer.semanticDim, 256)
        XCTAssertEqual(tokenizer.acousticCodebookSize, 262144)
        XCTAssertEqual(tokenizer.acousticDim, 36)
        XCTAssertEqual(tokenizer.convWeightNorm, true)
        XCTAssertEqual(tokenizer.causal, true)
        XCTAssertEqual(tokenizer.dim, 512)
        XCTAssertEqual(tokenizer.nHeads, 8)
        XCTAssertEqual(tokenizer.nKVHeads, 2)
        XCTAssertEqual(tokenizer.qkNorm, true)
        XCTAssertEqual(tokenizer.layerScale, true)
    }

    // MARK: - Computed Properties Tests

    func testAudioTokenizerComputedProperties() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        let tokenizer = config.audioTokenizer
        XCTAssertEqual(tokenizer.decoderTransformerLengths, [2, 2, 2, 2])
        XCTAssertEqual(tokenizer.decoderConvsKernels, [3, 4, 4, 4])
        XCTAssertEqual(tokenizer.decoderConvsStrides, [1, 2, 2, 2])
        XCTAssertEqual(tokenizer.latentDim, 256 + 36) // semantic + acoustic
        XCTAssertEqual(tokenizer.numDecoderBlocks, 4)
        XCTAssertEqual(tokenizer.totalUpsamplingFactor, 1 * 2 * 2 * 2) // 8
    }

    // MARK: - Convenience Accessors Tests

    func testConvenienceAccessors() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        XCTAssertEqual(config.bosTokenId, 1)
        XCTAssertEqual(config.audioModel.semanticCodebookSize, config.multimodal.audioModelArgs.semanticCodebookSize)
        XCTAssertEqual(config.audioTokenizer.samplingRate, config.multimodal.audioTokenizerArgs.samplingRate)
        XCTAssertEqual(config.flowMatching.nLayers, config.multimodal.audioModelArgs.acousticTransformerArgs.nLayers)
    }

    func testLlamaConfigGeneration() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        let llamaConfig = config.llamaConfig
        XCTAssertEqual(llamaConfig.vocabSize, 131072)
        XCTAssertEqual(llamaConfig.hiddenSize, 3072)
        XCTAssertEqual(llamaConfig.intermediateSize, 8192)
        XCTAssertEqual(llamaConfig.numHiddenLayers, 28)
        XCTAssertEqual(llamaConfig.numAttentionHeads, 24)
        XCTAssertEqual(llamaConfig.numKeyValueHeads, 8)
        XCTAssertEqual(llamaConfig.headDim, 128)
        XCTAssertEqual(llamaConfig.ropeTheta, 1000000.0)
        XCTAssertEqual(llamaConfig.rmsNormEps, 1e-5)
    }

    // MARK: - Load from File Tests

    func testLoadFromNonExistentFile() {
        let url = URL(fileURLWithPath: "/tmp/nonexistent_params.json")
        XCTAssertThrowsError(try VoxtralTTSConfiguration.load(from: url))
    }

    func testLoadFromValidFile() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_params.json")
        try sampleParamsJSON.data(using: .utf8)!.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let config = try VoxtralTTSConfiguration.load(from: tempURL)
        XCTAssertEqual(config.dim, 3072)
        XCTAssertEqual(config.vocabSize, 131072)
    }

    func testLoadFromInvalidJSON() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("invalid_params.json")
        try "{ invalid json }".data(using: .utf8)!.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        XCTAssertThrowsError(try VoxtralTTSConfiguration.load(from: tempURL))
    }

    // MARK: - Codable Conformance Tests

    func testConfigurationIsCodable() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        // Re-encode and decode
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: encoded)

        XCTAssertEqual(config.dim, decoded.dim)
        XCTAssertEqual(config.nLayers, decoded.nLayers)
        XCTAssertEqual(config.vocabSize, decoded.vocabSize)
        XCTAssertEqual(config.audioModel.semanticCodebookSize, decoded.audioModel.semanticCodebookSize)
        XCTAssertEqual(config.flowMatching.nLayers, decoded.flowMatching.nLayers)
    }

    // MARK: - Sendable Conformance

    func testConfigurationIsSendable() throws {
        let data = sampleParamsJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: data)

        // Verify Sendable by passing to another context
        let expectation = expectation(description: "Sendable check")
        Task {
            let _ = config.dim
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }

    // MARK: - Test Data

    private var sampleParamsJSON: String {
        """
        {
            "dim": 3072,
            "n_layers": 28,
            "head_dim": 128,
            "hidden_dim": 8192,
            "n_heads": 24,
            "n_kv_heads": 8,
            "use_biases": false,
            "rope_theta": 1000000.0,
            "norm_eps": 1e-5,
            "vocab_size": 131072,
            "tied_embeddings": false,
            "max_seq_len": 32768,
            "max_position_embeddings": 32768,
            "model_type": "voxtral_tts",
            "multimodal": {
                "bos_token_id": 1,
                "audio_model_args": {
                    "semantic_codebook_size": 2048,
                    "acoustic_codebook_size": 262144,
                    "n_acoustic_codebook": 36,
                    "audio_encoding_args": {
                        "codebook_pattern": "parallel",
                        "num_codebooks": 37,
                        "sampling_rate": 24000,
                        "frame_rate": 12.5
                    },
                    "audio_token_id": 10,
                    "begin_audio_token_id": 9,
                    "input_embedding_concat_type": "sum",
                    "acoustic_transformer_args": {
                        "input_dim": 3072,
                        "dim": 1024,
                        "n_layers": 3,
                        "head_dim": 64,
                        "hidden_dim": 4096,
                        "n_heads": 16,
                        "n_kv_heads": 4,
                        "use_biases": false,
                        "rope_theta": 10000.0,
                        "sigma": 1e-5,
                        "sigma_max": 2.0
                    },
                    "p_uncond": 0.2,
                    "condition_dropped_token_id": 0
                },
                "audio_tokenizer_args": {
                    "channels": 1,
                    "sampling_rate": 24000,
                    "pretransform_patch_size": 240,
                    "patch_proj_kernel_size": 7,
                    "semantic_codebook_size": 2048,
                    "semantic_dim": 256,
                    "acoustic_codebook_size": 262144,
                    "acoustic_dim": 36,
                    "conv_weight_norm": true,
                    "causal": true,
                    "attn_sliding_window_size": 512,
                    "half_attn_window_upon_downsampling": true,
                    "dim": 512,
                    "hidden_dim": 2048,
                    "head_dim": 64,
                    "n_heads": 8,
                    "n_kv_heads": 2,
                    "qk_norm_eps": 1e-6,
                    "qk_norm": true,
                    "use_biases": false,
                    "norm_eps": 1e-5,
                    "layer_scale": true,
                    "layer_scale_init": 0.01,
                    "decoder_transformer_lengths_str": "2,2,2,2",
                    "decoder_convs_kernels_str": "3,4,4,4",
                    "decoder_convs_strides_str": "1,2,2,2",
                    "voice": {
                        "neutral_female": 0,
                        "neutral_male": 1,
                        "casual_female": 2
                    }
                }
            }
        }
        """
    }
}
