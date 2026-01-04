/**
 * VoxtralConfigurationTests - Unit tests for VoxtralConfiguration
 */

import XCTest
@testable import VoxtralCore

final class VoxtralConfigurationTests: XCTestCase {

    // MARK: - VoxtralEncoderConfig Tests

    func testEncoderConfigDefaultValues() {
        let config = VoxtralEncoderConfig()

        XCTAssertEqual(config.vocab_size, 51866)
        XCTAssertEqual(config.hidden_size, 1280)
        XCTAssertEqual(config.intermediate_size, 5120)
        XCTAssertEqual(config.num_hidden_layers, 32)
        XCTAssertEqual(config.num_attention_heads, 20)
        XCTAssertEqual(config.scale_embedding, false)
        XCTAssertEqual(config.activation_function, "gelu")
        XCTAssertEqual(config.num_mel_bins, 128)
        XCTAssertEqual(config.max_source_positions, 1500)
        XCTAssertEqual(config.initializer_range, 0.02)
        XCTAssertEqual(config.attention_dropout, 0.0)
        XCTAssertEqual(config.dropout, 0.0)
        XCTAssertEqual(config.layerdrop, 0.0)
        XCTAssertEqual(config.activation_dropout, 0.0)
        XCTAssertEqual(config.pad_token_id, 0)
        XCTAssertEqual(config.head_dim, 64)
        XCTAssertEqual(config.num_key_value_heads, 20)
    }

    func testEncoderConfigCustomValues() {
        let config = VoxtralEncoderConfig(
            vocab_size: 1000,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 8,
            num_attention_heads: 8
        )

        XCTAssertEqual(config.vocab_size, 1000)
        XCTAssertEqual(config.hidden_size, 512)
        XCTAssertEqual(config.intermediate_size, 2048)
        XCTAssertEqual(config.num_hidden_layers, 8)
        XCTAssertEqual(config.num_attention_heads, 8)
    }

    func testEncoderConfigComputedProperties() {
        let config = VoxtralEncoderConfig(
            hidden_size: 256,
            intermediate_size: 1024,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            layerdrop: 0.1
        )

        // Test computed property aliases
        XCTAssertEqual(config.d_model, 256)
        XCTAssertEqual(config.encoder_layers, 4)
        XCTAssertEqual(config.encoder_attention_heads, 8)
        XCTAssertEqual(config.encoder_ffn_dim, 1024)
        XCTAssertEqual(config.encoder_layerdrop, 0.1)
    }

    func testEncoderConfigToDictRoundTrip() {
        let original = VoxtralEncoderConfig(
            vocab_size: 10000,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            scale_embedding: true,
            activation_function: "relu"
        )

        let dict = original.to_dict()
        let restored = VoxtralEncoderConfig.from_dict(dict)

        XCTAssertEqual(original.vocab_size, restored.vocab_size)
        XCTAssertEqual(original.hidden_size, restored.hidden_size)
        XCTAssertEqual(original.intermediate_size, restored.intermediate_size)
        XCTAssertEqual(original.num_hidden_layers, restored.num_hidden_layers)
        XCTAssertEqual(original.num_attention_heads, restored.num_attention_heads)
        XCTAssertEqual(original.scale_embedding, restored.scale_embedding)
        XCTAssertEqual(original.activation_function, restored.activation_function)
    }

    func testEncoderConfigFromDictWithMissingFields() {
        // Only provide a few fields, rest should use defaults
        let dict: [String: Any] = [
            "vocab_size": 5000,
            "hidden_size": 256
        ]

        let config = VoxtralEncoderConfig.from_dict(dict)

        XCTAssertEqual(config.vocab_size, 5000)
        XCTAssertEqual(config.hidden_size, 256)
        // Should use default values for missing fields
        XCTAssertEqual(config.intermediate_size, 5120)  // default
        XCTAssertEqual(config.num_hidden_layers, 32)    // default
    }

    func testEncoderConfigFromDictWithAllFields() {
        let dict = TestFixtures.sampleEncoderConfig
        let config = VoxtralEncoderConfig.from_dict(dict)

        XCTAssertEqual(config.vocab_size, 51866)
        XCTAssertEqual(config.hidden_size, 1280)
        XCTAssertEqual(config.intermediate_size, 5120)
        XCTAssertEqual(config.num_hidden_layers, 32)
        XCTAssertEqual(config.num_attention_heads, 20)
        XCTAssertEqual(config.num_mel_bins, 128)
        XCTAssertEqual(config.max_source_positions, 1500)
    }

    func testEncoderConfigToDictContainsAllKeys() {
        let config = VoxtralEncoderConfig()
        let dict = config.to_dict()

        let expectedKeys = [
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "scale_embedding", "activation_function",
            "num_mel_bins", "max_source_positions", "initializer_range",
            "attention_dropout", "dropout", "layerdrop", "activation_dropout",
            "pad_token_id", "head_dim", "num_key_value_heads"
        ]

        for key in expectedKeys {
            XCTAssertNotNil(dict[key], "Missing key: \(key)")
        }
    }

    func testEncoderConfigAttributeMap() {
        // Test that attribute map contains expected mappings
        let map = VoxtralEncoderConfig.attributeMap

        XCTAssertEqual(map["d_model"], "hidden_size")
        XCTAssertEqual(map["encoder_layers"], "num_hidden_layers")
        XCTAssertEqual(map["encoder_attention_heads"], "num_attention_heads")
        XCTAssertEqual(map["encoder_ffn_dim"], "intermediate_size")
        XCTAssertEqual(map["encoder_layerdrop"], "layerdrop")
    }

    func testEncoderConfigModelType() {
        XCTAssertEqual(VoxtralEncoderConfig.modelType, "voxtral_encoder")
    }

    // MARK: - VoxtralTextConfig Tests

    func testTextConfigDefaultValues() {
        let config = VoxtralTextConfig()

        XCTAssertEqual(config.vocab_size, 131072)
        XCTAssertEqual(config.hidden_size, 3072)
        XCTAssertEqual(config.intermediate_size, 8192)
        XCTAssertEqual(config.num_hidden_layers, 30)
        XCTAssertEqual(config.num_attention_heads, 32)
        XCTAssertEqual(config.num_key_value_heads, 8)
        XCTAssertEqual(config.head_dim, 128)
        XCTAssertEqual(config.max_position_embeddings, 131072)
        XCTAssertEqual(config.hidden_act, "silu")
        XCTAssertEqual(config.attention_bias, false)
        XCTAssertEqual(config.mlp_bias, false)
    }

    func testTextConfigCustomValues() {
        let config = VoxtralTextConfig(
            vocab_size: 50000,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 12,
            num_attention_heads: 16
        )

        XCTAssertEqual(config.vocab_size, 50000)
        XCTAssertEqual(config.hidden_size, 1024)
        XCTAssertEqual(config.intermediate_size, 4096)
        XCTAssertEqual(config.num_hidden_layers, 12)
        XCTAssertEqual(config.num_attention_heads, 16)
    }

    func testTextConfigToDictRoundTrip() {
        let original = VoxtralTextConfig(
            vocab_size: 32000,
            hidden_size: 2048,
            hidden_act: "gelu"
        )

        let dict = original.to_dict()
        let restored = VoxtralTextConfig.from_dict(dict)

        XCTAssertEqual(original.vocab_size, restored.vocab_size)
        XCTAssertEqual(original.hidden_size, restored.hidden_size)
        XCTAssertEqual(original.hidden_act, restored.hidden_act)
    }

    func testTextConfigFromDictWithMissingFields() {
        let dict: [String: Any] = [
            "vocab_size": 10000
        ]

        let config = VoxtralTextConfig.from_dict(dict)

        XCTAssertEqual(config.vocab_size, 10000)
        // Should use defaults for missing fields
        XCTAssertEqual(config.hidden_size, 3072)  // default
        XCTAssertEqual(config.hidden_act, "silu") // default
    }

    func testTextConfigWithBiasOptions() {
        let config = VoxtralTextConfig(
            attention_bias: true,
            mlp_bias: true
        )

        XCTAssertTrue(config.attention_bias)
        XCTAssertTrue(config.mlp_bias)
    }

    // MARK: - PythonVoxtralConfig Tests

    func testPythonConfigInitialization() {
        let audioConfig = VoxtralEncoderConfig()
        let textConfig = VoxtralTextConfig()

        let config = PythonVoxtralConfig(
            audio_config: audioConfig,
            text_config: textConfig,
            audio_token_id: 10,
            projector_hidden_act: "gelu"
        )

        XCTAssertEqual(config.audio_token_id, 10)
        XCTAssertEqual(config.projector_hidden_act, "gelu")
        XCTAssertEqual(config.audio_config.vocab_size, 51866)
        XCTAssertEqual(config.text_config.vocab_size, 131072)
    }

    func testPythonConfigDefaultAudioTokenId() {
        let config = PythonVoxtralConfig(
            audio_config: VoxtralEncoderConfig(),
            text_config: VoxtralTextConfig()
        )

        // Default audio_token_id should be 24 (from Python config)
        XCTAssertEqual(config.audio_token_id, 24)
    }

    func testPythonConfigToDictionary() {
        let config = PythonVoxtralConfig(
            audio_config: VoxtralEncoderConfig(),
            text_config: VoxtralTextConfig(),
            audio_token_id: 15
        )

        let dict = config.toDictionary()

        XCTAssertEqual(dict["audio_token_id"] as? Int, 15)
        XCTAssertEqual(dict["model_type"] as? String, "voxtral")
        XCTAssertNotNil(dict["audio_config"])
        XCTAssertNotNil(dict["text_config"])
    }

    // MARK: - Edge Cases

    func testConfigWithZeroValues() {
        let config = VoxtralEncoderConfig(
            vocab_size: 0,
            hidden_size: 0,
            num_hidden_layers: 0
        )

        XCTAssertEqual(config.vocab_size, 0)
        XCTAssertEqual(config.hidden_size, 0)
        XCTAssertEqual(config.num_hidden_layers, 0)
    }

    func testConfigWithLargeValues() {
        let config = VoxtralEncoderConfig(
            vocab_size: 1_000_000,
            hidden_size: 16384,
            num_hidden_layers: 128
        )

        XCTAssertEqual(config.vocab_size, 1_000_000)
        XCTAssertEqual(config.hidden_size, 16384)
        XCTAssertEqual(config.num_hidden_layers, 128)
    }

    func testConfigFromEmptyDict() {
        let dict: [String: Any] = [:]
        let config = VoxtralEncoderConfig.from_dict(dict)

        // Should use all default values
        XCTAssertEqual(config.vocab_size, 51866)
        XCTAssertEqual(config.hidden_size, 1280)
    }

    func testConfigFromDictWithWrongTypes() {
        // Dictionary with wrong types should fall back to defaults
        let dict: [String: Any] = [
            "vocab_size": "not_an_int",  // Wrong type
            "hidden_size": 512           // Correct type
        ]

        let config = VoxtralEncoderConfig.from_dict(dict)

        // vocab_size should use default because of wrong type
        XCTAssertEqual(config.vocab_size, 51866)  // default
        // hidden_size should use provided value
        XCTAssertEqual(config.hidden_size, 512)
    }
}
