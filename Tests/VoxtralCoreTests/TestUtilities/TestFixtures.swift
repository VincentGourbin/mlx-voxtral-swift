/**
 * TestFixtures - Shared test data and fixtures for VoxtralCore tests
 */

import Foundation
import MLX

/// Shared test fixtures for VoxtralCore tests
/// Note: nonisolated(unsafe) used for test fixtures that don't need concurrency safety
enum TestFixtures {

    // MARK: - Sample Configuration Data

    /// Sample encoder configuration dictionary
    nonisolated(unsafe) static let sampleEncoderConfig: [String: Any] = [
        "vocab_size": 51866,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "scale_embedding": false,
        "activation_function": "gelu",
        "num_mel_bins": 128,
        "max_source_positions": 1500,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "layerdrop": 0.0,
        "activation_dropout": 0.0,
        "pad_token_id": 0,
        "head_dim": 64,
        "num_key_value_heads": 20
    ]

    /// Sample text configuration dictionary
    nonisolated(unsafe) static let sampleTextConfig: [String: Any] = [
        "vocab_size": 131072,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 30,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000000.0,
        "hidden_act": "silu",
        "attention_bias": false,
        "mlp_bias": false
    ]

    /// Sample full Voxtral configuration
    nonisolated(unsafe) static let sampleFullConfig: [String: Any] = [
        "model_type": "voxtral",
        "audio_config": sampleEncoderConfig,
        "text_config": sampleTextConfig,
        "audio_token_id": 24,
        "projector_hidden_act": "gelu"
    ]

    /// Minimal configuration with only required fields
    nonisolated(unsafe) static let minimalEncoderConfig: [String: Any] = [
        "vocab_size": 1000,
        "hidden_size": 256
    ]

    /// Configuration with invalid types for error testing
    nonisolated(unsafe) static let invalidTypeConfig: [String: Any] = [
        "vocab_size": "not_an_int",
        "hidden_size": 256.5
    ]

    // MARK: - Sample Token Data

    /// Sample token IDs for testing
    static let sampleTokenIds: [Int] = [1, 100, 500, 1000, 5000, 10000]

    /// Sample special tokens dictionary
    static let sampleSpecialTokens: [String: Int] = [
        "<|begin_of_text|>": 1,
        "<|end_of_text|>": 2,
        "<|audio|>": 10,
        "<pad>": 0
    ]

    /// Sample BPE vocabulary
    static let sampleVocab: [String: Int] = [
        "hello": 100,
        "world": 101,
        "test": 102,
        "Ġhello": 103,
        "Ġworld": 104
    ]

    // MARK: - Sample Audio Data

    /// Generate synthetic audio data (sine wave)
    /// - Parameters:
    ///   - seconds: Duration in seconds
    ///   - sampleRate: Sample rate (default 16000)
    ///   - frequency: Frequency in Hz (default 440 = A4)
    /// - Returns: Array of Float samples
    static func syntheticAudioData(
        seconds: Float,
        sampleRate: Int = 16000,
        frequency: Float = 440.0
    ) -> [Float] {
        let numSamples = Int(seconds * Float(sampleRate))
        return (0..<numSamples).map { i in
            sin(2.0 * .pi * frequency * Float(i) / Float(sampleRate))
        }
    }

    /// Generate silent audio data
    /// - Parameters:
    ///   - seconds: Duration in seconds
    ///   - sampleRate: Sample rate (default 16000)
    /// - Returns: Array of zero samples
    static func silentAudioData(seconds: Float, sampleRate: Int = 16000) -> [Float] {
        let numSamples = Int(seconds * Float(sampleRate))
        return [Float](repeating: 0.0, count: numSamples)
    }

    /// Generate white noise audio data
    /// - Parameters:
    ///   - seconds: Duration in seconds
    ///   - sampleRate: Sample rate (default 16000)
    /// - Returns: Array of random samples between -1 and 1
    static func noiseAudioData(seconds: Float, sampleRate: Int = 16000) -> [Float] {
        let numSamples = Int(seconds * Float(sampleRate))
        return (0..<numSamples).map { _ in
            Float.random(in: -1.0...1.0)
        }
    }

    // MARK: - Sample Model Info

    /// Sample model info for testing
    static let sampleModelInfo: (id: String, repoId: String, name: String) = (
        id: "test-model",
        repoId: "test/test-model",
        name: "Test Model"
    )

    // MARK: - Sample Text Data

    /// Sample text for tokenization tests
    static let sampleTexts: [String] = [
        "Hello, world!",
        "Bonjour le monde!",
        "你好世界",
        "مرحبا بالعالم",
        "🎵 Music 🎶",
        "Test with numbers: 123 456.789",
        "Special chars: @#$%^&*()",
        ""  // Empty string
    ]

    /// Sample multilingual text
    static let multilingualText = "Hello Bonjour 你好 مرحبا"

    /// Very long text for stress testing
    static let veryLongText = String(repeating: "This is a test sentence. ", count: 1000)
}
