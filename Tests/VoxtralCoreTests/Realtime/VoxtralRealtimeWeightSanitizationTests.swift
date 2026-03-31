/**
 * VoxtralRealtimeWeightSanitizationTests - Tests for weight name mapping
 */

import XCTest
import MLX
@testable import VoxtralCore

final class VoxtralRealtimeWeightSanitizationTests: XCTestCase {

    // MARK: - Format A (Mistral consolidated.safetensors)

    func testFormatATokEmbeddings() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["decoder.tok_embeddings.weight"])
    }

    func testFormatANorm() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "norm.weight": [1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["decoder.norm.weight"])
    }

    func testFormatAEncoderConvLayers() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight": [1, 1, 1],
            "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.bias": [1],
            "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.weight": [1, 1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["encoder.conv_layers_0_conv.conv.weight"])
        XCTAssertNotNil(sanitized["encoder.conv_layers_0_conv.conv.bias"])
        XCTAssertNotNil(sanitized["encoder.conv_layers_1_conv.conv.weight"])
    }

    func testFormatAEncoderTransformerLayers() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wq.weight": [1, 1],
            "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.feed_forward.w1.weight": [1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["encoder.transformer_layers.0.attention.wq.weight"])
        XCTAssertNotNil(sanitized["encoder.transformer_layers.0.feed_forward_w1.weight"])
    }

    func testFormatAEncoderNorm() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight": [1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["encoder.transformer_norm.weight"])
    }

    func testFormatAAdapterProjection() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight": [1, 1],
            "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight": [1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["encoder.audio_language_projection_0.weight"])
        XCTAssertNotNil(sanitized["encoder.audio_language_projection_2.weight"])
    }

    func testFormatADecoderLayers() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "layers.0.attention.wq.weight": [1, 1],
            "layers.0.feed_forward.w1.weight": [1, 1],
            "layers.0.feed_forward.w2.weight": [1, 1],
            "layers.0.feed_forward.w3.weight": [1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["decoder.layers.0.attention.wq.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.feed_forward_w1.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.feed_forward_w2.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.feed_forward_w3.weight"])
    }

    func testFormatAAdaRMSNorm() {
        let weights = makeWeights([
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": [1, 1],
            "layers.0.ada_rms_norm_t_cond.0.weight": [1, 1],
            "layers.0.ada_rms_norm_t_cond.2.weight": [1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNotNil(sanitized["decoder.layers.0.ada_rms_norm_t_cond.ada_down.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.ada_rms_norm_t_cond.ada_up.weight"])
    }

    // MARK: - Format B (mlx-community)

    func testFormatBDecoderAttentionRemap() {
        let weights = makeWeights([
            "encoder.conv_layers_0_conv.conv.weight": [1, 1, 1],
            "decoder.layers.0.attention.wq.weight": [1, 1],
            "decoder.layers.0.attention.wk.weight": [1, 1],
            "decoder.layers.0.attention.wv.weight": [1, 1],
            "decoder.layers.0.attention.wo.weight": [1, 1],
            "decoder.tok_embeddings.weight": [1, 1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        // Encoder keys pass through
        XCTAssertNotNil(sanitized["encoder.conv_layers_0_conv.conv.weight"])
        XCTAssertNotNil(sanitized["decoder.tok_embeddings.weight"])
        // Decoder attention keys are remapped: wq→q_proj, etc.
        XCTAssertNotNil(sanitized["decoder.layers.0.attention.q_proj.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.attention.k_proj.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.attention.v_proj.weight"])
        XCTAssertNotNil(sanitized["decoder.layers.0.attention.o_proj.weight"])
    }

    // MARK: - Filtering

    func testRotaryEmbFiltered() {
        let weights = makeWeights([
            "layers.0.rotary_emb.weight": [1],
            "norm.weight": [1],
        ])
        let sanitized = sanitizeRealtimeWeights(weights)
        XCTAssertNil(sanitized["layers.0.rotary_emb.weight"])
    }

    func testEmptyWeights() {
        let sanitized = sanitizeRealtimeWeights([:])
        XCTAssertTrue(sanitized.isEmpty)
    }

    // MARK: - Helpers

    private func makeWeights(_ spec: [String: [Int]]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, shape) in spec {
            result[key] = MLXArray.zeros(shape)
        }
        return result
    }
}
