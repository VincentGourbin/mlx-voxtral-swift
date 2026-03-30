/**
 * VoxtralTTSWeightSanitizationTests - Unit tests for weight name sanitization
 */

import XCTest
import MLX
@testable import VoxtralCore

final class VoxtralTTSWeightSanitizationTests: XCTestCase {

    // MARK: - Format A (Original Mistral) Sanitization Tests

    func testFormatAAttentionWeightRenaming() {
        let weights = makeWeights([
            "layers.0.attention.wq.weight": [1, 1],
            "layers.0.attention.wk.weight": [1, 1],
            "layers.0.attention.wv.weight": [1, 1],
            "layers.0.attention.wo.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["layers.0.self_attn.k_proj.weight"])
        XCTAssertNotNil(sanitized["layers.0.self_attn.v_proj.weight"])
        XCTAssertNotNil(sanitized["layers.0.self_attn.o_proj.weight"])
    }

    func testFormatANormRenaming() {
        let weights = makeWeights([
            "layers.0.attention_norm.weight": [1],
            "layers.0.ffn_norm.weight": [1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["layers.0.input_layernorm.weight"])
        XCTAssertNotNil(sanitized["layers.0.post_attention_layernorm.weight"])
    }

    func testFormatAFFNRenaming() {
        let weights = makeWeights([
            "layers.0.feed_forward.w1.weight": [1, 1],
            "layers.0.feed_forward.w2.weight": [1, 1],
            "layers.0.feed_forward.w3.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["layers.0.mlp.gate_proj.weight"])
        XCTAssertNotNil(sanitized["layers.0.mlp.down_proj.weight"])
        XCTAssertNotNil(sanitized["layers.0.mlp.up_proj.weight"])
    }

    func testFormatAPreservesNonLayerKeys() {
        let weights = makeWeights([
            "norm.weight": [1],
            "mm_audio_embeddings.tok_embeddings.weight": [1, 1],
            "acoustic_transformer.layers.0.weight": [1],
            "audio_tokenizer.decoder.weight": [1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["norm.weight"])
        XCTAssertNotNil(sanitized["mm_audio_embeddings.tok_embeddings.weight"])
        XCTAssertNotNil(sanitized["acoustic_transformer.layers.0.weight"])
        XCTAssertNotNil(sanitized["audio_tokenizer.decoder.weight"])
    }

    // MARK: - Format B (MLX Community) Sanitization Tests

    func testFormatBStripsLanguageModelPrefix() {
        let weights = makeWeights([
            "language_model.model.model.layers.0.self_attn.q_proj.weight": [1, 1],
            "language_model.model.model.norm.weight": [1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["norm.weight"])
    }

    func testFormatBEmbedTokensMapping() {
        let weights = makeWeights([
            "language_model.model.model.embed_tokens.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["mm_audio_embeddings.tok_embeddings.weight"])
    }

    func testFormatBAudioCodebookEmbeddingsMapping() {
        let weights = makeWeights([
            "language_model.model.model.layers.0.self_attn.q_proj.weight": [1, 1],  // MLX community marker
            "audio_codebook_embeddings.semantic.weight": [1, 1],
            "audio_codebook_embeddings.acoustic.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["mm_audio_embeddings.audio_codebook_embeddings.semantic.weight"])
        XCTAssertNotNil(sanitized["mm_audio_embeddings.audio_codebook_embeddings.acoustic.weight"])
    }

    func testFormatBPreservesAcousticTransformer() {
        let weights = makeWeights([
            "language_model.model.model.layers.0.self_attn.q_proj.weight": [1, 1],  // MLX community marker
            "acoustic_transformer.layers.0.self_attn.q_proj.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["acoustic_transformer.layers.0.self_attn.q_proj.weight"])
    }

    func testFormatBPreservesAudioTokenizer() {
        let weights = makeWeights([
            "language_model.model.model.layers.0.self_attn.q_proj.weight": [1, 1],  // MLX community marker
            "audio_tokenizer.decoder.conv.weight": [1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["audio_tokenizer.decoder.conv.weight"])
    }

    // MARK: - Rotary Embedding Filtering

    func testRotaryEmbeddingsAreFiltered() {
        let weights = makeWeights([
            "layers.0.self_attn.rotary_emb.weight": [1],
            "layers.0.self_attn.q_proj.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNil(sanitized["layers.0.self_attn.rotary_emb.weight"])
        // q_proj should be preserved (but renamed from Format A or kept from Format B)
    }

    func testPositionIdsAreFiltered() {
        let weights = makeWeights([
            "layers.0.position_ids": [1],
            "norm.weight": [1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNil(sanitized["layers.0.position_ids"])
        XCTAssertNotNil(sanitized["norm.weight"])
    }

    // MARK: - Edge Cases

    func testEmptyWeightsReturnsEmpty() {
        let weights: [String: MLXArray] = [:]
        let sanitized = sanitizeTTSWeights(weights)
        XCTAssertTrue(sanitized.isEmpty)
    }

    func testWeightCountPreservedMinusFiltered() {
        let weights = makeWeights([
            "layers.0.attention.wq.weight": [1, 1],
            "layers.0.attention.wk.weight": [1, 1],
            "layers.0.attention.wv.weight": [1, 1],
            "layers.0.attention.wo.weight": [1, 1],
            "norm.weight": [1],
            "layers.0.self_attn.rotary_emb.weight": [1],  // Should be filtered
        ])

        let sanitized = sanitizeTTSWeights(weights)

        // 6 input - 1 filtered = 5 output
        XCTAssertEqual(sanitized.count, 5)
    }

    func testMultipleLayersAreHandled() {
        let weights = makeWeights([
            "layers.0.attention.wq.weight": [1, 1],
            "layers.1.attention.wq.weight": [1, 1],
            "layers.27.attention.wq.weight": [1, 1],
        ])

        let sanitized = sanitizeTTSWeights(weights)

        XCTAssertNotNil(sanitized["layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["layers.1.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["layers.27.self_attn.q_proj.weight"])
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
