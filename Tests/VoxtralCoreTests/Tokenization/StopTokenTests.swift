/**
 * StopTokenTests - K-4 (S-01, MLX-017)
 *
 * STT generation stops on special tokens only. Tekken maps text ranks to ids ≥ 1 000
 * (id = rank + 1 000), so a stop id ≥ 1 000 is a word: the historical 32000 is "␣Capital"
 * and cut every transcription at its first "Capital".
 */

import MLX
import MLXRandom
import XCTest
@testable import VoxtralCore

final class StopTokenTests: XCTestCase {

    /// Reduced Voxtral (2 text layers, random weights, vocabulary of 128).
    private func reducedModel() throws -> VoxtralForConditionalGeneration {
        let json = """
        {
          "model_type": "voxtral", "audio_token_id": 24, "projector_hidden_act": "gelu",
          "text_config": {
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 4096, "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0, "hidden_act": "silu", "attention_bias": false, "mlp_bias": false
          },
          "audio_config": {
            "hidden_size": 32, "intermediate_size": 128, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
            "max_source_positions": 1500, "num_mel_bins": 128, "vocab_size": 128
          }
        }
        """
        let configuration = try JSONDecoder().decode(VoxtralStandardConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(4)
        return VoxtralForConditionalGeneration(standardModel: VoxtralStandardModel(configuration: configuration))
    }

    func testStopTokensAreSpecialIds() throws {
        let model = try reducedModel()
        XCTAssertFalse(model.stopTokenIds.isEmpty)
        for id in model.stopTokenIds {
            XCTAssertLessThan(id, 1_000, "\(id) ≥ 1000: a Tekken text token, not a special one")
        }
    }

    func testGenerationStopsOnConfiguredStopTokens() throws {
        let model = try reducedModel()
        let inputIds = MLXArray((30 ..< 38).map { Int32($0) }).reshaped([1, 8])

        model.stopTokenIds = Array(0 ..< 128)  // every id stops
        XCTAssertEqual(try model.generateStream(inputIds: inputIds, maxNewTokens: 5, temperature: 0).count, 1)

        model.stopTokenIds = []  // nothing stops
        XCTAssertEqual(try model.generateStream(inputIds: inputIds, maxNewTokens: 5, temperature: 0).count, 5)
    }
}
