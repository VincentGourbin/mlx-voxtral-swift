/**
 * MLXErrorBoundaryTests - K-1 (MLX-021)
 *
 * An MLX error raised inside a public entry point surfaces as `VoxtralError.mlx` and the
 * process stays alive, instead of mlx-swift's default handler calling `fatalError`
 * (which kills the host app). Without the fix, (b) crashes the test runner (piège 38).
 */

import MLX
import MLXRandom
import XCTest
@testable import VoxtralCore

final class MLXErrorBoundaryTests: XCTestCase {

    // (a) the boundary converts a caught MLX error
    func testBroadcastErrorInsideBoundaryThrowsVoxtralError() {
        XCTAssertThrowsError(try withMLXErrors { _ -> MLXArray in
            let a = MLXArray(0 ..< 10, [2, 5])
            let b = MLXArray(0 ..< 15, [3, 5])
            return a + b
        }) { error in
            guard case VoxtralError.mlx = error else {
                return XCTFail("expected VoxtralError.mlx, got \(error)")
            }
        }
    }

    /// Reduced Voxtral (2 text layers, random weights) built from a JSON configuration.
    private func reducedModel() throws -> VoxtralForConditionalGeneration {
        let json = """
        {
          "model_type": "voxtral",
          "audio_token_id": 24,
          "projector_hidden_act": "gelu",
          "text_config": {
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 8192, "rms_norm_eps": 1e-5,
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
        MLXRandom.seed(1)
        return VoxtralForConditionalGeneration(standardModel: VoxtralStandardModel(configuration: configuration))
    }

    // (b) an MLX error inside generation: P-17, a bf16 model fed the hand-made fp32 additive mask
    // ("Mask type must promote to output type"), present until K-3. The P-03 trigger (prefill beyond a
    // rotating window) was used before K-2 removed the window from every preset.
    func testGenerationMLXErrorThrowsVoxtralError() throws {
        let model = try reducedModel()
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let inputIds = MLXArray((0 ..< 16).map { Int32(30 + $0) }).reshaped([1, 16])

        XCTAssertThrowsError(try model.generateStream(
            inputIds: inputIds, maxNewTokens: 1, memoryOptimization: .disabled)
        ) { error in
            guard case VoxtralError.mlx(let message) = error else {
                return XCTFail("expected VoxtralError.mlx, got \(error)")
            }
            XCTAssertTrue(message.lowercased().contains("mask"), message)
        }
    }
}
