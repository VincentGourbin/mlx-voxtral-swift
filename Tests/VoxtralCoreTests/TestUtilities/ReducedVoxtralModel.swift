import Foundation
import MLXRandom
@testable import VoxtralCore

/// Voxtral with 2 text layers, random weights (fixed seed) and a vocabulary of 128: exercises the
/// generation loops (prefill slicing, caches, stop tokens) without a real model.
func makeReducedVoxtralModel(seed: UInt64 = 1) throws -> VoxtralForConditionalGeneration {
    let json = """
    {
      "model_type": "voxtral", "audio_token_id": 24, "projector_hidden_act": "gelu",
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
    MLXRandom.seed(seed)
    return VoxtralForConditionalGeneration(standardModel: VoxtralStandardModel(configuration: configuration))
}
