/**
 * VoxtralSanitize - weight-name sanitization and weight loading on `Module`, used by the live loader
 * (`VoxtralStandardLoader`). Moved out of the legacy `VoxtralModelLoading.swift` (K-23).
 */

import Foundation
import MLX
import MLXNN

/**
 * Extension to support model loading operations not provided by MLX
 */
extension Module {
    func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        VoxtralDebug.log("Sanitizing \(weights.count) weights")

        var sanitized: [String: MLXArray] = [:]
        var rotaryCount = 0

        for (key, value) in weights {
            // Skip rotary embeddings and position_ids like Python
            if key.contains("rotary_emb") || key.contains("position_ids") {
                if key.contains("rotary_emb") { rotaryCount += 1 }
                continue
            }

            var newKey = key

            // CRITICAL MAPPINGS for VoxtralStandardModel structure:
            // language_model.lm_head.weight -> languageModel.lmHead.weight (for LanguageModelContainer)
            if newKey == "language_model.lm_head.weight" {
                newKey = "languageModel.lmHead.weight"
            }
            // CRITICAL FIX: Handle root-level lm_head.* keys (for quantized models)
            // lm_head.weight, lm_head.scales, lm_head.biases -> languageModel.lmHead.*
            else if newKey.hasPrefix("lm_head.") {
                let suffix = String(newKey.dropFirst("lm_head.".count))
                newKey = "languageModel.lmHead.\(suffix)"
            }
            // language_model.model.* -> languageModel.model.* (for LlamaStandardModel inside)
            else if newKey.hasPrefix("language_model.model.") {
                let suffix = String(newKey.dropFirst("language_model.model.".count))
                newKey = "languageModel.model.\(suffix)"
            }
            // language_model.* -> languageModel.model.* (other language_model components)
            else if newKey.hasPrefix("language_model.") {
                let suffix = String(newKey.dropFirst("language_model.".count))
                newKey = "languageModel.model.\(suffix)"
            }
            // Handle top-level components (snake_case to camelCase)
            else {
                newKey = newKey.replacingOccurrences(of: "multi_modal_projector", with: "multiModalProjector")
                newKey = newKey.replacingOccurrences(of: "audio_tower", with: "audioTower")
            }

            // Audio-specific conversions - ORDER MATTERS: longer patterns first!
            newKey = newKey.replacingOccurrences(of: "self_attn_layer_norm", with: "selfAttnLayerNorm")
            newKey = newKey.replacingOccurrences(of: "final_layer_norm", with: "finalLayerNorm")
            newKey = newKey.replacingOccurrences(of: "embed_positions", with: "embedPositions")
            newKey = newKey.replacingOccurrences(of: "embed_tokens", with: "embedTokens")
            newKey = newKey.replacingOccurrences(of: "self_attn", with: "selfAttn")
            newKey = newKey.replacingOccurrences(of: "layer_norm", with: "layerNorm")
            newKey = newKey.replacingOccurrences(of: "out_proj", with: "outProj")  // For audio
            newKey = newKey.replacingOccurrences(of: "o_proj", with: "oProj")      // For language model
            newKey = newKey.replacingOccurrences(of: "q_proj", with: "qProj")
            newKey = newKey.replacingOccurrences(of: "k_proj", with: "kProj")
            newKey = newKey.replacingOccurrences(of: "v_proj", with: "vProj")

            // Language model specific conversions
            newKey = newKey.replacingOccurrences(of: "input_layernorm", with: "inputLayerNorm")
            newKey = newKey.replacingOccurrences(of: "post_attention_layernorm", with: "postAttentionLayerNorm")
            newKey = newKey.replacingOccurrences(of: "gate_proj", with: "gateProj")
            newKey = newKey.replacingOccurrences(of: "up_proj", with: "upProj")
            newKey = newKey.replacingOccurrences(of: "down_proj", with: "downProj")

            // VOXTRAL SPECIFIC: Conv weight transpose if needed
            var finalValue = value
            if key.contains("conv") && key.contains("weight") && value.ndim == 3 {
                if value.shape[1] != 3 {  // kernel_size should be 3
                    finalValue = value.transposed(axes: [0, 2, 1])
                }
            }

            sanitized[newKey] = finalValue
        }

        // VOXTRAL SPECIFIC: Copy embed_tokens for sharing
        if let embedWeight = sanitized["languageModel.model.embedTokens.weight"],
           sanitized["embedTokens.weight"] == nil {
            sanitized["embedTokens.weight"] = embedWeight
        }

        VoxtralDebug.log("Sanitized to \(sanitized.count) weights")
        return sanitized
    }
}
