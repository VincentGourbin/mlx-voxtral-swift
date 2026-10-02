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
    
    func loadWeights(_ weightItems: [(key: String, value: MLXArray)], strict: Bool) throws {
        // Python equivalent: model.load_weights(list(weights.items()), strict=True)
        writeDebugToDump("\n🔍 DEBUG LOAD_WEIGHTS SWIFT: Loading \(weightItems.count) weight tensors into model\n")
        
        // DEBUG: Print model parameters to see what the Swift model expects
        let modelParams = self.parameters()
        let flatParams = modelParams.flattened()
        writeDebugToDump("🎯 SWIFT MODEL EXPECTED PARAMETERS (first 20):\n")
        let expectedSorted = Array(flatParams.map { $0.0 }.sorted().prefix(20))
        for (i, paramName) in expectedSorted.enumerated() {
            writeDebugToDump("  \(i+1). \(paramName)\n")
        }
        
        // Convert to dictionary format for MLX Swift
        var weightDict: [String: MLXArray] = [:]
        
        // Group weights by base name for quantized layers
        var quantizedWeights: [String: (weight: MLXArray?, scales: MLXArray?, biases: MLXArray?)] = [:]
        
        for (key, value) in weightItems {
            if key.hasSuffix(".scales") {
                let baseName = String(key.dropLast(7)) // Remove ".scales"
                if quantizedWeights[baseName] == nil {
                    quantizedWeights[baseName] = (nil, nil, nil)
                }
                quantizedWeights[baseName]?.scales = value
                writeDebugToDump("🔍 Found scales for: \(baseName)\n")
            } else if key.hasSuffix(".biases") {
                let baseName = String(key.dropLast(7)) // Remove ".biases"  
                if quantizedWeights[baseName] == nil {
                    quantizedWeights[baseName] = (nil, nil, nil)
                }
                quantizedWeights[baseName]?.biases = value
                writeDebugToDump("🔍 Found biases for: \(baseName)\n")
            } else if key.hasSuffix(".weight") {
                let baseName = String(key.dropLast(7)) // Remove ".weight"
                // Check if this has quantization components
                let hasScales = weightItems.contains { $0.key == "\(baseName).scales" }
                let hasBiases = weightItems.contains { $0.key == "\(baseName).biases" }
                
                if hasScales && hasBiases {
                    // This is a quantized weight - store for special handling
                    if quantizedWeights[baseName] == nil {
                        quantizedWeights[baseName] = (nil, nil, nil)
                    }
                    quantizedWeights[baseName]?.weight = value
                    writeDebugToDump("🔍 Found quantized weight for: \(baseName)\n")
                } else {
                    // Regular weight
                    weightDict[key] = value
                }
            } else {
                // Other parameters (bias, normalization, etc.)
                weightDict[key] = value
            }
        }
        
        writeDebugToDump("📊 QUANTIZED WEIGHTS SUMMARY:\n")
        writeDebugToDump("  - Found \(quantizedWeights.count) quantized layer groups\n")
        writeDebugToDump("  - Regular weights: \(weightDict.count)\n")
        
        // For quantized layers, we need special handling
        // MLX Python automatically handles this, but Swift needs explicit logic
        for (baseName, components) in quantizedWeights {
            if let weight = components.weight, let scales = components.scales, let biases = components.biases {
                // Store ALL components for QuantizedLinear: weight, scales, and biases
                weightDict["\(baseName).weight"] = weight
                weightDict["\(baseName).scales"] = scales
                weightDict["\(baseName).biases"] = biases
                writeDebugToDump("✅ Prepared quantized layer: \(baseName) with weight, scales and biases\n")
            }
        }
        
        // CRITICAL: Do NOT use apply() for quantized models as it breaks the QuantizedLinear structure
        // QuantizedLinear needs weight, scales, and biases to be set together
        writeDebugToDump("🔄 Updating parameters with loaded weights...\n")
        
        // Check if model has quantized layers
        let hasQuantizedLayers = !quantizedWeights.isEmpty
        
        if hasQuantizedLayers {
            writeDebugToDump("⚠️ Model has quantized layers - using update(parameters:) like Python\n")
            
            writeDebugToDump("🔄 Using SIMPLIFIED weight loading for quantized model\n")
            writeDebugToDump("📊 Attempting to load \(weightDict.count) quantized parameters\n")
            
            // SIMPLIFIED: Use apply with a direct parameter mapping to avoid complex nested structures
            // This preserves QuantizedLinear structure unlike the original apply()
            var successCount = 0
            let currentParams = self.parameters().flattened()
            
            for (paramName, _) in currentParams {
                if weightDict[paramName] != nil {
                    writeDebugToDump("✅ Loading quantized weight: \(paramName)\n")
                    // Direct parameter replacement - this should work for QuantizedLinear
                    successCount += 1
                } else {
                    writeDebugToDump("⚠️ No weight found for: \(paramName)\n")
                }
            }
            
            // CRITICAL FIX: Actually apply the weights to the model!
            writeDebugToDump("🔄 APPLYING \(weightDict.count) weights to quantized model...\n")
            
            // Convert [String: MLXArray] to NestedDictionary<String, MLXArray> (ModuleParameters)
            var nestedValues: [String: NestedItem<String, MLXArray>] = [:]
            for (key, array) in weightDict {
                nestedValues[key] = .value(array)
                writeDebugToDump("📋 Converting weight: \(key), shape: \(array.shape), dtype: \(array.dtype)\n")
            }
            let moduleParameters = NestedDictionary(values: nestedValues)
            
            // Apply weights using MLX Swift's update(parameters:) - equivalent to Python's load_weights()
            self.update(parameters: moduleParameters)
            writeDebugToDump("✅ Applied \(weightDict.count) weights using update(parameters:)\n")
            
            writeDebugToDump("📊 Successfully matched \(successCount) quantized parameters\n")
            
            writeDebugToDump("✅ Quantized model update(parameters:) completed successfully\n")
        } else {
            writeDebugToDump("Model has no quantized layers - using update(parameters:) like Python\n")
            
            writeDebugToDump("🔄 Using SIMPLIFIED weight loading for non-quantized model\n")
            writeDebugToDump("📊 Attempting to load \(weightDict.count) parameters\n")
            
            // SIMPLIFIED: Direct parameter loading
            var successCount = 0
            let currentParams = self.parameters().flattened()
            
            for (paramName, _) in currentParams {
                if weightDict[paramName] != nil {
                    writeDebugToDump("✅ Loading weight: \(paramName)\n")
                    successCount += 1
                }
            }
            
            // CRITICAL FIX: Actually apply the weights to the model!
            writeDebugToDump("🔄 APPLYING \(weightDict.count) weights to non-quantized model...\n")
            
            // Convert [String: MLXArray] to NestedDictionary<String, MLXArray> (ModuleParameters)
            var nestedValues: [String: NestedItem<String, MLXArray>] = [:]
            for (key, array) in weightDict {
                nestedValues[key] = .value(array)
                writeDebugToDump("📋 Converting weight: \(key), shape: \(array.shape), dtype: \(array.dtype)\n")
            }
            let moduleParameters = NestedDictionary(values: nestedValues)
            
            // Apply weights using MLX Swift's update(parameters:) - equivalent to Python's load_weights()
            self.update(parameters: moduleParameters)
            writeDebugToDump("✅ Applied \(weightDict.count) weights using update(parameters:)\n")
            
            writeDebugToDump("📊 Successfully matched \(successCount) parameters\n")
            
            writeDebugToDump("✅ update(parameters:) completed successfully\n")
        }
        
        if strict {
            // Verify all expected parameters were loaded
            let modelParams = self.parameters().flattened()
            let loadedKeys = Set(weightDict.keys)
            let expectedKeys = Set(modelParams.map { $0.0 })
            
            let missingKeys = expectedKeys.subtracting(loadedKeys)
            if !missingKeys.isEmpty {
                writeDebugToDump("❌ Missing required weights: \(missingKeys)\n")
                throw VoxtralError.loadingFailed("Missing required weights: \(missingKeys)")
            }
        }
        
        // Final verification: Check that QuantizedLinear structure is preserved
        writeDebugToDump("🔍 FINAL VERIFICATION - Checking model structure after update(parameters:)\n")
        if let voxtralModel = self as? VoxtralForConditionalGeneration {
            let proj1 = voxtralModel.multiModalProjector.linear1
            if let qLinear1 = proj1 as? QuantizedLinear {
                writeDebugToDump("✅ Projector linear_1 is still QuantizedLinear after update()\n")
                writeDebugToDump("  - weight shape: \(qLinear1.weight.shape), dtype: \(qLinear1.weight.dtype)\n")
                writeDebugToDump("  - scales shape: \(qLinear1.scales.shape), dtype: \(qLinear1.scales.dtype)\n")
                if let biases = qLinear1.biases {
                    writeDebugToDump("  - biases shape: \(biases.shape), dtype: \(biases.dtype)\n")
                }
            } else {
                writeDebugToDump("❌ CRITICAL: Projector linear_1 is NO LONGER QuantizedLinear after update()!\n")
                writeDebugToDump("  - Type is now: \(type(of: proj1))\n")
            }
        }
        
        
        writeDebugToDump("✅ SWIFT LOAD_WEIGHTS: Successfully loaded weights using update(parameters:)\n")
        writeDebugToDump("🔍 END DEBUG LOAD_WEIGHTS SWIFT\n\n")
    }
}
