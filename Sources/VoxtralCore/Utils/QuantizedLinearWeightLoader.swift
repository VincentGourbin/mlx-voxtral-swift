/**
 * QuantizedLinearWeightLoader - Replace QuantizedLinear with new instances
 * 
 * Since Swift MLX QuantizedLinear has immutable properties (unlike Python where Module is a dict),
 * we need to replace the entire module with a new one containing the correct weights.
 */

import Foundation
import MLX
import MLXNN

extension VoxtralForConditionalGeneration {
    
    /**
     * Replace ALL quantized linear modules with new instances containing the correct weights
     * 
     * Python debug shows 3 quantized modules:
     * 1. multi_modal_projector.linear_1 (6-bit) ✅ 
     * 2. multi_modal_projector.linear_2 (6-bit) ✅
     * 3. lm_head (6-bit) ❓ <- This was missing!
     */
    public func replaceAllQuantizedLinearWithWeights(_ moduleWeights: [String: [(String, MLXArray)]]) {
        writeDebugToDump("🔧 REPLACING ALL QUANTIZED LINEAR MODULES (including lm_head)\n")
        
        // Replace multi_modal_projector modules (already working)
        multi_modal_projector.replaceQuantizedLinearWithWeights(moduleWeights)
        
        // 🎯 NEW: Replace lm_head if it's quantized
        if let lmHeadWeights = moduleWeights["lm_head"],
           let currentLmHead = self.lm_head as? QuantizedLinear {
            
            writeDebugToDump("  📝 CRITICAL: Replacing lm_head QuantizedLinear\n")
            
            // Extract the weights from the list
            var newWeight: MLXArray? = nil
            var newScales: MLXArray? = nil
            var newBiases: MLXArray? = nil
            
            for (key, value) in lmHeadWeights {
                if key == "lm_head.weight" {
                    newWeight = value
                } else if key == "lm_head.scales" {
                    newScales = value
                } else if key == "lm_head.biases" {
                    newBiases = value
                }
            }
            
            // Create new QuantizedLinear with the correct weights
            if let weight = newWeight, let scales = newScales, let biases = newBiases {
                let newLmHead = QuantizedLinear(
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    groupSize: currentLmHead.groupSize,
                    bits: currentLmHead.bits
                )
                
                // Replace the lm_head module
                var updates = ModuleChildren()
                updates[unwrapping: "lm_head"] = newLmHead
                self.update(modules: updates)
                
                // Verify
                let verifyFlat = newLmHead.weight.flattened()
                var verifyFirst3: [String] = []
                for i in 0..<min(3, verifyFlat.count) {
                    let val = verifyFlat[i].item(UInt32.self)
                    verifyFirst3.append(String(val))
                }
                writeDebugToDump("    ✅ Replaced lm_head, first 3 values: \(verifyFirst3)\n")
                writeDebugToDump("    🎯 Expected: [\"3036416368\", \"413436762\", \"2260200846\"]\n")
            }
        } else {
            writeDebugToDump("  ℹ️ lm_head is not quantized or weights not found\n")
        }
        
        writeDebugToDump("✅ ALL QUANTIZED LINEAR REPLACEMENT COMPLETED\n")
    }
}

extension VoxtralMultiModalProjector {
    
    /**
     * Replace quantized linear modules with new instances containing the correct weights
     * 
     * This mimics what Python does: dst[k] = new_value in Module.update()
     * But since Swift doesn't have mutable dictionary-like access, we replace the whole module
     */
    public func replaceQuantizedLinearWithWeights(_ moduleWeights: [String: [(String, MLXArray)]]) {
        writeDebugToDump("🔧 REPLACING QUANTIZED LINEAR MODULES WITH CORRECT WEIGHTS\n")
        
        // Process linear_1 if it's quantized
        if let linear1Weights = moduleWeights["multi_modal_projector.linear_1"],
           let currentQL1 = self.linear_1 as? QuantizedLinear {
            
            writeDebugToDump("  📝 Replacing linear_1 QuantizedLinear\n")
            
            // Extract the weights from the list
            var newWeight: MLXArray? = nil
            var newScales: MLXArray? = nil
            var newBiases: MLXArray? = nil
            
            for (key, value) in linear1Weights {
                if key.hasSuffix(".weight") {
                    newWeight = value
                } else if key.hasSuffix(".scales") {
                    newScales = value
                } else if key.hasSuffix(".biases") {
                    newBiases = value
                }
            }
            
            // Create new QuantizedLinear with the correct weights
            if let weight = newWeight, let scales = newScales, let biases = newBiases {
                let newQL1 = QuantizedLinear(
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    groupSize: currentQL1.groupSize,
                    bits: currentQL1.bits
                )
                
                // Replace the module using Module.update
                var updates = ModuleChildren()
                updates[unwrapping: "linear_1"] = newQL1
                self.update(modules: updates)
                
                // Verify
                let verifyFlat = newQL1.weight.flattened()
                var verifyFirst3: [String] = []
                for i in 0..<min(3, verifyFlat.count) {
                    let val = verifyFlat[i].item(UInt32.self)
                    verifyFirst3.append(String(val))
                }
                writeDebugToDump("    ✅ Replaced linear_1, first 3 values: \(verifyFirst3)\n")
            }
        }
        
        // Process linear_2 if it's quantized
        if let linear2Weights = moduleWeights["multi_modal_projector.linear_2"],
           let currentQL2 = self.linear_2 as? QuantizedLinear {
            
            writeDebugToDump("  📝 Replacing linear_2 QuantizedLinear\n")
            
            // Extract the weights from the list
            var newWeight: MLXArray? = nil
            var newScales: MLXArray? = nil
            var newBiases: MLXArray? = nil
            
            for (key, value) in linear2Weights {
                if key.hasSuffix(".weight") {
                    newWeight = value
                } else if key.hasSuffix(".scales") {
                    newScales = value
                } else if key.hasSuffix(".biases") {
                    newBiases = value
                }
            }
            
            // Create new QuantizedLinear with the correct weights
            if let weight = newWeight, let scales = newScales, let biases = newBiases {
                let newQL2 = QuantizedLinear(
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    groupSize: currentQL2.groupSize,
                    bits: currentQL2.bits
                )
                
                // Replace the module
                var updates = ModuleChildren()
                updates[unwrapping: "linear_2"] = newQL2
                self.update(modules: updates)
                
                writeDebugToDump("    ✅ Replaced linear_2\n")
            }
        }
        
        writeDebugToDump("✅ QUANTIZED LINEAR REPLACEMENT COMPLETED\n")
    }
}