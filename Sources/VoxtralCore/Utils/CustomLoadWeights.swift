/**
 * CustomLoadWeights - Reimplementation of Python's load_weights logic
 * 
 * This file implements the exact Python chain:
 * load_weights() → tree_unflatten() → update()
 * 
 * The goal is to fix the issue where Swift's native loadWeights() corrupts
 * values on quantized models while Python's preserves them.
 */

import Foundation
import MLX
import MLXNN

extension VoxtralForConditionalGeneration {
    
    /**
     * Custom implementation of load_weights that matches Python behavior exactly
     * 
     * Python source: https://github.com/ml-explore/mlx/blob/dde3682b69dd78762a805e915f284aa5bf2d1639/python/mlx/nn/layers/base.py#L123
     * 
     * Key difference: Python uses tree_unflatten + update which preserves quantized structure
     */
    @available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). Use VoxtralPipeline.loadModel(), which verifies the weights.")
    public func customLoadWeights(_ weights: [(String, MLXArray)], strict: Bool = true) throws {
        writeDebugToDump("\n🔧 CUSTOM LOAD_WEIGHTS: Starting with \(weights.count) weights\n")
        
        if strict {
            // Python: new_weights = dict(weights)
            let newWeights = Dictionary(uniqueKeysWithValues: weights)
            
            // Python: curr_weights = dict(tree_flatten(self.parameters()))
            // Get current model parameters using flattened() like the rest of the codebase
            let flatParams = self.parameters().flattened()
            var currentWeights: [String: MLXArray] = [:]
            for (key, value) in flatParams {
                currentWeights[key] = value
            }
            
            // Check for extra weights not in model
            let extras = Set(newWeights.keys).subtracting(Set(currentWeights.keys))
            if !extras.isEmpty {
                let extrasList = extras.sorted().joined(separator: ",\n")
                throw VoxtralError.loadingFailed("Received \(extras.count) parameters not in model: \n\(extrasList)")
            }
            
            // Check for missing weights
            let missing = Set(currentWeights.keys).subtracting(Set(newWeights.keys))
            if !missing.isEmpty {
                throw VoxtralError.missingWeights(missing.sorted())
            }
            
            // Validate shapes match
            for (key, currentValue) in currentWeights {
                guard let newValue = newWeights[key] else { continue }
                if newValue.shape != currentValue.shape {
                    throw VoxtralError.loadingFailed("Expected shape \(currentValue.shape) but received shape \(newValue.shape) for parameter \(key)")
                }
            }
        }
        
        // Python: if len(weights) != 0: self.update(tree_unflatten(weights), strict=False)
        if !weights.isEmpty {
            writeDebugToDump("🔧 CUSTOM: Applying weights using targeted approach\n")
            try applyWeightsTargeted(weights)
        }
        
        writeDebugToDump("✅ CUSTOM LOAD_WEIGHTS: Completed successfully\n")
    }
    
    /**
     * Apply weights using a targeted approach that preserves quantized structure
     * 
     * Strategy: Load weights to specific modules individually to avoid corruption
     */
    @available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23).")
    private func applyWeightsTargeted(_ weights: [(String, MLXArray)]) throws {
        writeDebugToDump("🔧 APPLY_WEIGHTS_TARGETED: Processing \(weights.count) weights\n")
        
        // Group weights by module prefix
        var moduleWeights: [String: [(String, MLXArray)]] = [:]
        
        for (key, value) in weights {
            // Extract module prefix (e.g., "multiModalProjector.linear1" from "multiModalProjector.linear1.weight")
            let components = key.split(separator: ".")
            
            if key.hasPrefix("lm_head.") {
                // Special case for lm_head (top-level quantized module)
                if moduleWeights["lm_head"] == nil {
                    moduleWeights["lm_head"] = []
                }
                moduleWeights["lm_head"]?.append((key, value))
            } else if components.count >= 3 {
                // For nested modules like multiModalProjector.linear1
                let modulePrefix = components.prefix(2).joined(separator: ".")
                if moduleWeights[modulePrefix] == nil {
                    moduleWeights[modulePrefix] = []
                }
                moduleWeights[modulePrefix]?.append((key, value))
            } else {
                // For other top-level parameters
                if moduleWeights["root"] == nil {
                    moduleWeights["root"] = []
                }
                moduleWeights["root"]?.append((key, value))
            }
        }
        
        // For quantized modules, we need to REPLACE them entirely with new instances
        // Python debug shows 3 quantized modules: multiModalProjector.linear1, linear_2, and lm_head
        writeDebugToDump("  🎯 Python shows 3 quantized modules: multiModalProjector.linear1, linear_2, lm_head\n")
        
        // Check for ANY quantized modules in the model
        let hasQuantizedModules = (multiModalProjector.linear1 is QuantizedLinear || 
                                  multiModalProjector.linear2 is QuantizedLinear ||
                                  self.lm_head is QuantizedLinear)
        
        if hasQuantizedModules {
            writeDebugToDump("  🎯 Detected quantized layers - replacing ALL quantized modules\n")
            self.replaceAllQuantizedLinearWithWeights(moduleWeights)
        } else {
            writeDebugToDump("  ℹ️ No quantized modules detected - using standard loading\n")
            // For non-quantized modules, use standard loading
            if let projectorWeights = moduleWeights["multiModalProjector.linear1"] {
                writeDebugToDump("  🎯 Loading weights for multiModalProjector.linear1 (Linear)\n")
                try multiModalProjector.linear1.loadWeights(projectorWeights, strict: false)
            }
            
            if let projectorWeights = moduleWeights["multiModalProjector.linear2"] {
                writeDebugToDump("  🎯 Loading weights for multiModalProjector.linear2 (Linear)\n")
                try multiModalProjector.linear2.loadWeights(projectorWeights, strict: false)
            }
        }
        
        // Remove handled weights from the list (including lm_head)
        let handledPrefixes = ["multiModalProjector.linear1", "multiModalProjector.linear2", "lm_head"]
        let remainingWeights = weights.filter { weight in
            !handledPrefixes.contains { prefix in
                weight.0.hasPrefix(prefix + ".") || weight.0 == prefix || weight.0.hasPrefix(prefix + ".")
            }
        }
        
        // Load remaining weights using standard method
        if !remainingWeights.isEmpty {
            writeDebugToDump("  📝 Loading \(remainingWeights.count) remaining weights via standard loadWeights\n")
            
            // Load all remaining weights at once
            // This should work for non-quantized modules
            try self.loadWeights(remainingWeights, strict: false)
        }
        
        writeDebugToDump("✅ APPLY_WEIGHTS_TARGETED: Completed\n")
    }
}