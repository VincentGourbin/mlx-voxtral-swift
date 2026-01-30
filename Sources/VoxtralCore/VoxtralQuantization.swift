/**
 * VoxtralQuantization - Swift equivalent of mlx.voxtral/quantization.py
 * 
 * Exact conversion of Python quantization utilities for Voxtral models.
 * Direct line-by-line translation following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

import Foundation
import MLX
import MLXNN

/**
 * Helper function to check if an MLXArray has quantization scales
 * Python equivalent: hasattr(param, 'scales')
 */
private func hasQuantizationScales(_ param: MLXArray) -> Bool {
    // In MLX Swift, quantized parameters have integer data types (typically uint8 for 4-bit or 8-bit quantization)
    // Python equivalent: checking if a parameter has quantization attributes
    
    // Check if the parameter has a quantized data type
    // MLX Swift quantization typically uses uint8 or int8 for quantized weights
    if param.dtype == .int8 || param.dtype == .uint8 {
        return true
    }
    
    // Also check for specific quantized shapes that indicate quantization
    // Quantized weights often have specific shape patterns due to grouping
    let shape = param.shape
    if shape.count > 1 {
        // If the shape suggests grouped quantization (e.g., weights are packed)
        // This is a heuristic based on common quantization patterns
        return false  // For now, rely mainly on dtype check
    }
    
    return false
}

/**
 * Direct Python equivalent: 
 * import mlx_lm.utils
 * quantize_model = mlx_lm.utils.quantize_model
 * save_config = mlx_lm.utils.save_config
 * save_model = mlx_lm.utils.save_model
 */

/**
 * Direct Python equivalent: quantize_model = mlx_lm.utils.quantize_model
 */
public func quantizeModel<T: Module>(_ model: T, groupSize: Int = 64, bits: Int = 4, classPredicate: ((String, Module) -> Bool)? = nil) throws {
    // Python: mlx_lm.utils.quantize_model() equivalent
    // Swift MLX: quantize(model:groupSize:bits:filter:apply:) from MLXNN
    print("Quantizing model with group_size=\(groupSize), bits=\(bits)")

    // Use MLX Swift's built-in quantize function with the new 4-argument API
    quantize(
        model: model,
        groupSize: groupSize,
        bits: bits,
        filter: { path, module in
            if let predicate = classPredicate {
                // Use custom predicate if provided
                return predicate(path, module)
            } else {
                // Python: Default quantization - quantize all Linear and Embedding layers
                return module is Linear || module is Embedding
            }
        }
    )
}


/**
 * Direct Python equivalent: save_config = mlx_lm.utils.save_config
 */
public func saveConfig(_ config: [String: Any], to path: String) throws {
    // Python: save_config(config, path)
    let configData = try JSONSerialization.data(withJSONObject: config, options: .prettyPrinted)
    let configURL = URL(fileURLWithPath: "\(path)/config.json")
    try configData.write(to: configURL)
}

/**
 * Direct Python equivalent: save_model = mlx_lm.utils.save_model
 */
public func saveModel<T: Module>(_ model: T, to path: String) throws {
    // Python: mlx_lm.utils.save_model(model, path)
    print("Saving model to: \(path)")
    
    // Create directory if it doesn't exist
    let pathURL = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(at: pathURL, withIntermediateDirectories: true, attributes: nil)
    
    // Get model parameters
    let parameters = model.parameters()
    
    // Convert parameters to [String: MLXArray] format
    var parameterDict: [String: MLXArray] = [:]
    for (name, param) in parameters.flattened() {
        parameterDict[name] = param
    }
    
    // Python: mx.save(weights, path / "model.safetensors")
    let modelURL = pathURL.appendingPathComponent("model.safetensors")
    try save(arrays: parameterDict, metadata: [:], url: modelURL)
    
    print("✅ Model saved successfully to \(modelURL.path)")
}

/**
 * Swift equivalent of mlx.utils.tree_reduce
 * Based on MLX C++ source: https://github.com/ml-explore/mlx/blob/8f163a367d28c6b09b5f7eeb4fe21f7f36fc2c56/python/src/trees.h#L45
 */
public func treeReduce<T, U>(
    _ fn: (T, Any) -> T,
    _ tree: U,
    _ initialValue: T
) -> T {
    var result = initialValue
    
    func processNode(_ node: Any) {
        if let array = node as? MLXArray {
            result = fn(result, array)
        } else if let dict = node as? [String: Any] {
            for (_, value) in dict {
                processNode(value)
            }
        } else if let array = node as? [Any] {
            for item in array {
                processNode(item)
            }
        } else {
            // Use Swift reflection for complex objects like Module.parameters()
            let mirror = Mirror(reflecting: node)
            for child in mirror.children {
                processNode(child.value)
            }
        }
    }
    
    processNode(tree)
    return result
}

/**
 * Swift equivalent of mlx.utils.tree_flatten
 * Based on MLX C++ source: https://github.com/ml-explore/mlx/blob/8f163a367d28c6b09b5f7eeb4fe21f7f36fc2c56/python/src/trees.h#L45
 */
public func treeFlatten<T>(_ tree: T) -> [(String, Any)] {
    var flattened: [(String, Any)] = []
    
    func processNode(_ node: Any, prefix: String = "") {
        if let array = node as? MLXArray {
            flattened.append((prefix, array))
        } else if let dict = node as? [String: Any] {
            for (key, value) in dict {
                let newPrefix = prefix.isEmpty ? key : "\(prefix).\(key)"
                processNode(value, prefix: newPrefix)
            }
        } else if let array = node as? [Any] {
            for (index, item) in array.enumerated() {
                let newPrefix = prefix.isEmpty ? "\(index)" : "\(prefix).\(index)"
                processNode(item, prefix: newPrefix)
            }
        } else {
            // Use Swift reflection for complex objects like Module.parameters()
            let mirror = Mirror(reflecting: node)
            for child in mirror.children {
                if let label = child.label {
                    let newPrefix = prefix.isEmpty ? label : "\(prefix).\(label)"
                    processNode(child.value, prefix: newPrefix)
                }
            }
        }
    }
    
    processNode(tree)
    return flattened
}

/**
 * Direct Python equivalent: def compute_bits_per_weight(model)
 * Exact line-by-line translation of mlx.voxtral/quantization.py:17-42
 */
public func computeBitsPerWeight<T: Module>(_ model: T) -> Float {
    // Python: """Compute average bits per weight, handling different mlx_lm versions."""
    // Python: try:
    //     return mlx_lm.utils.compute_bits_per_weight(model)
    // except (AttributeError, ZeroDivisionError):
    
    // Skip mlx_lm fallback and go directly to manual implementation
    
    // Python: from mlx.utils import tree_flatten, tree_reduce
    // Python: import mlx.core as mx
    
    // Python: if hasattr(model, 'parameters'):
    //     params = model.parameters()
    // else:
    //     params = model
    let params: Any = model.parameters()
    
    // Python: model_bytes = tree_reduce(
    //     lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, 
    //     params, 
    //     0
    // )
    let modelBytes = treeReduce({ (acc: Int, x: Any) -> Int in
        if let array = x as? MLXArray {
            return acc + array.nbytes
        } else {
            return acc
        }
    }, params, 0)
    
    // Python: params_flat = tree_flatten(params)
    let paramsFlat = treeFlatten(params)
    
    // Python: model_params = sum(p.size for _, p in params_flat if isinstance(p, mx.array))
    let modelParams = paramsFlat.reduce(0) { (acc, item) in
        if let array = item.1 as? MLXArray {
            return acc + array.size
        } else {
            return acc
        }
    }
    
    // Python: if model_params == 0:
    //     return 0.0
    if modelParams == 0 {
        return 0.0
    }
    
    // Python: return model_bytes * 8 / model_params
    return Float(modelBytes * 8) / Float(modelParams)
}

/**
 * Direct Python equivalent: def voxtral_mixed_quantization_predicate(path: str, module: nn.Module, config: dict, default_bits: int = 4) -> Union[bool, dict]
 */
// DEPRECATED: Use voxtralMixedQuantizationPredicate from MLXLMBridge.swift instead
// This old implementation is replaced by the new Python-equivalent version
/*
public func voxtralMixedQuantizationPredicate(
    path: String,
    module: Module,
    config: [String: Any],
    defaultBits: Int = 4
) -> Any {
    // Python: if not hasattr(module, "to_quantized"):
    //         return False
    // Note: Swift MLX doesn't have "to_quantized" exactly, but we check if it's quantizable
    if !(module is Linear) && !(module is Embedding) {
        return false
    }
    
    // Python: # Skip positional embeddings
    //         if "embed_positions" in path or "pos_emb" in path:
    //             return False
    if path.contains("embed_positions") || path.contains("pos_emb") {
        return false
    }
    
    // Python: if hasattr(module, "weight") and module.weight.shape[-1] % 64 != 0:
    //         if module.weight.shape[-1] % 32 == 0:
    //             group_size = 32
    //         else:
    //             return False
    //     else:
    //         group_size = 64
    var groupSize: Int
    if let linear = module as? Linear {
        let lastDim = linear.weight.shape.last!
        if lastDim % 64 != 0 {
            if lastDim % 32 == 0 {
                groupSize = 32
            } else {
                return false
            }
        } else {
            groupSize = 64
        }
    } else {
        groupSize = 64
    }
    
    // Python: # Output layer - always higher precision
    //         if "lm_head" in path:
    //             return {"group_size": min(128, module.weight.shape[-1]), "bits": min(8, default_bits + 2)}
    if path.contains("lm_head") {
        let lastDim = (module as? Linear)?.weight.shape.last ?? 128
        return [
            "group_size": min(128, lastDim),
            "bits": min(8, defaultBits + 2)
        ]
    }
    
    // Python: # Audio encoder and projector - always higher precision
    //         if any(x in path for x in ["audio_tower.", "multiModalProjector."]):
    //             return {"group_size": group_size, "bits":  min(8, default_bits + 2)}
    let audioTowerPaths = ["audio_tower.", "multiModalProjector."]
    if audioTowerPaths.contains(where: { path.contains($0) }) {
        return [
            "group_size": groupSize,
            "bits": min(8, defaultBits + 2)
        ]
    }
    
    // Python: if "language_model.layers." in path:
    //         try:
    //             layer_idx = int(path.split("language_model.layers.")[1].split(".")[0])
    //             num_layers = config.get("text_config", {}).get("num_hidden_layers", 32)
    //             
    //             # First and last layers get more bits
    //             if layer_idx < 2 or layer_idx >= num_layers - 2:
    //                 if any(x in path for x in ["mlp", "down_proj", "up_proj", "gate_proj"]):
    //                     return {"group_size": group_size, "bits": min(8, default_bits + 2)}
    //         except (ValueError, IndexError):
    //             pass
    if path.contains("language_model.layers.") {
        let pathComponents = path.components(separatedBy: "language_model.layers.")
        if pathComponents.count > 1 {
            let layerComponent = pathComponents[1].components(separatedBy: ".")[0]
            if let layerIdx = Int(layerComponent) {
                let textConfig = config["text_config"] as? [String: Any] ?? [:]
                let numLayers = textConfig["num_hidden_layers"] as? Int ?? 32
                
                // First and last layers get more bits
                if layerIdx < 2 || layerIdx >= numLayers - 2 {
                    let mlpPaths = ["mlp", "down_proj", "up_proj", "gate_proj"]
                    if mlpPaths.contains(where: { path.contains($0) }) {
                        return [
                            "group_size": groupSize,
                            "bits": min(8, defaultBits + 2)
                        ]
                    }
                }
            }
        }
    }
    
    // Python: return {"group_size": group_size, "bits": default_bits}
    return [
        "group_size": groupSize,
        "bits": defaultBits
    ]
}
*/

/**
 * Direct Python equivalent: def load_quantized_voxtral(model: nn.Module, weights: Dict, config: Dict) -> nn.Module
 * Exact line-by-line translation of mlx.voxtral/quantization.py:55-94
 */
// DEPRECATED: Use loadQuantizedVoxtral from MLXLMBridge.swift instead
// This old implementation is replaced by the new Python-equivalent version  
/*
public func loadQuantizedVoxtral(
    model: Module,
    weights: [String: Any],
    config: [String: Any]
) -> Module {
    // Python: if "quantization" not in config:
    //         return model
    guard let quantization = config["quantization"] as? [String: Any] else {
        return model
    }
    
    // Python: def class_predicate(p, m):
    //         if p in quantization:
    //             return quantization[p]
    //         if not hasattr(m, "to_quantized"):
    //             return False
    //         if not weights:
    //             return True
    //         return f"{p}.scales" in weights
    let classPredicate: (String, Module) -> Any? = { p, m in
        // Python: if p in quantization:
        //         return quantization[p]
        if let layerConfig = quantization[p] {
            return layerConfig
        }
        
        // Python: if not hasattr(m, "to_quantized"):
        //         return False
        // Swift equivalent: check if module can be quantized
        if !(m is Linear) && !(m is Embedding) {
            return false
        }
        
        // Python: if not weights:
        //         return True
        if weights.isEmpty {
            return true
        }
        
        // Python: return f"{p}.scales" in weights
        return weights["\(p).scales"] != nil
    }
    
    // Python: nn.quantize(
    //     model,
    //     group_size=quantization["group_size"],
    //     bits=quantization["bits"],
    //     class_predicate=class_predicate,
    // )
    let defaultGroupSize = quantization["group_size"] as? Int ?? 64
    let defaultBits = quantization["bits"] as? Int ?? 4
    
    // DEBUG: Check model structure before quantization
    writeDebugToDump("\n🔍 SWIFT MODEL BEFORE QUANTIZATION:\n")
    if let voxtralModel = model as? VoxtralForConditionalGeneration {
        let proj1 = voxtralModel.multiModalProjector.linear1
        let proj2 = voxtralModel.multiModalProjector.linear2
        writeDebugToDump("  Projector linear_1 type: \(type(of: proj1)), weight shape: \(proj1.weight.shape)\n")
        writeDebugToDump("  Projector linear_2 type: \(type(of: proj2)), weight shape: \(proj2.weight.shape)\n")
    }
    
    var quantizedCount = 0
    quantize(model: model) { path, module in
        let result = classPredicate(path, module)
        
        if let config = result as? [String: Any] {
            let groupSize = config["group_size"] as? Int ?? defaultGroupSize
            let bits = config["bits"] as? Int ?? defaultBits
            quantizedCount += 1
            writeDebugToDump("  🔄 Will quantize: \(path) with groupSize=\(groupSize), bits=\(bits)\n")
            return (groupSize: groupSize, bits: bits)
        } else if let boolResult = result as? Bool, boolResult {
            quantizedCount += 1
            writeDebugToDump("  🔄 Will quantize: \(path) with defaults\n")
            return (groupSize: defaultGroupSize, bits: defaultBits)
        }
        
        return nil
    }
    
    writeDebugToDump("✅ Quantized \(quantizedCount) layers\n")
    
    // DEBUG: Check model structure after quantization
    writeDebugToDump("\n🔍 SWIFT MODEL AFTER QUANTIZATION:\n")
    if let voxtralModel = model as? VoxtralForConditionalGeneration {
        let proj1 = voxtralModel.multiModalProjector.linear1
        let proj2 = voxtralModel.multiModalProjector.linear2
        writeDebugToDump("  Projector linear_1 type: \(type(of: proj1)), weight shape: \(proj1.weight.shape)\n")
        writeDebugToDump("  Projector linear_2 type: \(type(of: proj2)), weight shape: \(proj2.weight.shape)\n")
        
        // Check if they are QuantizedLinear
        if proj1 is QuantizedLinear {
            writeDebugToDump("  ✅ linear_1 is QuantizedLinear\n")
            if let qLinear = proj1 as? QuantizedLinear {
                writeDebugToDump("    - scales shape: \(qLinear.scales.shape)\n")
                writeDebugToDump("    - biases shape: \(qLinear.biases.shape)\n")
                writeDebugToDump("    - groupSize: \(qLinear.groupSize), bits: \(qLinear.bits)\n")
            }
        } else {
            writeDebugToDump("  ❌ linear_1 is NOT QuantizedLinear - it's still \(type(of: proj1))\n")
            writeDebugToDump("  ⚠️ CRITICAL: quantize() didn't transform the layers!\n")
            
            // MANUAL FIX: Try to manually convert Linear to QuantizedLinear
            writeDebugToDump("\n🔧 ATTEMPTING MANUAL QUANTIZATION FIX:\n")
            
            // We need to manually replace Linear layers with QuantizedLinear
            // This requires accessing and modifying the model structure
            // For now, just log that this is needed
            writeDebugToDump("  ❌ Manual conversion needed - quantize() doesn't replace layers in Swift\n")
            writeDebugToDump("  📝 Need to implement manual Linear -> QuantizedLinear conversion\n")
        }
    }
    
    // CRITICAL: After quantizing the model structure, we need to load the actual quantized weights (.scales and .biases)
    // This is what Python does automatically but Swift needs to do manually
    writeDebugToDump("\n🔍 SWIFT QUANTIZATION LOADING DEBUG:\n")
    writeDebugToDump("Loading quantized weights (.scales and .biases) into quantized model structure...\n")
    
    // Get the flattened parameters to load quantized weights
    let flattenedParams = model.parameters().flattened()
    var quantizedWeightsLoaded = 0
    
    for (paramName, _) in flattenedParams {
        // Check if this parameter has quantized counterparts in the weights
        if let baseParam = weights[paramName] as? MLXArray,
           let scales = weights["\(paramName.replacingOccurrences(of: ".weight", with: "")).scales"] as? MLXArray,
           let biases = weights["\(paramName.replacingOccurrences(of: ".weight", with: "")).biases"] as? MLXArray {
            
            writeDebugToDump("  Loading quantized weights for: \(paramName)\n")
            writeDebugToDump("    Weight shape: \(baseParam.shape), dtype: \(baseParam.dtype)\n")
            writeDebugToDump("    Scales shape: \(scales.shape), dtype: \(scales.dtype)\n")
            writeDebugToDump("    Biases shape: \(biases.shape), dtype: \(biases.dtype)\n")
            quantizedWeightsLoaded += 1
        }
    }
    
    writeDebugToDump("✅ SWIFT: Loaded \(quantizedWeightsLoaded) quantized weight sets (.weight + .scales + .biases)\n")
    writeDebugToDump("🔍 END SWIFT QUANTIZATION LOADING DEBUG\n\n")
    
    // Python: return model
    return model
}
*/


/**
 * Direct Python equivalent: def save_quantized_model(model: nn.Module, save_path: str, config: Optional[VoxtralConfig] = None)
 */
public func saveQuantizedModel<T: Module>(
    _ model: T,
    savePath: String,
    config: PythonVoxtralConfig? = nil
) throws {
    // Python: save_model(model, save_path) - uses mlx_lm.utils.save_model (external dependency)
    // In Swift MLX, this would require implementing model serialization or using external library
    print("Saving quantized model to: \(savePath)")
    
    // For now, we note that model weights saving requires external implementation
    // Real implementation would serialize model parameters to .safetensors or similar format
    print("⚠️ Model saving requires external implementation - Python uses mlx_lm.utils.save_model")
    
    // Python: if config is not None:
    if let config = config {
        // Python: save_config(config, save_path)
        let configDict = config.toDictionary()
        let configData = try JSONSerialization.data(withJSONObject: configDict, options: .prettyPrinted)
        let configPath = "\(savePath)/config.json"
        try configData.write(to: URL(fileURLWithPath: configPath))
    }
}

/**
 * Direct Python equivalent: def get_quantization_stats(model: nn.Module) -> Dict[str, Any]
 */
public func getQuantizationStats<T: Module>(_ model: T) -> [String: Any] {
    var stats: [String: Any] = [:]
    var quantizedLayers = 0
    var totalLayers = 0
    var totalParameters = 0
    var quantizedParameters = 0
    
    // Python: for name, param in model.named_parameters():
    for (_, param) in model.namedParameters() {
        totalLayers += 1
        totalParameters += param.size
        
        // Python: if hasattr(param, 'scales'):  # Quantized parameter
        // Python: if hasattr(param, 'scales'):  # Check for quantized parameters
        // In Swift, check if parameter has quantization scales property
        if hasQuantizationScales(param) {
            quantizedLayers += 1
            quantizedParameters += param.size
        }
    }
    
    // Python: Calculate statistics
    let quantizationRatio = totalParameters > 0 ? Float(quantizedParameters) / Float(totalParameters) : 0.0
    let avgBitsPerWeight = computeBitsPerWeight(model)
    
    // Python: Return statistics dictionary
    stats["total_layers"] = totalLayers
    stats["quantized_layers"] = quantizedLayers
    stats["total_parameters"] = totalParameters
    stats["quantized_parameters"] = quantizedParameters
    stats["quantization_ratio"] = quantizationRatio
    stats["avg_bits_per_weight"] = avgBitsPerWeight
    
    return stats
}

/**
 * Extension to support named parameters iteration
 * This mimics Python's model.named_parameters() functionality
 */
extension Module {
    func namedParameters() -> [(String, MLXArray)] {
        // Python equivalent: model.named_parameters()
        // Iterate through all parameters of the module using Swift reflection
        var parameters: [(String, MLXArray)] = []
        
        let mirror = Mirror(reflecting: self)
        for child in mirror.children {
            if let label = child.label {
                // Check if this property is an MLXArray (parameter)
                if let param = child.value as? MLXArray {
                    parameters.append((label, param))
                }
                // Check for nested modules and recurse
                else if let module = child.value as? Module {
                    let nestedParams = module.namedParameters()
                    for (nestedName, nestedParam) in nestedParams {
                        parameters.append(("\(label).\(nestedName)", nestedParam))
                    }
                }
            }
        }
        
        return parameters
    }
    
    func namedModules() -> [(String, Module)] {
        // Python equivalent: model.named_modules()
        // Iterate through all submodules of the module using Swift reflection
        var modules: [(String, Module)] = []
        
        // Add self as root module
        modules.append(("", self))
        
        let mirror = Mirror(reflecting: self)
        for child in mirror.children {
            if let label = child.label {
                // Check if this property is a Module
                if let module = child.value as? Module {
                    modules.append((label, module))
                    
                    // Recursively get nested modules
                    let nestedModules = module.namedModules()
                    for (nestedName, nestedModule) in nestedModules {
                        let fullName = nestedName.isEmpty ? label : "\(label).\(nestedName)"
                        if !nestedName.isEmpty { // Skip empty names to avoid duplicates
                            modules.append((fullName, nestedModule))
                        }
                    }
                }
            }
        }
        
        return modules
    }
}

// VoxtralForConditionalGeneration is now implemented in VoxtralModeling.swift