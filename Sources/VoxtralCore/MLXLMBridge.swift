/**
 * MLXLMBridge.swift
 * 
 * ÉQUIVALENTS MLX-LM POUR SWIFT
 * =============================
 * 
 * Ce fichier contient les équivalents Swift des fonctions MLX-LM utilisées dans le code Python :
 * 
 * RÉFÉRENCES PYTHON MLX-LM :
 * - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
 * - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py  
 * - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py
 * - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py
 * 
 * CONVERSIONS IDENTIFIÉES :
 * 1. KVCache (mlx_lm.models.cache.KVCache)
 * 2. create_attention_mask (mlx_lm.models.base.create_attention_mask)
 * 3. scaled_dot_product_attention (mlx_lm.models.base.scaled_dot_product_attention)
 * 4. initialize_rope (mlx_lm.models.rope_utils.initialize_rope)
 * 5. quantize_model, save_config, save_model (mlx_lm.utils)
 * 6. get_model_path (mlx_lm.utils.get_model_path)
 */

import Foundation
import MLX
import MLXNN

// MARK: - 1. KVCache - DISABLED (Using MLXLMCommon KVCache instead)

/**
 * ARCHIVED: Our custom KVCache implementation has been disabled
 * We now use the official KVCache protocol from MLXLMCommon
 * Original implementation archived in MLXLMBridge.swift.archived
 * 
 * The conflict was: our class KVCache vs MLXLMCommon's protocol KVCache
 * This was causing the LanguageModel protocol conformance issues
 */

// Import MLXLMCommon to use their KVCache protocol
import MLXLMCommon

// MARK: - 2. create_attention_mask (mlx_lm.models.base.create_attention_mask)

/**
 * Direct Python equivalent: from mlx_lm.models.base import create_attention_mask
 * Python source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py
 */
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). The cache builds the masks (K-3).")
public func mlxLMCreateAttentionMask(_ x: MLXArray, cache: [any KVCache]? = nil) -> MLXArray? {
    // Python: def create_attention_mask(h: mx.array, cache: Optional[Any] = None, return_array: bool = False)
    // Exact translation of mlx_lm.models.llama.create_attention_mask
    
    let T = x.shape[1]  // sequence length
    
    // Python: if T > 1:
    if T > 1 {
        var offset = 0
        var windowSize: Int? = nil
        
        // Python: if cache is not None and cache[0] is not None:
        if let cache = cache, !cache.isEmpty, let firstCache = cache.first {
            // Python: c = cache[0]
            // Python: offset = c.offset
            offset = firstCache.offset
            
            // Python: if hasattr(c, "max_size"):
            //         window_size = c.max_size
            //         offset = min(window_size, offset)
            if let maxSize = firstCache.maxSize {
                windowSize = maxSize
                offset = min(maxSize, offset)
            }
        }
        
        // Python: if return_array:
        //         return create_causal_mask(T, offset, window_size=window_size)
        //     else:
        //         return "causal"
        
        // Always an array (Python returns "causal" here)
        // Create causal mask with offset
        return createCausalMask(N: T, offset: offset, windowSize: windowSize)
        
    } else {
        // Python: mask = None
        return nil
    }
}

// MARK: - 3. scaled_dot_product_attention (mlx_lm.models.base.scaled_dot_product_attention)

/**
 * Direct Python equivalent: from mlx_lm.models.base import scaled_dot_product_attention  
 * Python source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py
 */
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). Use MLXFast.scaledDotProductAttention.")
public func mlxLMScaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray, 
    values: MLXArray,
    mask: MLXArray? = nil,
    scale: Float? = nil
) -> MLXArray {
    // Python: def scaled_dot_product_attention(queries, keys, values, mask=None, scale=None)
    
    let _ = queries.shape[0] // batch size
    let _ = queries.shape[1] // sequence length  
    let _ = keys.shape[1]    // key sequence length
    let D = queries.shape[2] // head dimension
    
    // Calculate scale factor
    let actualScale = scale ?? (1.0 / sqrt(Float(D)))
    
    // Compute attention scores: Q @ K^T  
    let scores = MLX.matmul(queries, keys.swappedAxes(-2, -1)) * actualScale
    
    // Apply mask if provided
    var maskedScores = scores
    if let mask = mask {
        // Apply mask by setting masked positions to very negative values
        let maskedValue = MLXArray(-1e9, dtype: scores.dtype)
        maskedScores = MLX.where(mask, scores, maskedValue)
    }
    
    // Apply softmax to get attention weights
    let attentionWeights = MLX.softmax(maskedScores, axis: -1)
    
    // Apply attention weights to values: attention @ V
    let output = MLX.matmul(attentionWeights, values)
    
    return output
}

// MARK: - 4. initialize_rope (mlx_lm.models.rope_utils.initialize_rope)

/**
 * Direct Python equivalent: from mlx_lm.models.rope_utils import initialize_rope
 * Python source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py
 */
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). Use MLXNN.RoPE.")
public func mlxLMInitializeRope(
    headDim: Int,
    maxPositionEmbeddings: Int = 2048,
    theta: Float = 10000.0
) -> (cos: MLXArray, sin: MLXArray) {
    // Python: def initialize_rope(head_dim, max_position_embeddings=2048, theta=10000.0)
    
    // Create frequency bands
    let indices = MLXArray(0..<(headDim/2)).asType(.float32)
    let invFreq = pow(theta, -2.0 * indices / Float(headDim))
    
    // Create position indices  
    let positions = MLXArray(0..<maxPositionEmbeddings).asType(.float32)
    
    // Outer product to get all position-frequency combinations
    let freqs = matmul(positions.expandedDimensions(axis: -1), invFreq.expandedDimensions(axis: 0))
    
    // Compute cos and sin embeddings
    let cos = cos(freqs)
    let sin = sin(freqs)
    
    return (cos: cos, sin: sin)
}

// MARK: - 5. MLX-LM Utils (mlx_lm.utils)

/**
 * Direct Python equivalent: import mlx_lm.utils
 * Python source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py
 */

// These functions are already implemented in VoxtralQuantization.swift:
// - quantize_model → quantizeModel
// - save_config → saveConfig  
// - save_model → saveModel
// - compute_bits_per_weight → computeBitsPerWeight

/**
 * Direct Python equivalent: from mlx_lm.utils import get_model_path
 */
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). Use ModelDownloader.resolveModel(_:).")
public func mlxLMGetModelPath(_ modelName: String) -> String {
    // Python: def get_model_path(model_name: str) -> str
    // This function resolves model names to local paths or downloads them
    
    // Treats the argument as a local path; the Python version would:
    // 1. Check if it's a local path
    // 2. Check HuggingFace cache
    // 3. Download if needed
    return modelName
}

// MARK: - VoxtralModel (Corresponds to VoxtralForConditionalGeneration)

/**
 * Main Voxtral model class - equivalent to Python VoxtralForConditionalGeneration
 * This was previously in VoxtralChat.swift but belongs in the core conversion
 */
public typealias VoxtralModel = VoxtralForConditionalGeneration

/**
 * Extension to provide convenience initializer matching the deleted VoxtralChat.swift usage
 */
public extension VoxtralForConditionalGeneration {
    
    /**
     * Convenience initializer that loads a model from path
     * Python equivalent: model = VoxtralForConditionalGeneration.from_pretrained(path)
     */
    @available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). Use VoxtralPipeline.loadModel() or loadVoxtralStandardModel(modelPath:).")
    convenience init(path: String) throws {
        // Python: model, _ = load_voxtral_model(model_id, dtype=dtype)
        // Use the SAME function as Python - no multiple versions!
        let (loadedModel, _) = try loadVoxtralModel(modelPath: path, dtype: .float16)
        
        // Initialize self with the loaded model's config
        self.init(config: loadedModel.config)
        
        // CRITICAL: For quantized models, copy the entire structure not just modules
        // Copying individual modules breaks QuantizedLinear structure
        writeDebugToDump("\n🔍 MLX LM BRIDGE MODULE REPLACEMENT DEBUG:\n")
        writeDebugToDump("Checking if loadedModel has quantized layers...\n")
        
        let hasQuantizedLayers = loadedModel.children().contains { child in
            String(describing: child).contains("QuantizedLinear")
        }
        writeDebugToDump("Has quantized layers: \(hasQuantizedLayers)\n")
        
        // Debug loaded model structure BEFORE copying
        writeDebugToDump("🔍 LOADED MODEL STRUCTURE (before copying to self):\n")
        let loadedProj1 = loadedModel.multiModalProjector.linear1
        let loadedProj2 = loadedModel.multiModalProjector.linear2
        writeDebugToDump("  Loaded projector linear_1 type: \(type(of: loadedProj1))\n")
        writeDebugToDump("  Loaded projector linear_1 weight shape: \(loadedProj1.weight.shape)\n")
        writeDebugToDump("  Loaded projector linear_2 type: \(type(of: loadedProj2))\n")
        writeDebugToDump("  Loaded projector linear_2 weight shape: \(loadedProj2.weight.shape)\n")
        
        writeDebugToDump("🔧 SOLUTION: Using update(modules:) but checking if structure is preserved\n")
        
        // We MUST use update(modules:) - no other choice in Swift MLX
        let loadedModules = loadedModel.children()
        writeDebugToDump("About to call self.update(modules:) with \(loadedModules.count) modules\n")
        
        self.update(modules: loadedModules)
        writeDebugToDump("✅ update(modules:) completed\n")
        
        // Check what happened to our structure
        writeDebugToDump("🔍 POST-UPDATE CHECK: Verifying if QuantizedLinear structure survived\n")
        let postUpdateProj1 = self.multiModalProjector.linear1
        writeDebugToDump("After update() - projector linear_1 type: \(type(of: postUpdateProj1))\n")
        
        if String(describing: type(of: postUpdateProj1)).contains("QuantizedLinear") {
            writeDebugToDump("✅ SUCCESS: QuantizedLinear structure SURVIVED update(modules:)\n")
        } else {
            writeDebugToDump("❌ FAILURE: QuantizedLinear structure was LOST during update(modules:)\n")
            writeDebugToDump("This means the problem is deeper - update(modules:) itself breaks quantization\n")
        }
        
        // Debug SELF structure AFTER copying
        writeDebugToDump("🔍 SELF MODEL STRUCTURE (after copying from loaded):\n")
        let selfProj1 = self.multiModalProjector.linear1
        let selfProj2 = self.multiModalProjector.linear2
        writeDebugToDump("  Self projector linear_1 type: \(type(of: selfProj1))\n")
        writeDebugToDump("  Self projector linear_1 weight shape: \(selfProj1.weight.shape)\n")
        writeDebugToDump("  Self projector linear_2 type: \(type(of: selfProj2))\n")
        writeDebugToDump("  Self projector linear_2 weight shape: \(selfProj2.weight.shape)\n")
        writeDebugToDump("🔍 END MLX LM BRIDGE MODULE REPLACEMENT DEBUG\n\n")
        
        // 🎯 CRITICAL FIX: Copy the actual parameters from loadedModel
        // update(modules:) only copies the module structure, not the parameters!
        writeDebugToDump("🔧 FIXING: Copying parameters from loadedModel to self\n")
        let loadedParams = loadedModel.parameters()
        self.update(parameters: loadedParams)
        writeDebugToDump("✅ Parameters copied from loadedModel\n")
        
        // ✅ CRITICAL: DO NOT convert dtype - this destroys numerical accuracy!
        // The embed_tokens weights are already correct from loadedModel parameters
        writeDebugToDump("✅ PRESERVING NUMERICAL ACCURACY: No dtype conversion applied to embed_tokens\n")
        
        // Verify final state
        writeDebugToDump("🔍 FINAL STATE:\n")
        for (key, value) in self.parameters().flattened() {
            if key.contains("embed_tokens") {
                writeDebugToDump("  FINAL \(key) dtype=\(value.dtype)\n")
            }
        }
        
        // CRITICAL DEBUGGING: Focus on weight transformation process 
        debugSwiftWeightTransformation()
    }
    
    /**
     * Debug weight transformation process: raw safetensors → final model
     * Focus on the corruption happening between ÉTAPE 5 and final model
     */
    private func debugSwiftWeightTransformation() {
        writeDebugToDump("🎯 WEIGHT TRANSFORMATION AUDIT: Raw Safetensors → Final Model\n")
        writeDebugToDump("============================================================\n")
        
        // Focus on multiModalProjector.linear1 weight transformation
        let proj1 = self.multiModalProjector.linear1
        writeDebugToDump("\n📊 FINAL MODEL - multiModalProjector.linear1:\n")
        writeDebugToDump("   Type: \(type(of: proj1))\n")
        writeDebugToDump("   Weight shape: \(proj1.weight.shape), dtype: \(proj1.weight.dtype)\n")
        
        let finalWeightFirst3 = extractFirst3Values(from: proj1.weight)
        writeDebugToDump("   FINAL Weight first 3 values: \(finalWeightFirst3)\n")
        writeDebugToDump("   Expected (from raw safetensors): [\"762978296\", \"904439080\", \"2753987014\"]\n")
        
        // Check if values match raw
        let expectedRaw = ["762978296", "904439080", "2753987014"]
        let matches = finalWeightFirst3.enumerated().map { index, value in
            value == expectedRaw[index]
        }
        writeDebugToDump("   Values match raw safetensors: \(matches)\n")
        
        if !matches.allSatisfy({ $0 }) {
            writeDebugToDump("   🚨 CORRUPTION DETECTED: Values changed from raw to final!\n")
        } else {
            writeDebugToDump("   ✅ Values preserved from raw to final\n")
        }
        
        writeDebugToDump("\n✅ WEIGHT TRANSFORMATION AUDIT COMPLETE\n")
        writeDebugToDump("🎯 Expected Python pattern:\n")
        writeDebugToDump("  Raw safetensors: [\"762978296\", \"904439080\", \"2753987014\"]\n")
        writeDebugToDump("  Python final:    [\"762978296\", \"904439080\", \"2753987014\"]\n")
        writeDebugToDump("  Note: Python quantized step corrupts but final restores values\n")
    }
    
    // Helper functions for numerical debugging
    private func extractFirst10Values(from array: MLXArray) -> [String] {
        let count = min(10, array.count)
        var values: [String] = []
        for i in 0..<count {
            // Preserve exact values based on dtype
            switch array.dtype {
            case .uint32:
                let val = array[i].item(UInt32.self)
                values.append(String(val))
            case .float16:
                let val = array[i].asType(.float32).item(Float.self)
                values.append(String(val))
            case .float32:
                let val = array[i].item(Float.self)
                values.append(String(val))
            case .int32:
                let val = array[i].item(Int32.self)
                values.append(String(val))
            default:
                let val = array[i].asType(.float32).item(Float.self)
                values.append(String(val))
            }
        }
        return values
    }
    
    private func extractLast10Values(from array: MLXArray) -> [String] {
        let count = min(10, array.count)
        let startIndex = array.count - count
        var values: [String] = []
        for i in startIndex..<array.count {
            // Preserve exact values based on dtype
            switch array.dtype {
            case .uint32:
                let val = array[i].item(UInt32.self)
                values.append(String(val))
            case .float16:
                let val = array[i].asType(.float32).item(Float.self)
                values.append(String(val))
            case .float32:
                let val = array[i].item(Float.self)
                values.append(String(val))
            case .int32:
                let val = array[i].item(Int32.self)
                values.append(String(val))
            default:
                let val = array[i].asType(.float32).item(Float.self)
                values.append(String(val))
            }
        }
        return values
    }
    
    private func computeStats(for array: MLXArray) -> (min: Float, max: Float, mean: Float) {
        let minVal = array.min().asType(.float32).item(Float.self)
        let maxVal = array.max().asType(.float32).item(Float.self)
        let meanVal = array.mean().asType(.float32).item(Float.self)
        return (min: minVal, max: maxVal, mean: meanVal)
    }
    
    
    // Helper pour extraire 3 première valeurs d'un tensor
    private func extractFirst3Values(from array: MLXArray) -> [String] {
        let count = min(3, array.count)
        var values: [String] = []
        let flattened = array.flattened()
        
        for i in 0..<count {
            // Preserve exact values based on dtype
            switch array.dtype {
            case .uint32:
                // For uint32, extract exact integer value without Float conversion
                let val = flattened[i].item(UInt32.self)
                values.append(String(val))
            case .float16, .float32:
                // For float types, use Float conversion
                let val = flattened[i].asType(.float32).item(Float.self)
                values.append(String(val))
            case .int32:
                let val = flattened[i].item(Int32.self)
                values.append(String(val))
            case .int64:
                let val = flattened[i].item(Int64.self)
                values.append(String(val))
            default:
                // Default to Float conversion for other types
                let val = flattened[i].asType(.float32).item(Float.self)
                values.append(String(val))
            }
        }
        return values
    }
    
}

// MARK: - MLX Voxtral Quantization Functions
// Direct Swift equivalents of mlx.voxtral/quantization.py

/**
 * Direct Python equivalent: compute_bits_per_weight(model)
 * Python source: mlx.voxtral/quantization.py lines 17-42
 */
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23).")
public func computeBitsPerWeight(_ model: Module) -> Float {
    // Python: def compute_bits_per_weight(model)
    // Handle different mlx versions like Python does
    
    let params = model.parameters()
    
    // Python equivalent: tree_reduce to sum all nbytes
    var modelBytes = 0
    var modelParams = 0
    
    func processParameter(_ key: String, _ value: Any) {
        if let array = value as? MLXArray {
            modelBytes += array.nbytes
            modelParams += array.size
        } else if let dict = value as? [String: Any] {
            for (subKey, subValue) in dict {
                processParameter("\(key).\(subKey)", subValue)
            }
        }
    }
    
    // Traverse parameter tree like Python tree_flatten + tree_reduce
    for (key, value) in params {
        processParameter(key, value)
    }
    
    if modelParams == 0 {
        return 0.0
    }
    
    // Python: return model_bytes * 8 / model_params
    return Float(modelBytes * 8) / Float(modelParams)
}

/**
 * Direct Python equivalent: voxtral_mixed_quantization_predicate()
 * Python source: mlx.voxtral/quantization.py lines 97-139
 */
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23).")
public func voxtralMixedQuantizationPredicate(
    path: String,
    module: Module,
    config: [String: Any],
    defaultBits: Int = 4
) -> [String: Any]? {
    // Python: if not hasattr(module, "to_quantized"): return False
    // In Swift MLX, Linear and Embedding modules can be quantized
    guard module is Linear || module is Embedding else {
        return nil
    }
    
    // Python: Skip positional embeddings
    if path.contains("embed_positions") || path.contains("pos_emb") {
        return nil
    }
    
    // Python: Dynamic group_size based on weight dimensions
    var groupSize = 64  // Default
    if let linear = module as? Linear {
        let lastDim = linear.weight.shape.last ?? 0
        if lastDim % 64 != 0 {
            if lastDim % 32 == 0 {
                groupSize = 32
            } else {
                return nil  // Can't quantize
            }
        }
    }
    
    // Python: Output layer - always higher precision
    if path.contains("lm_head") {
        let weightLastDim = (module as? Linear)?.weight.shape.last ?? groupSize
        return [
            "group_size": min(128, weightLastDim),
            "bits": min(8, defaultBits + 2)
        ]
    }
    
    // Python: Audio encoder and projector - always higher precision  
    if path.contains("audio_tower.") || path.contains("multiModalProjector.") {
        return [
            "group_size": groupSize,
            "bits": min(8, defaultBits + 2)
        ]
    }
    
    // Python: Language model layers logic
    if path.contains("language_model.layers.") {
        // Extract layer index like Python
        let components = path.components(separatedBy: "language_model.layers.")
        if components.count > 1 {
            let layerPart = components[1].components(separatedBy: ".")[0]
            if let layerIdx = Int(layerPart) {
                let numLayers = (config["text_config"] as? [String: Any])?["num_hidden_layers"] as? Int ?? 32
                
                // Python: First and last layers get more bits
                if layerIdx < 2 || layerIdx >= numLayers - 2 {
                    if path.contains("mlp") || path.contains("down_proj") || path.contains("up_proj") || path.contains("gate_proj") {
                        return [
                            "group_size": groupSize,
                            "bits": min(8, defaultBits + 2)
                        ]
                    }
                }
            }
        }
    }
    
    // Python: Default quantization
    return ["group_size": groupSize, "bits": defaultBits]
}

/**
 * Direct Python equivalent: load_quantized_voxtral()
 * Python source: mlx.voxtral/quantization.py lines 55-94
 */
public func loadQuantizedVoxtral(
    model: Module,
    weights: [String: MLXArray],
    config: [String: Any]
) -> Module {
    // Python: if "quantization" not in config: return model
    guard let quantization = config["quantization"] as? [String: Any] else {
        return model
    }
    
    // Python: def class_predicate(p, m):
    func classPredicate(path: String, module: Module) -> [String: Any]? {
        // SKIP embed_tokens quantization - MLXLMCommon doesn't handle QuantizedEmbedding properly
        // This is a workaround for Swift MLX limitations
        if path.contains("embed_tokens") {
            return nil  // Don't quantize embed_tokens
        }
        
        // Python: if p in quantization: return quantization[p]
        if let layerConfig = quantization[path] as? [String: Any] {
            return layerConfig
        }
        
        // Python: if not hasattr(m, "to_quantized"): return False  
        // In Swift MLX, Linear and Embedding modules can be quantized
        guard module is Linear || module is Embedding else {
            return nil
        }
        
        // Python: if not weights: return True
        if weights.isEmpty {
            return [
                "group_size": quantization["group_size"]!,
                "bits": quantization["bits"]!
            ]
        }
        
        // Python: return f"{p}.scales" in weights
        let scalesKey = "\(path).scales"
        return weights[scalesKey] != nil ? [
            "group_size": quantization["group_size"]!,
            "bits": quantization["bits"]!
        ] : nil
    }
    
    // Python: nn.quantize(model, group_size=..., bits=..., class_predicate=...)
    // A malformed quantization block leaves the model unquantized: the verified weight load then reports the
    // shape mismatches as an error instead of a crash here (K-27)
    guard let globalGroupSize = quantization["group_size"] as? Int, let globalBits = quantization["bits"] as? Int else {
        VoxtralDebug.always("quantization config without integer group_size/bits: model left unquantized")
        return model
    }

    // Use the filter that returns per-layer (groupSize, bits, mode) for mixed quantization support
    MLXNN.quantize(
        model: model,
        filter: { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
            // Get per-layer config from classPredicate
            guard let quantParams = classPredicate(path: path, module: module) else {
                return nil  // Don't quantize this layer
            }

            // Use per-layer config if available, otherwise fall back to global
            let layerGroupSize = quantParams["group_size"] as? Int ?? globalGroupSize
            let layerBits = quantParams["bits"] as? Int ?? globalBits

            return (groupSize: layerGroupSize, bits: layerBits, mode: .affine)
        }
    )

    return model
}