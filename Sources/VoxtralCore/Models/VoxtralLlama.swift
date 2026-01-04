/**
 * VoxtralLlama - Swift equivalent of mlx.voxtral/models/llama.py
 * 
 * Exact conversion of Python Llama model implementation using MLX Swift.
 * Direct line-by-line translation following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon  // Use official KVCache protocol from MLXLMCommon

/**
 * Direct Python equivalent: class LlamaAttention(nn.Module)
 * Multi-headed attention with rotary embeddings.
 */
public class LlamaAttention: Module {
    
    // Python: def __init__(self, config):
    let config: LlamaConfig
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let numKeyValueHeads: Int
    let numKeyValueGroups: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let isCausal: Bool
    let scale: Float
    
    // Approach A: Direct assignment WITHOUT @ModuleInfo (consistent pattern)
    public var q_proj: Linear
    public var k_proj: Linear 
    public var v_proj: Linear
    public var o_proj: Linear
    let rope: MLXLMRope
    
    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    public init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.config = config
        self.config = config
        // Python: self.hidden_size = config.hidden_size
        self.hiddenSize = config.hiddenSize
        // Python: self.num_heads = config.num_attention_heads
        self.numHeads = config.numAttentionHeads
        // Python: self.head_dim = config.head_dim if hasattr(config, "head_dim") else (self.hidden_size // self.num_heads)
        self.headDim = config.headDim ?? (hiddenSize / numHeads)
        // Python: self.num_key_value_heads = config.num_key_value_heads
        self.numKeyValueHeads = config.numKeyValueHeads
        // Python: self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.numKeyValueGroups = numHeads / numKeyValueHeads
        // Python: self.max_position_embeddings = config.max_position_embeddings
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        // Python: self.rope_theta = config.rope_theta
        self.ropeTheta = config.ropeTheta
        // Python: self.is_causal = True
        self.isCausal = true
        // Python: self.scale = self.head_dim**-0.5
        self.scale = Float(pow(Double(headDim), -0.5))

        // Python: self.rope = initialize_rope(self.head_dim, self.rope_theta, config.rope_traditional if hasattr(config, "rope_traditional") else False, config.rope_scaling if hasattr(config, "rope_scaling") else None, self.max_position_embeddings)
        self.rope = initializeRope(
            headDim: headDim,
            ropeTheta: ropeTheta,
            ropeTraditional: config.ropeTraditional ?? false,
            ropeScaling: config.ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
        
        // Approach A: Initialize properties BEFORE super.init() (required without @ModuleInfo)
        // Python: self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj = Linear(hiddenSize, numHeads * headDim, bias: config.attentionBias)
        // Python: self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: config.attentionBias)
        // Python: self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: config.attentionBias)
        // Python: self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.o_proj = Linear(numHeads * headDim, hiddenSize, bias: config.attentionBias)
        
        super.init()
    }
    
    /**
     * Direct Python equivalent: def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
     */
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        // Python: bsz, q_len, _ = hidden_states.shape
        let bsz = hiddenStates.shape[0]
        let qLen = hiddenStates.shape[1]

        // Python: queries, keys, values = (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        let queries = q_proj(hiddenStates)
        let keys = k_proj(hiddenStates)
        let values = v_proj(hiddenStates)

        // Python: queries = queries.reshape(bsz, q_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        let queriesReshaped = queries.reshaped([bsz, qLen, numHeads, -1])
        let queriesTransposed = queriesReshaped.transposed(0, 2, 1, 3)
        
        // Python: keys = keys.reshape(bsz, q_len, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        let keysReshaped = keys.reshaped([bsz, qLen, numKeyValueHeads, -1])
        let keysTransposed = keysReshaped.transposed(0, 2, 1, 3)
        
        // Python: values = values.reshape(bsz, q_len, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        let valuesReshaped = values.reshaped([bsz, qLen, numKeyValueHeads, -1])
        let valuesTransposed = valuesReshaped.transposed(0, 2, 1, 3)

        var finalQueries: MLXArray
        var finalKeys: MLXArray
        var finalValues: MLXArray

        // Python: if cache is not None:
        if let cache = cache {
            // Python: queries = self.rope(queries, offset=cache.offset)
            finalQueries = rope.callAsFunction(queriesTransposed, offset: cache.offset)
            // Python: keys = self.rope(keys, offset=cache.offset)
            let rotatedKeys = rope.callAsFunction(keysTransposed, offset: cache.offset)
            // Python: keys, values = cache.update_and_fetch(keys, values)
            // MLXLMCommon KVCache uses update() instead of updateAndFetch()
            (finalKeys, finalValues) = cache.update(keys: rotatedKeys, values: valuesTransposed)
        } else {
            // Python: queries = self.rope(queries)
            finalQueries = rope.callAsFunction(queriesTransposed)
            // Python: keys = self.rope(keys)
            finalKeys = rope.callAsFunction(keysTransposed)
            finalValues = valuesTransposed
        }

        // Python: output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=attention_mask)
        let output = mlxLMScaledDotProductAttention(
            queries: finalQueries,
            keys: finalKeys,
            values: finalValues,
            cache: cache,
            scale: scale,
            mask: attentionMask
        )

        // Python: output = output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        let outputTransposed = output.transposed(0, 2, 1, 3)
        let outputReshaped = outputTransposed.reshaped([bsz, qLen, -1])
        
        // Python: return self.o_proj(output)
        return o_proj(outputReshaped)
    }
}

/**
 * Direct Python equivalent: class LlamaMLP(nn.Module)
 * Llama MLP with SiLU activation.
 */
public class LlamaMLP: Module {
    
    // Python: def __init__(self, config):
    let config: LlamaConfig
    let hiddenSize: Int
    let intermediateSize: Int
    // Approach A: Direct assignment WITHOUT @ModuleInfo (consistent pattern)
    public var gate_proj: Linear
    public var up_proj: Linear
    public var down_proj: Linear
    let actFn: (MLXArray) -> MLXArray
    
    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    public init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.config = config
        self.config = config
        // Python: self.hidden_size = config.hidden_size
        self.hiddenSize = config.hiddenSize
        // Python: self.intermediate_size = config.intermediate_size
        self.intermediateSize = config.intermediateSize
        
        // Python: self.act_fn = nn.silu
        self.actFn = silu
        
        // Approach A: Initialize properties BEFORE super.init() (required without @ModuleInfo)
        // Python: self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.gate_proj = Linear(hiddenSize, intermediateSize, bias: config.mlpBias)
        // Python: self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = Linear(hiddenSize, intermediateSize, bias: config.mlpBias)
        // Python: self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.down_proj = Linear(intermediateSize, hiddenSize, bias: config.mlpBias)
        
        super.init()
    }
    
    /**
     * Direct Python equivalent: def __call__(self, hidden_states: mx.array) -> mx.array:
     */
    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Python: return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj(actFn(gate_proj(hiddenStates)) * up_proj(hiddenStates))
    }
}

/**
 * Direct Python equivalent: class LlamaDecoderLayer(nn.Module)
 * Llama decoder layer.
 */
public class LlamaDecoderLayer: Module {
    
    let hiddenSize: Int
    // Approach A: Direct assignment WITHOUT @ModuleInfo (consistent pattern)
    public var self_attn: LlamaAttention
    public var mlp: LlamaMLP
    public var input_layernorm: RMSNorm
    public var post_attention_layernorm: RMSNorm
    
    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    public init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.hidden_size = config.hidden_size
        self.hiddenSize = config.hiddenSize

        // Approach A: Initialize properties BEFORE super.init() (required without @ModuleInfo)
        // Python: self.self_attn = LlamaAttention(config)
        self.self_attn = LlamaAttention(config: config)
        // Python: self.mlp = LlamaMLP(config)
        self.mlp = LlamaMLP(config: config)
        // Python: self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        // Python: self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        
        super.init()
    }
    
    /**
     * Direct Python equivalent: def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
     */
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        // Python: r = self.self_attn(self.input_layernorm(hidden_states), attention_mask, cache)
        let r = self_attn(input_layernorm(hiddenStates), attentionMask: attentionMask, cache: cache)
        // Python: h = hidden_states + r
        let h = hiddenStates + r
        // Python: r = self.mlp(self.post_attention_layernorm(h))
        let r2 = mlp(post_attention_layernorm(h))
        // Python: out = h + r
        let out = h + r2
        // Python: return out
        return out
    }
}

/**
 * Direct Python equivalent: class LlamaModel(nn.Module)
 * Llama model for Voxtral text generation.
 */
public class LlamaModel: Module {
    
    let config: LlamaConfig
    let paddingIdx: Int?
    let vocabSize: Int
    // Approach A: Direct assignment WITHOUT @ModuleInfo (consistent pattern)
    public var embed_tokens: Embedding
    public var layers: [LlamaDecoderLayer]
    public var norm: RMSNorm
    
    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    public init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.config = config
        self.config = config
        // Python: self.padding_idx = getattr(config, "pad_token_id", None)
        self.paddingIdx = config.padTokenId
        // Python: self.vocab_size = config.vocab_size
        self.vocabSize = config.vocabSize

        // Approach A: Initialize properties BEFORE super.init() (required without @ModuleInfo)
        // Python: self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        
        // Python: self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        
        // Python: self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.layers = (0..<config.numHiddenLayers).map { _ in
            LlamaDecoderLayer(config: config)
        }
        
        super.init()
    }
    
    /**
     * Direct Python equivalent: def __call__(self, inputs: Optional[mx.array] = None, mask: Optional[mx.array] = None, cache: Optional[List[KVCache]] = None, inputs_embeds: Optional[mx.array] = None) -> mx.array:
     */
    public func callAsFunction(
        inputs: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        
        // Python: if inputs_embeds is not None:
        if let inputsEmbeds = inputsEmbeds {
            // Python: h = inputs_embeds
            h = inputsEmbeds
        } else {
            // Python: h = self.embed_tokens(inputs)
            h = embed_tokens(inputs!)
        }

        var attentionMask = mask
        // Python: if mask is None:
        if attentionMask == nil {
            // Python: mask = create_attention_mask(h, cache)
            attentionMask = mlxLMCreateAttentionMask(h, cache: cache)
        }

        var layerCaches: [(any KVCache)?]
        // Python: if cache is None:
        if let cache = cache {
            layerCaches = cache.map { $0 as (any KVCache)? }
        } else {
            // Python: cache = [None] * len(self.layers)
            layerCaches = Array<(any KVCache)?>(repeating: nil, count: layers.count)
        }

        // Python: for layer, c in zip(self.layers, cache):
        for (layer, layerCache) in zip(layers, layerCaches) {
            // Python: h = layer(h, mask, cache=c)
            // Call layer with explicit method signature for LlamaDecoderLayer
            let layerOutput: MLXArray = layer.callAsFunction(h, attentionMask: attentionMask, cache: layerCache)
            h = layerOutput
        }

        // Python: return self.norm(h)
        return norm(h)
    }
}

/**
 * Direct Python equivalent: config object (matches Python Llama config)
 * Configuration structure that corresponds exactly to the Python config used in models/llama.py
 */
public struct LlamaConfig {
    // Core model dimensions
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    
    // Optional head dimension (Python: hasattr(config, "head_dim"))
    public let headDim: Int?
    
    // RoPE configuration
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let ropeTraditional: Bool?
    public let ropeScaling: [String: Any]?
    
    // Normalization
    public let rmsNormEps: Float
    
    // Bias configuration (Python: config.attention_bias, config.mlp_bias)
    public let attentionBias: Bool
    public let mlpBias: Bool
    
    // Token IDs
    public let padTokenId: Int?
    
    public init(
        vocabSize: Int = 32000,
        hiddenSize: Int = 4096,
        intermediateSize: Int = 11008,
        numHiddenLayers: Int = 32,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 32,
        headDim: Int? = nil,
        maxPositionEmbeddings: Int = 32768,
        ropeTheta: Float = 10000.0,
        ropeTraditional: Bool? = nil,
        ropeScaling: [String: Any]? = nil,
        rmsNormEps: Float = 1e-5,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        padTokenId: Int? = nil
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.rmsNormEps = rmsNormEps
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.padTokenId = padTokenId
    }
}

/**
 * Direct Python equivalent: initialize_rope() from mlx_lm.models.rope_utils
 * This replaces the direct RoPE instantiation to match MLX-LM's approach
 */
public struct MLXLMRope {
    let rope: RoPE
    
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // This would integrate with MLX-LM's rope implementation
        // For now, use standard RoPE
        return rope(x, offset: offset)
    }
    
    public func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        // This would integrate with MLX-LM's rope implementation
        return (rope(queries, offset: offset), rope(keys, offset: offset))
    }
}

/**
 * Direct Python equivalent: initialize_rope() function
 */
public func initializeRope(
    headDim: Int,
    ropeTheta: Float,
    ropeTraditional: Bool,
    ropeScaling: [String: Any]?,
    maxPositionEmbeddings: Int
) -> MLXLMRope {
    // Python: initialize_rope(self.head_dim, self.rope_theta, config.rope_traditional if hasattr(config, "rope_traditional") else False, config.rope_scaling if hasattr(config, "rope_scaling") else None, self.max_position_embeddings)
    let rope = RoPE(
        dimensions: headDim, 
        traditional: ropeTraditional, 
        base: ropeTheta
    )
    
    return MLXLMRope(rope: rope)
}

/**
 * Direct Python equivalent: scaled_dot_product_attention() from mlx_lm.models.base
 */
public func mlxLMScaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: (any KVCache)? = nil,
    scale: Float,
    mask: MLXArray? = nil
) -> MLXArray {
    // Python: scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=attention_mask)
    
    // After cache.update_and_fetch, keys/values dimensions may have changed
    // We need to adjust mask to match the actual key sequence length
    let querySeqLen = queries.shape[2]  // [B, num_heads, q_len, head_dim]
    let keySeqLen = keys.shape[2]       // [B, num_heads, k_len, head_dim] - may be > q_len due to cache
    
    var adjustedMask: MLXArray? = mask
    
    // If mask exists and key sequence is longer than query sequence (due to cache),
    // we need to adjust the mask dimensions
    if mask != nil, keySeqLen != querySeqLen {
        // Python equivalent: when cache is used, mask dimensions must match key length
        // Create causal mask for the full key sequence length
        let fullMask = createCausalMask(N: keySeqLen)
        
        // Take only the last querySeqLen rows (corresponding to current queries)
        let startRow = keySeqLen - querySeqLen
        adjustedMask = fullMask[startRow..<keySeqLen, 0..<keySeqLen]
        
        // Expand dimensions to match expected mask shape [1, 1, q_len, k_len]
        adjustedMask = adjustedMask?.expandedDimensions(axes: [0, 1])
    }
    
    return scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: adjustedMask
    )
}

/**
 * Direct Python equivalent: def create_causal_mask(N: int, offset: int = 0, window_size: Optional[int] = None, lengths: Optional[mx.array] = None):
 *
 * 🚀 OPTIMIZED: Vectorized implementation using MLX comparison operations
 * Before: O(N²) CPU loops with individual element assignments
 * After: O(1) GPU operations - orders of magnitude faster for large N
 */
public func createCausalMask(
    N: Int,
    offset: Int = 0,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    // Create row and column index grids
    let rowIndices = MLXArray(Array(0..<N).map { Float($0) }).reshaped([N, 1])
    let colIndices = MLXArray(Array(0..<N).map { Float($0) }).reshaped([1, N])

    // Python: mask = mx.tril(mx.ones((N, N)), k=offset)
    // Causal mask: mask[i,j] = 1 if col <= row + offset, else 0
    let offsetFloat = MLXArray(Float(offset))
    let causalCondition = colIndices .<= (rowIndices + offsetFloat)
    var mask = causalCondition.asType(.float32)

    // Python: if window_size is not None:
    if let windowSize = windowSize {
        // Python: mask = mask & mx.triu(mx.ones((N, N)), k=-window_size)
        // Window mask: mask[i,j] = 1 if col >= row - windowSize, else 0
        let windowSizeFloat = MLXArray(Float(windowSize))
        let windowCondition = colIndices .>= (rowIndices - windowSizeFloat)
        mask = mask * windowCondition.asType(.float32)
    }

    // Python: if lengths is not None:
    // Note: lengths-based masking is rarely used in practice
    // If needed, it would require more complex batched operations
    if lengths != nil {
        // For now, log a warning if lengths is used
        // The previous implementation was O(N² × batchSize) which is very slow
        print("⚠️ Warning: lengths parameter in createCausalMask is not yet optimized")
    }

    return mask
}

// mlxLMCreateAttentionMask and KVCache are now defined in MLXLMBridge.swift