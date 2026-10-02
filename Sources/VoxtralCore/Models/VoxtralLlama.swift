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
class LlamaAttention: Module {
    
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
    
    // @ModuleInfo required for quantization support (MLX needs to replace Linear with QuantizedLinear)
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: MLXLMRope
    
    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    init(config: LlamaConfig) {
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
        
        // Initialize with @ModuleInfo wrapper
        // Python: self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: config.attentionBias)
        // Python: self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: config.attentionBias)
        // Python: self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: config.attentionBias)
        // Python: self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: config.attentionBias)

        super.init()
    }
    
    /**
     * Direct Python equivalent: def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
     */
    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        // Python: bsz, q_len, _ = hidden_states.shape
        let bsz = hiddenStates.shape[0]
        let qLen = hiddenStates.shape[1]

        // Python: queries, keys, values = (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        let queries = qProj(hiddenStates)
        let keys = kProj(hiddenStates)
        let values = vProj(hiddenStates)

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

        // Match Python's mask="causal" by using MLXFast's .causal mode
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask = attentionMask {
            maskMode = .array(mask)
        } else if qLen > 1 {
            // Prefill: use .causal (matches Python's mask="causal" string)
            maskMode = .causal
        } else {
            // Single token decode: no mask needed
            maskMode = .none
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: finalQueries,
            keys: finalKeys,
            values: finalValues,
            scale: scale,
            mask: maskMode
        )

        // Python: output = output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        let outputTransposed = output.transposed(0, 2, 1, 3)
        let outputReshaped = outputTransposed.reshaped([bsz, qLen, -1])
        
        // Python: return self.o_proj(output)
        return oProj(outputReshaped)
    }
}

/**
 * Direct Python equivalent: class LlamaMLP(nn.Module)
 * Llama MLP with SiLU activation.
 */
class LlamaMLP: Module {

    // Python: def __init__(self, config):
    let config: LlamaConfig
    let hiddenSize: Int
    let intermediateSize: Int
    // @ModuleInfo required for quantization support (MLX needs to replace Linear with QuantizedLinear)
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let actFn: (MLXArray) -> MLXArray

    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.config = config
        self.config = config
        // Python: self.hidden_size = config.hidden_size
        self.hiddenSize = config.hiddenSize
        // Python: self.intermediate_size = config.intermediate_size
        self.intermediateSize = config.intermediateSize

        // Python: self.act_fn = nn.silu
        self.actFn = silu

        // Initialize with @ModuleInfo wrapper
        // Python: self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: config.mlpBias)
        // Python: self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: config.mlpBias)
        // Python: self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: config.mlpBias)

        super.init()
    }

    /**
     * Direct Python equivalent: def __call__(self, hidden_states: mx.array) -> mx.array:
     */
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Python: return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return downProj(actFn(gateProj(hiddenStates)) * upProj(hiddenStates))
    }
}

/**
 * Direct Python equivalent: class LlamaDecoderLayer(nn.Module)
 * Llama decoder layer.
 */
class LlamaDecoderLayer: Module {

    let hiddenSize: Int
    // @ModuleInfo required for weight loading
    @ModuleInfo(key: "self_attn") var selfAttn: LlamaAttention
    @ModuleInfo var mlp: LlamaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.hidden_size = config.hidden_size
        self.hiddenSize = config.hiddenSize

        // Initialize with @ModuleInfo wrapper
        // Python: self.self_attn = LlamaAttention(config)
        self._selfAttn.wrappedValue = LlamaAttention(config: config)
        // Python: self.mlp = LlamaMLP(config)
        self._mlp.wrappedValue = LlamaMLP(config: config)
        // Python: self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        // Python: self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /**
     * Direct Python equivalent: def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
     */
    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        // Python: r = self.self_attn(self.input_layernorm(hidden_states), attention_mask, cache)
        let r = selfAttn(inputLayernorm(hiddenStates), attentionMask: attentionMask, cache: cache)
        // Python: h = hidden_states + r
        let h = hiddenStates + r
        // Python: r = self.mlp(self.post_attention_layernorm(h))
        let r2 = mlp(postAttentionLayernorm(h))
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
class LlamaModel: Module {
    
    let config: LlamaConfig
    let paddingIdx: Int?
    let vocabSize: Int
    // @ModuleInfo required for weight loading
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [LlamaDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    /**
     * Direct Python equivalent: def __init__(self, config):
     */
    init(config: LlamaConfig) {
        // Python: super().__init__()
        // Python: self.config = config
        self.config = config
        // Python: self.padding_idx = getattr(config, "pad_token_id", None)
        self.paddingIdx = config.padTokenId
        // Python: self.vocab_size = config.vocab_size
        self.vocabSize = config.vocabSize

        // Initialize with @ModuleInfo wrapper
        // Python: self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Python: self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        // Python: self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            LlamaDecoderLayer(config: config)
        }

        super.init()
    }
    
    /**
     * Direct Python equivalent: def __call__(self, inputs: Optional[mx.array] = None, mask: Optional[mx.array] = None, cache: Optional[List[KVCache]] = None, inputs_embeds: Optional[mx.array] = None) -> mx.array:
     */
    func callAsFunction(
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
            h = embedTokens(inputs!)
        }

        // No hand-made mask: nil lets LlamaAttention use .causal for a prefill at any cache offset and no mask for
        // one token. The [T, T] mask built here stopped the second prefill chunk (keys = offset + T) (K-3)
        let attentionMask = mask

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
struct LlamaConfig {
    // Core model dimensions
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    
    // Optional head dimension (Python: hasattr(config, "head_dim"))
    let headDim: Int?
    
    // RoPE configuration
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let ropeTraditional: Bool?
    let ropeScaling: [String: Any]?
    
    // Normalization
    let rmsNormEps: Float
    
    // Bias configuration (Python: config.attention_bias, config.mlp_bias)
    let attentionBias: Bool
    let mlpBias: Bool
    
    // Token IDs
    let padTokenId: Int?
    
    init(
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
struct MLXLMRope {
    let rope: RoPE
    
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Standard RoPE (no MLX-LM rope scaling on this legacy path)
        return rope(x, offset: offset)
    }
    
    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        // Standard RoPE
        return (rope(queries, offset: offset), rope(keys, offset: offset))
    }
}

/**
 * Direct Python equivalent: initialize_rope() function
 */
func initializeRope(
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

/**
 * Direct Python equivalent: def create_causal_mask(N: int, offset: int = 0, window_size: Optional[int] = None, lengths: Optional[mx.array] = None):
 *
 * 🚀 OPTIMIZED: Vectorized implementation using MLX comparison operations
 * Before: O(N²) CPU loops with individual element assignments
 * After: O(1) GPU operations - orders of magnitude faster for large N
 */

// mlxLMCreateAttentionMask and KVCache are now defined in MLXLMBridge.swift