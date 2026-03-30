/**
 * VoxtralRealtimeDecoder - LLM decoder with Ada RMS-Norm for Voxtral Realtime
 *
 * 26-layer decoder-only transformer with:
 * - GQA (32 query heads, 8 KV heads, head_dim=128)
 * - Sliding window attention (8192)
 * - Interleaved RoPE (theta=1M)
 * - Adaptive RMSNorm with time conditioning (delay-aware)
 * - Tied embeddings (tok_embeddings used as both input and LM head)
 * - SwiGLU FFN, no biases
 *
 * Reference: mlx-audio/stt/models/voxtral_realtime/decoder.py
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Time Embedding

/// Sinusoidal time embedding for Ada RMS-Norm conditioning.
/// t_value: number of delay tokens (e.g. 6.0 for 480ms) → [dim] vector
public func computeTimeEmbedding(tValue: Float, dim: Int, theta: Float = 10000.0) -> MLXArray {
    let halfDim = dim / 2
    let invFreq = MLX.exp(
        -log(theta) * MLXArray(stride(from: 0, to: halfDim, by: 1).map { Float($0) / Float(halfDim) })
    )
    let emb = MLXArray(tValue) * invFreq
    return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)])  // [dim]
}

// MARK: - Ada RMS-Norm

/// Adaptive RMSNorm with time conditioning.
/// Per-layer MLP: Linear(dim→bottleneck) → GELU → Linear(bottleneck→dim)
/// Applied as: rms_norm(x) * (1 + ada_scale)
public class AdaRMSNorm: Module {
    @ModuleInfo(key: "ada_down") var adaDown: Linear
    @ModuleInfo(key: "ada_up") var adaUp: Linear

    public init(dim: Int, bottleneckDim: Int) {
        self._adaDown.wrappedValue = Linear(dim, bottleneckDim, bias: false)
        self._adaUp.wrappedValue = Linear(bottleneckDim, dim, bias: false)
        super.init()
    }

    /// Precompute ada_scale from time conditioning. Returns [dim].
    public func computeScale(_ tCond: MLXArray) -> MLXArray {
        adaUp(gelu(adaDown(tCond)))
    }

    /// Apply adaptive scaling: x * (1 + ada_scale)
    public func callAsFunction(_ x: MLXArray, adaScale: MLXArray) -> MLXArray {
        x * (1.0 + adaScale)
    }
}

// MARK: - Decoder Layer

/// Single decoder transformer layer with Ada RMS-Norm on FFN branch.
/// Reuses LlamaAttention for standard GQA attention (no biases, RoPE).
public class RealtimeDecoderLayer: Module {

    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo var attention: LlamaAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ada_rms_norm_t_cond") var adaNorm: AdaRMSNorm?

    // SwiGLU FFN (no biases)
    @ModuleInfo(key: "feed_forward_w1") var w1: Linear
    @ModuleInfo(key: "feed_forward_w2") var w2: Linear
    @ModuleInfo(key: "feed_forward_w3") var w3: Linear

    public init(decoderConfig: RealtimeDecoderConfig) {
        let llamaConfig = LlamaConfig(
            vocabSize: decoderConfig.vocabSize,
            hiddenSize: decoderConfig.dim,
            intermediateSize: decoderConfig.hiddenDim,
            numHiddenLayers: decoderConfig.nLayers,
            numAttentionHeads: decoderConfig.nHeads,
            numKeyValueHeads: decoderConfig.nKVHeads,
            headDim: decoderConfig.headDim,
            maxPositionEmbeddings: 131072,
            ropeTheta: decoderConfig.ropeTheta,
            ropeTraditional: true,
            rmsNormEps: decoderConfig.normEps,
            attentionBias: false,
            mlpBias: false
        )

        self._attentionNorm.wrappedValue = RMSNorm(dimensions: decoderConfig.dim, eps: decoderConfig.normEps)
        self._attention.wrappedValue = LlamaAttention(config: llamaConfig)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: decoderConfig.dim, eps: decoderConfig.normEps)

        if decoderConfig.adaRmsNormTCond {
            self._adaNorm.wrappedValue = AdaRMSNorm(
                dim: decoderConfig.dim,
                bottleneckDim: decoderConfig.adaRmsNormTCondDim
            )
        }

        // SwiGLU FFN
        self._w1.wrappedValue = Linear(decoderConfig.dim, decoderConfig.hiddenDim, bias: false)
        self._w3.wrappedValue = Linear(decoderConfig.dim, decoderConfig.hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(decoderConfig.hiddenDim, decoderConfig.dim, bias: false)

        super.init()
    }

    /// Forward pass.
    /// embeds: [1, seq, dim], adaScale: precomputed [dim] or nil
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil,
        adaScale: MLXArray? = nil
    ) -> MLXArray {
        // Attention
        let r = attention(attentionNorm(hiddenStates), attentionMask: attentionMask, cache: cache)
        var x = hiddenStates + r

        // FFN with Ada RMS-Norm
        var h = ffnNorm(x)
        if let adaNorm, let adaScale {
            h = adaNorm(h, adaScale: adaScale)
        }
        let gate = silu(w1(h))
        let up = w3(h)
        x = x + w2(gate * up)

        return x
    }
}

// MARK: - Decoder

/// Full LLM decoder with tied embeddings and Ada RMS-Norm.
public class VoxtralRealtimeDecoder: Module {

    let config: RealtimeDecoderConfig

    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo var layers: [RealtimeDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    /// Precomputed ada_scales per layer (set via precomputeAdaScales)
    var adaScales: [MLXArray?] = []

    public init(config: RealtimeDecoderConfig) {
        self.config = config

        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dim
        )
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            RealtimeDecoderLayer(decoderConfig: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        super.init()
    }

    /// Precompute Ada RMS-Norm scales from delay conditioning.
    /// Call once after model loading with the chosen delay.
    public func precomputeAdaScales(tCond: MLXArray) {
        adaScales = layers.map { layer in
            layer.adaNorm?.computeScale(tCond)
        }
        // Evaluate eagerly
        for scale in adaScales {
            if let scale { MLX.eval(scale) }
        }
    }

    /// Embed a single token ID.
    public func embedToken(_ tokenId: Int) -> MLXArray {
        tokEmbeddings.weight[tokenId]
    }

    /// Embed token IDs.
    public func embedTokens(_ tokenIds: MLXArray) -> MLXArray {
        tokEmbeddings(tokenIds)
    }

    /// Compute logits via tied embeddings: h @ tok_embeddings^T
    public func logits(_ h: MLXArray) -> MLXArray {
        MLX.matmul(h, tokEmbeddings.weight.transposed())
    }

    /// Forward pass through all layers.
    /// embeds: [seq, dim] input embeddings (audio + text summed)
    /// Returns: hidden states [seq, dim]
    public func forward(
        embeds: MLXArray,
        cache: [any KVCache]? = nil
    ) -> (MLXArray, [any KVCache]) {
        // Add batch dimension: [seq, dim] → [1, seq, dim]
        var h = embeds.expandedDimensions(axis: 0)

        var newCache: [any KVCache] = []
        for (i, layer) in layers.enumerated() {
            let layerCache: (any KVCache)? = cache.map { $0[i] }
            let adaScale = i < adaScales.count ? adaScales[i] : nil
            h = layer(h, cache: layerCache, adaScale: adaScale)
            newCache.append(layerCache ?? KVCacheSimple())
        }

        h = norm(h)
        return (h.squeezed(axis: 0), newCache)  // back to [seq, dim]
    }
}
