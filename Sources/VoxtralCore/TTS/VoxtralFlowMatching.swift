/**
 * VoxtralFlowMatching - Flow Matching Acoustic Transformer
 *
 * Generates acoustic codes from LLM hidden states using 8-step Euler flow matching
 * with classifier-free guidance (alpha=1.2).
 *
 * CRITICAL: Inputs are STACKED as a sequence of 3 (acoustic, time, llm), NOT summed.
 * The velocity is extracted at position 0 of the transformer output.
 * Semantic prediction is DIRECT from LLM hidden state (no transformer pass).
 *
 * Reference: mlx-audio PR #607, acoustic_head.py
 */

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Bidirectional Attention (no positional encoding)

/// Multi-head attention without causal mask, no KV cache, no positional encoding.
/// Weight keys: attention.{wq, wk, wv, wo}
class BidirectionalAttention: Module {

    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(dim: Int, nHeads: Int, nKVHeads: Int, headDim: Int, useBiases: Bool = false) {
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: useBiases)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: useBiases)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: useBiases)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: useBiases)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        let q = wq(x).reshaped(B, T, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // GQA: repeat K, V
        if nKVHeads < nHeads {
            let repeat_ = nHeads / nKVHeads
            k = MLX.repeated(k, count: repeat_, axis: 1)
            v = MLX.repeated(v, count: repeat_, axis: 1)
        }

        // Full bidirectional attention (no causal mask)
        // Swift scalar: a Float32 MLXArray would promote bf16 scores to fp32 (K-39)
        let scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let weights = MLX.softmax(scores, axis: -1)
        var out = MLX.matmul(weights, v)
        out = out.transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return wo(out)
    }
}

// MARK: - Acoustic Transformer Block

/// Weight keys: layers.N.{attention_norm, attention, ffn_norm, feed_forward}
class AcousticTransformerBlock: Module {

    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var attention: BidirectionalAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FMFeedForward

    init(config: VoxtralTTSConfiguration.FlowMatchingConfiguration) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.sigma)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.sigma)
        self._attention.wrappedValue = BidirectionalAttention(
            dim: config.dim,
            nHeads: config.nHeads,
            nKVHeads: config.nKVHeads,
            headDim: config.headDim,
            useBiases: config.useBiases
        )
        self._feedForward.wrappedValue = FMFeedForward(
            dim: config.dim, hiddenDim: config.hiddenDim, bias: config.useBiases
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(attentionNorm(x))
        h = h + feedForward(ffnNorm(h))
        return h
    }
}

// MARK: - Feed Forward (SwiGLU)

/// Weight keys: feed_forward.{w1, w2, w3}
class FMFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int, bias: Bool = false) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: bias)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Time Embedding

/// Sinusoidal time embedding matching vllm-omni convention: (cos, sin) order.
/// Reference: acoustic_head.py lines 112-129
class TimeEmbedding: Module {
    // Computed constant, not a weight: the `_` prefix keeps it out of parameters() (verified loading, K-7)
    let _invFreq: MLXArray

    init(dim: Int, theta: Float = 10000.0) {
        let half = dim / 2
        self._invFreq = MLX.exp(
            MLXArray(-log(theta)) * MLXArray(0..<half).asType(.float32) / MLXArray(Float(half))
        )
        super.init()
    }

    /// t: (B,) → (B, dim)
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        var tInput = t.asType(.float32)
        if tInput.ndim == 1 {
            tInput = MLX.expandedDimensions(tInput, axis: -1)  // (B, 1)
        }
        let emb = tInput * _invFreq  // (B, half)
        // (cos, sin) order — matches vllm-omni convention
        return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)], axis: -1)
    }
}

// MARK: - Flow Matching Audio Transformer

/// Generates acoustic codes via flow matching with classifier-free guidance.
///
/// Weight keys: acoustic_transformer.{input_projection, llm_projection, time_projection,
///              layers.N.*, norm, semantic_codebook_output, acoustic_codebook_output}
///
/// CRITICAL ARCHITECTURE NOTES:
/// 1. Semantic prediction is DIRECT from LLM hidden (no transformer pass)
/// 2. Velocity prediction: inputs are STACKED as (B, 3, dim), output taken at position 0
/// 3. _run_transformer = layers + norm
class FlowMatchingAudioTransformer: Module {

    let semanticCodebookSize: Int
    let acousticCodebookSize: Int
    let nAcousticCodebook: Int
    let nDenoisingSteps: Int
    let cfgAlpha: Float
    let sigmaMax: Float

    /// false: the whole frame in fp32, the path before K-39 (teacher-forced parity reference)
    var computeInWeightDType = true

    // Input projections
    @ModuleInfo(key: "input_projection") var inputProjection: Linear   // acoustic x_t → dim
    @ModuleInfo(key: "llm_projection") var llmProjection: Linear       // LLM hidden → dim
    @ModuleInfo(key: "time_projection") var timeProjection: Linear     // time emb → dim

    // Time embedding module (NOT a weight — stores inv_freq)
    let timeEmbedding: TimeEmbedding

    // Transformer layers
    @ModuleInfo var layers: [AcousticTransformerBlock]

    // Final norm
    @ModuleInfo var norm: RMSNorm

    // Output heads
    @ModuleInfo(key: "semantic_codebook_output") var semanticCodebookOutput: Linear   // [8320, dim]
    @ModuleInfo(key: "acoustic_codebook_output") var acousticCodebookOutput: Linear   // [36, dim]

    init(config: VoxtralTTSConfiguration) {
        let fmConfig = config.flowMatching
        self.semanticCodebookSize = config.audioModel.semanticCodebookSize
        self.acousticCodebookSize = config.audioModel.acousticCodebookSize
        self.nAcousticCodebook = config.audioModel.nAcousticCodebook
        self.nDenoisingSteps = 8
        self.cfgAlpha = 1.2
        self.sigmaMax = fmConfig.sigmaMax

        self._inputProjection.wrappedValue = Linear(config.audioModel.nAcousticCodebook, fmConfig.dim, bias: false)
        self._llmProjection.wrappedValue = Linear(fmConfig.inputDim, fmConfig.dim, bias: false)
        self._timeProjection.wrappedValue = Linear(fmConfig.dim, fmConfig.dim, bias: false)

        self.timeEmbedding = TimeEmbedding(dim: fmConfig.dim)

        self._layers.wrappedValue = (0..<fmConfig.nLayers).map { _ in
            AcousticTransformerBlock(config: fmConfig)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: fmConfig.dim, eps: fmConfig.sigma)

        // Semantic output: padded to 8320 = (8192 // 128 + 1) * 128
        let semanticPadded = (config.audioModel.semanticCodebookSize / 128 + 1) * 128
        self._semanticCodebookOutput.wrappedValue = Linear(fmConfig.dim, semanticPadded, bias: false)
        self._acousticCodebookOutput.wrappedValue = Linear(fmConfig.dim, config.audioModel.nAcousticCodebook, bias: false)

        super.init()
    }

    /// dtype of the transformer layers (K-39): the weights' dtype for an unquantized head (bf16 pack), float32
    /// for a quantized one. Measured with teacher forcing: quantized layers in bf16 (from `scales`) changed 1.4 to
    /// 1.9 % of the acoustic codes by up to 2 levels (gate: < 1 %, one level), for no gain in 4 bits.
    var layerComputeDType: DType {
        guard computeInWeightDType, let wq = layers.first?.attention.wq, !(wq is QuantizedLinear) else {
            return .float32
        }
        return wq.weight.dtype
    }

    /// Run input through all transformer layers + norm.
    private func runTransformer(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return norm(h)
    }

    /// Predict velocity field for one Euler step.
    ///
    /// CRITICAL: Inputs are STACKED as sequence of 3, velocity extracted at position 0.
    /// Reference: acoustic_head.py lines 163-176
    private func predictVelocity(xt: MLXArray, t: MLXArray, llmOutput: MLXArray, dtype: DType) -> MLXArray {
        // Projections and the acoustic head in float32 (small); only the transformer layers in `dtype` (K-39)
        let timeEmb = timeProjection(timeEmbedding(t))     // (B, dim)
        let llmEmb = llmProjection(llmOutput)              // (B, dim)
        let acousticEmb = inputProjection(xt)              // (B, dim)

        // Stack as sequence: (B, 3, dim) — acoustic at position 0
        let x = MLX.stacked([acousticEmb, timeEmb, llmEmb], axis: 1).asType(dtype)
        let out = runTransformer(x)

        // Velocity extracted at position 0 (the acoustic position)
        return acousticCodebookOutput(out[0..., 0, 0...].asType(.float32))
    }

    /// Predict semantic codebook index DIRECTLY from LLM hidden state.
    /// No transformer pass — just a linear projection + argmax with masking.
    /// Reference: acoustic_head.py lines 178-191
    func predictSemantic(_ llmOutput: MLXArray, debug: Bool = false) -> MLXArray {
        var logits = semanticCodebookOutput(llmOutput).asType(.float32)  // (B, 8320)
        if debug {
            let raw = logits[0]
            let topK = MLX.argSort(raw, axis: -1)
            let top5 = topK[(-5)...]
            var line = "  [DEBUG] Top-5 semantic logits: "
            for i in 0..<5 {
                let idx = top5[4 - i].item(Int32.self)
                let val = raw[Int(idx)].item(Float.self)
                line += "\(idx)(\(String(format: "%.2f", val))) "
            }
            VoxtralDebug.console(line)  // `debug: true` asked for it
        }
        // Mask padding and empty_audio in a single pass to reduce GPU allocations
        let totalDim = logits.dim(-1)
        let maskStart = semanticCodebookSize + 2
        let padSize = totalDim - maskStart
        // Build mask: index 0 = -inf (empty_audio), indices >= maskStart = -inf (padding)
        var maskValues = [Float](repeating: 0.0, count: totalDim)
        maskValues[0] = -1e9
        if padSize > 0 {
            for i in maskStart..<totalDim {
                maskValues[i] = -1e9
            }
        }
        logits = logits + MLXArray(maskValues)
        return MLX.argMax(logits, axis: -1)
    }

    /// Generate one frame: semantic code (argmax) + acoustic codes (flow matching).
    ///
    /// Returns codes WITH special token offsets applied:
    /// - Semantic: raw index in [2, 8193] (0 and 1 are masked → min is 2)
    /// - Acoustic: FSQ index [0..20] + 2 = [2..22]
    ///
    /// Reference: acoustic_head.py lines 193-234
    func decodeOneFrame(_ llmOutput: MLXArray) -> MLXArray {
        let B = llmOutput.dim(0)
        let N_SPECIAL: Int32 = 2  // empty_audio=0, end_audio=1

        // The flow-matching transformer layers run in `layerComputeDType` (K-39: bf16 for the bf16 pack); the
        // projections, both heads, the Euler state `xt`, the velocity and the CFG combination stay in float32.
        // Before K-39 the whole frame ran in float32 (`computeInWeightDType = false`), like mlx-audio. Semantic
        // head in bf16 flipped 1.7 to 2.6 % of the codes on near-ties (teacher-forced parity).
        let dtype = layerComputeDType
        let llm = llmOutput.asType(.float32)

        // Semantic code (direct prediction from LLM hidden state)
        let semanticCodes = predictSemantic(llm)  // (B,) in [0, 8319]

        // Euler integration for acoustic codes (state in float32 for stability)
        var xt = MLXRandom.normal([B, nAcousticCodebook]).asType(.float32) * MLXArray(sigmaMax)

        let timesteps: [Float] = (0..<nDenoisingSteps).map { Float($0) / Float(nDenoisingSteps - 1) }

        // Classifier-free guidance: the conditional and unconditional velocity
        // passes are independent, so run them as a single batch-2 forward
        // instead of two sequential passes (halves the transformer forwards).
        let llmZeros = MLX.zeros(like: llm)
        let llmBoth = MLX.concatenated([llm, llmZeros], axis: 0)  // (2B, dim)

        for step in 0..<(nDenoisingSteps - 1) {
            let tVal = timesteps[step]
            let dt = timesteps[step + 1] - tVal
            let tBoth = MLX.full([2 * B], values: MLXArray(tVal))
            let xtBoth = MLX.concatenated([xt, xt], axis: 0)          // (2B, nAcoustic)

            let vBoth = predictVelocity(xt: xtBoth, t: tBoth, llmOutput: llmBoth, dtype: dtype).asType(.float32)
            let vCond = vBoth[0 ..< B]
            let vUncond = vBoth[B ..< (2 * B)]
            let v = MLXArray(cfgAlpha) * vCond + MLXArray(1.0 - cfgAlpha) * vUncond

            xt = xt + v * MLXArray(dt)
        }
        // One sync for the whole 7-step Euler integration instead of one per
        // step — the graph is shallow (7 small transformer passes) and a
        // single eval avoids ~6 GPU round-trips per frame.
        MLX.eval(xt)

        // Quantize to FSQ + add special token offset
        let clamped = MLX.clip(xt, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))
        let scaled = MLX.round((clamped + 1.0) * MLXArray(Float(acousticCodebookSize - 1) / 2.0))
        let acousticCodes = MLX.clip(scaled, min: MLXArray(Float(0)), max: MLXArray(Float(acousticCodebookSize - 1)))
            .asType(.int32) + MLXArray(N_SPECIAL)

        // Combine: (B, 1+36) = (B, 37)
        return MLX.concatenated([
            MLX.expandedDimensions(semanticCodes, axis: -1),
            acousticCodes
        ], axis: -1)
    }
}

// MARK: - Utilities

/// Quantize continuous values to FSQ levels (without special token offset).
func quantizeToFSQ(_ x: MLXArray, levels: Int) -> MLXArray {
    let clamped = MLX.clip(x, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))
    let scaled = (clamped + MLXArray(Float(1.0))) * MLXArray(Float(levels - 1) / 2.0)
    return MLX.round(scaled).asType(.int32)
}

/// Dequantize FSQ indices back to continuous values in [-1, 1].
func dequantizeFSQ(_ indices: MLXArray, levels: Int) -> MLXArray {
    return (2.0 * indices.asType(.float32) / MLXArray(Float(levels - 1))) - 1.0
}
