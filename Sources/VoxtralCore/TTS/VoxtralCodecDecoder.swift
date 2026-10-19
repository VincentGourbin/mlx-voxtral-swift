/**
 * VoxtralCodecDecoder - Decodes quantized codes to 24kHz waveform
 *
 * Reference: mlx-audio PR #607, audio_tokenizer.py
 *
 * Decoder: alternating weight-normed conv blocks and ALiBi transformer blocks.
 * Strides [1,2,2,2] give 8x upsampling; output_proj maps dim -> 240 samples per frame.
 * Sliding windows: [2, 4, 8, 16] (encoder reversed).
 *
 * Codes arrive WITH +2 special token offset — stripped before codebook lookup.
 */

import Foundation
import MLX
import MLXNN

// MARK: - Invariant cache

/// Value computed from a module's parameters (K-58, P-39), cleared by that module's `update(parameters:)`
/// override. `update` copies new arrays into the existing `MLXArray` objects (`_updateInternal`), so neither object
/// identity nor the handle tells that they changed. A class, not an `MLXArray` property: module reflection ignores it.
final class DerivedArrayCache: @unchecked Sendable {
    private let lock = NSLock()
    private var value: MLXArray?

    func value(_ compute: () -> MLXArray) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let value { return value }
        let computed = compute()
        value = computed
        return computed
    }

    func invalidate() {
        lock.lock()
        value = nil
        lock.unlock()
    }
}

// MARK: - Weight-Normalized Conv1d

/// Container for weight norm parameters at the "weight" level.
/// Matches: parametrizations.weight.{original0, original1}
class WeightNormParams: Module {
    var original0: MLXArray  // (out_ch, 1, 1) — gain
    var original1: MLXArray  // (out_ch, in_ch, K) — direction

    private let weightCache = DerivedArrayCache()

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self.original0 = MLX.ones([outChannels, 1, 1])
        self.original1 = MLX.zeros([outChannels, inChannels, kernelSize])
        super.init()
    }

    /// w = g * v / ||v||, computed once per set of parameters (K-58)
    var weight: MLXArray {
        weightCache.value {
            let vNorm = MLX.sqrt((original1 * original1).sum(axes: [1, 2], keepDims: true) + 1e-12)
            return original0 * original1 / vNorm
        }
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        weightCache.invalidate()
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}

/// Container for parametrizations.
/// Matches: parametrizations.weight.*
class WeightParametrizations: Module {
    @ModuleInfo(key: "weight") var weight: WeightNormParams

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self._weight.wrappedValue = WeightNormParams(outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize)
        super.init()
    }
}

/// Causal Conv1d with weight normalization stored as parametrizations.
/// Weight keys: conv.parametrizations.weight.{original0, original1}
///
/// Reference: audio_tokenizer.py lines 68-142
class WeightNormConv: Module {

    @ModuleInfo var parametrizations: WeightParametrizations

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self._parametrizations.wrappedValue = WeightParametrizations(
            outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize
        )
        super.init()
    }

    /// Reconstruct weight from weight norm: w = g * v / ||v|| (cached by the parameters' module, K-58)
    private func getWeight() -> MLXArray { parametrizations.weight.weight }

    /// Apply causal or transposed convolution.
    /// Reference: audio_tokenizer.py lines 97-142
    func callAsFunction(_ x: MLXArray, stride: Int = 1, transpose: Bool = false) -> MLXArray {
        let weight = getWeight()  // (out_ch, in_ch, K)
        if transpose {
            return convTranspose1d(x, weight: weight, stride: stride)
        }
        return conv1d(x, weight: weight, stride: stride)
    }

    /// Causal 1D convolution with left padding.
    private func conv1d(_ x: MLXArray, weight: MLXArray, stride: Int) -> MLXArray {
        let K = weight.dim(2)
        var padded = x
        if K > 1 {
            let B = x.dim(0)
            let C = x.dim(2)
            padded = MLX.concatenated([MLX.zeros([B, K - 1, C], dtype: x.dtype), x], axis: 1)
        }
        // mx.conv1d weight: (out_ch, K, in_ch)
        let w = weight.transposed(0, 2, 1)
        return MLX.conv1d(padded, w, stride: stride)
    }

    /// Causal transposed 1D convolution.
    /// Reference: audio_tokenizer.py lines 123-142
    private func convTranspose1d(_ x: MLXArray, weight: MLXArray, stride: Int) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Checkpoint weight: (out_ch, in_ch, K) — for ConvTranspose this is (C_in_transpose, C_out_transpose, K)
        // mx.conv_transpose1d weight: (C_out, K, C_in)
        let w = weight.transposed(1, 2, 0)  // (in_ch, K, out_ch) = (C_out, K, C_in)

        var out = MLX.convTransposed1d(x, w, stride: stride, padding: 0)

        // Causal trim: keep first T*stride elements
        out = out[0..., ..<(T * stride), 0...]

        return out
    }
}

// MARK: - Conv Block

/// A conv block matching decoder_blocks.{even_index}.conv
class ConvBlock: Module {
    @ModuleInfo var conv: WeightNormConv

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self._conv.wrappedValue = WeightNormConv(outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize)
        super.init()
    }
}

// MARK: - ALiBi Slopes

/// Compute ALiBi slopes for n heads.
/// Reference: audio_tokenizer.py lines 158-170
func getAlibiSlopes(nHeads: Int) -> MLXArray {
    func getSlopesPowerOf2(_ n: Int) -> [Float] {
        let start = pow(2.0, -(pow(2.0, -(log2(Float(n)) - 3.0))))
        return (0..<n).map { start * pow(start, Float($0)) }
    }

    let slopes: [Float]
    if log2(Float(nHeads)).truncatingRemainder(dividingBy: 1.0) == 0 {
        slopes = getSlopesPowerOf2(nHeads)
    } else {
        let closest = Int(pow(2.0, floor(log2(Float(nHeads)))))
        var s = getSlopesPowerOf2(closest)
        let extra = getSlopesPowerOf2(2 * closest)
        let needed = nHeads - closest
        for i in stride(from: 0, to: min(needed * 2, extra.count), by: 2) {
            s.append(extra[i])
        }
        slopes = s
    }
    return MLXArray(slopes)
}

// MARK: - Codec Attention (ALiBi + QK-norm + sliding window)

/// Reference: audio_tokenizer.py lines 173-243
class CodecAttention: Module {

    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    /// Queries per block of the banded path (K-41).
    static let bandBlockSize = 64

    /// Sliding-window attention by blocks of queries instead of a (T, T) score matrix (K-41). A long decode reaches
    /// 18 000 positions at the last stage, where the full scores, ALiBi and masks took about 10 GB each. Exact up to
    /// rounding: every key outside the band carries the −1e9 mask, whose weight is exactly 0 after the softmax.
    /// Internal switch for the parity tests; the full path stays for `windowSize == 0`.
    var bandedAttention = true

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self.nHeads = config.nHeads
        self.nKVHeads = config.nKVHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        self._wq.wrappedValue = Linear(config.dim, config.nHeads * config.headDim, bias: false)
        self._wk.wrappedValue = Linear(config.dim, config.nKVHeads * config.headDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, config.nKVHeads * config.headDim, bias: false)
        self._wo.wrappedValue = Linear(config.nHeads * config.headDim, config.dim, bias: false)

        if config.qkNorm {
            // QK-norm over full projected dim (not per-head)
            self._qNorm.wrappedValue = RMSNorm(dimensions: config.nHeads * config.headDim, eps: config.qkNormEps)
            self._kNorm.wrappedValue = RMSNorm(dimensions: config.nKVHeads * config.headDim, eps: config.qkNormEps)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int = 0) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Project Q, K, V
        var q = wq(x)
        var k = wk(x)
        let v_ = wv(x)

        // QK-norm BEFORE reshaping to heads
        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }

        // Reshape to heads
        let qh = q.reshaped(B, T, nHeads, headDim).transposed(0, 2, 1, 3)
        var kh = k.reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var vh = v_.reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // GQA repeat
        if nKVHeads < nHeads {
            let repeat_ = nHeads / nKVHeads
            kh = MLX.repeated(kh, count: repeat_, axis: 1)
            vh = MLX.repeated(vh, count: repeat_, axis: 1)
        }

        if bandedAttention && windowSize > 0 && windowSize <= Self.bandBlockSize {
            let out = banded(qh, kh, vh, alibiSlopes: alibiSlopes, windowSize: windowSize, dtype: x.dtype)
            return wo(out.transposed(0, 2, 1, 3).reshaped(B, T, -1))
        }

        var scores = MLX.matmul(qh, kh.transposed(0, 1, 3, 2)) * scale

        // ALiBi bias: dist[i,j] = j - i (negative for past)
        let positions = MLXArray(0..<Int32(T))
        let dist = MLX.expandedDimensions(positions, axis: 0) - MLX.expandedDimensions(positions, axis: 1)
        let alibi = MLX.expandedDimensions(MLX.expandedDimensions(alibiSlopes, axis: -1), axis: -1) * dist.asType(.float32)

        // Causal mask (upper triangular = future positions masked)
        // dist[i,j] = j - i: future positions have dist > 0
        let causalMaskValues = MLX.where(dist .> MLXArray(Int32(0)), MLXArray(Float(-1e9)), MLXArray(Float(0)))

        // Sliding window mask
        var combinedMask = causalMaskValues
        if windowSize > 0 {
            let windowMask = MLX.where(dist .< MLXArray(Int32(-windowSize)), MLXArray(Float(-1e9)), MLXArray(Float(0)))
            combinedMask = combinedMask + windowMask
        }

        scores = scores + alibi + combinedMask

        // Softmax in float32 for precision
        let weights = MLX.softmax(scores.asType(.float32), axis: -1).asType(x.dtype)
        var out = MLX.matmul(weights, vh)
        out = out.transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return wo(out)
    }

    /// Banded attention: the T queries are cut into blocks of `bandBlockSize`; block b attends to the W keys before
    /// it and to its own keys, (Bq + W) keys in all, with the same ALiBi and causal / window masks as the full path.
    /// Queries and keys are padded to a whole number of blocks (padded keys lie in the future of every real query and
    /// are masked; padded queries are dropped), and W zero keys before the first block are masked too.
    /// qh, kh, vh: (B, H, T, D) → (B, H, T, D)
    private func banded(
        _ qh: MLXArray, _ kh: MLXArray, _ vh: MLXArray, alibiSlopes: MLXArray, windowSize W: Int, dtype: DType
    ) -> MLXArray {
        let B = qh.dim(0), H = qh.dim(1), T = qh.dim(2), D = qh.dim(3)
        let Bq = Self.bandBlockSize
        let nb = (T + Bq - 1) / Bq
        let Tp = nb * Bq

        func padTime(_ a: MLXArray, before: Int, after: Int) -> MLXArray {
            before == 0 && after == 0 ? a : MLX.padded(a, widths: [0, 0, .init((before, after)), 0])
        }
        let qb = padTime(qh, before: 0, after: Tp - T).reshaped(B, H, nb, Bq, D)
        // Per block: the W keys before it (from the keys shifted by W), then its own Bq keys.
        func window(_ a: MLXArray) -> MLXArray {
            let own = padTime(a, before: 0, after: Tp - T).reshaped(B, H, nb, Bq, D)
            let before = padTime(a, before: W, after: Tp - T)[0..., 0..., ..<Tp, 0...]
                .reshaped(B, H, nb, Bq, D)[0..., 0..., 0..., ..<W, 0...]
            return MLX.concatenated([before, own], axis: 3)  // (B, H, nb, W + Bq, D)
        }
        let kb = window(kh)
        let vb = window(vh)

        var scores = MLX.matmul(qb, kb.transposed(0, 1, 2, 4, 3)) * scale  // (B, H, nb, Bq, W + Bq)

        // Query i of block b sits at b·Bq + i, key j at b·Bq − W + j: dist = j − W − i, as in the full path.
        let i = MLX.expandedDimensions(MLXArray(0..<Int32(Bq)), axis: 1)
        let j = MLX.expandedDimensions(MLXArray(0..<Int32(W + Bq)), axis: 0)
        let dist = j - MLXArray(Int32(W)) - i  // (Bq, W + Bq)
        let alibi = alibiSlopes.reshaped(H, 1, 1, 1) * dist.asType(.float32)  // (H, 1, Bq, W + Bq)
        let outside = MLX.logicalOr(dist .> MLXArray(Int32(0)), dist .< MLXArray(Int32(-W)))
        let bandMask = MLX.where(outside, MLXArray(Float(-1e9)), MLXArray(Float(0)))
        // Keys before position 0 (the W padded keys of block 0).
        let block = MLXArray(0..<Int32(nb)).reshaped(nb, 1, 1)
        let beforeStart = MLX.logicalAnd(block .== MLXArray(Int32(0)), j.reshaped(1, 1, W + Bq) .< MLXArray(Int32(W)))
        let startMask = MLX.where(beforeStart, MLXArray(Float(-1e9)), MLXArray(Float(0)))  // (nb, 1, W + Bq)

        scores = scores + alibi + (bandMask + startMask)

        let weights = MLX.softmax(scores.asType(.float32), axis: -1).asType(dtype)
        let out = MLX.matmul(weights, vb).reshaped(B, H, Tp, D)
        return Tp == T ? out : out[0..., 0..., ..<T, 0...]
    }
}

// MARK: - Codec Transformer Layer (with LayerScale)

/// Reference: audio_tokenizer.py lines 253-285
class CodecTransformerLayer: Module {

    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var attention: CodecAttention
    @ModuleInfo(key: "feed_forward") var feedForward: CodecFeedForward

    let useLayerScale: Bool
    var attention_scale: MLXArray
    var ffn_scale: MLXArray

    init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self.useLayerScale = config.layerScale

        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = CodecAttention(config: config)
        self._feedForward.wrappedValue = CodecFeedForward(dim: config.dim, hiddenDim: config.hiddenDim)

        if config.layerScale {
            self.attention_scale = MLX.full([config.dim], values: MLXArray(config.layerScaleInit))
            self.ffn_scale = MLX.full([config.dim], values: MLXArray(config.layerScaleInit))
        } else {
            self.attention_scale = MLX.ones([config.dim])
            self.ffn_scale = MLX.ones([config.dim])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int = 0) -> MLXArray {
        var h = attention(attentionNorm(x), alibiSlopes: alibiSlopes, windowSize: windowSize)
        if useLayerScale { h = h * attention_scale }
        var out = x + h

        h = feedForward(ffnNorm(out))
        if useLayerScale { h = h * ffn_scale }
        out = out + h
        return out
    }

    /// The layer over time slices of `timeChunk` positions (K-41): slice [s, e) is computed from x[s − W, e) and
    /// keeps its last e − s rows. Exact up to rounding: norms and the feed-forward act per position, ALiBi depends on
    /// j − i only, and a query of the slice sees its W previous keys. Bounds the transient of a long decode (fp32
    /// activations of 15 000 positions at the last stage) to one slice. Each slice is evaluated: inference only, not
    /// for a differentiated graph.
    func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int, timeChunk: Int) -> MLXArray {
        let T = x.dim(1)
        guard timeChunk > 0, windowSize > 0, T > timeChunk else {
            return self(x, alibiSlopes: alibiSlopes, windowSize: windowSize)
        }
        var parts: [MLXArray] = []
        var start = 0
        while start < T {
            let end = min(start + timeChunk, T)
            let lo = max(0, start - windowSize)
            let y = self(x[0..., lo ..< end, 0...], alibiSlopes: alibiSlopes, windowSize: windowSize)
            let part = y[0..., (start - lo)..., 0...]
            // Evaluated here: left lazy, MLX schedules the slices together and their transients add up.
            MLX.eval(part)
            parts.append(part)
            start = end
        }
        return MLX.concatenated(parts, axis: 1)
    }
}

/// SwiGLU feed-forward matching feed_forward.{w1, w2, w3}
class CodecFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Transformer Block (N layers)

class CodecTransformerBlock: Module {
    @ModuleInfo var layers: [CodecTransformerLayer]

    init(nLayers: Int, config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self._layers.wrappedValue = (0..<nLayers).map { _ in CodecTransformerLayer(config: config) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int = 0) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, alibiSlopes: alibiSlopes, windowSize: windowSize)
        }
        return h
    }

    /// Inference path (K-41): each layer by time slices, evaluated before the next one, so that a long decode never
    /// holds more than one slice's transient. Not for a differentiated graph (voice enrollment uses the lazy path).
    func evaluatedByLayer(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int, timeChunk: Int) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, alibiSlopes: alibiSlopes, windowSize: windowSize, timeChunk: timeChunk)
            MLX.eval(h)
        }
        return h
    }
}

// MARK: - Codebooks

/// EMA-based semantic codebook. Decode in float32 for precision.
/// Reference: audio_tokenizer.py lines 311-338
class SemanticCodebook: Module {
    var cluster_usage: MLXArray
    var embedding_sum: MLXArray

    init(codebookSize: Int, dim: Int) {
        self.cluster_usage = MLX.ones([codebookSize])
        self.embedding_sum = MLX.zeros([codebookSize, dim])
        super.init()
    }

    private let codebookCache = DerivedArrayCache()

    /// Centroid table (codebookSize, dim) in float32, computed once per set of parameters (K-58).
    /// Exposed for voice enrollment, which needs a differentiable
    /// soft lookup (probabilities × codebook) alongside the hard one.
    var codebook: MLXArray {
        codebookCache.value {
            embedding_sum.asType(.float32) / MLX.maximum(
                MLX.expandedDimensions(cluster_usage.asType(.float32), axis: -1),
                MLXArray(Float(1e-8))
            )
        }
    }

    func decode(_ indices: MLXArray) -> MLXArray {
        return codebook[indices]
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        codebookCache.invalidate()
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}

/// FSQ acoustic codebook — no learned parameters.
/// Reference: audio_tokenizer.py lines 341-351
class AcousticCodebook: Module {
    let codebookSize: Int

    init(codebookSize: Int) {
        self.codebookSize = codebookSize
        super.init()
    }

    func decode(_ indices: MLXArray) -> MLXArray {
        // [0, codebookSize-1] → [-1, 1]
        return (2.0 * indices.asType(.float32) / MLXArray(Float(codebookSize - 1))) - 1.0
    }
}

/// Combined semantic + acoustic codebook.
/// Reference: audio_tokenizer.py lines 354-386
class MistralAudioCodebook: Module {
    @ModuleInfo(key: "semantic_codebook") var semanticCodebook: SemanticCodebook

    let acousticCodebook: AcousticCodebook

    init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self._semanticCodebook.wrappedValue = SemanticCodebook(
            codebookSize: config.semanticCodebookSize, dim: config.semanticDim
        )
        self.acousticCodebook = AcousticCodebook(codebookSize: config.acousticCodebookSize)
        super.init()
    }

    /// Decode codes WITH special token offset.
    /// codes: (B, T, 37) where semantic in [2..8193], acoustic in [2..22]
    func decode(_ codes: MLXArray) -> MLXArray {
        let N_SPECIAL: Int32 = 2
        // Strip special token offset
        let semanticCodes = codes[0..., 0..., 0] - MLXArray(N_SPECIAL)    // (B, T) in [0..8191]
        let acousticCodes = codes[0..., 0..., 1...] - MLXArray(N_SPECIAL)  // (B, T, 36) in [0..20]

        let semanticEmb = semanticCodebook.decode(semanticCodes)    // (B, T, 256)
        let acousticEmb = acousticCodebook.decode(acousticCodes)    // (B, T, 36)

        return MLX.concatenated([semanticEmb, acousticEmb], axis: -1)  // (B, T, 292)
    }
}

// MARK: - VoxtralCodecDecoder (audio_tokenizer)

/// Complete audio tokenizer decoder.
/// Reference: audio_tokenizer.py lines 394-471
///
/// decoder_blocks = [conv0, xformer0, conv1, xformer1, conv2, xformer2, conv3, xformer3]
/// Sliding windows: [2, 4, 8, 16] (encoder reversed)
/// Strides: [1, 2, 2, 2] → 8x total upsampling
class VoxtralCodecDecoder: Module {

    let config: VoxtralTTSConfiguration.AudioTokenizerConfiguration

    @ModuleInfo var quantizer: MistralAudioCodebook
    @ModuleInfo(key: "decoder_blocks") var decoderBlocks: [Module]
    @ModuleInfo(key: "output_proj") var outputProj: ConvBlock

    // Computed constant, not a weight: the `_` prefix keeps it out of parameters() (verified loading, K-7)
    let _alibiSlopes: MLXArray
    let strides: [Int]

    init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self.config = config
        self._alibiSlopes = getAlibiSlopes(nHeads: config.nHeads)
        self.strides = config.decoderConvsStrides

        self._quantizer.wrappedValue = MistralAudioCodebook(config: config)

        let kernels = config.decoderConvsKernels
        let transformerLengths = config.decoderTransformerLengths

        var blocks: [Module] = []
        for i in 0..<config.numDecoderBlocks {
            let inCh = i == 0 ? config.latentDim : config.dim
            blocks.append(ConvBlock(outChannels: config.dim, inChannels: inCh, kernelSize: kernels[i]))
            blocks.append(CodecTransformerBlock(nLayers: transformerLengths[i], config: config))
        }
        self._decoderBlocks.wrappedValue = blocks

        self._outputProj.wrappedValue = ConvBlock(
            outChannels: config.pretransformPatchSize,
            inChannels: config.dim,
            kernelSize: config.patchProjKernelSize
        )
        super.init()
    }

    /// Decode audio codes to waveform.
    /// codes: (B, T, 37) — with +2 special token offset
    /// The network runs in the dtype of its weights (bf16 in the packs, K-58): the float32 centroids and FSQ levels are
    /// cast after the lookup; the waveform comes back in float32.
    func decode(_ codes: MLXArray) -> MLXArray {
        let embeddings = quantizer.decode(codes).asType(computeDType)
        return forwardEmbeddings(embeddings, timeChunk: bandedInference ? Self.inferenceTimeChunk : 0).asType(.float32)
    }

    /// Dtype of the decoder weights (the output projection's direction tensor).
    var computeDType: DType { outputProj.conv.parametrizations.weight.original1.dtype }

    /// Sliding windows of the transformer stages: [2, 4, 8, 16] — encoder reversed
    static let windowSizes = [2, 4, 8, 16]

    /// Output samples per frame of codes (strides × patch size: 8 × 240 = 1 920, 80 ms at 24 kHz).
    var samplesPerFrame: Int { strides.reduce(1, *) * config.pretransformPatchSize }

    /// Causal receptive field in frames (K-43): the samples of frame t depend on the codes of frames t − R … t only.
    /// Walked back from the output: the projection conv (K − 1 positions), each transformer stage (n layers × W
    /// keys), each stage conv (K − 1 positions, or ⌈(r + K − 1) / s⌉ input positions for a causal transposed conv of
    /// stride s). Pack (2,2,2,2 layers; kernels 3,4,4,4; projection 7): 22 frames.
    var receptiveFieldFrames: Int {
        let kernels = config.decoderConvsKernels, lengths = config.decoderTransformerLengths
        var r = config.patchProjKernelSize - 1
        for stage in stride(from: lengths.count - 1, through: 0, by: -1) {
            let window = stage < Self.windowSizes.count ? Self.windowSizes[stage] : 16
            r += lengths[stage] * window
            let s = strides[stage]
            r = s > 1 ? (r + kernels[stage] - 1 + s - 1) / s : r + kernels[stage] - 1
        }
        return r
    }

    /// Samples of frames `start ..< codes.dim(1)`, decoded from at most `context` frames before `start` (K-43):
    /// equal to the matching slice of `decode(codes)` up to rounding when `context ≥ receptiveFieldFrames`.
    /// codes: (B, T, 37) → (B, (T − start) × samplesPerFrame)
    func decode(_ codes: MLXArray, from start: Int, context: Int) -> MLXArray {
        let lo = max(0, start - context)
        let waveform = decode(lo == 0 ? codes : codes[0..., lo..., 0...])
        return waveform[0..., ((start - lo) * samplesPerFrame)...]
    }

    /// Time slice of the inference path (K-41); 0 = the lazy whole-sequence path.
    static let inferenceTimeChunk = 2048
    /// Internal switch for the parity tests: false decodes with the lazy whole-sequence path.
    var bandedInference = true

    /// Run the decoder network on continuous quantizer-space embeddings.
    /// Equivalent to Python's `_forward_decoder` — this is the entry point
    /// used by voice enrollment, where gradients flow through `embeddings`
    /// while the decoder weights stay frozen.
    /// embeddings: (B, T, 292) → waveform (B, T*1920)
    /// timeChunk > 0 (inference only, `decode`): transformer layers by time slices, evaluated layer by layer.
    func forwardEmbeddings(_ embeddings: MLXArray, timeChunk: Int = 0) -> MLXArray {
        var x = embeddings
        // Slices and per-layer evals only pay off once the last stage is longer than a slice: below that, the lazy
        // graph is small and one eval per layer costs more than it saves (short text, streaming chunks).
        let upsampling = strides.reduce(1, *)
        let timeChunk = timeChunk > 0 && embeddings.dim(1) * upsampling > timeChunk ? timeChunk : 0

        let windowSizes = Self.windowSizes

        for i in stride(from: 0, to: decoderBlocks.count, by: 2) {
            let stageIdx = i / 2
            let stride_ = strides[stageIdx]
            let isTranspose = stride_ > 1

            // Conv block (even index)
            if let convBlock = decoderBlocks[i] as? ConvBlock {
                x = convBlock.conv(x, stride: stride_, transpose: isTranspose)
            }

            // Transformer block (odd index)
            let window = stageIdx < windowSizes.count ? windowSizes[stageIdx] : 16
            if let xformerBlock = decoderBlocks[i + 1] as? CodecTransformerBlock {
                if timeChunk > 0 {
                    x = xformerBlock.evaluatedByLayer(x, alibiSlopes: _alibiSlopes, windowSize: window, timeChunk: timeChunk)
                } else {
                    x = xformerBlock(x, alibiSlopes: _alibiSlopes, windowSize: window)
                }
            }
        }

        // Output projection
        x = outputProj.conv(x, stride: 1, transpose: false)  // (B, T_up, 240)

        // Reshape to waveform
        let B = x.dim(0)
        return x.reshaped(B, -1)
    }
}
