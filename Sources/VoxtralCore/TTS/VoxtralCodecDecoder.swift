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

// MARK: - Weight-Normalized Conv1d

/// Container for weight norm parameters at the "weight" level.
/// Matches: parametrizations.weight.{original0, original1}
public class WeightNormParams: Module {
    var original0: MLXArray  // (out_ch, 1, 1) — gain
    var original1: MLXArray  // (out_ch, in_ch, K) — direction

    public init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self.original0 = MLX.ones([outChannels, 1, 1])
        self.original1 = MLX.zeros([outChannels, inChannels, kernelSize])
        super.init()
    }
}

/// Container for parametrizations.
/// Matches: parametrizations.weight.*
public class WeightParametrizations: Module {
    @ModuleInfo(key: "weight") var weight: WeightNormParams

    public init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self._weight.wrappedValue = WeightNormParams(outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize)
        super.init()
    }
}

/// Causal Conv1d with weight normalization stored as parametrizations.
/// Weight keys: conv.parametrizations.weight.{original0, original1}
///
/// Reference: audio_tokenizer.py lines 68-142
public class WeightNormConv: Module {

    @ModuleInfo var parametrizations: WeightParametrizations

    public init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self._parametrizations.wrappedValue = WeightParametrizations(
            outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize
        )
        super.init()
    }

    /// Reconstruct weight from weight norm: w = g * v / ||v||
    private func getWeight() -> MLXArray {
        let g = parametrizations.weight.original0  // (out, 1, 1)
        let v = parametrizations.weight.original1  // (out, in, K)
        let vNorm = MLX.sqrt((v * v).sum(axes: [1, 2], keepDims: true) + 1e-12)
        return g * v / vNorm
    }

    /// Apply causal or transposed convolution.
    /// Reference: audio_tokenizer.py lines 97-142
    public func callAsFunction(_ x: MLXArray, stride: Int = 1, transpose: Bool = false) -> MLXArray {
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
            padded = MLX.concatenated([MLX.zeros([B, K - 1, C]), x], axis: 1)
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
public class ConvBlock: Module {
    @ModuleInfo var conv: WeightNormConv

    public init(outChannels: Int, inChannels: Int, kernelSize: Int) {
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
public class CodecAttention: Module {

    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    public init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
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

    public func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int = 0) -> MLXArray {
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

        var scores = MLX.matmul(qh, kh.transposed(0, 1, 3, 2)) * MLXArray(scale)

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
}

// MARK: - Codec Transformer Layer (with LayerScale)

/// Reference: audio_tokenizer.py lines 253-285
public class CodecTransformerLayer: Module {

    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var attention: CodecAttention
    @ModuleInfo(key: "feed_forward") var feedForward: CodecFeedForward

    let useLayerScale: Bool
    var attention_scale: MLXArray
    var ffn_scale: MLXArray

    public init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
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

    public func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int = 0) -> MLXArray {
        var h = attention(attentionNorm(x), alibiSlopes: alibiSlopes, windowSize: windowSize)
        if useLayerScale { h = h * attention_scale }
        var out = x + h

        h = feedForward(ffnNorm(out))
        if useLayerScale { h = h * ffn_scale }
        out = out + h
        return out
    }
}

/// SwiGLU feed-forward matching feed_forward.{w1, w2, w3}
public class CodecFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Transformer Block (N layers)

public class CodecTransformerBlock: Module {
    @ModuleInfo var layers: [CodecTransformerLayer]

    public init(nLayers: Int, config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self._layers.wrappedValue = (0..<nLayers).map { _ in CodecTransformerLayer(config: config) }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, alibiSlopes: MLXArray, windowSize: Int = 0) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, alibiSlopes: alibiSlopes, windowSize: windowSize)
        }
        return h
    }
}

// MARK: - Codebooks

/// EMA-based semantic codebook. Decode in float32 for precision.
/// Reference: audio_tokenizer.py lines 311-338
public class SemanticCodebook: Module {
    var cluster_usage: MLXArray
    var embedding_sum: MLXArray

    public init(codebookSize: Int, dim: Int) {
        self.cluster_usage = MLX.ones([codebookSize])
        self.embedding_sum = MLX.zeros([codebookSize, dim])
        super.init()
    }

    public func decode(_ indices: MLXArray) -> MLXArray {
        // Compute centroids in float32 for precision
        let codebook = embedding_sum.asType(.float32) / MLX.maximum(
            MLX.expandedDimensions(cluster_usage.asType(.float32), axis: -1),
            MLXArray(Float(1e-8))
        )
        return codebook[indices]
    }
}

/// FSQ acoustic codebook — no learned parameters.
/// Reference: audio_tokenizer.py lines 341-351
public class AcousticCodebook: Module {
    let codebookSize: Int

    public init(codebookSize: Int) {
        self.codebookSize = codebookSize
        super.init()
    }

    public func decode(_ indices: MLXArray) -> MLXArray {
        // [0, codebookSize-1] → [-1, 1]
        return (2.0 * indices.asType(.float32) / MLXArray(Float(codebookSize - 1))) - 1.0
    }
}

/// Combined semantic + acoustic codebook.
/// Reference: audio_tokenizer.py lines 354-386
public class MistralAudioCodebook: Module {
    @ModuleInfo(key: "semantic_codebook") var semanticCodebook: SemanticCodebook

    let acousticCodebook: AcousticCodebook

    public init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self._semanticCodebook.wrappedValue = SemanticCodebook(
            codebookSize: config.semanticCodebookSize, dim: config.semanticDim
        )
        self.acousticCodebook = AcousticCodebook(codebookSize: config.acousticCodebookSize)
        super.init()
    }

    /// Decode codes WITH special token offset.
    /// codes: (B, T, 37) where semantic in [2..8193], acoustic in [2..22]
    public func decode(_ codes: MLXArray) -> MLXArray {
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
public class VoxtralCodecDecoder: Module {

    let config: VoxtralTTSConfiguration.AudioTokenizerConfiguration

    @ModuleInfo var quantizer: MistralAudioCodebook
    @ModuleInfo(key: "decoder_blocks") var decoderBlocks: [Module]
    @ModuleInfo(key: "output_proj") var outputProj: ConvBlock

    let alibiSlopes: MLXArray
    let strides: [Int]

    public init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration) {
        self.config = config
        self.alibiSlopes = getAlibiSlopes(nHeads: config.nHeads)
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
    public func decode(_ codes: MLXArray) -> MLXArray {
        var x = quantizer.decode(codes)  // (B, T, 292)

        // Sliding windows: [2, 4, 8, 16] — encoder reversed
        let windowSizes = [2, 4, 8, 16]

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
                x = xformerBlock(x, alibiSlopes: alibiSlopes, windowSize: window)
            }
        }

        // Output projection
        x = outputProj.conv(x, stride: 1, transpose: false)  // (B, T_up, 240)

        // Reshape to waveform
        let B = x.dim(0)
        return x.reshaped(B, -1)
    }
}
