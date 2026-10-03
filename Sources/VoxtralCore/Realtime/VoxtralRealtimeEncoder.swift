/**
 * VoxtralRealtimeEncoder - Causal audio encoder for Voxtral Realtime
 *
 * 32-layer causal transformer with:
 * - Causal conv1d stem (128→1280 stride 1, 1280→1280 stride 2)
 * - Interleaved RoPE (theta=1M)
 * - Sliding window attention (750)
 * - SwiGLU FFN
 * - Selective biases (wq/wv/wo yes, wk no)
 * - 4x downsample + adapter MLP (5120→3072)
 *
 * Reference: mlx-audio/stt/models/voxtral_realtime/encoder.py
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Causal Conv1d

/// Left-padded causal 1D convolution.
class RealtimeCausalConv1d: Module {
    @ModuleInfo var conv: Conv1d
    let padding: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.padding = kernelSize - stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, seq, channels] (MLX conv1d NLC format)
        var padded = x
        if padding > 0 {
            let B = x.dim(0)
            let C = x.dim(2)
            padded = MLX.concatenated([MLX.zeros([B, padding, C]), x], axis: 1)
        }
        return conv(padded)
    }
}

// MARK: - Interleaved RoPE Helper

/// Apply interleaved (GPT-J style) RoPE to a 2D tensor.
/// Rotates consecutive pairs: (x[0], x[1]), (x[2], x[3]), ...
/// x: [seq, n_heads * head_dim], cos/sin: [seq, head_dim / 2]
func interleavedRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray, nHeads: Int, headDim: Int) -> MLXArray {
    let seqLen = x.dim(0)
    let halfDim = headDim / 2
    // Reshape to [seq, heads, half_dim, 2] to split even/odd pairs
    let reshaped = x.reshaped(seqLen, nHeads, halfDim, 2)
    let x1 = reshaped[0..., 0..., 0..., 0]  // even indices [seq, heads, hd/2]
    let x2 = reshaped[0..., 0..., 0..., 1]  // odd indices  [seq, heads, hd/2]

    let cosExp = MLX.expandedDimensions(cos, axis: 1)  // [seq, 1, hd/2]
    let sinExp = MLX.expandedDimensions(sin, axis: 1)

    let o1 = x1 * cosExp - x2 * sinExp
    let o2 = x2 * cosExp + x1 * sinExp

    // Interleave back: stack on last axis then reshape
    let stacked = MLX.stacked([o1, o2], axis: -1)  // [seq, heads, hd/2, 2]
    return stacked.reshaped(seqLen, nHeads * headDim)
}

/// Compute RoPE cos/sin frequencies.
/// positions: [seq_len] int array → (cos, sin) each [seq_len, head_dim/2]
func computeRoPEFreqs(positions: MLXArray, headDim: Int, theta: Float) -> (cos: MLXArray, sin: MLXArray) {
    let freqs = 1.0 / MLX.pow(
        MLXArray(theta),
        MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
    )
    let angles = MLX.expandedDimensions(positions.asType(.float32), axis: -1) * freqs
    return (MLX.cos(angles), MLX.sin(angles))
}

// MARK: - Encoder Attention (Selective Biases)

/// Multi-head attention with selective biases: wq/wv/wo have bias, wk does NOT.
/// Cannot reuse LlamaAttention which uses a single attentionBias for all projections.
class RealtimeEncoderAttention: Module {

    let nHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: RealtimeEncoderConfig) {
        self.nHeads = config.nHeads
        self.headDim = config.headDim
        self.slidingWindow = config.slidingWindow
        self.ropeTheta = config.ropeTheta
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let attnDim = config.nHeads * config.headDim

        // Selective biases: wq, wv, wo have bias; wk does NOT
        self._wq.wrappedValue = Linear(config.dim, attnDim, bias: true)
        self._wk.wrappedValue = Linear(config.dim, attnDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, attnDim, bias: true)
        self._wo.wrappedValue = Linear(attnDim, config.dim, bias: true)

        super.init()
    }

    /// Forward pass.
    /// x: [seq, dim], ropeCos/ropeSin: precomputed [seq, head_dim/2]
    /// mask: precomputed additive mask [seq, kv_len] or nil
    /// cache: optional (keys, values) for streaming
    /// maskMode: explicit mask (sliding-window chunks, K-13); takes precedence over `mask`
    func callAsFunction(
        _ x: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil,
        maskMode explicitMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        let seqLen = x.dim(0)

        var q = wq(x)
        var k = wk(x)
        let v = wv(x)

        // Apply interleaved RoPE
        q = interleavedRoPE(q, cos: ropeCos, sin: ropeSin, nHeads: nHeads, headDim: headDim)
        k = interleavedRoPE(k, cos: ropeCos, sin: ropeSin, nHeads: nHeads, headDim: headDim)

        // Reshape for SDPA: [1, heads, seq, head_dim]
        var qh = q.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        var kh = k.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        var vh = v.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)

        // Update KV cache if provided
        if let cache {
            (kh, vh) = cache.update(keys: kh, values: vh)
        }

        // Attention
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let explicitMask {
            maskMode = explicitMask
        } else if let mask {
            maskMode = .array(mask.expandedDimensions(axes: [0, 1]))
        } else if seqLen > 1 && cache == nil {
            maskMode = .causal
        } else {
            maskMode = .none
        }

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh,
            scale: scale, mask: maskMode
        )

        // [1, heads, seq, head_dim] → [seq, heads * head_dim]
        let output = attnOut.transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)
        return wo(output)
    }
}

// MARK: - Encoder SwiGLU FFN

/// SwiGLU feed-forward for encoder. w1=gate(no bias), w3=up(no bias), w2=down(bias).
class RealtimeEncoderFFN: Module {
    @ModuleInfo(key: "feed_forward_w1") var w1: Linear
    @ModuleInfo(key: "feed_forward_w2") var w2: Linear
    @ModuleInfo(key: "feed_forward_w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Encoder Layer

/// Single causal encoder transformer layer: RMSNorm + attention + SwiGLU FFN.
/// FFN weights are flat on the layer (feed_forward_w1/w2/w3), matching Python structure.
class RealtimeEncoderLayer: Module {

    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo var attention: RealtimeEncoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    // SwiGLU FFN directly on layer (not nested in sub-module)
    @ModuleInfo(key: "feed_forward_w1") var w1: Linear
    @ModuleInfo(key: "feed_forward_w2") var w2: Linear
    @ModuleInfo(key: "feed_forward_w3") var w3: Linear

    init(config: RealtimeEncoderConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = RealtimeEncoderAttention(config: config)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        // w1=gate (no bias), w3=up (no bias), w2=down (bias)
        self._w1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._w3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        // Attention with pre-norm
        var h = attentionNorm(x)
        h = attention(h, ropeCos: ropeCos, ropeSin: ropeSin, mask: mask, cache: cache, maskMode: maskMode)
        var out = x + h

        // SwiGLU FFN with pre-norm
        h = ffnNorm(out)
        let gate = silu(w1(h))
        let up = w3(h)
        out = out + w2(gate * up)

        return out
    }
}

// MARK: - Audio Encoder

/// Complete causal audio encoder: conv stem → transformer → norm → 4x downsample+project.
class VoxtralRealtimeEncoder: Module {

    let config: RealtimeEncoderConfig

    // Conv stem
    @ModuleInfo(key: "conv_layers_0_conv") var conv1: RealtimeCausalConv1d
    @ModuleInfo(key: "conv_layers_1_conv") var conv2: RealtimeCausalConv1d

    // Transformer layers
    @ModuleInfo(key: "transformer_layers") var layers: [RealtimeEncoderLayer]
    @ModuleInfo(key: "transformer_norm") var transformerNorm: RMSNorm

    // Adapter MLP (4x downsample → project to decoder dim)
    @ModuleInfo(key: "audio_language_projection_0") var adapterLinear1: Linear
    @ModuleInfo(key: "audio_language_projection_2") var adapterLinear2: Linear

    init(config: RealtimeEncoderConfig, decoderDim: Int) {
        self.config = config

        self._conv1.wrappedValue = RealtimeCausalConv1d(
            inChannels: config.audioEncodingArgs.numMelBins,
            outChannels: config.dim,
            kernelSize: 3,
            stride: 1
        )
        self._conv2.wrappedValue = RealtimeCausalConv1d(
            inChannels: config.dim,
            outChannels: config.dim,
            kernelSize: 3,
            stride: 2
        )

        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            RealtimeEncoderLayer(config: config)
        }
        self._transformerNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        let adapterInputDim = config.dim * config.downsampleFactor  // 1280 * 4 = 5120
        self._adapterLinear1.wrappedValue = Linear(adapterInputDim, decoderDim, bias: false)
        self._adapterLinear2.wrappedValue = Linear(decoderDim, decoderDim, bias: false)

        super.init()
    }

    // MARK: - Conv Stem

    /// Run conv layers on mel spectrogram. Aligns output to downsample factor.
    /// mel: [mel_bins, frames] → returns [seq, dim]
    func convStem(_ mel: MLXArray) -> MLXArray {
        // mel: [128, frames] → [1, frames, 128] (NLC for conv1d)
        var x = mel.transposed(1, 0).expandedDimensions(axis: 0)
        x = gelu(conv1(x))
        x = gelu(conv2(x))
        x = x.squeezed(axis: 0)  // [seq, 1280]

        // Trim to align with downsample factor
        let trunc = x.dim(0) % config.downsampleFactor
        if trunc > 0 {
            x = x[trunc..., 0...]
        }
        return x
    }

    /// The conv stem evaluated in slices of `chunkFrames` mel frames (even), each with 4 frames of left context: conv1
    /// (k 3, s 1) and conv2 (k 3, s 2) are causal, so output j only reads mel frames 2j-3 … 2j+1 and the result equals
    /// `convStem(mel)`. Each slice is evaluated, and a cancelled caller stops between slices instead of waiting for
    /// the whole audio (≈ 3 s on 11 min, K-15). Returns nil when cancelled.
    func convStemChunked(_ mel: MLXArray, chunkFrames: Int = 6000) -> MLXArray? {
        let frames = mel.dim(1)
        precondition(chunkFrames > 0 && chunkFrames % 2 == 0, "chunkFrames must be even")
        let context = 4
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: frames, by: chunkFrames) {
            if VoxtralCancellation.isCancelled { return nil }
            let end = min(start + chunkFrames, frames)
            let from = max(0, start - context)
            var x = mel[0..., from ..< end].transposed(1, 0).expandedDimensions(axis: 0)
            x = gelu(conv1(x))
            x = gelu(conv2(x))
            x = x.squeezed(axis: 0)
            // Outputs computed from the left context (zero-padded inside this slice) are dropped: 2 when there is one
            let piece = start == 0 ? x : x[((start - from) / 2)...]
            MLX.eval(piece)
            pieces.append(piece)
        }
        var x = MLX.concatenated(pieces, axis: 0)
        let trunc = x.dim(0) % config.downsampleFactor
        if trunc > 0 {
            x = x[trunc..., 0...]
        }
        return x
    }

    // MARK: - Transformer + Adapter

    /// Downsample 4x and project to decoder dimension.
    /// encoded: [seq, dim] → [seq/4, decoder_dim]
    func downsampleAndProject(_ encoded: MLXArray) -> MLXArray {
        let seqLen = encoded.dim(0)
        let ds = config.downsampleFactor
        let dsLen = seqLen / ds
        guard dsLen > 0 else { return encoded[..<0] }

        let x = encoded[..<(dsLen * ds)].reshaped(dsLen, config.dim * ds)
        return adapterLinear2(gelu(adapterLinear1(x)))
    }

    /// Full encode: conv stem → transformer → norm → downsample → project.
    /// Beyond the sliding window the encoder attends within the window only (K-13, P-62): a full causal
    /// pass would attend past 15 s, a context the model never saw, and degenerate.
    /// mel: [mel_bins, frames] → [seq/4, decoder_dim]
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // Within the window (mel ≤ 2 × window frames): one pass, full causal attention
        if mel.dim(1) <= 2 * config.slidingWindow {
            let convOut = convStem(mel)
            if convOut.dim(0) <= config.slidingWindow {
                return encodeFull(convOut)
            }
            MLX.eval(convOut)
            return downsampleAndProject(encodeChunked(convOut))
        }
        // Long audio: the conv stem slice by slice, then the transformer chunk by chunk, so a cancelled caller stops
        // at the next slice or chunk boundary (K-15); the pipeline then throws
        guard let convOut = convStemChunked(mel) else { return MLXArray.zeros([0, config.dim]) }  // cancelled
        return downsampleAndProject(encodeChunked(convOut))
    }

    /// Encode window-sized chunks with a rotating KV cache per layer (mlx-audio `encode_chunks`):
    /// query i attends to keys i-window+1 … i, at absolute RoPE positions. Memory is O(window) per chunk.
    /// convOut: [seq, dim] → normed [seq, dim]
    func encodeChunked(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.dim(0)
        let window = config.slidingWindow
        let caches = layers.map { _ in RotatingKVCache(maxSize: window, keep: 0) }
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: seqLen, by: window) {
            // A cancelled caller stops at a chunk boundary (≈ 0,4 s); the pipeline then throws (K-15)
            if !chunks.isEmpty, VoxtralCancellation.isCancelled { break }
            let end = min(start + window, seqLen)
            let (ropeCos, ropeSin) = computeRoPEFreqs(
                positions: MLXArray(Int32(start) ..< Int32(end)), headDim: config.headDim, theta: config.ropeTheta
            )
            // The mask depends only on the chunk length and the cache offset: shared by every layer
            let maskMode = caches[0].makeMask(n: end - start, windowSize: window, returnArray: false)
            var x = convOut[start ..< end]
            for (layer, cache) in zip(layers, caches) {
                x = layer(x, ropeCos: ropeCos, ropeSin: ropeSin, cache: cache, maskMode: maskMode)
            }
            x = transformerNorm(x)
            MLX.eval(x)
            chunks.append(x)
        }
        return MLX.concatenated(chunks, axis: 0)
    }

    /// Encode using full-sequence causal attention (for audio within sliding window).
    /// convOut: [seq, dim] → [seq/4, decoder_dim]
    func encodeFull(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.dim(0)
        let positions = MLXArray(0..<Int32(seqLen))
        let (ropeCos, ropeSin) = computeRoPEFreqs(
            positions: positions, headDim: config.headDim, theta: config.ropeTheta
        )

        var x = convOut
        for layer in layers {
            x = layer(x, ropeCos: ropeCos, ropeSin: ropeSin)
        }
        x = transformerNorm(x)
        return downsampleAndProject(x)
    }
}
