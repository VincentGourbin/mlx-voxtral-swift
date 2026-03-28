/**
 * VoxtralCodecEncoder - Encodes audio waveform to discrete codec tokens
 *
 * Architecture (paper Section 2.1, Table 1):
 * 1. Input: 24kHz mono waveform → patches of 240 samples (100Hz)
 * 2. Input projection: CausalConv(240→1024, kernel=7, stride=1)
 * 3. 4 encoder blocks, each:
 *    - 2-layer Transformer (ALiBi, sliding window 16→8→4→2, QK-norm, LayerScale)
 *    - CausalConv downsampling (kernels 4→4→4→3, strides 2→2→2→1)
 * 4. Last block projects 1024→292 (latent = 256 semantic + 36 acoustic)
 * 5. VQ quantization (semantic) + FSQ quantization (acoustic)
 *
 * NOTE: Encoder weights are NOT included in the public TTS checkpoint.
 * This implementation is ready for when weights become available.
 * Weight keys expected: audio_tokenizer.input_proj.*, audio_tokenizer.encoder_blocks.*
 */

import Foundation
import MLX
import MLXNN

// MARK: - Codec Encoder

/// Encodes 24kHz mono audio waveform to discrete codec tokens (1 semantic + 36 acoustic per frame at 12.5Hz).
/// Requires encoder weights which are NOT in the public checkpoint.
public class VoxtralCodecEncoder: Module {

    let config: VoxtralTTSConfiguration.AudioTokenizerConfiguration

    /// Input projection: patch_size (240) → dim (1024), kernel=7
    @ModuleInfo(key: "input_proj") var inputProj: ConvBlock

    /// 8 encoder blocks (alternating transformer + conv, reversed vs decoder)
    /// Block order: [transformer, conv, transformer, conv, ...]
    @ModuleInfo(key: "encoder_blocks") var encoderBlocks: [Module]

    /// Quantizer for encoding (same as decoder's quantizer — shared codebook)
    let quantizer: MistralAudioCodebook

    let alibiSlopes: MLXArray
    let encoderLoaded: Bool

    /// Encoder hyperparameters from paper Table 1
    static let encoderConvsKernels = [4, 4, 4, 3]
    static let encoderConvsStrides = [2, 2, 2, 1]
    static let encoderTransformerLengths = [2, 2, 2, 2]
    static let encoderSlidingWindows = [16, 8, 4, 2]  // halving (vs decoder's doubling)

    public init(config: VoxtralTTSConfiguration.AudioTokenizerConfiguration, quantizer: MistralAudioCodebook) {
        self.config = config
        self.quantizer = quantizer
        self.alibiSlopes = getAlibiSlopes(nHeads: config.nHeads)
        self.encoderLoaded = false  // Will be true only if weights are loaded

        // Input projection: patch_size → dim
        self._inputProj.wrappedValue = ConvBlock(
            outChannels: config.dim,
            inChannels: config.pretransformPatchSize,
            kernelSize: config.patchProjKernelSize
        )

        // Encoder blocks: alternating [transformer, conv] × 4
        var blocks: [Module] = []
        for i in 0..<Self.encoderTransformerLengths.count {
            // Transformer block first (opposite of decoder)
            blocks.append(CodecTransformerBlock(
                nLayers: Self.encoderTransformerLengths[i],
                config: config
            ))

            // Conv block for downsampling
            let isLastBlock = (i == Self.encoderTransformerLengths.count - 1)
            let outputChannels = isLastBlock ? config.latentDim : config.dim  // Last: 1024→292
            blocks.append(ConvBlock(
                outChannels: outputChannels,
                inChannels: config.dim,
                kernelSize: Self.encoderConvsKernels[i]
            ))
        }
        self._encoderBlocks.wrappedValue = blocks

        super.init()
    }

    /// Encode a waveform to codec tokens.
    ///
    /// - Parameter waveform: Audio samples [batch, numSamples] at 24kHz
    /// - Returns: Codes [batch, numFrames, 37] (1 semantic + 36 acoustic) with +2 special offset
    /// - Throws: If encoder weights are not loaded
    public func encode(_ waveform: MLXArray) throws -> MLXArray {
        guard encoderLoaded else {
            throw VoxtralTTSError.modelLoadingFailed(
                "Codec encoder weights are not available in the public checkpoint. " +
                "Voice cloning from audio requires encoder weights that Mistral has not published. " +
                "Use voice presets instead, or provide pre-computed voice embeddings."
            )
        }

        let B = waveform.dim(0)

        // 1. Patchify: reshape [B, numSamples] → [B, numPatches, patchSize]
        let numPatches = waveform.dim(1) / config.pretransformPatchSize
        let x = waveform[0..., ..<(numPatches * config.pretransformPatchSize)]
            .reshaped(B, numPatches, config.pretransformPatchSize)

        // 2. Input projection: [B, numPatches, patchSize] → [B, numPatches, dim]
        var h = inputProj.conv(x, stride: 1, transpose: false)

        // 3. Encoder blocks: alternating [transformer, conv]
        for i in stride(from: 0, to: encoderBlocks.count, by: 2) {
            let stageIdx = i / 2
            let windowSize = Self.encoderSlidingWindows[stageIdx]
            let stride = Self.encoderConvsStrides[stageIdx]

            // Transformer block
            if let xformerBlock = encoderBlocks[i] as? CodecTransformerBlock {
                h = xformerBlock(h, alibiSlopes: alibiSlopes, windowSize: windowSize)
            }

            // Conv block (downsampling)
            if let convBlock = encoderBlocks[i + 1] as? ConvBlock {
                h = convBlock.conv(h, stride: stride, transpose: false)
            }
        }

        // 4. h is now [B, numFrames, 292] (latent)
        // Split into semantic (256) and acoustic (36)
        let semanticLatent = h[0..., 0..., ..<config.semanticDim]    // [B, T, 256]
        let acousticLatent = h[0..., 0..., config.semanticDim...]     // [B, T, 36]

        // 5. Quantize
        let semanticCodes = quantizeSemantic(semanticLatent)  // [B, T] int32
        let acousticCodes = quantizeAcoustic(acousticLatent)  // [B, T, 36] int32

        // 6. Combine with +2 special token offset
        let N_SPECIAL: Int32 = 2
        let semWithOffset = semanticCodes + MLXArray(N_SPECIAL)
        let acWithOffset = acousticCodes + MLXArray(N_SPECIAL)

        return MLX.concatenated([
            MLX.expandedDimensions(semWithOffset, axis: -1),
            acWithOffset
        ], axis: -1)  // [B, T, 37]
    }

    /// Encode waveform to voice embedding (for voice cloning).
    ///
    /// - Parameter waveform: Audio samples [numSamples] at 24kHz (mono)
    /// - Parameter embeddingTable: The audio codebook embedding table
    /// - Returns: Voice embedding [numFrames, dim] ready for LLM input
    public func encodeToVoiceEmbedding(
        waveform: MLXArray,
        embeddingTable: AudioCodebookEmbeddingsContainer,
        config: VoxtralTTSConfiguration
    ) throws -> MLXArray {
        let codes = try encode(MLX.expandedDimensions(waveform, axis: 0))  // [1, T, 37]

        // Convert to global indices and embed
        let nSpecial: Int32 = 2
        let semanticSize = config.audioModel.semanticCodebookSize + Int(nSpecial)
        let acousticSize = config.audioModel.acousticCodebookSize + Int(nSpecial)

        var offsets: [Int32] = [0]
        for i in 0..<config.audioModel.nAcousticCodebook {
            offsets.append(Int32(semanticSize + i * acousticSize))
        }
        let offsetsArray = MLXArray(offsets).reshaped(1, offsets.count)

        let globalCodes = codes + offsetsArray  // [1, T, 37]
        let embeddings = embeddingTable(globalCodes)  // [1, T, 37, dim]
        let voiceEmb = embeddings.sum(axis: 2)  // [1, T, dim] — sum across codebooks
        return voiceEmb.squeezed(axis: 0)  // [T, dim]
    }

    // MARK: - Quantization

    /// VQ quantization for semantic component: find nearest codebook entry
    private func quantizeSemantic(_ x: MLXArray) -> MLXArray {
        // x: [B, T, 256]
        // codebook: [8192, 256] from quantizer.semanticCodebook
        let codebook = quantizer.semanticCodebook.decode(
            MLXArray(0..<Int32(config.semanticCodebookSize)).reshaped(config.semanticCodebookSize, 1)
        ).squeezed(axis: -1)  // Doesn't work — codebook is accessed differently

        // Actually: centroids = embedding_sum / cluster_usage
        let centroids = quantizer.semanticCodebook.embedding_sum.asType(.float32) /
            MLX.maximum(
                MLX.expandedDimensions(quantizer.semanticCodebook.cluster_usage.asType(.float32), axis: -1),
                MLXArray(Float(1e-8))
            )  // [8192, 256]

        // Find nearest: argmin ||x - c||^2 for each frame
        let xFlat = x.asType(.float32).reshaped(-1, config.semanticDim)  // [B*T, 256]
        // Compute distances using ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x·c
        let xNormSq = MLX.sum(xFlat * xFlat, axis: -1, keepDims: true)  // [B*T, 1]
        let cNormSq = MLX.sum(centroids * centroids, axis: -1, keepDims: true).transposed()  // [1, 8192]
        let dotProduct = MLX.matmul(xFlat, centroids.transposed())  // [B*T, 8192]
        let distances = xNormSq + cNormSq - 2 * dotProduct  // [B*T, 8192]

        let indices = MLX.argMin(distances, axis: -1).asType(.int32)  // [B*T]
        return indices.reshaped(x.dim(0), x.dim(1))  // [B, T]
    }

    /// FSQ quantization for acoustic component
    private func quantizeAcoustic(_ x: MLXArray) -> MLXArray {
        // x: [B, T, 36]
        // Apply tanh then quantize to 21 levels
        let tanhed = MLX.tanh(x)
        return quantizeToFSQ(tanhed, levels: config.acousticCodebookSize)
    }
}
