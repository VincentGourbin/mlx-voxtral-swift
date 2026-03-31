/**
 * VoxtralVoiceSLERP - Spherical Linear Interpolation for voice embeddings
 *
 * Voice embeddings are 2D tensors [N, 3072] where N varies per voice (67–218).
 * SLERP is applied row-wise across the 3072-dim hidden space after length alignment.
 *
 * Reference: MushroomFleet/ZeroVoice-Voxtral-mini-4b/slerp_voices.py
 */

import Foundation
import MLX

// MARK: - Length Alignment

/// Resample a [N, D] tensor to [targetLen, D] via linear interpolation.
public func resampleVoiceEmbedding(_ t: MLXArray, targetLen: Int) -> MLXArray {
    let currentLen = t.dim(0)
    if currentLen == targetLen { return t }

    // Compute source positions for each target position
    let srcPositions = MLXArray(
        (0..<targetLen).map { Float($0) * Float(currentLen - 1) / Float(max(targetLen - 1, 1)) }
    )
    let srcFloor = MLX.floor(srcPositions).asType(.int32)
    let srcCeil = MLX.minimum(srcFloor + 1, MLXArray(Int32(currentLen - 1)))
    let frac = MLX.expandedDimensions(srcPositions - srcFloor.asType(.float32), axis: -1)

    let floorRows = t[srcFloor]
    let ceilRows = t[srcCeil]
    return floorRows.asType(.float32) * (1.0 - frac) + ceilRows.asType(.float32) * frac
}

/// Align two voice embeddings to the same sequence length by resampling shorter to longer.
public func alignVoiceLengths(_ a: MLXArray, _ b: MLXArray) -> (MLXArray, MLXArray) {
    let Na = a.dim(0)
    let Nb = b.dim(0)
    if Na == Nb { return (a, b) }

    let targetLen = max(Na, Nb)
    let aAligned = Na < targetLen ? resampleVoiceEmbedding(a, targetLen: targetLen) : a
    let bAligned = Nb < targetLen ? resampleVoiceEmbedding(b, targetLen: targetLen) : b
    return (aAligned, bAligned)
}

// MARK: - SLERP

/// Row-wise Spherical Linear Interpolation between two voice embeddings.
public func slerpVoices(_ a: MLXArray, _ b: MLXArray, t: Float) -> MLXArray {
    let aF = a.asType(.float32)
    let bF = b.asType(.float32)

    // Row norms for magnitude restoration
    let aMag = MLX.sqrt(MLX.maximum((aF * aF).sum(axis: -1, keepDims: true), MLXArray(Float(1e-16))))
    let bMag = MLX.sqrt(MLX.maximum((bF * bF).sum(axis: -1, keepDims: true), MLXArray(Float(1e-16))))

    // Normalize to unit sphere
    let aNorm = aF / aMag
    let bNorm = bF / bMag

    // Per-row cosine similarity → angle
    let dot = MLX.clip((aNorm * bNorm).sum(axis: -1, keepDims: true),
                        min: MLXArray(Float(-1.0 + 1e-7)), max: MLXArray(Float(1.0 - 1e-7)))
    let omega = MLX.acos(dot)
    let sinOmega = MLX.sin(omega)

    // SLERP weights
    let safeSin = MLX.maximum(sinOmega, MLXArray(Float(1e-8)))
    let weightA = MLX.sin((1.0 - t) * omega) / safeSin
    let weightB = MLX.sin(MLXArray(t) * omega) / safeSin
    var result = weightA * aF + weightB * bF

    // LERP fallback for near-parallel rows (sinOmega ≈ 0)
    let useLinear = sinOmega .< MLXArray(Float(1e-6))
    let lerpResult = (1.0 - t) * aF + MLXArray(t) * bF
    result = MLX.where(useLinear, lerpResult, result)

    // Restore interpolated magnitude
    let targetMag = (1.0 - t) * aMag + MLXArray(t) * bMag
    let resultMag = MLX.sqrt(MLX.maximum((result * result).sum(axis: -1, keepDims: true), MLXArray(Float(1e-16))))
    result = result * (targetMag / resultMag)

    return result
}

// MARK: - Norm Calibration

/// Rescale rows so the mean row norm matches the preset average (~4.48).
public func calibrateVoiceNorms(_ result: MLXArray, targetMeanNorm: Float = 4.48) -> MLXArray {
    let f = result.asType(.float32)
    let rowNorms = MLX.sqrt(MLX.maximum((f * f).sum(axis: -1, keepDims: true), MLXArray(Float(1e-16))))
    let currentMean = rowNorms.mean()
    let scale = MLXArray(targetMeanNorm) / MLX.maximum(currentMean, MLXArray(Float(1e-8)))
    return f * scale
}

// MARK: - Convenience

/// Blend two voice embeddings: align lengths → SLERP → calibrate norms.
public func blendVoices(voiceA: MLXArray, voiceB: MLXArray, t: Float) -> MLXArray {
    let t = max(0.0, min(1.0, t))
    let (aAligned, bAligned) = alignVoiceLengths(voiceA, voiceB)
    let blended = slerpVoices(aAligned, bAligned, t: t)
    return calibrateVoiceNorms(blended)
}
