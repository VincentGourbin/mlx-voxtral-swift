/**
 * VoxtralTTSProcessor - Input processing and output utilities for TTS
 *
 * Contains: WAVWriter, TTSSynthesisResult
 * Input construction is now handled directly by VoxtralTTSModel.
 */

import Foundation
import MLX

// MARK: - TTS Synthesis Result

public struct TTSSynthesisResult: @unchecked Sendable {
    public let waveform: MLXArray
    /// Number of generated codec token frames at 12.5 Hz — NOT audio samples.
    /// One frame = 80 ms = 1920 samples at 24 kHz; multiplying `numFrames` by
    /// anything other than 1920 under-estimates the audio length. For the
    /// audio's wall-clock length use `duration`, which is computed from the
    /// decoded waveform.
    public let numFrames: Int
    public let sampleRate: Int
    public let generationTime: TimeInterval
    /// Time to first token: prefill + first frame generation
    public let timeToFirstToken: TimeInterval

    public var duration: TimeInterval {
        Double(waveform.dim(0)) / Double(sampleRate)
    }

    public var realTimeFactor: Double {
        guard duration > 0 else { return 0 }
        return generationTime / duration
    }

    public var framesPerSecond: Double {
        guard generationTime > 0 else { return 0 }
        return Double(numFrames) / generationTime
    }
}

// MARK: - Silence Trimming

/// Per-frame (80 ms) RMS threshold RELATIVE to the clip's peak amplitude.
///
/// A fixed absolute threshold fails on cloned/enrolled voices: the enrollment
/// optimization reproduces the reference recording's noise floor, so the
/// "silence" frames of a synthesis never drop below an absolute value tuned
/// on the clean presets (measured: enrolled-voice silences at −32.5 dBFS vs
/// −35.8 dBFS for presets — an absolute 0.025 never triggered). Relative to
/// peak, silence and speech stay separable regardless of the voice's floor.
/// `absoluteFloor` keeps near-digital-silence clips from producing a
/// meaninglessly low threshold.
private func silenceThreshold(
    _ samples: MLXArray, relativeThresholdDB: Float, absoluteFloor: Float
) -> Float {
    let peak = MLX.abs(samples).max().item(Float.self)
    return max(absoluteFloor, peak * Foundation.pow(10, relativeThresholdDB / 20))
}

private let samplesPerFrame = 1920  // 80ms at 24kHz (8x upsample * 240 patch)

private func frameRMS(_ samples: MLXArray, frame: Int, totalSamples: Int) -> Float {
    let start = frame * samplesPerFrame
    let end = min(start + samplesPerFrame, totalSamples)
    let chunk = samples[start..<end]
    return MLX.sqrt(MLX.mean(chunk * chunk)).item(Float.self)
}

/// Trim low-energy lead-in frames from waveform.
/// Voxtral TTS often generates a few silence/transition frames before speech starts,
/// especially noticeable with non-English voices. This removes them for cleaner output.
/// The threshold is relative to the clip's peak (default peak − 25 dB) — see
/// `silenceThreshold` for why absolute thresholds fail on enrolled voices.
public func trimLeadInSilence(
    _ waveform: MLXArray,
    sampleRate: Int = 24000,
    relativeThresholdDB: Float = -25,
    absoluteFloor: Float = 0.001
) -> MLXArray {
    let totalSamples = waveform.dim(0)
    let totalFrames = totalSamples / samplesPerFrame

    let samples = waveform.asType(.float32)
    let threshold = silenceThreshold(
        samples, relativeThresholdDB: relativeThresholdDB, absoluteFloor: absoluteFloor)

    // Scan frame-by-frame, find the first frame above threshold RMS
    var trimFrames = 0
    for i in 0..<min(totalFrames, 20) {  // Check at most first 20 frames (1.6s)
        if frameRMS(samples, frame: i, totalSamples: totalSamples) >= threshold {
            break
        }
        trimFrames += 1
    }

    if trimFrames > 0 {
        let trimSamples = trimFrames * samplesPerFrame
        if trimSamples < totalSamples {
            return waveform[trimSamples...]
        }
    }
    return waveform
}

/// Deprecated absolute-threshold variant, kept for source compatibility.
@available(*, deprecated, message: "A fixed absolute threshold never triggers on enrolled voices; use trimLeadInSilence(_:sampleRate:relativeThresholdDB:absoluteFloor:)")
public func trimLeadInSilence(_ waveform: MLXArray, sampleRate: Int = 24000, threshold: Float) -> MLXArray {
    let totalSamples = waveform.dim(0)
    let totalFrames = totalSamples / samplesPerFrame
    let samples = waveform.asType(.float32)
    var trimFrames = 0
    for i in 0..<min(totalFrames, 20) {
        if frameRMS(samples, frame: i, totalSamples: totalSamples) >= threshold { break }
        trimFrames += 1
    }
    if trimFrames > 0, trimFrames * samplesPerFrame < totalSamples {
        return waveform[(trimFrames * samplesPerFrame)...]
    }
    return waveform
}

/// Trim low-energy trailing frames from waveform (fade-out / hang after the
/// last word). Same relative threshold as `trimLeadInSilence`; scans at most
/// the last 20 frames (1.6 s) and always keeps at least one frame.
public func trimTrailingSilence(
    _ waveform: MLXArray,
    sampleRate: Int = 24000,
    relativeThresholdDB: Float = -25,
    absoluteFloor: Float = 0.001
) -> MLXArray {
    let totalSamples = waveform.dim(0)
    let totalFrames = totalSamples / samplesPerFrame
    guard totalFrames > 1 else { return waveform }

    let samples = waveform.asType(.float32)
    let threshold = silenceThreshold(
        samples, relativeThresholdDB: relativeThresholdDB, absoluteFloor: absoluteFloor)

    // Scan from the end, find the last frame above threshold RMS. The final
    // partial frame (< 80 ms remainder) is treated as part of the last frame.
    var trimFrames = 0
    for i in stride(from: totalFrames - 1, through: max(0, totalFrames - 20), by: -1) {
        if frameRMS(samples, frame: i, totalSamples: totalSamples) >= threshold {
            break
        }
        trimFrames += 1
    }

    if trimFrames > 0, trimFrames < totalFrames {
        // Cut at the frame boundary; drop the trailing partial frame too.
        let keepSamples = (totalFrames - trimFrames) * samplesPerFrame
        return waveform[0..<keepSamples]
    }
    return waveform
}

// MARK: - TTS Streaming Chunk

/// A chunk of decoded audio from the streaming TTS pipeline.
public struct TTSStreamingChunk: @unchecked Sendable {
    /// Decoded waveform samples for this chunk (float32 PCM, 24kHz mono)
    public let waveform: MLXArray
    /// Index of the first frame in this chunk
    public let frameIndex: Int
    /// Number of new frames decoded in this chunk
    public let frameCount: Int
    /// Total frames generated so far
    public let totalFrames: Int
    /// Sample rate
    public let sampleRate: Int
    /// Whether this is the first chunk (use for TTFT measurement)
    public let isFirst: Bool
    /// Whether this is the final chunk
    public let isFinal: Bool
    /// Time elapsed since generation started
    public let elapsed: TimeInterval

    /// Duration of audio in this chunk
    public var duration: TimeInterval {
        Double(waveform.dim(0)) / Double(sampleRate)
    }
}

// MARK: - WAV File Writer

public struct WAVWriter {

    public static func write(
        waveform: MLXArray,
        to url: URL,
        sampleRate: Int = 24000,
        bitDepth: Int = 16
    ) throws {
        let samples = waveform.asType(.float32)
        let numSamples = samples.dim(0)
        let numChannels: Int = 1

        let maxVal = Float(Int16.max)
        let clipped = MLX.clip(samples, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))
        let scaled = (clipped * MLXArray(maxVal)).asType(.int16)

        var data = Data()

        let dataSize = numSamples * numChannels * (bitDepth / 8)
        let fileSize = 36 + dataSize

        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(numChannels).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        let byteRate = sampleRate * numChannels * (bitDepth / 8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
        let blockAlign = numChannels * (bitDepth / 8)
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(bitDepth).littleEndian) { Array($0) })

        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

        // Bulk transfer GPU→CPU: single copy instead of per-sample .item() calls
        MLX.eval(scaled)
        scaled.asArray(Int16.self).withUnsafeBufferPointer { buffer in
            data.append(buffer)
        }

        try data.write(to: url)
    }
}
