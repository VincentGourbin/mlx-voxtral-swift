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

/// Samples per 80 ms codec frame at the given rate (1920 at 24 kHz:
/// 8x upsample * 240 patch).
private func samplesPerFrame(at sampleRate: Int) -> Int {
    sampleRate * 2 / 25
}

private func rms(_ samples: MLXArray, _ start: Int, _ end: Int) -> Float {
    let chunk = samples[start..<end]
    return MLX.sqrt(MLX.mean(chunk * chunk)).item(Float.self)
}

/// Scan cap: 50 frames = 4 s. Enrolled voices can generate well over 20
/// frames (1.6 s) of noise-floor lead-in — measured 22 on a real clone —
/// so the historical 20-frame cap left audible noise at the start.
private let maxTrimScanFrames = 50

/// Shared scan-and-count core: number of consecutive sub-threshold frames
/// from the start (or, reversed, from the end), capped at `maxTrimScanFrames`.
/// When scanning from the end, the final partial frame (< 80 ms remainder)
/// is folded into the last frame so its energy is never dropped unseen.
private func quietFrameCount(
    _ samples: MLXArray, totalSamples: Int, frameSize: Int, totalFrames: Int,
    threshold: Float, fromEnd: Bool
) -> Int {
    let indices = fromEnd
        ? Array((max(0, totalFrames - maxTrimScanFrames) ..< totalFrames).reversed())
        : Array(0 ..< min(totalFrames, maxTrimScanFrames))
    var quiet = 0
    for i in indices {
        let start = i * frameSize
        let end = (fromEnd && i == totalFrames - 1) ? totalSamples : min(start + frameSize, totalSamples)
        if rms(samples, start, end) >= threshold { break }
        quiet += 1
    }
    return quiet
}

/// Trim low-energy lead-in frames from waveform.
/// Voxtral TTS often generates a few silence/transition frames before speech starts,
/// especially noticeable with non-English voices. This removes them for cleaner output.
/// The threshold is relative to the clip's peak (default peak − 25 dB) — see
/// `silenceThreshold` for why absolute thresholds fail on enrolled voices.
///
/// - Note: Behavior change vs the old absolute threshold (0.025), in both
///   directions: on a full-scale clip the relative threshold sits higher
///   (~0.056), so a very soft onset frame that used to survive may now be
///   trimmed; on a quiet clip (peak < 0.44) it sits lower, so faint lead-in
///   ambience that used to be cut is now kept.
public func trimLeadInSilence(
    _ waveform: MLXArray,
    sampleRate: Int = 24000,
    relativeThresholdDB: Float = -25,
    absoluteFloor: Float = 0.001
) -> MLXArray {
    let totalSamples = waveform.dim(0)
    let frameSize = samplesPerFrame(at: sampleRate)
    let totalFrames = totalSamples / frameSize
    guard totalFrames > 0 else { return waveform }

    let samples = waveform.asType(.float32)
    let threshold = silenceThreshold(
        samples, relativeThresholdDB: relativeThresholdDB, absoluteFloor: absoluteFloor)
    return trimLead(waveform, samples: samples, totalSamples: totalSamples,
                    frameSize: frameSize, totalFrames: totalFrames, threshold: threshold)
}

private func trimLead(
    _ waveform: MLXArray, samples: MLXArray, totalSamples: Int,
    frameSize: Int, totalFrames: Int, threshold: Float
) -> MLXArray {
    let trimFrames = quietFrameCount(
        samples, totalSamples: totalSamples, frameSize: frameSize,
        totalFrames: totalFrames, threshold: threshold, fromEnd: false)
    if trimFrames > 0, trimFrames * frameSize < totalSamples {
        return waveform[(trimFrames * frameSize)...]
    }
    return waveform
}

/// Deprecated absolute-threshold variant. Note it only preserves the old
/// semantics for calls that spell out `threshold:` — calls that omitted the
/// argument resolve to the new relative-threshold function above.
@available(*, deprecated, message: "A fixed absolute threshold never triggers on enrolled voices; use trimLeadInSilence(_:sampleRate:relativeThresholdDB:absoluteFloor:)")
public func trimLeadInSilence(_ waveform: MLXArray, sampleRate: Int = 24000, threshold: Float) -> MLXArray {
    let totalSamples = waveform.dim(0)
    let frameSize = samplesPerFrame(at: sampleRate)
    let totalFrames = totalSamples / frameSize
    guard totalFrames > 0 else { return waveform }
    return trimLead(waveform, samples: waveform.asType(.float32), totalSamples: totalSamples,
                    frameSize: frameSize, totalFrames: totalFrames, threshold: threshold)
}

/// Trim low-energy trailing frames from waveform (fade-out / hang after the
/// last word). Same relative threshold as `trimLeadInSilence`; scans at most
/// the last 50 frames (4 s) and always keeps at least one frame. The final
/// partial frame (< 80 ms remainder) is folded into the last frame's RMS, so
/// audible content there prevents the trim instead of being dropped unseen.
public func trimTrailingSilence(
    _ waveform: MLXArray,
    sampleRate: Int = 24000,
    relativeThresholdDB: Float = -25,
    absoluteFloor: Float = 0.001
) -> MLXArray {
    let totalSamples = waveform.dim(0)
    let frameSize = samplesPerFrame(at: sampleRate)
    let totalFrames = totalSamples / frameSize
    guard totalFrames > 1 else { return waveform }

    let samples = waveform.asType(.float32)
    let threshold = silenceThreshold(
        samples, relativeThresholdDB: relativeThresholdDB, absoluteFloor: absoluteFloor)

    let trimFrames = quietFrameCount(
        samples, totalSamples: totalSamples, frameSize: frameSize,
        totalFrames: totalFrames, threshold: threshold, fromEnd: true)

    if trimFrames > 0, trimFrames < totalFrames {
        let keepSamples = (totalFrames - trimFrames) * frameSize
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
