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

// MARK: - Lead-in Silence Trimming

/// Trim low-energy lead-in frames from waveform.
/// Voxtral TTS often generates a few silence/transition frames before speech starts,
/// especially noticeable with non-English voices. This removes them for cleaner output.
public func trimLeadInSilence(_ waveform: MLXArray, sampleRate: Int = 24000, threshold: Float = 0.025) -> MLXArray {
    let samplesPerFrame = 1920  // 80ms at 24kHz (8x upsample * 240 patch)
    let totalSamples = waveform.dim(0)
    let totalFrames = totalSamples / samplesPerFrame

    // Scan frame-by-frame, find the first frame above threshold RMS
    let samples = waveform.asType(.float32)
    var trimFrames = 0

    for i in 0..<min(totalFrames, 20) {  // Check at most first 20 frames (1.6s)
        let start = i * samplesPerFrame
        let end = min(start + samplesPerFrame, totalSamples)
        let chunk = samples[start..<end]
        let rms = MLX.sqrt(MLX.mean(chunk * chunk)).item(Float.self)
        if rms >= threshold {
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
