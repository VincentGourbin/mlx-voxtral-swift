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

        let int16Array: [Int16] = (0..<numSamples).map { i in
            scaled[i].item(Int16.self)
        }
        int16Array.withUnsafeBufferPointer { buffer in
            data.append(buffer)
        }

        try data.write(to: url)
    }
}
