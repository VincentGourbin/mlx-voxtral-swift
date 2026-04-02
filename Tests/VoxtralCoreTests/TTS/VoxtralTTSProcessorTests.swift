/**
 * VoxtralTTSProcessorTests - Unit tests for TTS processor, WAVWriter, and TTSSynthesisResult
 */

import XCTest
import MLX
@testable import VoxtralCore

final class VoxtralTTSProcessorTests: XCTestCase {

    // MARK: - TTSSynthesisResult Tests

    func testSynthesisResultProperties() {
        let waveform = MLXArray.zeros([24000])  // 1 second at 24kHz
        let result = TTSSynthesisResult(
            waveform: waveform,
            numFrames: 13,
            sampleRate: 24000,
            generationTime: 0.5,
            timeToFirstToken: 0.1
        )

        XCTAssertEqual(result.numFrames, 13)
        XCTAssertEqual(result.sampleRate, 24000)
        XCTAssertEqual(result.generationTime, 0.5)
    }

    func testSynthesisResultDuration() {
        let waveform = MLXArray.zeros([48000])  // 2 seconds at 24kHz
        let result = TTSSynthesisResult(
            waveform: waveform,
            numFrames: 25,
            sampleRate: 24000,
            generationTime: 1.0,
            timeToFirstToken: 0.1
        )

        XCTAssertEqual(result.duration, 2.0, accuracy: 0.001)
    }

    func testSynthesisResultRealTimeFactor() {
        let waveform = MLXArray.zeros([24000])  // 1 second
        let result = TTSSynthesisResult(
            waveform: waveform,
            numFrames: 13,
            sampleRate: 24000,
            generationTime: 0.5,
            timeToFirstToken: 0.1
        )

        // RTF = generationTime / duration = 0.5 / 1.0 = 0.5
        XCTAssertEqual(result.realTimeFactor, 0.5, accuracy: 0.001)
    }

    func testSynthesisResultRealTimeFactorWithZeroDuration() {
        let waveform = MLXArray.zeros([0])
        let result = TTSSynthesisResult(
            waveform: waveform,
            numFrames: 0,
            sampleRate: 24000,
            generationTime: 1.0,
            timeToFirstToken: 0
        )

        XCTAssertEqual(result.realTimeFactor, 0)
    }

    func testSynthesisResultFramesPerSecond() {
        let waveform = MLXArray.zeros([24000])
        let result = TTSSynthesisResult(
            waveform: waveform,
            numFrames: 25,
            sampleRate: 24000,
            generationTime: 2.0,
            timeToFirstToken: 0.1
        )

        // FPS = 25 / 2.0 = 12.5
        XCTAssertEqual(result.framesPerSecond, 12.5, accuracy: 0.001)
    }

    func testSynthesisResultFramesPerSecondWithZeroTime() {
        let waveform = MLXArray.zeros([24000])
        let result = TTSSynthesisResult(
            waveform: waveform,
            numFrames: 25,
            sampleRate: 24000,
            generationTime: 0,
            timeToFirstToken: 0
        )

        XCTAssertEqual(result.framesPerSecond, 0)
    }

    // MARK: - WAVWriter Tests

    func testWAVWriterCreatesFile() throws {
        let waveform = MLXArray(Array(repeating: Float(0.0), count: 2400))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_output_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL)

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
    }

    func testWAVWriterFileSize() throws {
        let numSamples = 24000  // 1 second at 24kHz
        let waveform = MLXArray(Array(repeating: Float(0.0), count: numSamples))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_size_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL)

        let data = try Data(contentsOf: outputURL)
        // WAV header = 44 bytes, PCM data = numSamples * 2 bytes (16-bit)
        let expectedSize = 44 + numSamples * 2
        XCTAssertEqual(data.count, expectedSize)
    }

    func testWAVWriterRIFFHeader() throws {
        let waveform = MLXArray(Array(repeating: Float(0.0), count: 100))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_header_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL)

        let data = try Data(contentsOf: outputURL)

        // Check RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        XCTAssertEqual(riff, "RIFF")

        // Check WAVE format
        let wave = String(data: data[8..<12], encoding: .ascii)
        XCTAssertEqual(wave, "WAVE")

        // Check fmt chunk
        let fmt = String(data: data[12..<16], encoding: .ascii)
        XCTAssertEqual(fmt, "fmt ")

        // Check data chunk
        let dataChunk = String(data: data[36..<40], encoding: .ascii)
        XCTAssertEqual(dataChunk, "data")
    }

    func testWAVWriterSampleRate() throws {
        let waveform = MLXArray(Array(repeating: Float(0.0), count: 100))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_sr_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL, sampleRate: 24000)

        let data = try Data(contentsOf: outputURL)

        // Sample rate is at bytes 24-27 (little-endian)
        let sampleRate = data[24...27].withUnsafeBytes { $0.load(as: UInt32.self) }
        XCTAssertEqual(sampleRate, 24000)
    }

    func testWAVWriterMonoChannel() throws {
        let waveform = MLXArray(Array(repeating: Float(0.0), count: 100))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_mono_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL)

        let data = try Data(contentsOf: outputURL)

        // Number of channels at bytes 22-23 (little-endian)
        let channels = data[22...23].withUnsafeBytes { $0.load(as: UInt16.self) }
        XCTAssertEqual(channels, 1)
    }

    func testWAVWriterBitDepth() throws {
        let waveform = MLXArray(Array(repeating: Float(0.0), count: 100))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_bits_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL, bitDepth: 16)

        let data = try Data(contentsOf: outputURL)

        // Bits per sample at bytes 34-35 (little-endian)
        let bitsPerSample = data[34...35].withUnsafeBytes { $0.load(as: UInt16.self) }
        XCTAssertEqual(bitsPerSample, 16)
    }

    func testWAVWriterWithSineWave() throws {
        // Generate a sine wave
        let sampleRate = 24000
        let numSamples = sampleRate  // 1 second
        let frequency: Float = 440.0
        let samples = (0..<numSamples).map { i in
            sin(2.0 * .pi * frequency * Float(i) / Float(sampleRate)) * 0.5
        }
        let waveform = MLXArray(samples)

        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_sine_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL, sampleRate: sampleRate)

        let data = try Data(contentsOf: outputURL)
        XCTAssertEqual(data.count, 44 + numSamples * 2)
    }

    func testWAVWriterClampsValues() throws {
        // Values outside [-1, 1] should be clipped
        let samples: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let waveform = MLXArray(samples)

        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_clamp_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        // Should not throw even with out-of-range values
        try WAVWriter.write(waveform: waveform, to: outputURL)

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
    }

    func testWAVWriterCustomSampleRate() throws {
        let waveform = MLXArray(Array(repeating: Float(0.0), count: 100))
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_custom_sr_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try WAVWriter.write(waveform: waveform, to: outputURL, sampleRate: 16000)

        let data = try Data(contentsOf: outputURL)
        let sampleRate = data[24...27].withUnsafeBytes { $0.load(as: UInt32.self) }
        XCTAssertEqual(sampleRate, 16000)
    }
}
