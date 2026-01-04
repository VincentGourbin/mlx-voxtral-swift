/**
 * AudioUtilsTests - Unit tests for audio utility functions
 */

import XCTest
import MLX
@testable import VoxtralCore

final class AudioUtilsTests: XCTestCase {

    // MARK: - Test Fixtures Audio Generation Tests

    func testSyntheticAudioDataLength() {
        let sampleRate = 16000
        let seconds: Float = 1.0
        let audio = TestFixtures.syntheticAudioData(seconds: seconds, sampleRate: sampleRate)

        XCTAssertEqual(audio.count, 16000, "1 second at 16kHz should have 16000 samples")
    }

    func testSyntheticAudioDataRange() {
        let audio = TestFixtures.syntheticAudioData(seconds: 0.5, sampleRate: 16000, frequency: 440.0)

        let minVal = audio.min() ?? 0
        let maxVal = audio.max() ?? 0

        XCTAssertGreaterThanOrEqual(minVal, -1.0, "Sine wave should be >= -1")
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Sine wave should be <= 1")
    }

    func testSilentAudioDataLength() {
        let audio = TestFixtures.silentAudioData(seconds: 2.0, sampleRate: 16000)

        XCTAssertEqual(audio.count, 32000, "2 seconds at 16kHz should have 32000 samples")
    }

    func testSilentAudioDataIsZero() {
        let audio = TestFixtures.silentAudioData(seconds: 0.5, sampleRate: 16000)

        for sample in audio {
            XCTAssertEqual(sample, 0.0, "Silent audio should be all zeros")
        }
    }

    func testNoiseAudioDataLength() {
        let audio = TestFixtures.noiseAudioData(seconds: 1.5, sampleRate: 16000)

        XCTAssertEqual(audio.count, 24000, "1.5 seconds at 16kHz should have 24000 samples")
    }

    func testNoiseAudioDataRange() {
        let audio = TestFixtures.noiseAudioData(seconds: 0.5, sampleRate: 16000)

        let minVal = audio.min() ?? 0
        let maxVal = audio.max() ?? 0

        XCTAssertGreaterThanOrEqual(minVal, -1.0, "Noise should be >= -1")
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Noise should be <= 1")
    }

    func testNoiseAudioDataIsRandom() {
        let audio1 = TestFixtures.noiseAudioData(seconds: 0.1, sampleRate: 16000)
        let audio2 = TestFixtures.noiseAudioData(seconds: 0.1, sampleRate: 16000)

        // Two noise samples should be different (with very high probability)
        XCTAssertNotEqual(audio1, audio2, "Noise should be random each time")
    }

    // MARK: - MLXArray Audio Conversion Tests

    func testMLXArrayFromFloatArray() {
        let floatData: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        let array = MLXArray(floatData)
        eval(array)

        XCTAssertEqual(array.shape[0], 5, "Array should have correct length")
    }

    func testMLXArrayFromSyntheticAudio() {
        let audioData = TestFixtures.syntheticAudioData(seconds: 0.1, sampleRate: 16000)
        let array = MLXArray(audioData)
        eval(array)

        XCTAssertEqual(array.shape[0], 1600, "0.1 seconds should be 1600 samples")
    }

    // MARK: - Audio Normalization Tests

    func testAudioNormalizationBounds() {
        // Create audio with values outside [-1, 1]
        var audio = TestFixtures.syntheticAudioData(seconds: 0.1, sampleRate: 16000)
        audio = audio.map { $0 * 2.0 }  // Scale to [-2, 2]

        // Convert to MLX and normalize
        let mlxAudio = MLXArray(audio)
        let maxAbs = abs(mlxAudio).max()
        eval(maxAbs)
        let normalized = mlxAudio / maxAbs
        eval(normalized)

        let minVal = normalized.min().item(Float.self)
        let maxVal = normalized.max().item(Float.self)

        XCTAssertGreaterThanOrEqual(minVal, -1.0, "Normalized min should be >= -1")
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Normalized max should be <= 1")
    }

    // MARK: - Sample Rate Tests

    func testStandardSampleRate() {
        // Voxtral expects 16kHz
        XCTAssertEqual(SAMPLE_RATE, 16000, "Standard sample rate should be 16kHz")
    }

    // MARK: - Audio Duration Calculation Tests

    func testAudioDurationCalculation() {
        let sampleRate = 16000
        let numSamples = 48000
        let expectedDuration: Float = Float(numSamples) / Float(sampleRate)

        XCTAssertEqual(expectedDuration, 3.0, "48000 samples at 16kHz is 3 seconds")
    }

    func testFrameToTimeConversion() {
        // n_frames = n_samples / hop_length
        // time = frame * hop_length / sample_rate
        let frame = 100
        let hopLength = HOP_LENGTH
        let sampleRate = SAMPLE_RATE
        let time = Float(frame * hopLength) / Float(sampleRate)

        XCTAssertEqual(time, 1.0, "Frame 100 with hop 160 at 16kHz is 1.0 seconds")
    }

    // MARK: - Audio Shape Tests

    func testMonoAudioShape() {
        let audio = TestFixtures.syntheticAudioData(seconds: 1.0, sampleRate: 16000)
        let mlxAudio = MLXArray(audio)
        eval(mlxAudio)

        XCTAssertEqual(mlxAudio.ndim, 1, "Mono audio should be 1D")
        XCTAssertEqual(mlxAudio.shape[0], 16000, "1 second should be 16000 samples")
    }

    // MARK: - Frequency Tests

    func testNyquistFrequency() {
        // Nyquist frequency is half the sample rate
        let nyquist = SAMPLE_RATE / 2

        XCTAssertEqual(nyquist, 8000, "Nyquist for 16kHz is 8kHz")
    }

    func testFFTFrequencyBins() {
        // Number of unique frequency bins from FFT
        let freqBins = N_FFT / 2 + 1

        XCTAssertEqual(freqBins, 201, "400-point FFT has 201 frequency bins")
    }

    // MARK: - Chunk Processing Tests

    func testChunkSamples() {
        // 30 seconds * 16000 samples/second
        let chunkSamples = CHUNK_LENGTH * SAMPLE_RATE

        XCTAssertEqual(chunkSamples, 480000, "30 second chunk has 480000 samples")
    }

    func testChunkFrames() {
        // Number of frames in a chunk
        let chunkFrames = N_SAMPLES / HOP_LENGTH

        XCTAssertEqual(chunkFrames, 3000, "30 second chunk has 3000 frames")
    }

    // MARK: - Edge Cases

    func testEmptyAudioArray() {
        let audio: [Float] = []
        let mlxAudio = MLXArray(audio)
        eval(mlxAudio)

        XCTAssertEqual(mlxAudio.shape[0], 0, "Empty audio should have 0 samples")
    }

    func testSingleSampleAudio() {
        let audio: [Float] = [0.5]
        let mlxAudio = MLXArray(audio)
        eval(mlxAudio)

        XCTAssertEqual(mlxAudio.shape[0], 1, "Single sample audio should have 1 sample")
    }

    func testVeryLongAudio() {
        // Test with 60 seconds of audio (longer than chunk length)
        let audio = TestFixtures.syntheticAudioData(seconds: 60.0, sampleRate: 16000)

        XCTAssertEqual(audio.count, 960000, "60 seconds should have 960000 samples")
    }
}
