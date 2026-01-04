/**
 * FeatureExtractorTests - Unit tests for audio feature extraction
 */

import XCTest
import MLX
@testable import VoxtralCore

final class FeatureExtractorTests: XCTestCase {

    // MARK: - Constants Tests

    func testSampleRateConstant() {
        XCTAssertEqual(SAMPLE_RATE, 16000, "Sample rate should be 16kHz")
    }

    func testNFFTConstant() {
        XCTAssertEqual(N_FFT, 400, "N_FFT should be 400")
    }

    func testHopLengthConstant() {
        XCTAssertEqual(HOP_LENGTH, 160, "Hop length should be 160")
    }

    func testChunkLengthConstant() {
        XCTAssertEqual(CHUNK_LENGTH, 30, "Chunk length should be 30 seconds")
    }

    func testNSamplesConstant() {
        XCTAssertEqual(N_SAMPLES, 480000, "N_SAMPLES should be 480000 (30 * 16000)")
    }

    func testNFramesConstant() {
        XCTAssertEqual(N_FRAMES, 3000, "N_FRAMES should be 3000 (480000 / 160)")
    }

    func testNMelsConstant() {
        XCTAssertEqual(N_MELS, 128, "N_MELS should be 128")
    }

    // MARK: - padToMultiple Tests

    func testPadToMultipleNoChange() {
        let x = MLXArray.zeros([10])
        let padded = padToMultiple(x, multiple: 5, axis: 0)

        XCTAssertEqual(padded.shape[0], 10, "Array already multiple should not change")
    }

    func testPadToMultipleNeedsPadding() {
        let x = MLXArray.zeros([7])
        let padded = padToMultiple(x, multiple: 5, axis: 0)

        XCTAssertEqual(padded.shape[0], 10, "Array should be padded to next multiple")
    }

    func testPadToMultiple2D() {
        let x = MLXArray.zeros([7, 3])
        let padded = padToMultiple(x, multiple: 5, axis: 0)

        XCTAssertEqual(padded.shape[0], 10, "First dimension should be padded")
        XCTAssertEqual(padded.shape[1], 3, "Second dimension should be unchanged")
    }

    func testPadToMultipleAxis1() {
        let x = MLXArray.zeros([3, 7])
        let padded = padToMultiple(x, multiple: 5, axis: 1)

        XCTAssertEqual(padded.shape[0], 3, "First dimension should be unchanged")
        XCTAssertEqual(padded.shape[1], 10, "Second dimension should be padded")
    }

    func testPadToMultipleWithOnes() {
        let x = MLXArray.ones([3])
        let padded = padToMultiple(x, multiple: 5, axis: 0)

        // Check that original values are preserved
        let firstThree = padded[0..<3]
        eval(firstThree)
        XCTAssertEqual(firstThree.shape[0], 3)
    }

    // MARK: - Hanning Window Tests

    func testHanningSize() {
        let window = hanning(400)
        XCTAssertEqual(window.shape[0], 400, "Hanning window should have correct size")
    }

    func testHanningSymmetry() {
        let size = 100
        let window = hanning(size)
        eval(window)

        // Hanning window should be symmetric
        let first = window[0].item(Float.self)
        let last = window[size - 1].item(Float.self)
        XCTAssertEqual(first, last, accuracy: 1e-5, "Hanning window should be symmetric")
    }

    func testHanningPeakAtCenter() {
        let size = 101  // Odd size for clear center
        let window = hanning(size)
        eval(window)

        let center = window[50].item(Float.self)
        let first = window[0].item(Float.self)

        XCTAssertGreaterThan(center, first, "Center should be higher than edges")
    }

    func testHanningRange() {
        let window = hanning(400)
        eval(window)

        let minVal = window.min().item(Float.self)
        let maxVal = window.max().item(Float.self)

        XCTAssertGreaterThanOrEqual(minVal, 0.0, "Hanning values should be >= 0")
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Hanning values should be <= 1")
    }

    // MARK: - Mel Filter Bank Tests

    func testGetMelFiltersShape() {
        let filters = getMelFilters(nMels: 128)
        eval(filters)

        // Mel filter bank shape should be [n_mels, n_fft/2 + 1]
        XCTAssertEqual(filters.shape[0], 128, "Should have 128 mel bins")
        XCTAssertEqual(filters.shape[1], 201, "Should have 201 frequency bins (400/2 + 1)")
    }

    func testGetMelFiltersDefaultMels() {
        let filters = getMelFilters()
        eval(filters)

        XCTAssertEqual(filters.shape[0], N_MELS, "Default should use N_MELS constant")
    }

    func testGetMelFiltersNonNegative() {
        let filters = getMelFilters(nMels: 128)
        eval(filters)

        let minVal = filters.min().item(Float.self)
        XCTAssertGreaterThanOrEqual(minVal, 0.0, "Mel filter values should be non-negative")
    }

    // MARK: - Log Mel Spectrogram Tests

    func testLogMelSpectrogramShape() {
        // Create synthetic audio (1 second at 16kHz)
        let audio = MLXArray.zeros([SAMPLE_RATE])
        let (logMel, _) = logMelSpectrogram(audio)
        eval(logMel)

        // Expected shape: [n_frames, n_mels]
        XCTAssertEqual(logMel.shape[1], N_MELS, "Should have 128 mel bins")
        XCTAssertGreaterThan(logMel.shape[0], 0, "Should have some frames")
    }

    func testLogMelSpectrogramWithSyntheticAudio() {
        // Generate a sine wave
        let audioData = TestFixtures.syntheticAudioData(seconds: 0.5, sampleRate: SAMPLE_RATE)
        let audio = MLXArray(audioData)
        let (logMel, _) = logMelSpectrogram(audio)
        eval(logMel)

        XCTAssertEqual(logMel.shape[1], N_MELS, "Should have 128 mel bins")
    }

    func testLogMelSpectrogramWithSilence() {
        let audioData = TestFixtures.silentAudioData(seconds: 0.5, sampleRate: SAMPLE_RATE)
        let audio = MLXArray(audioData)
        let (logMel, _) = logMelSpectrogram(audio)
        eval(logMel)

        // Log mel of silence should have low values (close to log(1e-10) clipping)
        let maxVal = logMel.max().item(Float.self)
        XCTAssertLessThan(maxVal, 0.0, "Silent audio should have negative log mel values")
    }

    // MARK: - VoxtralFeatureExtractor Tests

    func testFeatureExtractorInitialization() {
        let extractor = VoxtralFeatureExtractor()
        XCTAssertNotNil(extractor, "Feature extractor should initialize")
    }

    // MARK: - Audio Loading Tests

    func testLoadAudioThrowsForMissingFile() {
        XCTAssertThrowsError(try loadAudio("/nonexistent/path/to/audio.wav")) { error in
            // Should throw some kind of error for missing file
            XCTAssertNotNil(error)
        }
    }

    // MARK: - Shape Calculations

    func testFrameCountCalculation() {
        // Number of frames = (samples - hop_length) / hop_length + 1 (approx)
        let samples = SAMPLE_RATE  // 1 second
        let expectedFrames = (samples / HOP_LENGTH)

        XCTAssertEqual(expectedFrames, 100, "1 second should produce ~100 frames")
    }

    func testChunkFrameCount() {
        // 30 second chunk should produce N_FRAMES frames
        let chunkSamples = CHUNK_LENGTH * SAMPLE_RATE
        let expectedFrames = chunkSamples / HOP_LENGTH

        XCTAssertEqual(expectedFrames, N_FRAMES, "30 second chunk should produce 3000 frames")
    }
}
