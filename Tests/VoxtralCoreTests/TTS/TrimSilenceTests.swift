/**
 * TrimSilenceTests - Regression tests for lead-in / trailing silence trimming.
 *
 * Guards the enrolled-voice bug: a cloned voice reproduces its reference's
 * noise floor, so "silence" frames sit ABOVE any absolute RMS threshold tuned
 * on the clean presets (measured in production: -32.5 dBFS silences, the old
 * absolute 0.025 never triggered on the whole clip). The threshold must be
 * relative to the clip's peak.
 */

import XCTest
import MLX
@testable import VoxtralCore

final class TrimSilenceTests: XCTestCase {

    private let frame = 1920  // samples per 80 ms codec frame at 24 kHz

    /// `count` frames of alternating ±amplitude (RMS == amplitude).
    private func noiseFrames(_ count: Int, amplitude: Float) -> [Float] {
        (0 ..< count * frame).map { $0 % 2 == 0 ? amplitude : -amplitude }
    }

    /// `count` frames of a 440 Hz sine at `amplitude` peak.
    private func speechFrames(_ count: Int, amplitude: Float = 0.8) -> [Float] {
        (0 ..< count * frame).map {
            amplitude * Float(sin(2 * Double.pi * 440 * Double($0) / 24_000))
        }
    }

    /// The production case: lead-in "silence" whose RMS (0.03) is above the
    /// old absolute threshold (0.025) but far below peak − 25 dB (0.045).
    func testRelativeThresholdTrimsEnrolledVoiceNoiseFloor() {
        let lead = noiseFrames(3, amplitude: 0.03)
        let speech = speechFrames(5)
        let waveform = MLXArray(lead + speech)

        let trimmed = trimLeadInSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 5 * frame, "noise-floor lead-in not trimmed")
    }

    /// A clip that is quiet throughout must NOT be trimmed: its own noise is
    /// its signal, and the threshold (relative to its own peak) sits below it.
    func testUniformQuietClipUntouched() {
        let waveform = MLXArray(noiseFrames(10, amplitude: 0.02))
        let trimmed = trimLeadInSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), waveform.dim(0))
    }

    /// Digital-silence lead-in (the original preset case) still trims.
    func testDigitalSilenceLeadTrimmed() {
        let waveform = MLXArray([Float](repeating: 0, count: 2 * frame) + speechFrames(4))
        let trimmed = trimLeadInSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 4 * frame)
    }

    /// No trim when speech starts immediately.
    func testNoTrimWhenSpeechStartsImmediately() {
        let waveform = MLXArray(speechFrames(4))
        let trimmed = trimLeadInSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 4 * frame)
    }

    /// Trailing quiet frames (same noise floor as the lead-in case) are cut.
    func testTrailingSilenceTrimmed() {
        let waveform = MLXArray(speechFrames(5) + noiseFrames(3, amplitude: 0.03))
        let trimmed = trimTrailingSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 5 * frame, "noise-floor tail not trimmed")
    }

    /// A fully quiet clip survives tail trimming untouched.
    func testTrailingTrimKeepsUniformQuietClip() {
        let waveform = MLXArray(noiseFrames(5, amplitude: 0.02))
        let trimmed = trimTrailingSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), waveform.dim(0))
    }

    /// Lead + tail composed, as the pipeline applies them with trimTail on.
    func testLeadAndTailComposed() {
        let waveform = MLXArray(
            noiseFrames(2, amplitude: 0.03) + speechFrames(4) + noiseFrames(2, amplitude: 0.03))
        let trimmed = trimTrailingSilence(
            trimLeadInSilence(waveform, sampleRate: 24_000), sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 4 * frame)
    }

    /// An empty waveform must pass through untouched — the threshold pass
    /// must not run on a zero-size array (MLX max() aborts on empty input).
    func testEmptyWaveformUntouched() {
        let empty = MLXArray([Float]())
        XCTAssertEqual(trimLeadInSilence(empty, sampleRate: 24_000).dim(0), 0)
        XCTAssertEqual(trimTrailingSilence(empty, sampleRate: 24_000).dim(0), 0)
    }

    /// Audible content in the trailing partial frame (< 80 ms remainder)
    /// must prevent the tail trim: the remainder is folded into the last
    /// frame's RMS, not dropped unseen.
    func testTrailingPartialFrameWithSpeechPreventsTrim() {
        let remainder = Array(speechFrames(1)[0 ..< 900])
        let waveform = MLXArray(speechFrames(4) + noiseFrames(1, amplitude: 0.03) + remainder)
        let trimmed = trimTrailingSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 4 * frame + frame + 900, "audible remainder was dropped")
    }

    /// A quiet partial remainder after a quiet last frame is trimmed with it.
    func testTrailingQuietPartialFrameTrimmed() {
        let remainder = Array(noiseFrames(1, amplitude: 0.03)[0 ..< 900])
        let waveform = MLXArray(speechFrames(4) + noiseFrames(1, amplitude: 0.03) + remainder)
        let trimmed = trimTrailingSilence(waveform, sampleRate: 24_000)
        XCTAssertEqual(trimmed.dim(0), 4 * frame)
    }
}
