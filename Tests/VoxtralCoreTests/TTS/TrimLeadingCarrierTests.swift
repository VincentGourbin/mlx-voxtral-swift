/**
 * TrimLeadingCarrierTests — fast, model-free unit tests for the A6b warm-up
 * carrier trim. Builds synthetic [carrier | gap | content] waveforms and
 * checks the cut lands at the content onset, plus the safe fallbacks.
 */

import XCTest
import MLX
@testable import VoxtralCore

final class TrimLeadingCarrierTests: XCTestCase {

    private let frame = 1920  // samples per 80 ms frame at 24 kHz

    /// Build a waveform from a list of (frameCount, amplitude) segments.
    private func waveform(_ segments: [(Int, Float)]) -> MLXArray {
        var samples: [Float] = []
        for (count, amp) in segments {
            for _ in 0 ..< (count * frame) {
                // Alternate sign so RMS ≈ amp (not a DC level).
                samples.append(samples.count % 2 == 0 ? amp : -amp)
            }
        }
        return MLXArray(samples)
    }

    func testCutsAtContentOnsetAfterGap() {
        // 10 carrier frames, 4 silent frames, 30 content frames.
        let wav = waveform([(10, 0.3), (4, 0.0), (30, 0.3)])
        let (trimmed, cut) = trimLeadingCarrier(wav, gapMinFrames: 3)
        XCTAssertEqual(cut, 14, "should cut at the content onset (10 carrier + 4 gap)")
        XCTAssertEqual(trimmed.dim(0), 30 * frame, "trimmed clip is the 30 content frames")
    }

    func testShortGapBelowThresholdIsNotACut() {
        // A 2-frame gap (< gapMinFrames) is a comma, not a boundary → no cut.
        let wav = waveform([(10, 0.3), (2, 0.0), (30, 0.3)])
        let (_, cut) = trimLeadingCarrier(wav, gapMinFrames: 3)
        XCTAssertEqual(cut, 0, "a sub-threshold gap must not trigger a cut")
    }

    func testFirstQualifyingGapWins() {
        // carrier, boundary gap, content, sentence gap, more content:
        // the FIRST qualifying gap (carrier boundary) is the cut.
        let wav = waveform([(8, 0.3), (4, 0.0), (20, 0.3), (5, 0.0), (20, 0.3)])
        let (_, cut) = trimLeadingCarrier(wav, gapMinFrames: 3)
        XCTAssertEqual(cut, 12, "cut at the first boundary, not a later sentence gap")
    }

    func testNoGapReturnsUnchanged() {
        // Continuous speech, no gap → safe fallback, nothing trimmed.
        let wav = waveform([(40, 0.3)])
        let (trimmed, cut) = trimLeadingCarrier(wav, gapMinFrames: 3)
        XCTAssertEqual(cut, 0)
        XCTAssertEqual(trimmed.dim(0), wav.dim(0))
    }

    func testTrailingSilenceIsNotMistakenForBoundary() {
        // carrier speech then only trailing silence (no resumed speech) → no cut
        // (would otherwise blank the whole clip).
        let wav = waveform([(10, 0.3), (10, 0.0)])
        let (_, cut) = trimLeadingCarrier(wav, gapMinFrames: 3)
        XCTAssertEqual(cut, 0, "a gap with no resumed speech is trailing silence, not a boundary")
    }
}
