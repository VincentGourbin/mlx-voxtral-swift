/**
 * VoiceEnrollmentReferenceTests - Regression tests for reference-audio
 * preparation.
 *
 * Guards the bug where enrollment silently produced an all-zero reference
 * when the input needed UPSAMPLING (e.g. a 22.05 kHz LibriVox clip → 24 kHz),
 * which collapsed the optimization to a degenerate, input-independent voice.
 */

import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class VoiceEnrollmentReferenceTests: XCTestCase {

    private func sine(_ freq: Double, seconds: Double, rate: Double) -> [Float] {
        let n = Int(seconds * rate)
        return (0 ..< n).map { Float(0.5 * sin(2 * .pi * freq * Double($0) / rate)) }
    }

    private func rms(_ x: [Float]) -> Float {
        x.isEmpty ? 0 : (x.reduce(0) { $0 + $1 * $1 } / Float(x.count)).squareRoot()
    }

    /// Upsampling (22.05 kHz → 24 kHz) must preserve the signal, not silence
    /// it — this is the exact regression.
    func testUpsampleDoesNotSilence() {
        let src = sine(220, seconds: 1.0, rate: 22_050)
        let out = VoxtralVoiceEnrollment.resampleLinear(src, from: 22_050, to: 24_000)

        XCTAssertGreaterThan(out.count, src.count, "upsample should produce more samples")
        XCTAssertEqual(Double(out.count), 24_000, accuracy: 5, "wrong output length")
        // Energy preserved (linear interp of a sine keeps ~the same RMS).
        XCTAssertEqual(rms(out), rms(src), accuracy: 0.05)
        XCTAssertGreaterThan(rms(out), 0.2, "output is silent — the regression")
    }

    /// Downsampling must also preserve the signal.
    func testDownsamplePreservesSignal() {
        let src = sine(220, seconds: 1.0, rate: 44_100)
        let out = VoxtralVoiceEnrollment.resampleLinear(src, from: 44_100, to: 24_000)

        XCTAssertLessThan(out.count, src.count)
        XCTAssertEqual(Double(out.count), 24_000, accuracy: 5)
        XCTAssertGreaterThan(rms(out), 0.2)
    }

    /// Equal rates are a no-op.
    func testSameRateIsIdentity() {
        let src = sine(220, seconds: 0.5, rate: 24_000)
        let out = VoxtralVoiceEnrollment.resampleLinear(src, from: 24_000, to: 24_000)
        XCTAssertEqual(out.count, src.count)
        XCTAssertEqual(out, src)
    }
}
