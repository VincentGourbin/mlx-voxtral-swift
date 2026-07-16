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

    /// Downsampling must attenuate content above the target Nyquist instead
    /// of aliasing it back into the band.
    func testDownsampleAntiAliases() {
        // 16 kHz tone at 48 kHz — above the 12 kHz Nyquist of 24 kHz.
        let src = sine(16_000, seconds: 0.5, rate: 48_000)
        let out = VoxtralVoiceEnrollment.resampleLinear(src, from: 48_000, to: 24_000)
        // The low-pass should strongly attenuate a >Nyquist tone.
        XCTAssertLessThan(rms(out), rms(src) * 0.5, "16kHz tone not attenuated — aliasing")
        // A tone comfortably in-band must survive.
        let inBand = sine(1_000, seconds: 0.5, rate: 48_000)
        let inBandOut = VoxtralVoiceEnrollment.resampleLinear(inBand, from: 48_000, to: 24_000)
        XCTAssertGreaterThan(rms(inBandOut), rms(inBand) * 0.7, "1kHz tone wrongly attenuated")
    }

    /// Equal rates are a no-op.
    func testSameRateIsIdentity() {
        let src = sine(220, seconds: 0.5, rate: 24_000)
        let out = VoxtralVoiceEnrollment.resampleLinear(src, from: 24_000, to: 24_000)
        XCTAssertEqual(out.count, src.count)
        XCTAssertEqual(out, src)
    }

    // MARK: - Reference clean-up (high-pass + gate)

    /// Quiet-only windows must come out of the gate as EXACT silence — the
    /// optimization learns whatever the reference contains, so a noise floor
    /// left in the "silences" is baked into the cloned voice.
    func testGateZeroesNoiseOnlyRegions() {
        let speech = sine(440, seconds: 0.5, rate: 24_000)
        // Noise floor at 0.01 (loudest window RMS ≈ 0.35 → −30 dB ≈ 0.011).
        let noise = (0 ..< 12_000).map { Float($0 % 2 == 0 ? 0.01 : -0.01) }
        let out = VoxtralVoiceEnrollment.gate(speech + noise, sampleRate: 24_000, thresholdDB: -30)

        // Away from the closing ramp, the noise region is exactly zero.
        let tail = Array(out[(speech.count + 480)...])
        XCTAssertEqual(tail.reduce(0) { max($0, abs($1)) }, 0, "noise floor survived the gate")
        // The speech region keeps its energy (ramps only touch 5 ms).
        XCTAssertGreaterThan(rms(Array(out[0 ..< speech.count])), 0.3)
    }

    /// A fully voiced signal passes through the gate untouched.
    func testGateIsIdentityOnVoicedSignal() {
        let speech = sine(440, seconds: 0.5, rate: 24_000)
        let out = VoxtralVoiceEnrollment.gate(speech, sampleRate: 24_000, thresholdDB: -30)
        XCTAssertEqual(out, speech)
    }

    /// A single full-scale click must NOT inflate the gate threshold above
    /// genuine soft speech — the threshold is relative to the loudest WINDOW
    /// RMS, not the sample peak. (With a sample-peak threshold, one click in
    /// a soft recording gated the entire voice to silence and the 30-minute
    /// optimization targeted nothing.)
    func testGateSurvivesTransientClick() {
        // Soft speech at RMS ≈ 0.028: below 1.0 × 10^(−30/20) ≈ 0.032, so a
        // sample-peak threshold would gate ALL of it because of the click.
        var signal = sine(440, seconds: 1.0, rate: 24_000).map { $0 * 0.08 }
        signal[12_000] = 1.0  // full-scale click
        let out = VoxtralVoiceEnrollment.gate(signal, sampleRate: 24_000, thresholdDB: -30)

        // Speech well away from the click must survive ungated.
        XCTAssertGreaterThan(rms(Array(out[0 ..< 6_000])), 0.02, "soft speech gated by a click")
        XCTAssertGreaterThan(rms(Array(out[18_000 ..< 24_000])), 0.02, "soft speech gated by a click")
    }

    /// The high-pass must remove DC/rumble while keeping the voice band.
    func testHighPassRemovesDCKeepsVoiceBand() {
        let tone = sine(1_000, seconds: 0.5, rate: 24_000)
        let withDC = tone.map { $0 + 0.3 }
        let out = VoxtralVoiceEnrollment.highPass(withDC, cutoff: 70, sampleRate: 24_000)

        // Assert away from the FIR edges (64 taps).
        let mid = Array(out[100 ..< out.count - 100])
        let mean = mid.reduce(0, +) / Float(mid.count)
        XCTAssertEqual(mean, 0, accuracy: 0.01, "DC offset not removed")
        XCTAssertGreaterThan(rms(mid), 0.3, "1 kHz tone wrongly attenuated")
    }
}
