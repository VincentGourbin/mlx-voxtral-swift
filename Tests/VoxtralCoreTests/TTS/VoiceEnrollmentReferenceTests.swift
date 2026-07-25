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

        // Assert away from the filter's edge transients.
        let mid = Array(out[2_000 ..< out.count - 2_000])
        let mean = mid.reduce(0, +) / Float(mid.count)
        XCTAssertEqual(mean, 0, accuracy: 0.01, "DC offset not removed")
        XCTAssertGreaterThan(rms(mid), 0.3, "1 kHz tone wrongly attenuated")
    }

    /// Frequency-response guard for the high-pass. The old 64-tap complementary
    /// FIR could not resolve a 70 Hz corner at 24 kHz and attenuated a male
    /// voice's fundamental (100–120 Hz) by 24–27 dB — the root cause of the
    /// "thin" cloned timbre. A tone at 1 kHz alone (as the previous test used)
    /// does not catch this; these near-corner tones do.
    func testHighPassPreservesFundamentalRemovesRumble() {
        // Measured attenuation (dB, positive = cut) of a pure tone through the
        // filter, over its steady-state middle to avoid edge transients.
        func attenuationDB(_ freq: Double) -> Float {
            let x = sine(freq, seconds: 1.0, rate: 24_000)
            let y = VoxtralVoiceEnrollment.highPass(x, cutoff: 70, sampleRate: 24_000)
            let lo = x.count / 5, hi = x.count * 4 / 5
            return -20 * log10(rms(Array(y[lo ..< hi])) / rms(Array(x[lo ..< hi])))
        }

        XCTAssertGreaterThan(attenuationDB(30), 20, "30 Hz rumble not removed")
        XCTAssertLessThan(attenuationDB(100), 2.5, "100 Hz fundamental over-attenuated")
        XCTAssertLessThan(attenuationDB(120), 1.5, "120 Hz fundamental over-attenuated")
        XCTAssertLessThan(attenuationDB(1_000), 0.5, "1 kHz passband not flat")
    }

    // MARK: - Soft gate (attenuationDB) + loudness normalization

    /// With an attenuation floor, gated regions must be ATTENUATED, not
    /// zeroed: exact zeros are learned by the enrollment and reproduce as
    /// hard-chopped micro-gaps in every synthesis (measured −inf noise floor
    /// on enrolled voices).
    func testSoftGateAttenuatesInsteadOfZeroing() {
        let speech = sine(440, seconds: 0.5, rate: 24_000)
        let noise = (0 ..< 12_000).map { Float($0 % 2 == 0 ? 0.01 : -0.01) }
        let out = VoxtralVoiceEnrollment.gate(
            speech + noise, sampleRate: 24_000, thresholdDB: -30, attenuationDB: -24)

        // Away from the ramp, the noise region is attenuated by ≈24 dB…
        let tail = Array(out[(speech.count + 480)...])
        let expected = Float(0.01 * pow(10, -24.0 / 20))  // ≈ 0.00063
        XCTAssertEqual(tail.reduce(0) { max($0, abs($1)) }, expected, accuracy: expected * 0.1)
        // …but NOT zeroed.
        XCTAssertGreaterThan(rms(tail), 0, "soft gate produced exact zeros")
        // Speech untouched away from the ramps.
        XCTAssertGreaterThan(rms(Array(out[0 ..< speech.count])), 0.3)
    }

    /// nil attenuation preserves the legacy exact-zero behavior.
    func testNilAttenuationKeepsLegacyZeroGate() {
        let speech = sine(440, seconds: 0.5, rate: 24_000)
        let noise = (0 ..< 12_000).map { Float($0 % 2 == 0 ? 0.01 : -0.01) }
        let out = VoxtralVoiceEnrollment.gate(
            speech + noise, sampleRate: 24_000, thresholdDB: -30, attenuationDB: nil)
        let tail = Array(out[(speech.count + 480)...])
        XCTAssertEqual(tail.reduce(0) { max($0, abs($1)) }, 0)
    }

    /// Normalization must bring the ACTIVE-speech RMS to the target level —
    /// long silences in the reference must not inflate the gain.
    func testNormalizeActiveRMSHitsTarget() {
        // Quiet speech (RMS ≈ −38 dB ≈ 0.0126) followed by 1 s of silence.
        let speech = sine(440, seconds: 1.0, rate: 24_000).map { $0 * 0.0178 }
        let silence = [Float](repeating: 0, count: 24_000)
        let out = VoxtralVoiceEnrollment.normalizeActiveRMS(
            speech + silence, sampleRate: 24_000, targetDB: -20)

        // Active (speech) RMS lands on −20 dB = 0.1, regardless of the silence.
        let speechRMS = rms(Array(out[0 ..< speech.count]))
        XCTAssertEqual(speechRMS, 0.1, accuracy: 0.01, "active RMS missed the −20 dB target")
        // Silence stays silent (pure gain — no offset).
        XCTAssertEqual(rms(Array(out[speech.count...])), 0)
    }

    /// The normalization gain is capped so no sample can clip.
    func testNormalizeActiveRMSPeakGuard() {
        // A very quiet signal with one large transient: reaching −20 dB RMS
        // would need ×8 gain, but the 0.5 peak only allows ×1.96.
        var signal = sine(440, seconds: 1.0, rate: 24_000).map { $0 * 0.0178 }
        signal[12_000] = 0.5
        let out = VoxtralVoiceEnrollment.normalizeActiveRMS(
            signal, sampleRate: 24_000, targetDB: -20)
        let maxAbs = out.reduce(0) { max($0, abs($1)) }
        XCTAssertLessThanOrEqual(maxAbs, 0.98001, "peak guard failed")
    }

    /// An already-correct level is a near-identity transform.
    func testNormalizeActiveRMSIdentityAtTarget() {
        // sine() has amplitude 0.5 → ×0.2828 gives amplitude 0.1414, RMS ≈ 0.1.
        let speech = sine(440, seconds: 1.0, rate: 24_000).map { $0 * 0.2828 }
        let out = VoxtralVoiceEnrollment.normalizeActiveRMS(
            speech, sampleRate: 24_000, targetDB: -20)
        XCTAssertEqual(rms(out), rms(speech), accuracy: 0.005)
    }
}
