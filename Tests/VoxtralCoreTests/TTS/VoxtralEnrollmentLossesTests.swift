/**
 * VoxtralEnrollmentLossesTests - Numerical parity of the MLX enrollment
 * losses against the PyTorch reference (upstream training_script.py).
 *
 * Reference values were produced on a deterministic signal (closed-form
 * sines, no RNG, reproduced identically below) with:
 *   REF_L1   = 0.092424
 *   REF_STFT = 0.986005
 *   REF_MEL  = 0.625629
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class VoxtralEnrollmentLossesTests: XCTestCase {

    private let N = 12_000
    private let sr = 24_000

    private func makeSignals() -> (MLXArray, MLXArray) {
        var pred = [Float](repeating: 0, count: N)
        var targ = [Float](repeating: 0, count: N)
        for i in 0..<N {
            let n = Float(i)
            pred[i] = 0.1 * sin(2 * .pi * 220 * n / 24000) + 0.05 * sin(2 * .pi * 440 * n / 24000)
            targ[i] = 0.1 * sin(2 * .pi * 230 * n / 24000) + 0.05 * cos(2 * .pi * 450 * n / 24000)
        }
        return (MLXArray(pred), MLXArray(targ))
    }

    func testL1MatchesPyTorch() {
        let (pred, targ) = makeSignals()
        let computer = EnrollmentLossComputer(signalLength: N, sampleRate: sr)
        let value = computer.l1Loss(pred, targ).item(Float.self)
        XCTAssertEqual(value, 0.092424, accuracy: 1e-4)
    }

    func testMultiResSTFTMatchesPyTorch() {
        let (pred, targ) = makeSignals()
        let computer = EnrollmentLossComputer(signalLength: N, sampleRate: sr)
        let value = computer.multiResolutionSTFTLoss(pred, targ).item(Float.self)
        // Tolerance covers FFT/window float32 differences across backends.
        XCTAssertEqual(value, 0.986005, accuracy: 0.03)
    }

    func testMelMatchesPyTorch() {
        let (pred, targ) = makeSignals()
        let computer = EnrollmentLossComputer(signalLength: N, sampleRate: sr)
        let value = computer.melLoss(pred, targ).item(Float.self)
        XCTAssertEqual(value, 0.625629, accuracy: 0.03)
    }

    /// The whole loss chain must remain differentiable end to end.
    func testLossesAreDifferentiable() {
        let (_, targ) = makeSignals()
        let computer = EnrollmentLossComputer(signalLength: N, sampleRate: sr)

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            let p = inputs[0]
            return [computer.l1Loss(p, targ)
                + computer.multiResolutionSTFTLoss(p, targ)
                + computer.melLoss(p, targ)]
        }

        let x = MLXRandom.normal([N]) * 0.1
        let grads = MLX.grad(loss)([x])
        MLX.eval(grads)
        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite)
        XCTAssertGreaterThan(gradNorm, 0)
    }
}
