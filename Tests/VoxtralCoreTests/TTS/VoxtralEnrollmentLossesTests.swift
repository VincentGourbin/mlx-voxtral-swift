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

    func testL1MatchesPyTorch() throws {
        let (pred, targ) = makeSignals()
        let computer = try EnrollmentLossComputer(validating: targ, sampleRate: sr)
        let value = computer.l1Loss(pred).item(Float.self)
        XCTAssertEqual(value, 0.092424, accuracy: 1e-4)
    }

    func testMultiResSTFTMatchesPyTorch() throws {
        let (pred, targ) = makeSignals()
        let computer = try EnrollmentLossComputer(validating: targ, sampleRate: sr)
        let value = computer.multiResolutionSTFTLoss(pred).item(Float.self)
        // Tolerance covers FFT/window float32 differences across backends.
        XCTAssertEqual(value, 0.986005, accuracy: 0.03)
    }

    func testMelMatchesPyTorch() throws {
        let (pred, targ) = makeSignals()
        let computer = try EnrollmentLossComputer(validating: targ, sampleRate: sr)
        let value = computer.melLoss(pred).item(Float.self)
        XCTAssertEqual(value, 0.625629, accuracy: 0.03)
    }

    /// The whole loss chain must remain differentiable end to end.
    func testLossesAreDifferentiable() throws {
        let (_, targ) = makeSignals()
        let computer = try EnrollmentLossComputer(validating: targ, sampleRate: sr)

        func loss(_ inputs: [MLXArray]) -> [MLXArray] {
            let p = inputs[0]
            return [computer.l1Loss(p)
                + computer.multiResolutionSTFTLoss(p)
                + computer.melLoss(p)]
        }

        let x = MLXRandom.normal([N]) * 0.1
        let grads = MLX.grad(loss)([x])
        MLX.eval(grads)
        let gradNorm = MLX.sqrt(MLX.sum(grads[0] * grads[0])).item(Float.self)
        XCTAssertTrue(gradNorm.isFinite)
        XCTAssertGreaterThan(gradNorm, 0)
    }

    // K-26: the block framing gives the gather framing's values, with a deterministic backward
    func testBlockFramingEqualsGatherFraming() {
        let signal = MLXRandom.normal([24_000], key: MLXRandom.key(3))
        for nFFT in EnrollmentLossComputer.fftSizes + [2048] {
            let resolution = STFTResolution(nFFT: nFFT, signalLength: signal.dim(0))
            let padded = EnrollmentLossComputer.reflectPad(signal, pad: nFFT / 2)
            let indices = (0 ..< resolution.numFrames).flatMap { f in (0 ..< nFFT).map { Int32(f * resolution.hop + $0) } }
            let gathered = MLX.take(padded, MLXArray(indices), axis: 0).reshaped(resolution.numFrames, nFFT)
            let blocks = EnrollmentLossComputer.frames(padded, resolution: resolution)
            XCTAssertEqual(blocks.shape, gathered.shape, "nFFT \(nFFT)")
            XCTAssertTrue(MLX.arrayEqual(blocks, gathered).item(Bool.self), "nFFT \(nFFT)")
        }
    }

    func testLossGradientIsDeterministic() throws {
        let reference = MLXRandom.normal([24_000], key: MLXRandom.key(4)) * 0.3
        let prediction = MLXRandom.normal([24_000], key: MLXRandom.key(5)) * 0.3
        let losses = try EnrollmentLossComputer(validating: reference)
        func gradient() -> MLXArray {
            let grads = MLX.grad({ (p: [MLXArray]) -> [MLXArray] in
                [losses.multiResolutionSTFTLoss(p[0]) + losses.melLoss(p[0])]
            })([prediction])
            MLX.eval(grads[0])
            return grads[0]
        }
        let first = gradient()
        for _ in 0 ..< 3 {
            XCTAssertTrue(MLX.arrayEqual(gradient(), first).item(Bool.self), "two backward passes must agree bit for bit")
        }
    }
}
