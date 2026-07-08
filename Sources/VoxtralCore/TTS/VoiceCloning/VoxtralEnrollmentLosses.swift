/**
 * VoxtralEnrollmentLosses - Differentiable audio losses for voice enrollment.
 *
 * MLX ports of the reconstruction losses used by the Python reference
 * pipeline (upstream training_script.py): L1 waveform, multi-resolution
 * STFT (spectral convergence + log-magnitude), and log-mel L1.
 *
 * The STFT mirrors torch.stft(center=true, pad_mode="reflect",
 * hop=n_fft/4, hann window); the mel filterbank mirrors torchaudio's
 * default (HTK scale, no normalization, power=1.0).
 *
 * Everything is built from basic differentiable ops (gather, matmul,
 * rfft) so gradients flow back to the learnable codes. Note that MLX on
 * Apple Silicon does not share the PyTorch MPS bug where torch.stft
 * backward silently corrupts gradients past ~2.5 s of signal — this was
 * verified by the enrollment validation run.
 */

import Foundation
import MLX
import MLXFFT

/// Precomputed constants for one STFT resolution over a fixed-length signal.
struct STFTResolution {
    let nFFT: Int
    let hop: Int
    let window: MLXArray        // (nFFT)
    let frameIndices: MLXArray  // (numFrames * nFFT) gather indices into the padded signal
    let numFrames: Int
    let paddedLength: Int

    init(nFFT: Int, signalLength: Int) {
        self.nFFT = nFFT
        self.hop = nFFT / 4

        // Periodic Hann window (torch.hann_window default)
        var w = [Float](repeating: 0, count: nFFT)
        for n in 0..<nFFT {
            w[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / Float(nFFT)))
        }
        self.window = MLXArray(w)

        // torch.stft(center=true): pad nFFT/2 on both sides,
        // frames start at k*hop in the padded signal.
        self.paddedLength = signalLength + nFFT
        self.numFrames = 1 + signalLength / hop

        var indices = [Int32]()
        indices.reserveCapacity(numFrames * nFFT)
        for f in 0..<numFrames {
            let start = Int32(f * hop)
            for n in 0..<nFFT {
                indices.append(start + Int32(n))
            }
        }
        self.frameIndices = MLXArray(indices)
    }
}

/// Differentiable losses over a fixed-length waveform pair.
/// Index tables and filterbanks are precomputed once for the signal length.
public final class EnrollmentLossComputer {

    public let signalLength: Int
    public let sampleRate: Int

    /// Multi-resolution FFT sizes — identical to the Python reference.
    static let fftSizes = [2296, 1418, 876, 542, 334, 206, 126, 76]

    private let resolutions: [STFTResolution]

    // Mel loss configuration (torchaudio MelSpectrogram defaults, power=1.0)
    private let melResolution: STFTResolution
    private let melFilterbank: MLXArray  // (nFreqs, nMels)

    public init(signalLength: Int, sampleRate: Int = 24_000, nMels: Int = 128) {
        self.signalLength = signalLength
        self.sampleRate = sampleRate
        self.resolutions = Self.fftSizes
            .filter { $0 <= signalLength }
            .map { STFTResolution(nFFT: $0, signalLength: signalLength) }
        self.melResolution = STFTResolution(nFFT: 2048, signalLength: signalLength)
        self.melFilterbank = Self.makeHTKMelFilterbank(
            nFreqs: 2048 / 2 + 1, nMels: nMels, sampleRate: sampleRate
        )
    }

    // MARK: - Public losses

    /// Mean absolute error between raw waveforms.
    public func l1Loss(_ pred: MLXArray, _ target: MLXArray) -> MLXArray {
        MLX.mean(MLX.abs(pred - target))
    }

    /// Multi-resolution STFT loss: spectral convergence + log-magnitude L1,
    /// averaged over resolutions. Mirrors multi_resolution_stft_loss.
    public func multiResolutionSTFTLoss(_ pred: MLXArray, _ target: MLXArray) -> MLXArray {
        var total = MLXArray(Float(0))
        for res in resolutions {
            let magPred = magnitudeSpectrogram(pred, resolution: res)
            let magTrue = magnitudeSpectrogram(target, resolution: res)

            let scLoss = frobeniusNorm(magTrue - magPred) / (frobeniusNorm(magTrue) + 1e-8)
            let logMagLoss = MLX.mean(MLX.abs(
                MLX.log(magPred + 1e-5) - MLX.log(magTrue + 1e-5)
            ))
            total = total + scLoss + logMagLoss
        }
        return total / Float(resolutions.count)
    }

    /// L1 over log-mel spectrograms. Mirrors mel_spectrogram_loss
    /// (n_fft 2048, hop 512, 128 HTK mels, power 1.0).
    public func melLoss(_ pred: MLXArray, _ target: MLXArray) -> MLXArray {
        let melPred = MLX.matmul(magnitudeSpectrogram(pred, resolution: melResolution), melFilterbank)
        let melTrue = MLX.matmul(magnitudeSpectrogram(target, resolution: melResolution), melFilterbank)
        return MLX.mean(MLX.abs(
            MLX.log(melPred + 1e-5) - MLX.log(melTrue + 1e-5)
        ))
    }

    // MARK: - Internals

    /// (numFrames, nFFT/2+1) magnitude spectrogram of a (signalLength,) waveform.
    private func magnitudeSpectrogram(_ x: MLXArray, resolution: STFTResolution) -> MLXArray {
        let padded = reflectPad(x, pad: resolution.nFFT / 2)
        let frames = MLX.take(padded, resolution.frameIndices, axis: 0)
            .reshaped(resolution.numFrames, resolution.nFFT)
        let spec = MLXFFT.rfft(frames * resolution.window, axis: -1)
        return MLX.abs(spec)
    }

    /// Reflect padding on both sides (torch pad_mode="reflect").
    private func reflectPad(_ x: MLXArray, pad: Int) -> MLXArray {
        let n = x.dim(0)
        let leftIdx = MLXArray((1...pad).reversed().map { Int32($0) })
        let rightIdx = MLXArray(((n - pad - 1)..<(n - 1)).reversed().map { Int32($0) })
        return MLX.concatenated(
            [MLX.take(x, leftIdx, axis: 0), x, MLX.take(x, rightIdx, axis: 0)],
            axis: 0
        )
    }

    private func frobeniusNorm(_ x: MLXArray) -> MLXArray {
        MLX.sqrt(MLX.sum(x * x))
    }

    /// HTK-scale triangular mel filterbank, no normalization
    /// (torchaudio melscale_fbanks defaults). Shape (nFreqs, nMels) so the
    /// spectrogram (frames, nFreqs) can be projected with one matmul.
    static func makeHTKMelFilterbank(nFreqs: Int, nMels: Int, sampleRate: Int) -> MLXArray {
        func hzToMel(_ hz: Double) -> Double { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Double) -> Double { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        let fMax = Double(sampleRate) / 2.0
        let melPoints = (0..<(nMels + 2)).map {
            melToHz(hzToMel(fMax) * Double($0) / Double(nMels + 1))
        }
        let freqs = (0..<nFreqs).map { Double($0) * fMax / Double(nFreqs - 1) }

        var fb = [Float](repeating: 0, count: nFreqs * nMels)
        for m in 0..<nMels {
            let (fLeft, fCenter, fRight) = (melPoints[m], melPoints[m + 1], melPoints[m + 2])
            for k in 0..<nFreqs {
                let f = freqs[k]
                let up = (f - fLeft) / (fCenter - fLeft)
                let down = (fRight - f) / (fRight - fCenter)
                fb[k * nMels + m] = Float(max(0.0, min(up, down)))
            }
        }
        return MLXArray(fb).reshaped(nFreqs, nMels)
    }
}
