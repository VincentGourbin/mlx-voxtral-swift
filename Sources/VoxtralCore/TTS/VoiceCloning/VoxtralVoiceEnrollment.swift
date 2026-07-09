/**
 * VoxtralVoiceEnrollment - Swift/MLX port of the STE voice-cloning loop.
 *
 * Recovers Voxtral TTS voice codes from a reference recording by gradient
 * descent through the FROZEN codec decoder (the encoder was never
 * published by Mistral). Learnable parameters are the discrete codes,
 * relaxed for gradients via Gumbel-Softmax + straight-through estimators;
 * the decoder weights never change.
 *
 * Port of upstream/training_script.py. Two deliberate differences:
 *   - Spectral losses run natively in MLX (no PyTorch MPS torch.stft
 *     backward bug — see VoxtralEnrollmentLosses).
 *   - Adam is implemented inline over the two free parameter tensors
 *     rather than through a Module optimizer.
 *
 * Output is a voice embedding [T+1, 3072] (END_AUDIO frame appended),
 * ready for VoxtralTTSPipeline.synthesize(text:voiceEmbedding:).
 */

import Foundation
import AVFoundation
import MLX
import MLXRandom

@available(macOS 14.0, *)
public final class VoxtralVoiceEnrollment {

    public struct Config {
        public var numFrames: Int = 100          // 100 frames @ 12.5 Hz = 8 s
        public var epochs: Int = 5000
        public var learningRate: Float = 0.1
        public var reconstructionWeight: Float = 0.5
        public var perceptualWeight: Float = 1.0  // multi-res STFT
        public var melWeight: Float = 1.0
        public var temperature: Float = 2.0
        public var temperatureDecay: Float = 0.99
        public var minTemperature: Float = 0.3
        public var gradClip: Float = 1.0        // global-norm gradient clip
        public var logEvery: Int = 500
        public init() {}
    }

    public struct Progress {
        public let epoch: Int
        public let totalLoss: Float
        public let reconLoss: Float
    }

    static let samplesPerFrame = 1920  // 24000 / 12.5
    static let semanticVocab = 8192
    static let nAcoustic = 36
    static let nSpecial = 2

    let model: VoxtralTTSModel
    let config: Config
    let numSamples: Int

    public init(model: VoxtralTTSModel, config: Config = Config()) {
        self.model = model
        self.config = config
        self.numSamples = config.numFrames * Self.samplesPerFrame
    }

    // MARK: - Reference preparation

    /// Load a recording as mono 24 kHz, trimmed to `numFrames` worth of
    /// samples and ending on the quietest window in the last 1.5 s (fade +
    /// trailing silence). A reference cut mid-speech destabilizes the start
    /// of later syntheses.
    public func prepareReference(url: URL) throws -> MLXArray {
        // Read the file at its NATIVE rate as mono float32 (channel mix only —
        // no sample-rate conversion here, which is where AVAudioConverter is
        // unreliable for upsampling), then resample to 24 kHz ourselves.
        let native = try readMonoFloat(url: url)
        var samples = Self.resampleLinear(native.samples, from: native.sampleRate, to: 24_000)
        if samples.count < numSamples {
            throw VoxtralTTSError.invalidConfiguration(
                "Reference too short: \(String(format: "%.1f", Double(samples.count) / 24_000))s "
                + "< \(String(format: "%.1f", Double(numSamples) / 24_000))s required"
            )
        }
        samples = Array(samples[0 ..< numSamples])

        // Cut at the quietest 50 ms window in the last 1.5 s.
        let sr = 24_000, win = 24_000 / 20
        let searchStart = max(0, numSamples - Int(1.5 * Double(sr)))
        var cut = numSamples, bestRMS = Float.greatestFiniteMagnitude
        var s = searchStart
        while s < numSamples - win {
            var acc: Float = 0
            for i in s ..< s + win { acc += samples[i] * samples[i] }
            let rms = (acc / Float(win)).squareRoot()
            if rms < bestRMS { bestRMS = rms; cut = s + win / 2 }
            s += sr / 100
        }
        let fade = sr / 50
        for i in max(0, cut - fade) ..< cut {
            samples[i] *= Float(cut - i) / Float(fade)
        }
        for i in cut ..< numSamples { samples[i] = 0 }
        return MLXArray(samples)
    }

    /// Read an audio file as mono float32 at its native sample rate.
    /// Only mixes channels — no sample-rate conversion (done separately).
    private func readMonoFloat(url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let file = try AVAudioFile(forReading: url)
        let fmt = file.processingFormat
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt,
                                         frameCapacity: AVAudioFrameCount(file.length)) else {
            throw VoxtralTTSError.invalidConfiguration("Cannot allocate audio buffer")
        }
        try file.read(into: buf)
        let n = Int(buf.frameLength)
        guard n > 0, let chans = buf.floatChannelData else {
            throw VoxtralTTSError.invalidConfiguration("Empty or non-float audio: \(url.lastPathComponent)")
        }
        let channelCount = Int(fmt.channelCount)
        var mono = [Float](repeating: 0, count: n)
        for c in 0 ..< channelCount {
            let ch = chans[c]
            for i in 0 ..< n { mono[i] += ch[i] }
        }
        if channelCount > 1 {
            let inv = 1 / Float(channelCount)
            for i in 0 ..< n { mono[i] *= inv }
        }
        return (mono, fmt.sampleRate)
    }

    /// Linear-interpolation resample. Adequate for enrollment references
    /// (the reference is an optimization target, not played back).
    /// Static + internal so it can be unit-tested without loading the model —
    /// this is the path that must never silence the signal when upsampling.
    static func resampleLinear(_ x: [Float], from srcRate: Double, to dstRate: Double) -> [Float] {
        if srcRate == dstRate || x.isEmpty { return x }
        let ratio = srcRate / dstRate
        let outCount = Int(Double(x.count) / ratio)
        var out = [Float](repeating: 0, count: outCount)
        for i in 0 ..< outCount {
            let pos = Double(i) * ratio
            let i0 = Int(pos)
            let frac = Float(pos - Double(i0))
            let a = x[i0]
            let b = i0 + 1 < x.count ? x[i0 + 1] : a
            out[i] = a + (b - a) * frac
        }
        return out
    }

    // MARK: - Forward (STE relaxation → decoder → waveform)

    /// Build the (1, T, 292) quantizer-space embedding from the learnable
    /// parameters using straight-through estimators, then decode to a
    /// waveform. Gradients flow to `semanticLogits` / `acousticValues`.
    private func synthesize(
        semanticLogits: MLXArray,   // (T, 8192)
        acousticValues: MLXArray,   // (T, 36)
        temperature: Float,
        training: Bool
    ) -> MLXArray {
        let T = config.numFrames

        // --- Semantic: Gumbel-Softmax + STE ---
        var logits = semanticLogits
        if training {
            let u = MLXRandom.uniform(0.0 ..< 1.0, [T, Self.semanticVocab])
            let gumbel = -MLX.log(-MLX.log(u + 1e-10) + 1e-10)
            logits = (semanticLogits + gumbel) / temperature
        } else {
            logits = semanticLogits / temperature
        }
        let probs = MLX.softmax(logits, axis: -1)                 // (T, 8192)

        let semCodebook = model.audioTokenizer.quantizer.semanticCodebook.codebook  // (8192, 256)
        let softEmb = MLX.matmul(probs, semCodebook)              // (T, 256)
        let hardCodes = probs.argMax(axis: -1)                    // (T)
        let hardEmb = MLX.take(semCodebook, hardCodes, axis: 0)   // (T, 256)
        let semanticEmb = softEmb + MLX.stopGradient(hardEmb - softEmb)

        // --- Acoustic: FSQ with tanh + STE ---
        let levels = Float(model.config.audioModel.acousticCodebookSize)  // 21
        let normalized = MLX.tanh(acousticValues)                 // (T, 36) in [-1, 1]
        let scaled = ((normalized + 1) / 2) * (levels - 1)        // [0, 20]
        let quantized = MLX.round(scaled)
        let acousticCodes = scaled + MLX.stopGradient(quantized - scaled)
        let acousticEmb = (acousticCodes * 2 / (levels - 1)) - 1  // (T, 36)

        // --- Combine → (1, T, 292) → frozen decoder ---
        let fullEmb = MLX.concatenated([semanticEmb, acousticEmb], axis: -1)  // (T, 292)
        let embeddings = MLX.expandedDimensions(fullEmb, axis: 0)             // (1, T, 292)
        let waveform = model.audioTokenizer.forwardEmbeddings(embeddings)     // (1, T*1920)
        return waveform.reshaped(-1)
    }

    // MARK: - Optimization

    /// Run the enrollment loop and return the learned discrete codes (T, 37).
    public func optimize(
        reference: MLXArray,                       // (numSamples,) 24 kHz mono
        progress: ((Progress) -> Void)? = nil
    ) -> MLXArray {
        let T = config.numFrames
        let ref = reference[0 ..< numSamples]
        let losses = EnrollmentLossComputer(signalLength: numSamples)

        // Learnable parameters (match Python init).
        var semanticLogits = MLXRandom.normal([T, Self.semanticVocab])
        var acousticValues = MLXRandom.normal([T, Self.nAcoustic]) * 0.1

        // Adam state.
        var mS = MLX.zeros(like: semanticLogits), vS = MLX.zeros(like: semanticLogits)
        var mA = MLX.zeros(like: acousticValues), vA = MLX.zeros(like: acousticValues)
        let beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8
        let minLR = config.learningRate * 0.01

        var temperature = config.temperature

        for epoch in 0 ..< config.epochs {
            let temp = temperature

            func lossFn(_ p: [MLXArray]) -> [MLXArray] {
                let wav = synthesize(
                    semanticLogits: p[0], acousticValues: p[1],
                    temperature: temp, training: true
                )
                let recon = losses.l1Loss(wav, ref)
                let percept = losses.multiResolutionSTFTLoss(wav, ref)
                let mel = losses.melLoss(wav, ref)
                let total = config.reconstructionWeight * recon
                    + config.perceptualWeight * percept
                    + config.melWeight * mel
                return [total, recon]
            }

            // Gradient w.r.t. BOTH free parameters (default is only arg 0).
            let (values, grads) = MLX.valueAndGrad(lossFn, argumentNumbers: [0, 1])(
                [semanticLogits, acousticValues]
            )

            // Global-norm gradient clipping (matches the Python reference's
            // grad_clip=1.0). Without it the semantic logits diverge on long
            // runs and the codes collapse to a single repeated frame.
            var gS = grads[0], gA = grads[1]
            let globalNorm = MLX.sqrt(MLX.sum(gS * gS) + MLX.sum(gA * gA))
            let scale = MLXArray(config.gradClip) / MLX.maximum(globalNorm, MLXArray(config.gradClip))
            gS = gS * scale
            gA = gA * scale

            // Cosine-annealed learning rate.
            let cos = 0.5 * (1 + Foundation.cos(Float.pi * Float(epoch) / Float(config.epochs)))
            let lr = minLR + (config.learningRate - minLR) * cos
            let t = Float(epoch + 1)

            (semanticLogits, mS, vS) = adamStep(
                param: semanticLogits, grad: gS, m: mS, v: vS,
                lr: lr, beta1: beta1, beta2: beta2, eps: eps, t: t
            )
            (acousticValues, mA, vA) = adamStep(
                param: acousticValues, grad: gA, m: mA, v: vA,
                lr: lr, beta1: beta1, beta2: beta2, eps: eps, t: t
            )

            MLX.eval(semanticLogits, acousticValues, mS, vS, mA, vA)

            temperature = max(config.minTemperature, temperature * config.temperatureDecay)

            if let progress, (epoch + 1) % config.logEvery == 0 || epoch == 0 {
                progress(Progress(
                    epoch: epoch + 1,
                    totalLoss: values[0].item(Float.self),
                    reconLoss: values[1].item(Float.self)
                ))
            }
        }

        return discreteCodes(semanticLogits: semanticLogits, acousticValues: acousticValues)
    }

    private func adamStep(
        param: MLXArray, grad: MLXArray, m: MLXArray, v: MLXArray,
        lr: Float, beta1: Float, beta2: Float, eps: Float, t: Float
    ) -> (MLXArray, MLXArray, MLXArray) {
        let mNew = beta1 * m + (1 - beta1) * grad
        let vNew = beta2 * v + (1 - beta2) * (grad * grad)
        let mHat = mNew / (1 - Foundation.pow(beta1, t))
        let vHat = vNew / (1 - Foundation.pow(beta2, t))
        let updated = param - lr * mHat / (MLX.sqrt(vHat) + eps)
        return (updated, mNew, vNew)
    }

    /// Purely discrete codes (T, 37): [semantic | 36 acoustic], no offset.
    private func discreteCodes(semanticLogits: MLXArray, acousticValues: MLXArray) -> MLXArray {
        let semantic = semanticLogits.argMax(axis: -1)               // (T)
        let levels = Float(model.config.audioModel.acousticCodebookSize)
        let normalized = MLX.tanh(acousticValues)
        let scaled = ((normalized + 1) / 2) * (levels - 1)
        let acoustic = MLX.round(scaled).asType(.int32)              // (T, 36)
        let sem2d = MLX.expandedDimensions(semantic.asType(.int32), axis: -1)
        let codes = MLX.concatenated([sem2d, acoustic], axis: -1)    // (T, 37)
        MLX.eval(codes)
        return codes
    }

    // MARK: - Export (codes → voice embedding [T+1, 3072])

    /// Convert discrete codes (T, 37) to a voice embedding, appending the
    /// END_AUDIO terminator frame (required — all official presets have it).
    /// Mirrors upstream codes_to_embeddings.py --add-end-token.
    public func codesToVoiceEmbedding(_ codes: MLXArray) -> MLXArray {
        let table = model.mmAudioEmbeddings.audioCodebookEmbeddings.weightTable()  // (9088, 3072)
        let T = codes.dim(0)

        // Per-codebook base offsets: semantic block (8192+2) then 36 acoustic
        // blocks of (21+2). Codes carry a +2 special-token offset at lookup.
        let acousticSize = model.config.audioModel.acousticCodebookSize + Self.nSpecial  // 23
        var offsets = [Int32](repeating: 0, count: 1 + Self.nAcoustic)
        offsets[0] = 0
        var running: Int32 = Int32(Self.semanticVocab + Self.nSpecial)  // 8194
        for k in 1 ... Self.nAcoustic {
            offsets[k] = running
            running += Int32(acousticSize)
        }
        let offsetsMx = MLXArray(offsets)  // (37)

        // Absolute row = offset_k + code_k + 2, summed over the 37 codebooks.
        let rows = codes + offsetsMx + MLXArray(Int32(Self.nSpecial))     // (T, 37)
        let looked = MLX.take(table, rows.reshaped(-1), axis: 0)          // (T*37, 3072)
        let voiceEmb = looked.reshaped(T, 1 + Self.nAcoustic, -1).sum(axis: 1)  // (T, 3072)

        // END_AUDIO frame: semantic=END(1), all acoustic=EMPTY(0).
        var endRows = [Int32](repeating: 0, count: 1 + Self.nAcoustic)
        endRows[0] = offsets[0] + 1
        for k in 1 ... Self.nAcoustic { endRows[k] = offsets[k] + 0 }
        let endFrame = MLX.take(table, MLXArray(endRows), axis: 0)
            .sum(axis: 0, keepDims: true)                                  // (1, 3072)

        let result = MLX.concatenated([voiceEmb, endFrame], axis: 0)       // (T+1, 3072)
        MLX.eval(result)
        return result
    }
}
