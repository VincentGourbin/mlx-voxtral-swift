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

        // Reference clean-up (see prepareReference). The recording's noise
        // floor is otherwise LEARNED as part of the voice: every synthesis
        // then reproduces it, and downstream silence detection (lead-in trim,
        // lip-sync alignment) stops finding any silence at all.
        /// High-pass cutoff (Hz) applied to the reference before optimization;
        /// removes rumble/DC below the voice band. `nil` disables.
        public var referenceHighPassHz: Float? = 70
        /// Gate reference windows whose RMS falls below `gateThresholdDB`
        /// (relative to the LOUDEST 20 ms window's RMS — not the sample peak,
        /// so a single click or plosive spike cannot inflate the threshold
        /// and gate genuine speech) to true silence. Set `false` to keep the
        /// raw recording, ambience included.
        public var gateReference: Bool = true
        /// Gate threshold relative to the loudest 20 ms window RMS, in dB.
        /// −30 is deliberately conservative: gating that eats soft speech is
        /// far worse than a slightly higher learned noise floor (the output
        /// trim is peak-relative and handles the latter downstream).
        public var gateThresholdDB: Float = -30

        public init() {}
    }

    public struct Progress {
        public let epoch: Int
        public let totalLoss: Float
        public let reconLoss: Float
    }

    static let samplesPerFrame = 1920  // 24000 / 12.5
    static let nSpecial = 2             // EMPTY_AUDIO=0, END_AUDIO=1 per codebook

    let model: VoxtralTTSModel
    let config: Config
    let numSamples: Int

    // Codebook geometry read from the model config, not hardcoded, so variant
    // checkpoints produce correct offsets.
    let semanticVocab: Int   // e.g. 8192
    let nAcoustic: Int       // number of acoustic codebooks, e.g. 36
    let acousticLevels: Int  // FSQ levels per acoustic codebook, e.g. 21

    public init(model: VoxtralTTSModel, config: Config = Config()) {
        self.model = model
        self.config = config
        self.numSamples = config.numFrames * Self.samplesPerFrame
        self.semanticVocab = model.config.audioModel.semanticCodebookSize
        self.nAcoustic = model.config.audioModel.nAcousticCodebook
        self.acousticLevels = model.config.audioModel.acousticCodebookSize
    }

    // MARK: - Reference preparation

    /// Load a recording as mono 24 kHz, trimmed to `numFrames` worth of
    /// samples and ending on the quietest window in the last 1.5 s (fade +
    /// trailing silence). A reference cut mid-speech destabilizes the start
    /// of later syntheses.
    ///
    /// Unless disabled in `Config`, the reference is also high-passed and
    /// noise-gated first: whatever is in the reference — noise floor
    /// included — becomes the optimization target and is baked into the
    /// cloned voice, so silences must be true silence going in.
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

        if let hp = config.referenceHighPassHz, hp > 0 {
            samples = Self.highPass(samples, cutoff: Double(hp), sampleRate: 24_000)
        }
        if config.gateReference {
            samples = Self.gate(samples, sampleRate: 24_000, thresholdDB: config.gateThresholdDB)
        }

        // Cut at the quietest 50 ms window in the last 1.5 s. `<=` so that
        // among equally quiet windows the LATEST wins: gating produces exact
        // zeros, and with strict `<` a gated inter-word pause early in the
        // search range would beat the trailing silence — cutting mid-pause
        // and silencing the final word(s) of the reference.
        let sr = 24_000, win = 24_000 / 20
        let searchStart = max(0, numSamples - Int(1.5 * Double(sr)))
        var cut = numSamples, bestRMS = Float.greatestFiniteMagnitude
        var s = searchStart
        while s < numSamples - win {
            let rms = Self.windowRMS(samples, s ..< s + win)
            if rms <= bestRMS { bestRMS = rms; cut = s + win / 2 }
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

    /// Resample to `dstRate`. When downsampling, a windowed-sinc low-pass is
    /// applied first so content above the target Nyquist doesn't alias into
    /// the band — the reference IS the optimization target, so aliasing would
    /// bake artifacts into the cloned voice. Static + internal so it can be
    /// unit-tested without loading the model; must never silence the signal.
    static func resampleLinear(_ x: [Float], from srcRate: Double, to dstRate: Double) -> [Float] {
        if srcRate == dstRate || x.isEmpty { return x }
        let source = dstRate < srcRate
            ? lowPass(x, cutoff: dstRate / 2, sampleRate: srcRate)
            : x
        let ratio = srcRate / dstRate
        let outCount = Int(Double(source.count) / ratio)
        var out = [Float](repeating: 0, count: outCount)
        for i in 0 ..< outCount {
            let pos = Double(i) * ratio
            let i0 = Int(pos)
            let frac = Float(pos - Double(i0))
            let a = source[i0]
            let b = i0 + 1 < source.count ? source[i0 + 1] : a
            out[i] = a + (b - a) * frac
        }
        return out
    }

    /// FIR high-pass: spectral complement of `lowPass` (x − LP(x)), so it
    /// shares the low-pass's linear phase and unity passband. Removes DC and
    /// rumble below the voice band that the optimization would otherwise
    /// learn as part of the voice. Static + internal for unit testing.
    static func highPass(_ x: [Float], cutoff: Double, sampleRate: Double) -> [Float] {
        let lp = lowPass(x, cutoff: cutoff, sampleRate: sampleRate)
        var out = x
        for i in 0 ..< out.count { out[i] -= lp[i] }
        return out
    }

    /// RMS of one window of samples. Shared by the gate and the tail-cut
    /// scan in `prepareReference` so "quiet" means the same thing in both.
    static func windowRMS(_ x: [Float], _ range: Range<Int>) -> Float {
        guard !range.isEmpty else { return 0 }
        var acc: Float = 0
        for i in range { acc += x[i] * x[i] }
        return (acc / Float(range.count)).squareRoot()
    }

    /// Noise gate: 20 ms windows whose RMS falls below `thresholdDB` relative
    /// to the LOUDEST window's RMS are pushed to true silence, with 5 ms
    /// linear ramps at every open/close so gating never clicks. The reference
    /// level is a window RMS, not the sample peak: a single click or plosive
    /// spike would inflate a sample-peak threshold above genuine speech
    /// windows and gate the voice itself — the one failure mode a reference
    /// gate must never have, since the gated audio IS the optimization
    /// target. Frames that are quiet in the reference must be EXACTLY zero —
    /// a learned noise floor shows up in every later synthesis. Static +
    /// internal for unit testing.
    static func gate(_ x: [Float], sampleRate: Int, thresholdDB: Float) -> [Float] {
        guard !x.isEmpty else { return x }

        let win = sampleRate / 50  // 20 ms
        let numWindows = (x.count + win - 1) / win

        var windowLevels = [Float](repeating: 0, count: numWindows)
        for w in 0 ..< numWindows {
            windowLevels[w] = windowRMS(x, w * win ..< min((w + 1) * win, x.count))
        }
        guard let loudest = windowLevels.max(), loudest > 0 else { return x }
        let threshold = loudest * Foundation.pow(10, thresholdDB / 20)

        // Per-window open/closed decision.
        var open = [Bool](repeating: true, count: numWindows)
        for w in 0 ..< numWindows {
            open[w] = windowLevels[w] >= threshold
        }

        // Per-sample gain with short ramps at open/close boundaries.
        let fade = sampleRate / 200  // 5 ms
        var out = x
        for w in 0 ..< numWindows where !open[w] {
            let start = w * win
            let end = min(start + win, x.count)
            for i in start ..< end { out[i] = 0 }
        }
        for w in 0 ..< numWindows {
            guard open[w] else { continue }
            let start = w * win
            let end = min(start + win, x.count)
            // Ramp in if the previous window is closed, out if the next is.
            if w > 0, !open[w - 1] {
                for i in 0 ..< min(fade, end - start) {
                    out[start + i] *= Float(i) / Float(fade)
                }
            }
            if w + 1 < numWindows, !open[w + 1] {
                for i in 0 ..< min(fade, end - start) {
                    out[end - 1 - i] *= Float(i) / Float(fade)
                }
            }
        }
        return out
    }

    /// Zero-phase-ish FIR low-pass (Hann-windowed sinc, 64 taps) applied as a
    /// centered convolution. Cutoff and sampleRate are in Hz.
    static func lowPass(_ x: [Float], cutoff: Double, sampleRate: Double) -> [Float] {
        let taps = 64
        let fc = cutoff / sampleRate  // normalized cutoff (cycles/sample)
        let half = taps / 2
        var kernel = [Float](repeating: 0, count: taps + 1)
        var sum: Float = 0
        for i in 0 ... taps {
            let m = Double(i - half)
            let sinc = m == 0 ? 2 * fc : sin(2 * .pi * fc * m) / (.pi * m)
            let hann = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(taps))
            let v = Float(sinc * hann)
            kernel[i] = v
            sum += v
        }
        for i in 0 ... taps { kernel[i] /= sum }  // unity DC gain

        let n = x.count
        var out = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            var acc: Float = 0
            for k in 0 ... taps {
                let j = i + k - half
                if j >= 0 && j < n { acc += x[j] * kernel[k] }
            }
            out[i] = acc
        }
        return out
    }

    // MARK: - Forward (STE relaxation → decoder → waveform)

    /// Build the (1, T, 292) quantizer-space embedding from the learnable
    /// parameters using straight-through estimators, then decode to a
    /// waveform. Gradients flow to `semanticLogits` / `acousticValues`.
    private func synthesize(
        semanticLogits: MLXArray,   // (T, semanticVocab)
        acousticValues: MLXArray,   // (T, nAcoustic)
        semCodebook: MLXArray,      // (semanticVocab, semanticDim) — precomputed once
        temperature: Float
    ) -> MLXArray {
        let T = config.numFrames

        // --- Semantic: Gumbel-Softmax + STE ---
        let u = MLXRandom.uniform(0.0 ..< 1.0, [T, semanticVocab])
        let gumbel = -MLX.log(-MLX.log(u + 1e-10) + 1e-10)
        let logits = (semanticLogits + gumbel) / temperature
        let probs = MLX.softmax(logits, axis: -1)                 // (T, semanticVocab)

        let softEmb = MLX.matmul(probs, semCodebook)              // (T, semanticDim)
        let hardCodes = probs.argMax(axis: -1)                    // (T)
        let hardEmb = MLX.take(semCodebook, hardCodes, axis: 0)   // (T, semanticDim)
        let semanticEmb = softEmb + MLX.stopGradient(hardEmb - softEmb)

        // --- Acoustic: FSQ with tanh + STE ---
        let levels = Float(acousticLevels)
        let (scaled, quantized) = acousticScaledQuantized(acousticValues)
        let acousticCodes = scaled + MLX.stopGradient(quantized - scaled)
        let acousticEmb = (acousticCodes * 2 / (levels - 1)) - 1  // (T, nAcoustic)

        // --- Combine → (1, T, 292) → frozen decoder ---
        let fullEmb = MLX.concatenated([semanticEmb, acousticEmb], axis: -1)  // (T, 292)
        let embeddings = MLX.expandedDimensions(fullEmb, axis: 0)             // (1, T, 292)
        let waveform = model.audioTokenizer.forwardEmbeddings(embeddings)     // (1, T*1920)
        return waveform.reshaped(-1)
    }

    /// Shared FSQ mapping used by both the STE forward and the final discrete
    /// export, so the codes written to disk always match what was optimized.
    /// Returns the continuous `scaled` value and its rounded `quantized` form.
    private func acousticScaledQuantized(_ acousticValues: MLXArray) -> (scaled: MLXArray, quantized: MLXArray) {
        let levels = Float(acousticLevels)
        let normalized = MLX.tanh(acousticValues)                 // [-1, 1]
        let scaled = ((normalized + 1) / 2) * (levels - 1)        // [0, levels-1]
        return (scaled, MLX.round(scaled))
    }

    // MARK: - Optimization

    /// Run the enrollment loop and return the learned discrete codes (T, 37).
    public func optimize(
        reference: MLXArray,                       // (numSamples,) 24 kHz mono
        progress: ((Progress) -> Void)? = nil
    ) -> MLXArray {
        optimizeCore(reference: reference, progress: progress, shouldContinue: nil).codes
    }

    /// Cancellable variant: `shouldContinue` is polled at the top of every
    /// epoch (an epoch is ~100–500 ms); the first `false` stops the loop and
    /// throws `CancellationError`. Cancellation is decided by that single
    /// in-loop poll — the partial codes never escape, so a non-latching
    /// predicate cannot leak an under-trained result to the caller.
    public func optimize(
        reference: MLXArray,                       // (numSamples,) 24 kHz mono
        progress: ((Progress) -> Void)? = nil,
        shouldContinue: @escaping () -> Bool
    ) throws -> MLXArray {
        let (codes, cancelled) = optimizeCore(
            reference: reference, progress: progress, shouldContinue: shouldContinue)
        if cancelled { throw CancellationError() }
        return codes
    }

    private func optimizeCore(
        reference: MLXArray,
        progress: ((Progress) -> Void)?,
        shouldContinue: (() -> Bool)?
    ) -> (codes: MLXArray, cancelled: Bool) {
        let T = config.numFrames
        precondition(reference.dim(0) >= numSamples,
                     "reference must have at least \(numSamples) samples, got \(reference.dim(0))")
        let ref = reference[0 ..< numSamples]
        let losses = EnrollmentLossComputer(reference: ref)

        // Semantic centroid table is constant across the run — compute once.
        let semCodebook = model.audioTokenizer.quantizer.semanticCodebook.codebook  // (semanticVocab, semanticDim)
        MLX.eval(semCodebook)

        // Learnable parameters (match Python init).
        var semanticLogits = MLXRandom.normal([T, semanticVocab])
        var acousticValues = MLXRandom.normal([T, nAcoustic]) * 0.1

        // Adam state.
        var mS = MLX.zeros(like: semanticLogits), vS = MLX.zeros(like: semanticLogits)
        var mA = MLX.zeros(like: acousticValues), vA = MLX.zeros(like: acousticValues)
        let beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8
        let minLR = config.learningRate * 0.01

        var temperature = config.temperature

        var cancelled = false
        for epoch in 0 ..< config.epochs {
            if let shouldContinue, !shouldContinue() { cancelled = true; break }
            let temp = temperature

            func lossFn(_ p: [MLXArray]) -> [MLXArray] {
                let wav = synthesize(
                    semanticLogits: p[0], acousticValues: p[1],
                    semCodebook: semCodebook, temperature: temp
                )
                let recon = losses.l1Loss(wav)
                let percept = losses.multiResolutionSTFTLoss(wav)
                let mel = losses.melLoss(wav)
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

        return (discreteCodes(semanticLogits: semanticLogits, acousticValues: acousticValues),
                cancelled)
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
        let (_, quantized) = acousticScaledQuantized(acousticValues)
        let acoustic = quantized.asType(.int32)                      // (T, nAcoustic)
        let sem2d = MLX.expandedDimensions(semantic.asType(.int32), axis: -1)
        let codes = MLX.concatenated([sem2d, acoustic], axis: -1)    // (T, 1+nAcoustic)
        MLX.eval(codes)
        return codes
    }

    // MARK: - Export (codes → voice embedding [T+1, 3072])

    /// Convert discrete codes (T, 37) to a voice embedding, appending the
    /// END_AUDIO terminator frame (required — all official presets have it).
    /// Mirrors upstream codes_to_embeddings.py --add-end-token.
    public func codesToVoiceEmbedding(_ codes: MLXArray) -> MLXArray {
        let T = codes.dim(0)
        let cb = 1 + nAcoustic  // codebooks per frame

        // Per-codebook base offsets: semantic block (semanticVocab+2) then
        // nAcoustic blocks of (acousticLevels+2). Codes carry a +2 special-
        // token offset at lookup.
        let acousticSize = acousticLevels + Self.nSpecial
        var offsets = [Int32](repeating: 0, count: cb)
        offsets[0] = 0
        var running = Int32(semanticVocab + Self.nSpecial)
        for k in 1 ..< cb {
            offsets[k] = running
            running += Int32(acousticSize)
        }
        let offsetsMx = MLXArray(offsets)

        // Absolute row = offset_k + code_k + 2 for every codebook, plus the
        // END_AUDIO frame (semantic=END(1), acoustic=EMPTY(0)). Gather all the
        // needed rows in one shot so only those are dequantized (not the whole
        // 9088-row table).
        let codeRows = (codes + offsetsMx + MLXArray(Int32(Self.nSpecial))).reshaped(-1)  // (T*cb)
        var endRows = [Int32](repeating: 0, count: cb)
        endRows[0] = offsets[0] + 1
        for k in 1 ..< cb { endRows[k] = offsets[k] + 0 }
        let allRows = MLX.concatenated([codeRows, MLXArray(endRows)], axis: 0)             // (T*cb + cb)

        let looked = model.mmAudioEmbeddings.audioCodebookEmbeddings.rows(allRows).asType(.float32)
        let voiceEmb = looked[0 ..< (T * cb)].reshaped(T, cb, -1).sum(axis: 1)   // (T, dim)
        let endFrame = looked[(T * cb)...].sum(axis: 0, keepDims: true)          // (1, dim)

        let result = MLX.concatenated([voiceEmb, endFrame], axis: 0)             // (T+1, dim)
        MLX.eval(result)
        return result
    }
}
