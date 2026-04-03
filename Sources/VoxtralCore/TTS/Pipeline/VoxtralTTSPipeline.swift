/**
 * VoxtralTTSPipeline - Simplified facade API for Voxtral TTS
 *
 * Usage:
 * ```swift
 * let pipeline = VoxtralTTSPipeline()
 * try await pipeline.loadModel()
 * let result = try await pipeline.synthesize(text: "Hello!", voice: .neutralFemale)
 * try WAVWriter.write(waveform: result.waveform, to: outputURL)
 * pipeline.unload()
 * ```
 */

import Foundation
import MLX

@available(macOS 14.0, *)
public class VoxtralTTSPipeline: @unchecked Sendable {

    // MARK: - Configuration

    public struct Configuration: Sendable {
        public var maxFrames: Int
        public var temperature: Float
        public var cfgAlpha: Float
        public var flowSteps: Int
        /// Sanitize text before synthesis (lowercase ALL-CAPS, add terminal punctuation, etc.)
        /// Disable if you need precise control over intonation via casing/punctuation.
        public var sanitizeText: Bool
        /// Trim low-energy lead-in silence frames from the beginning of generated audio.
        public var trimLeadIn: Bool

        public static var `default`: Configuration {
            Configuration(maxFrames: 2500, temperature: 0.0, cfgAlpha: 1.2, flowSteps: 8, sanitizeText: true, trimLeadIn: true)
        }

        public init(maxFrames: Int = 2500, temperature: Float = 0.0, cfgAlpha: Float = 1.2, flowSteps: Int = 8, sanitizeText: Bool = true, trimLeadIn: Bool = true) {
            self.maxFrames = maxFrames
            self.temperature = temperature
            self.cfgAlpha = cfgAlpha
            self.flowSteps = flowSteps
            self.sanitizeText = sanitizeText
            self.trimLeadIn = trimLeadIn
        }
    }

    // MARK: - State

    public enum State: Sendable {
        case unloaded, loading, ready, synthesizing, error(String)

        var isUnloaded: Bool { if case .unloaded = self { return true }; return false }
        var isReady: Bool { if case .ready = self { return true }; return false }
    }

    // MARK: - Properties

    public var configuration: Configuration
    public private(set) var state: State = .unloaded
    public let sampleRate: Int = 24000

    private var ttsModel: VoxtralTTSModel?
    private var tokenizer: TekkenTokenizer?
    private let voiceManager: VoxtralVoicePresetManager
    private var modelDirectory: URL?
    private var voiceEmbeddings: [String: MLXArray] = [:]

    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    // MARK: - Initialization

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.voiceManager = VoxtralVoicePresetManager()
    }

    // MARK: - Model Loading

    public func loadModel(modelInfo: VoxtralTTSModelInfo? = nil, progress: ProgressCallback? = nil) async throws {
        guard state.isUnloaded || { if case .error = state { return true }; return false }() else {
            throw VoxtralTTSError.invalidConfiguration("Model already loaded or loading")
        }

        state = .loading

        do {
            progress?(0.05, "Resolving TTS model...")
            let modelInfo = modelInfo ?? VoxtralTTSRegistry.defaultModel
            let modelDir = try await ModelDownloader.downloadTTSModel(modelInfo) { p, msg in
                progress?(0.05 + p * 0.35, msg)
            }
            self.modelDirectory = modelDir

            progress?(0.40, "Loading TTS model...")
            let model = try loadVoxtralTTSModel(from: modelDir) { p, msg in
                progress?(0.40 + Double(p) * 0.40, msg)
            }
            self.ttsModel = model

            progress?(0.85, "Loading tokenizer...")
            // TekkenTokenizer expects the MODEL DIRECTORY, not the tekken.json file path
            self.tokenizer = TekkenTokenizer(modelPath: modelDir.path)

            // Load voice embeddings
            progress?(0.90, "Loading voice embeddings...")
            let voiceDir = modelDir.appendingPathComponent("voice_embedding")
            for voice in VoxtralVoice.allCases {
                let safetensorsPath = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
                if FileManager.default.fileExists(atPath: safetensorsPath.path) {
                    let data = try MLX.loadArrays(url: safetensorsPath)
                    if let emb = data["embedding"] ?? data.values.first {
                        voiceEmbeddings[voice.rawValue] = emb
                    }
                }
            }

            progress?(1.0, "TTS model ready (\(voiceEmbeddings.count) voices loaded)")
            state = .ready

        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    // MARK: - Synthesis

    public func synthesize(
        text: String,
        voice: VoxtralVoice = .neutralFemale
    ) async throws -> TTSSynthesisResult {
        guard state.isReady, let model = ttsModel, let tokenizer else {
            throw VoxtralTTSError.invalidConfiguration("Model not loaded")
        }

        guard let voiceEmb = voiceEmbeddings[voice.rawValue] else {
            throw VoxtralTTSError.voiceNotFound("Voice '\(voice.rawValue)' not loaded")
        }

        state = .synthesizing
        let startTime = Date()

        do {
            // Generate audio codes
            let (codes, numFrames, ttft) = model.generate(
                text: text,
                voiceEmbedding: voiceEmb,
                tokenizer: tokenizer,
                maxTokens: configuration.maxFrames,
                sanitize: configuration.sanitizeText
            )

            guard numFrames > 0 else {
                state = .ready
                throw VoxtralTTSError.synthesisError("No audio frames generated")
            }

            // Decode to waveform, optionally trim lead-in silence
            let rawWaveform = model.decodeToWaveform(codes)
            MLX.eval(rawWaveform)
            let waveform = configuration.trimLeadIn ? trimLeadInSilence(rawWaveform, sampleRate: sampleRate) : rawWaveform

            let generationTime = Date().timeIntervalSince(startTime)
            state = .ready

            return TTSSynthesisResult(
                waveform: waveform,
                numFrames: numFrames,
                sampleRate: sampleRate,
                generationTime: generationTime,
                timeToFirstToken: ttft
            )
        } catch {
            state = .ready
            throw error
        }
    }

    @discardableResult
    public func synthesizeToFile(
        text: String,
        voice: VoxtralVoice = .neutralFemale,
        outputURL: URL
    ) async throws -> TTSSynthesisResult {
        let result = try await synthesize(text: text, voice: voice)
        try WAVWriter.write(waveform: result.waveform, to: outputURL, sampleRate: sampleRate)
        return result
    }

    // MARK: - ZeroVoice Synthesis

    /// Synthesize speech using a ZeroVoice coordinate (procedural voice).
    public func synthesize(
        text: String,
        voiceCoordinate: (x: Int, y: Int, z: Int)
    ) async throws -> TTSSynthesisResult {
        guard let voiceEmb = zeroVoice?.voiceAt(x: voiceCoordinate.x, y: voiceCoordinate.y, z: voiceCoordinate.z) else {
            throw VoxtralTTSError.voiceNotFound("Could not generate voice for coordinate (\(voiceCoordinate.x), \(voiceCoordinate.y), \(voiceCoordinate.z))")
        }
        return try await synthesize(text: text, voiceEmbedding: voiceEmb)
    }

    /// Synthesize speech using a pre-computed blended voice embedding.
    public func synthesize(
        text: String,
        voiceEmbedding: MLXArray
    ) async throws -> TTSSynthesisResult {
        guard state.isReady, let model = ttsModel, let tokenizer else {
            throw VoxtralTTSError.invalidConfiguration("Model not loaded")
        }

        state = .synthesizing
        let startTime = Date()

        do {
            let (codes, numFrames, ttft) = model.generate(
                text: text,
                voiceEmbedding: voiceEmbedding,
                tokenizer: tokenizer,
                maxTokens: configuration.maxFrames,
                sanitize: configuration.sanitizeText
            )

            guard numFrames > 0 else {
                state = .ready
                throw VoxtralTTSError.synthesisError("No audio frames generated")
            }

            let rawWaveform = model.decodeToWaveform(codes)
            MLX.eval(rawWaveform)
            let waveform = configuration.trimLeadIn ? trimLeadInSilence(rawWaveform, sampleRate: sampleRate) : rawWaveform

            let generationTime = Date().timeIntervalSince(startTime)
            state = .ready

            return TTSSynthesisResult(
                waveform: waveform,
                numFrames: numFrames,
                sampleRate: sampleRate,
                generationTime: generationTime,
                timeToFirstToken: ttft
            )
        } catch {
            state = .ready
            throw error
        }
    }

    /// ZeroVoice generator (lazy-initialized from loaded voice embeddings).
    public var zeroVoice: VoxtralZeroVoice? {
        guard !voiceEmbeddings.isEmpty else { return nil }
        return VoxtralZeroVoice(voiceEmbeddings: voiceEmbeddings)
    }

    /// Get the recipe for a ZeroVoice coordinate (metadata, no computation).
    public func voiceRecipe(x: Int, y: Int, z: Int) -> VoiceRecipe? {
        zeroVoice?.voiceRecipe(x: x, y: y, z: z)
    }

    /// Blend two named voice presets.
    public func blendVoicePresets(_ voiceA: VoxtralVoice, _ voiceB: VoxtralVoice, t: Float) -> MLXArray? {
        guard let embA = voiceEmbeddings[voiceA.rawValue],
              let embB = voiceEmbeddings[voiceB.rawValue] else { return nil }
        return blendVoices(voiceA: embA, voiceB: embB, t: t)
    }

    // MARK: - Streaming Synthesis

    /// Synthesize speech as a stream of audio chunks, enabling real-time playback.
    ///
    /// Each chunk contains decoded PCM audio that can be immediately scheduled on an audio player.
    /// The first chunk measures time-to-first-token (TTFT).
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice preset
    ///   - chunkSize: Number of frames per chunk (1 frame = 80ms audio). Default 10 = 800ms chunks.
    public func synthesizeStreaming(
        text: String,
        voice: VoxtralVoice = .neutralFemale,
        chunkSize: Int = 10
    ) -> AsyncThrowingStream<TTSStreamingChunk, Error> {
        guard state.isReady, let model = ttsModel, let tokenizer else {
            return AsyncThrowingStream { $0.finish(throwing: VoxtralTTSError.invalidConfiguration("Model not loaded")) }
        }
        guard let voiceEmb = voiceEmbeddings[voice.rawValue] else {
            return AsyncThrowingStream { $0.finish(throwing: VoxtralTTSError.voiceNotFound("Voice '\(voice.rawValue)' not loaded")) }
        }

        state = .synthesizing
        let startTime = Date()

        let capturedMaxFrames = configuration.maxFrames
        let capturedSampleRate = sampleRate
        let capturedSanitize = configuration.sanitizeText

        // Box non-Sendable captures for Swift 6 strict concurrency
        final class StreamContext: @unchecked Sendable {
            let model: VoxtralTTSModel
            let tokenizer: TekkenTokenizer
            let voiceEmb: MLXArray
            let pipeline: VoxtralTTSPipeline
            init(model: VoxtralTTSModel, tokenizer: TekkenTokenizer, voiceEmb: MLXArray, pipeline: VoxtralTTSPipeline) {
                self.model = model; self.tokenizer = tokenizer; self.voiceEmb = voiceEmb; self.pipeline = pipeline
            }
        }
        let ctx = StreamContext(model: model, tokenizer: tokenizer, voiceEmb: voiceEmb, pipeline: self)

        return AsyncThrowingStream { continuation in
            Task {
                var previousSampleCount = 0
                var isFirst = true

                do {
                    let codeStream = ctx.model.generateStreaming(
                        text: text,
                        voiceEmbedding: ctx.voiceEmb,
                        tokenizer: ctx.tokenizer,
                        maxTokens: capturedMaxFrames,
                        chunkSize: chunkSize,
                        sanitize: capturedSanitize
                    )

                    for try await chunk in codeStream {
                        // Decode all accumulated codes to get full waveform
                        let fullWaveform = ctx.model.decodeToWaveform(chunk.accumulatedCodes)
                        MLX.eval(fullWaveform)

                        let totalSamples = fullWaveform.dim(0)

                        // Extract only the new samples
                        let newWaveform: MLXArray
                        if previousSampleCount > 0 && previousSampleCount < totalSamples {
                            newWaveform = fullWaveform[previousSampleCount...]
                        } else {
                            newWaveform = fullWaveform
                        }

                        let elapsed = Date().timeIntervalSince(startTime)

                        continuation.yield(TTSStreamingChunk(
                            waveform: newWaveform,
                            frameIndex: chunk.totalFrames - chunk.newFrameCount,
                            frameCount: chunk.newFrameCount,
                            totalFrames: chunk.totalFrames,
                            sampleRate: capturedSampleRate,
                            isFirst: isFirst,
                            isFinal: chunk.isFinal,
                            elapsed: elapsed
                        ))

                        previousSampleCount = totalSamples
                        isFirst = false
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }

                ctx.pipeline.state = .ready
            }
        }
    }

    // MARK: - Resource Management

    public func unload() {
        ttsModel = nil
        tokenizer = nil
        voiceEmbeddings = [:]
        modelDirectory = nil
        state = .unloaded
    }

    public var isReady: Bool { state.isReady }

    public var availableVoices: [VoxtralVoice] {
        VoxtralVoice.allCases.filter { voiceEmbeddings[$0.rawValue] != nil }
    }
}
