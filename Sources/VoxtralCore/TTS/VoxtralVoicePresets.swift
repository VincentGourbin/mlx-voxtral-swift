/**
 * VoxtralVoicePresets - Voice embedding management for TTS
 *
 * Handles downloading and loading voice preset embeddings from HuggingFace.
 * The TTS model uses 20 voice presets stored as PyTorch .pt files
 * in the voice_embedding/ directory of the model repo.
 *
 * Note: .pt files are PyTorch pickle format. For MLX Swift compatibility,
 * these should be converted to .safetensors format. A Python conversion
 * script is provided for this purpose.
 */

import Foundation
import MLX

// MARK: - Voice Preset Enum

/// Available voice presets for Voxtral TTS
public enum VoxtralVoice: String, CaseIterable, Sendable {
    case casualFemale = "casual_female"
    case casualMale = "casual_male"
    case cheerfulFemale = "cheerful_female"
    case neutralFemale = "neutral_female"
    case neutralMale = "neutral_male"
    case ptMale = "pt_male"
    case ptFemale = "pt_female"
    case nlMale = "nl_male"
    case nlFemale = "nl_female"
    case itMale = "it_male"
    case itFemale = "it_female"
    case frMale = "fr_male"
    case frFemale = "fr_female"
    case esMale = "es_male"
    case esFemale = "es_female"
    case deMale = "de_male"
    case deFemale = "de_female"
    case arMale = "ar_male"
    case hiMale = "hi_male"
    case hiFemale = "hi_female"

    /// Display name for UI
    public var displayName: String {
        rawValue.replacingOccurrences(of: "_", with: " ").capitalized
    }

    /// Embedding file name on HuggingFace
    public var embeddingFileName: String {
        "voice_embedding/\(rawValue).pt"
    }

    /// Safetensors file name (after conversion)
    public var safetensorsFileName: String {
        "voice_embedding/\(rawValue).safetensors"
    }

    /// Language hint based on voice name
    public var language: String? {
        switch self {
        case .frMale, .frFemale: return "fr"
        case .esMale, .esFemale: return "es"
        case .deMale, .deFemale: return "de"
        case .itMale, .itFemale: return "it"
        case .ptMale, .ptFemale: return "pt"
        case .nlMale, .nlFemale: return "nl"
        case .arMale: return "ar"
        case .hiMale, .hiFemale: return "hi"
        case .casualFemale, .casualMale, .cheerfulFemale,
             .neutralFemale, .neutralMale: return "en"
        }
    }
}

// MARK: - Voice Preset Manager

/// Manages downloading and caching voice embeddings.
public class VoxtralVoicePresetManager: @unchecked Sendable {

    /// Directory where voice embeddings are cached
    public let cacheDirectory: URL

    /// The model repo for downloading voice embeddings
    public let modelRepoId: String

    public init(
        cacheDirectory: URL? = nil,
        modelRepoId: String = "mistralai/Voxtral-4B-TTS-2603"
    ) {
        if let cacheDirectory {
            self.cacheDirectory = cacheDirectory
        } else {
            #if os(iOS) || os(tvOS) || os(visionOS)
            let baseDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            #else
            let baseDir = FileManager.default.homeDirectoryForCurrentUser
            #endif
            self.cacheDirectory = baseDir.appendingPathComponent(".voxtral/voices")
        }
        self.modelRepoId = modelRepoId
    }

    /// Load a voice embedding from the model directory.
    ///
    /// Attempts to load in this order:
    /// 1. .safetensors format (pre-converted)
    /// 2. Raw MLX format
    ///
    /// - Parameters:
    ///   - voice: The voice preset to load
    ///   - modelDirectory: Directory containing the model files
    /// - Returns: Voice embedding tensor
    public func loadVoiceEmbedding(
        voice: VoxtralVoice,
        from modelDirectory: URL
    ) throws -> MLXArray {
        // Try safetensors first
        let safetensorsPath = modelDirectory.appendingPathComponent(voice.safetensorsFileName)
        if FileManager.default.fileExists(atPath: safetensorsPath.path) {
            let arrays = try MLX.loadArrays(url: safetensorsPath)
            guard let embedding = arrays["embedding"] ?? arrays.values.first else {
                throw VoxtralTTSError.voiceNotFound("No embedding found in \(safetensorsPath.lastPathComponent)")
            }
            return embedding
        }

        // Try .pt file (may work if it's a simple tensor)
        let ptPath = modelDirectory.appendingPathComponent(voice.embeddingFileName)
        if FileManager.default.fileExists(atPath: ptPath.path) {
            // MLX can sometimes load simple .pt files
            let arrays = try MLX.loadArrays(url: ptPath)
            guard let embedding = arrays.values.first else {
                throw VoxtralTTSError.voiceNotFound("Could not load .pt file for voice \(voice.rawValue). Convert to .safetensors first.")
            }
            return embedding
        }

        throw VoxtralTTSError.voiceNotFound(
            "Voice embedding not found for '\(voice.rawValue)'. "
            + "Expected at: \(safetensorsPath.path) or \(ptPath.path)"
        )
    }

    /// Check if a voice embedding is available locally.
    public func isVoiceAvailable(_ voice: VoxtralVoice, in modelDirectory: URL) -> Bool {
        let safetensorsPath = modelDirectory.appendingPathComponent(voice.safetensorsFileName)
        let ptPath = modelDirectory.appendingPathComponent(voice.embeddingFileName)
        return FileManager.default.fileExists(atPath: safetensorsPath.path)
            || FileManager.default.fileExists(atPath: ptPath.path)
    }

    /// List all voices available in a model directory.
    public func availableVoices(in modelDirectory: URL) -> [VoxtralVoice] {
        VoxtralVoice.allCases.filter { isVoiceAvailable($0, in: modelDirectory) }
    }
}
