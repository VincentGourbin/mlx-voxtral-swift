/**
 * VoxtralZeroVoice - Procedural voice generation from 3D coordinates
 *
 * Maps (x, y, z) integer coordinates to deterministic voice embeddings via
 * position hashing, coherent noise, and cross-family SLERP blending.
 * Creates ~3.2 billion unique voices from 20 base presets with 0 bytes extra data.
 *
 * Reference: MushroomFleet/ZeroVoice-Voxtral-mini-4b/zerovoice.py
 */

import Foundation
import MLX

// MARK: - Voice Families

/// Voice family groupings by language region
public enum VoiceFamily: CaseIterable, Sendable {
    case english
    case european
    case asianArabic

    public var voices: [VoxtralVoice] {
        switch self {
        case .english:
            return [.casualFemale, .casualMale, .cheerfulFemale, .neutralFemale, .neutralMale]
        case .european:
            return [.frMale, .frFemale, .esMale, .esFemale, .deMale, .deFemale,
                    .itMale, .itFemale, .ptMale, .ptFemale, .nlMale, .nlFemale]
        case .asianArabic:
            return [.arMale, .hiMale, .hiFemale]
        }
    }

    /// Select family based on z-coordinate
    public static func fromZ(_ z: Int) -> VoiceFamily {
        if z < 100 { return .english }
        if z < 200 { return .european }
        return .asianArabic
    }

    /// Get the cross-family partner for pairing
    public func crossFamily(z: Int) -> VoiceFamily {
        switch self {
        case .english: return .european
        case .european: return z < 150 ? .english : .asianArabic
        case .asianArabic: return .european
        }
    }
}

// MARK: - Voice Recipe (metadata)

/// Describes how a voice was constructed without computing the embedding
public struct VoiceRecipe: Sendable {
    public let voiceA: VoxtralVoice
    public let voiceB: VoxtralVoice
    public let blendWeight: Float
    public let coordinate: (x: Int, y: Int, z: Int)
}

// MARK: - ZeroVoice Generator

/// Procedural voice generator using 3D coordinate-based SLERP blending.
public class VoxtralZeroVoice: @unchecked Sendable {

    private let voiceEmbeddings: [String: MLXArray]
    /// Maximum SLERP blend weight (prevents voice identity from collapsing)
    public let maxBlendWeight: Float = 0.20

    public init(voiceEmbeddings: [String: MLXArray]) {
        self.voiceEmbeddings = voiceEmbeddings
    }

    // MARK: - Public API

    /// Generate a deterministic voice embedding from 3D coordinates.
    ///
    /// - Parameters:
    ///   - x: X coordinate (variation within family)
    ///   - y: Y coordinate (variation within family)
    ///   - z: Z coordinate (family selection: <100=EN, 100-199=EU, >=200=Asia/Arabic)
    /// - Returns: blended voice embedding [N, 3072]
    public func voiceAt(x: Int, y: Int, z: Int) -> MLXArray? {
        let recipe = voiceRecipe(x: x, y: y, z: z)

        guard let embA = voiceEmbeddings[recipe.voiceA.rawValue],
              let embB = voiceEmbeddings[recipe.voiceB.rawValue] else {
            return nil
        }

        return blendVoices(voiceA: embA, voiceB: embB, t: recipe.blendWeight)
    }

    /// Get the recipe (metadata) for a coordinate without computing the embedding.
    public func voiceRecipe(x: Int, y: Int, z: Int) -> VoiceRecipe {
        let familyA = VoiceFamily.fromZ(z)
        let familyB = familyA.crossFamily(z: z)

        let hash = positionHash(x: x, y: y, z: z)

        // Select voice A from primary family
        let voicesA = familyA.voices
        let idxA = Int(hash % UInt64(voicesA.count))
        let voiceA = voicesA[idxA]

        // Select voice B from cross-family (use rotated hash)
        let voicesB = familyB.voices
        let hash2 = positionHash(x: x &+ 7919, y: y &+ 104729, z: z &+ 1299721)
        let idxB = Int(hash2 % UInt64(voicesB.count))
        let voiceB = voicesB[idxB]

        // Coherent noise for smooth blend weight
        let rawNoise = coherentValue(x: x, y: y)
        let blendWeight = rawNoise * maxBlendWeight

        return VoiceRecipe(
            voiceA: voiceA,
            voiceB: voiceB,
            blendWeight: blendWeight,
            coordinate: (x, y, z)
        )
    }

    /// List all voice families and their members.
    public var availableFamilies: [(family: VoiceFamily, voices: [VoxtralVoice], loaded: Int)] {
        VoiceFamily.allCases.map { family in
            let loaded = family.voices.filter { voiceEmbeddings[$0.rawValue] != nil }.count
            return (family, family.voices, loaded)
        }
    }

    // MARK: - Hashing

    /// FNV-1a 64-bit hash of packed (x, y, z) coordinates.
    /// Deterministic, fast, good distribution — no external dependency needed.
    private func positionHash(x: Int, y: Int, z: Int) -> UInt64 {
        var hash: UInt64 = 14695981039346656037  // FNV offset basis
        let prime: UInt64 = 1099511628211          // FNV prime

        for byte in withUnsafeBytes(of: Int64(x), Array.init) {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        for byte in withUnsafeBytes(of: Int64(y), Array.init) {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        for byte in withUnsafeBytes(of: Int64(z), Array.init) {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }

    // MARK: - Coherent Noise

    /// Multi-octave coherent noise on X/Y plane producing smooth values in [0, 1].
    /// Adjacent coordinates produce similar values (perceptual coherence).
    private func coherentValue(x: Int, y: Int, octaves: Int = 4) -> Float {
        var value: Float = 0
        var amplitude: Float = 1.0
        var totalAmplitude: Float = 0

        for octave in 0..<octaves {
            let freq = 1 << octave  // 1, 2, 4, 8
            let sx = x / max(freq, 1)
            let sy = y / max(freq, 1)

            // Smooth hash-based noise at grid points
            let n00 = gridNoise(sx, sy, octave)
            let n10 = gridNoise(sx + 1, sy, octave)
            let n01 = gridNoise(sx, sy + 1, octave)
            let n11 = gridNoise(sx + 1, sy + 1, octave)

            // Bilinear interpolation
            let fx = Float(abs(x) % max(freq, 1)) / Float(max(freq, 1))
            let fy = Float(abs(y) % max(freq, 1)) / Float(max(freq, 1))

            let nx0 = n00 * (1 - fx) + n10 * fx
            let nx1 = n01 * (1 - fx) + n11 * fx
            let n = nx0 * (1 - fy) + nx1 * fy

            value += n * amplitude
            totalAmplitude += amplitude
            amplitude *= 0.5
        }

        return max(0, min(1, value / totalAmplitude))
    }

    /// Deterministic noise at a grid point, returns [0, 1].
    private func gridNoise(_ x: Int, _ y: Int, _ octave: Int) -> Float {
        let hash = positionHash(x: x, y: y, z: octave &+ 999)
        return Float(hash % 10000) / 10000.0
    }
}
