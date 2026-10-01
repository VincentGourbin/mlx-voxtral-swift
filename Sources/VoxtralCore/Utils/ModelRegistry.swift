/**
 * ModelRegistry - Registry of available Voxtral models from HuggingFace
 *
 * Lists all compatible quantized Voxtral models with metadata.
 */

import Foundation

/// Represents a Voxtral model available for download
/// Swift 6: Sendable because all properties are immutable value types
public struct VoxtralModelInfo: Identifiable, Codable, Sendable {
    public let id: String
    public let repoId: String
    public let name: String
    public let description: String
    public let size: String
    public let quantization: String
    public let parameters: String
    public let recommended: Bool
    /// Hub revision (commit or tag) to download; nil = `main`
    public let revision: String?
    /// Exact bytes of the weight files downloaded (Hub, 2026-09-27; docs/Weights.md, K-24)
    public let approximateBytes: Int64?

    public init(
        id: String,
        repoId: String,
        name: String,
        description: String,
        size: String,
        quantization: String,
        parameters: String,
        recommended: Bool = false,
        revision: String? = nil,
        approximateBytes: Int64? = nil
    ) {
        self.id = id
        self.repoId = repoId
        self.name = name
        self.description = description
        self.size = size
        self.quantization = quantization
        self.parameters = parameters
        self.recommended = recommended
        self.revision = revision
        self.approximateBytes = approximateBytes
    }
}

/// Registry of all available Voxtral models
public enum ModelRegistry {

    /// All available models
    public static let models: [VoxtralModelInfo] = [
        // Official Mistral models (full precision - require more memory)
        VoxtralModelInfo(
            id: "mini-3b",
            repoId: "mistralai/Voxtral-Mini-3B-2507",
            name: "Voxtral Mini 3B (Official)",
            description: "Official Mistral model - full precision",
            size: "9.36 GB",
            quantization: "bfloat16",
            parameters: "3B",
            approximateBytes: 9_356_474_312
        ),
        VoxtralModelInfo(
            id: "small-24b",
            repoId: "mistralai/Voxtral-Small-24B-2507",
            name: "Voxtral Small 24B (Official)",
            description: "Official Mistral model - full precision (48.5 GB of weights)",
            size: "48.5 GB",
            quantization: "bfloat16",
            parameters: "24B",
            approximateBytes: 48_527_546_144
        ),

        // Mini 3B quantized models (recommended for most users)
        VoxtralModelInfo(
            id: "mini-3b-8bit",
            repoId: "mzbac/voxtral-mini-3b-8bit",
            name: "Voxtral Mini 3B (8-bit)",
            description: "Best quality/size balance for the mini model",
            size: "5.40 GB",
            quantization: "8-bit",
            parameters: "3B",
            recommended: true,
            approximateBytes: 5_404_054_476
        ),
        VoxtralModelInfo(
            id: "mini-3b-4bit",
            repoId: "mzbac/voxtral-mini-3b-4bit-mixed",
            name: "Voxtral Mini 3B (4-bit mixed)",
            description: "Smaller footprint, slightly lower quality",
            size: "3.20 GB",
            quantization: "4-bit mixed",
            parameters: "3B",
            approximateBytes: 3_195_753_212
        ),

        // Small 24B quantized models (higher quality, more resources needed)
        VoxtralModelInfo(
            id: "small-24b-8bit",
            repoId: "VincentGOURBIN/voxtral-small-8bit",
            name: "Voxtral Small 24B (8-bit)",
            description: "Higher quality, requires more memory (26.5 GB of weights)",
            size: "26.5 GB",
            quantization: "8-bit",
            parameters: "24B",
            approximateBytes: 26_499_134_369
        ),
        VoxtralModelInfo(
            id: "small-4bit",
            repoId: "VincentGOURBIN/voxtral-small-4bit-mixed",
            name: "Voxtral Small (4-bit mixed)",
            description: "Memory efficient large model (14.9 GB of weights)",
            size: "14.9 GB",
            quantization: "4-bit mixed",
            parameters: "24B",
            approximateBytes: 14_857_318_962
        ),
    ]

    /// Get the default/recommended model
    public static var defaultModel: VoxtralModelInfo {
        models.first(where: { $0.recommended }) ?? models[0]
    }

    /// Find a model by ID
    public static func model(withId id: String) -> VoxtralModelInfo? {
        models.first(where: { $0.id == id })
    }

    /// Find a model by repo ID
    public static func model(withRepoId repoId: String) -> VoxtralModelInfo? {
        models.first(where: { $0.repoId == repoId })
    }

    /// Get official Mistral models
    public static var officialModels: [VoxtralModelInfo] {
        models.filter { $0.repoId.hasPrefix("mistralai/") }
    }

    /// Get all mini models (3B) - quantized only
    public static var miniModels: [VoxtralModelInfo] {
        models.filter { $0.parameters == "3B" && !$0.repoId.hasPrefix("mistralai/") }
    }

    /// Get all small/large models (24B) - quantized only
    public static var smallModels: [VoxtralModelInfo] {
        models.filter { $0.parameters == "24B" && !$0.repoId.hasPrefix("mistralai/") }
    }

    /// Print formatted list of available models
    public static func printAvailableModels() {
        print("\n" + String(repeating: "=", count: 70))
        print("AVAILABLE VOXTRAL MODELS")
        print(String(repeating: "=", count: 70))

        print("\n--- Official Mistral Models (full precision) ---")
        for model in officialModels {
            print("  \(model.id): \(model.name)")
            print("    Repo: \(model.repoId)")
            print("    Size: \(model.size) | Precision: \(model.quantization)")
            print("    \(model.description)")
            print()
        }

        print("--- Mini Models (3B parameters, quantized) ---")
        for model in miniModels {
            let recommended = model.recommended ? " [RECOMMENDED]" : ""
            print("  \(model.id): \(model.name)\(recommended)")
            print("    Repo: \(model.repoId)")
            print("    Size: \(model.size) | Quantization: \(model.quantization)")
            print("    \(model.description)")
            print()
        }

        print("--- Small Models (24B parameters, quantized) ---")
        for model in smallModels {
            print("  \(model.id): \(model.name)")
            print("    Repo: \(model.repoId)")
            print("    Size: \(model.size) | Quantization: \(model.quantization)")
            print("    \(model.description)")
            print()
        }

        print(String(repeating: "=", count: 70))
    }
}
