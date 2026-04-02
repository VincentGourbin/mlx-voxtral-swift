/**
 * VoxtralTTSRegistry - Registry of available Voxtral TTS models from HuggingFace
 */

import Foundation

public struct VoxtralTTSModelInfo: Identifiable, Sendable {
    public let id: String
    public let repoId: String
    public let name: String
    public let description: String
    public let size: String
    public let quantization: String
    public let parameters: String
    public let recommended: Bool

    public init(
        id: String, repoId: String, name: String, description: String,
        size: String, quantization: String, parameters: String, recommended: Bool = false
    ) {
        self.id = id; self.repoId = repoId; self.name = name; self.description = description
        self.size = size; self.quantization = quantization; self.parameters = parameters; self.recommended = recommended
    }
}

public enum VoxtralTTSRegistry {

    public static let models: [VoxtralTTSModelInfo] = [
        VoxtralTTSModelInfo(
            id: "tts-4b-mlx",
            repoId: "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
            name: "Voxtral TTS 4B (MLX)",
            description: "MLX-optimized, pre-sanitized weights, recommended",
            size: "~8 GB",
            quantization: "bfloat16",
            parameters: "4B",
            recommended: true
        ),
        VoxtralTTSModelInfo(
            id: "tts-4b",
            repoId: "mistralai/Voxtral-4B-TTS-2603",
            name: "Voxtral TTS 4B (Original)",
            description: "Original Mistral weights — requires sanitization",
            size: "~8 GB",
            quantization: "bfloat16",
            parameters: "4B"
        ),
        VoxtralTTSModelInfo(
            id: "tts-4b-4bit",
            repoId: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            name: "Voxtral TTS 4B (4-bit)",
            description: "MLX 4-bit quantized, smallest footprint",
            size: "~2.5 GB",
            quantization: "4-bit",
            parameters: "4B"
        ),
        VoxtralTTSModelInfo(
            id: "tts-4b-6bit",
            repoId: "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
            name: "Voxtral TTS 4B (6-bit)",
            description: "MLX 6-bit quantized, balanced quality/size",
            size: "~3.5 GB",
            quantization: "6-bit",
            parameters: "4B"
        ),
    ]

    public static var defaultModel: VoxtralTTSModelInfo {
        models.first(where: { $0.recommended }) ?? models[0]
    }

    public static func model(withId id: String) -> VoxtralTTSModelInfo? {
        models.first(where: { $0.id == id })
    }

    public static func printAvailableModels() {
        print("Available Voxtral TTS models:")
        for model in models {
            let marker = model.recommended ? " [recommended]" : ""
            print("  \(model.id)\(marker) - \(model.name) (\(model.size))")
        }
        print("\nAvailable voices:")
        for v in VoxtralVoice.allCases {
            print("  \(v.rawValue)")
        }
    }
}
