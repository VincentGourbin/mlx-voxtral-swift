/**
 * VoxtralRealtimeRegistry - Registry of available Voxtral Realtime models
 */

import Foundation

public struct VoxtralRealtimeModelInfo: Identifiable, Sendable {
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

public enum VoxtralRealtimeRegistry {

    public static let models: [VoxtralRealtimeModelInfo] = [
        VoxtralRealtimeModelInfo(
            id: "realtime-4b-4bit",
            repoId: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            name: "Voxtral Realtime 4B (4-bit)",
            description: "4-bit quantized, best speed/memory balance",
            size: "~3 GB",
            quantization: "4-bit",
            parameters: "4B",
            recommended: true
        ),
        VoxtralRealtimeModelInfo(
            id: "realtime-4b-fp16",
            repoId: "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
            name: "Voxtral Realtime 4B (FP16)",
            description: "Full precision, highest quality",
            size: "~8 GB",
            quantization: "float16",
            parameters: "4B"
        ),
        VoxtralRealtimeModelInfo(
            id: "realtime-4b",
            repoId: "mistralai/Voxtral-Mini-4B-Realtime-2602",
            name: "Voxtral Realtime 4B (Original)",
            description: "Original Mistral weights — requires sanitization",
            size: "~8 GB",
            quantization: "bfloat16",
            parameters: "4B"
        ),
    ]

    public static var defaultModel: VoxtralRealtimeModelInfo {
        models.first(where: { $0.recommended }) ?? models[0]
    }

    public static func model(withId id: String) -> VoxtralRealtimeModelInfo? {
        models.first(where: { $0.id == id })
    }

    public static func printAvailableModels() {
        print("Available Voxtral Realtime models:")
        for model in models {
            let marker = model.recommended ? " [recommended]" : ""
            print("  \(model.id)\(marker) - \(model.name) (\(model.size))")
        }
    }
}
