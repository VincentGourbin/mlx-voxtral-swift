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
    /// Hub revision (commit or tag) to download; nil = `main`
    public let revision: String?
    /// Exact bytes of the weight files downloaded (Hub, 2026-09-27; docs/Weights.md, K-24)
    public let approximateBytes: Int64?

    public init(
        id: String, repoId: String, name: String, description: String,
        size: String, quantization: String, parameters: String, recommended: Bool = false,
        revision: String? = nil, approximateBytes: Int64? = nil
    ) {
        self.id = id; self.repoId = repoId; self.name = name; self.description = description
        self.size = size; self.quantization = quantization; self.parameters = parameters; self.recommended = recommended
        self.revision = revision; self.approximateBytes = approximateBytes
    }
}

public enum VoxtralRealtimeRegistry {

    public static let models: [VoxtralRealtimeModelInfo] = [
        VoxtralRealtimeModelInfo(
            id: "realtime-4b-4bit",
            repoId: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            name: "Voxtral Realtime 4B (4-bit)",
            description: "4-bit quantized, best speed/memory balance",
            size: "3.13 GB",
            quantization: "4-bit",
            parameters: "4B",
            recommended: true,
            approximateBytes: 3_133_798_126
        ),
        VoxtralRealtimeModelInfo(
            id: "realtime-4b-fp16",
            repoId: "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
            name: "Voxtral Realtime 4B (FP16)",
            description: "Full precision, highest quality",
            size: "8.87 GB",
            quantization: "float16",
            parameters: "4B",
            approximateBytes: 8_870_608_794
        ),
        VoxtralRealtimeModelInfo(
            id: "realtime-4b",
            repoId: "mistralai/Voxtral-Mini-4B-Realtime-2602",
            name: "Voxtral Realtime 4B (Original)",
            description: "Original Mistral weights — requires sanitization",
            size: "8.86 GB",
            quantization: "bfloat16",
            parameters: "4B",
            approximateBytes: 8_859_446_848
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
