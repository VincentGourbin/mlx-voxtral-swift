/**
 * PackQuantization - the quantization block of a pack's config.json, decoded like MLXLMCommon (K-8).
 *
 * `BaseConfiguration` reads the global `group_size`/`bits`/`mode`, the per-layer entries (`false` = skip, object =
 * own parameters, `true` = defaults) and skips the metadata keys (`quant_method`, …). The three loaders (STT,
 * Realtime, TTS) use it, so the mode reaches `quantize`. Non-affine modes (mxfp4, mxfp8, nvfp4) load as experimental,
 * with a warning and no profile (ASK-21 = B). The NVFP4 global scale is absent from mlx-swift 0.31.6: a pack that
 * has one is refused.
 */

import Foundation
import MLX
import MLXLMCommon

struct PackQuantization: Sendable {
    let perLayer: BaseConfiguration.PerLayerQuantization

    /// Global parameters (the layers the config does not name)
    var defaults: BaseConfiguration.Quantization? { perLayer.quantization }

    /// The modes the pack uses, defaults and per-layer entries
    var modes: Set<QuantizationMode> {
        var modes = Set(defaults.map { [$0.mode] } ?? [])
        for case .quantize(let q) in perLayer.perLayerQuantization.values { modes.insert(q.mode) }
        return modes
    }

    /// Parameters of a layer named as in the config (`nil` = not quantized)
    func quantization(layer: String) -> BaseConfiguration.Quantization? {
        perLayer.quantization(layer: layer)
    }

    /// Explicit entry for a layer, matched exactly or by suffix in either direction (weight names and config names
    /// may differ by a prefix); `nil` when the config does not name it
    func explicitOption(layer: String) -> BaseConfiguration.QuantizationOption? {
        if let option = perLayer.perLayerQuantization[layer] { return option }
        return perLayer.perLayerQuantization.first { layer.hasSuffix($0.key) || $0.key.hasSuffix(layer) }?.value
    }

    /// The first block found among `keys` (`quantization`, then `quantization_config` for the TTS); nil if none.
    /// A block that cannot be decoded (unknown mode, missing `bits`, …) throws `makeError(reason)`.
    static func decode(
        configData: Data, source: String, keys: [String] = ["quantization"],
        makeError: (String) -> Error
    ) throws -> PackQuantization? {
        guard let json = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any] else {
            throw makeError("\(source): config.json is not a JSON object")
        }
        guard let key = keys.first(where: { json[$0] != nil }) else { return nil }
        guard let block = json[key] as? [String: Any] else {
            throw makeError("\(source): \"\(key)\" is not an object")
        }
        do {
            let wrapped = try JSONSerialization.data(withJSONObject: ["model_type": "voxtral", "quantization": block])
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: wrapped)
            guard let perLayer = base.perLayerQuantization, perLayer.quantization != nil else {
                throw makeError("\(source): \"\(key)\" without group_size/bits")
            }
            let quantization = PackQuantization(perLayer: perLayer)
            quantization.warnIfExperimental(source: source)
            return quantization
        } catch let error as DecodingError {
            throw makeError("\(source): unreadable \"\(key)\" (\(Self.describe(error)))")
        }
    }

    /// Weights carrying an NVFP4 global scale cannot be loaded with mlx-swift 0.31.6
    static func checkSupported(weightKeys: some Sequence<String>, source: String, makeError: (String) -> Error) throws {
        if let key = weightKeys.first(where: { $0.hasSuffix("global_scale") }) {
            throw makeError("\(source): \(key): NVFP4 global scale is not supported by mlx-swift 0.31.6")
        }
    }

    private func warnIfExperimental(source: String) {
        let experimental = modes.subtracting([.affine]).map(\.rawValue).sorted()
        guard !experimental.isEmpty else { return }
        VoxtralDebug.always(
            "warning: \(source): quantization mode \(experimental.joined(separator: ", ")) is experimental "
                + "(no profile, quality not measured)")
    }

    private static func describe(_ error: DecodingError) -> String {
        switch error {
        case .dataCorrupted(let context), .keyNotFound(_, let context), .typeMismatch(_, let context),
            .valueNotFound(_, let context):
            let path = context.codingPath.map(\.stringValue).joined(separator: ".")
            return path.isEmpty ? context.debugDescription : "\(path): \(context.debugDescription)"
        @unknown default:
            return "\(error)"
        }
    }
}
