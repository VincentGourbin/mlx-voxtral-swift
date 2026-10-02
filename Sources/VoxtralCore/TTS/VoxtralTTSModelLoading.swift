/**
 * VoxtralTTSModelLoading - Weight loading for Voxtral TTS model
 *
 * Supports two weight formats:
 * - Format A: Original Mistral (consolidated.safetensors) — needs full sanitization
 * - Format B: mlx-community (model-00001/00002-of-00002.safetensors) — pre-sanitized
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - TTS Model Loading

func loadVoxtralTTSModel(
    from modelDirectory: URL,
    progressCallback: ((Float, String) -> Void)? = nil
) throws -> VoxtralTTSModel {

    // Step 1: Load configuration
    progressCallback?(0.1, "Loading configuration...")
    let paramsURL = modelDirectory.appendingPathComponent("params.json")
    guard FileManager.default.fileExists(atPath: paramsURL.path) else {
        // Try config.json (mlx-community format has both)
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw VoxtralTTSError.fileNotFound("Neither params.json nor config.json found")
        }
        // mlx-community config.json is the same format as params.json
        let config = try VoxtralTTSConfiguration.load(from: configURL)
        return try loadWithConfig(config, modelDirectory: modelDirectory, progressCallback: progressCallback)
    }
    let config = try VoxtralTTSConfiguration.load(from: paramsURL)
    return try loadWithConfig(config, modelDirectory: modelDirectory, progressCallback: progressCallback)
}

private func loadWithConfig(
    _ config: VoxtralTTSConfiguration,
    modelDirectory: URL,
    progressCallback: ((Float, String) -> Void)?
) throws -> VoxtralTTSModel {

    // Step 2: Create model
    progressCallback?(0.2, "Creating model structure...")
    let model = VoxtralTTSModel(config: config)

    // Step 3: Load weights (single or sharded)
    progressCallback?(0.3, "Loading weights...")
    let rawWeights = try loadAllTTSWeights(from: modelDirectory)
    progressCallback?(0.6, "Mapping weight names...")

    // Step 4: Detect format and sanitize
    let sanitizedWeights = sanitizeTTSWeights(rawWeights)

    // Step 4b: If quantized weights detected, quantize model layers that have scales; per layer and mode,
    // `quantization`, else `quantization_config`, decoded like MLXLMCommon (K-8)
    if let quantization = try loadQuantizationConfig(from: modelDirectory), let defaults = quantization.defaults {
        progressCallback?(0.65, "Applying \(defaults.bits)-bit quantization (\(defaults.mode.rawValue))...")
        try PackQuantization.checkSupported(
            weightKeys: sanitizedWeights.keys, source: modelDirectory.path, makeError: VoxtralTTSError.invalidConfiguration)
        // Build set of quantized layer prefixes from weight keys
        let quantizedPrefixes = detectQuantizedLayers(in: sanitizedWeights)
        quantize(model: model) { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
            // Only quantize layers that actually have scales in the weights
            guard quantizedPrefixes.contains(where: { path.hasSuffix($0) || path.contains($0) }) else { return nil }
            switch quantization.explicitOption(layer: path) {
            case .skip: return nil
            case .quantize(let layer): return layer.asTuple
            case nil: return defaults.asTuple
            }
        }
    }

    progressCallback?(0.7, "Applying weights to model...")

    // Step 5: Apply
    let parameters = ModuleParameters.unflattened(sanitizedWeights)
    try model.updateVerified(parameters: parameters)

    progressCallback?(1.0, "Model loaded successfully")
    return model
}

// MARK: - Quantization Config

/// The pack's quantization: `quantization`, else `quantization_config` (some converters write only the latter);
/// nil without config.json or without either block. An unreadable config or block throws instead of loading
/// quantized weights into an unquantized model (K-27, K-8)
func loadQuantizationConfig(from modelDirectory: URL) throws -> PackQuantization? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: configURL.path) else { return nil }
    let data: Data
    do {
        data = try Data(contentsOf: configURL)
    } catch {
        throw VoxtralTTSError.invalidConfiguration("\(configURL.path): \(error.localizedDescription)")
    }
    return try PackQuantization.decode(
        configData: data, source: configURL.path, keys: ["quantization", "quantization_config"],
        makeError: VoxtralTTSError.invalidConfiguration)
}

/// Detect which layers have quantized weights by finding keys with `.scales` suffix.
/// Returns layer path prefixes (after sanitization) that should be quantized in the model.
private func detectQuantizedLayers(in weights: [String: MLXArray]) -> Set<String> {
    var prefixes = Set<String>()
    for key in weights.keys where key.hasSuffix(".scales") {
        // "layers.0.self_attn.q_proj.scales" → "layers.0.self_attn.q_proj"
        let layerPath = String(key.dropLast(".scales".count))
        prefixes.insert(layerPath)
    }
    return prefixes
}

// MARK: - Load All Weights (single or sharded)

func loadAllTTSWeights(from directory: URL) throws -> [String: MLXArray] {
    // Sharded model with an index: every shard it names is required
    let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
    if let data = try? Data(contentsOf: indexURL),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let weightMap = json["weight_map"] as? [String: String], !weightMap.isEmpty {
        var weights: [String: MLXArray] = [:]
        for shard in Set(weightMap.values).sorted() {
            let url = directory.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw VoxtralTTSError.fileNotFound("\(shard) (listed in model.safetensors.index.json) not found in \(directory.path)")
            }
            for (k, v) in try MLX.loadArrays(url: url) { weights[k] = v }
        }
        return weights
    }

    // Check for sharded model first
    let shard1 = directory.appendingPathComponent("model-00001-of-00002.safetensors")
    let shard2 = directory.appendingPathComponent("model-00002-of-00002.safetensors")

    if FileManager.default.fileExists(atPath: shard1.path) {
        var weights = try MLX.loadArrays(url: shard1)
        if FileManager.default.fileExists(atPath: shard2.path) {
            let weights2 = try MLX.loadArrays(url: shard2)
            for (k, v) in weights2 {
                weights[k] = v
            }
        }
        return weights
    }

    // Single file: model.safetensors (quantized mlx-community) or consolidated.safetensors (original Mistral)
    let singleModel = directory.appendingPathComponent("model.safetensors")
    if FileManager.default.fileExists(atPath: singleModel.path) {
        return try MLX.loadArrays(url: singleModel)
    }

    let consolidated = directory.appendingPathComponent("consolidated.safetensors")
    guard FileManager.default.fileExists(atPath: consolidated.path) else {
        throw VoxtralTTSError.fileNotFound("No safetensors files found in \(directory.path)")
    }
    return try MLX.loadArrays(url: consolidated)
}

// MARK: - Weight Name Sanitization

/// Auto-detects format and sanitizes weight names.
///
/// Format A (consolidated.safetensors):
///   layers.N.attention.wq → layers.N.self_attn.q_proj
///   mm_audio_embeddings.tok_embeddings → mm_audio_embeddings.tok_embeddings
///
/// Format B (mlx-community):
///   language_model.model.model.layers.N.self_attn.q_proj → layers.N.self_attn.q_proj
///   language_model.model.model.embed_tokens → mm_audio_embeddings.tok_embeddings
///   language_model.model.model.norm → norm
///   audio_codebook_embeddings.* → mm_audio_embeddings.audio_codebook_embeddings.*
func sanitizeTTSWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    // Detect format
    let isMLXCommunity = weights.keys.contains(where: { $0.hasPrefix("language_model.model.model.") })

    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
        if key.contains("rotary_emb") || key.contains("position_ids") { continue }

        var newKey = key

        if isMLXCommunity {
            // Format B: mlx-community pre-sanitized weights
            if newKey.hasPrefix("language_model.model.model.") {
                // Strip the wrapper prefix
                let suffix = String(newKey.dropFirst("language_model.model.model.".count))
                if suffix.hasPrefix("embed_tokens.") {
                    // embed_tokens → mm_audio_embeddings.tok_embeddings
                    newKey = "mm_audio_embeddings.tok_embeddings.\(String(suffix.dropFirst("embed_tokens.".count)))"
                } else {
                    // layers.N.* and norm.* stay at top level
                    newKey = suffix
                }
            } else if newKey.hasPrefix("audio_codebook_embeddings.") {
                // audio_codebook_embeddings.* → mm_audio_embeddings.audio_codebook_embeddings.*
                newKey = "mm_audio_embeddings.\(newKey)"
            }
            // acoustic_transformer.* and audio_tokenizer.* are unchanged
        } else {
            // Format A: Original Mistral consolidated.safetensors
            if newKey.hasPrefix("layers.") {
                newKey = newKey.replacingOccurrences(of: ".attention.", with: ".self_attn.")
                newKey = newKey.replacingOccurrences(of: ".attention_norm.", with: ".input_layernorm.")
                newKey = newKey.replacingOccurrences(of: ".ffn_norm.", with: ".post_attention_layernorm.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward.", with: ".mlp.")
                newKey = newKey.replacingOccurrences(of: ".wq.", with: ".q_proj.")
                newKey = newKey.replacingOccurrences(of: ".wk.", with: ".k_proj.")
                newKey = newKey.replacingOccurrences(of: ".wv.", with: ".v_proj.")
                newKey = newKey.replacingOccurrences(of: ".wo.", with: ".o_proj.")
                newKey = newKey.replacingOccurrences(of: ".w1.", with: ".gate_proj.")
                newKey = newKey.replacingOccurrences(of: ".w2.", with: ".down_proj.")
                newKey = newKey.replacingOccurrences(of: ".w3.", with: ".up_proj.")
            }
            // mm_audio_embeddings.*, norm.*, acoustic_transformer.*, audio_tokenizer.* unchanged
        }

        sanitized[newKey] = value
    }

    return sanitized
}

// MARK: - Errors

public enum VoxtralTTSError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidConfiguration(String)
    case modelLoadingFailed(String)
    case synthesisError(String)
    case voiceNotFound(String)
    /// Another operation (enrollment, synthesis, loading) holds the pipeline (K-11)
    case busy(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let msg): return "File not found: \(msg)"
        case .invalidConfiguration(let msg): return "Invalid configuration: \(msg)"
        case .modelLoadingFailed(let msg): return "Model loading failed: \(msg)"
        case .synthesisError(let msg): return "Synthesis error: \(msg)"
        case .voiceNotFound(let msg): return "Voice not found: \(msg)"
        case .busy(let msg): return "Pipeline busy: \(msg)"
        }
    }
}
