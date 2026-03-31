/**
 * VoxtralRealtimeModelLoading - Weight loading and sanitization
 *
 * Supports two formats:
 * - Format A: consolidated.safetensors (Mistral original) with mm_streams_embeddings prefix
 * - Format B: mlx-community quantized with encoder/decoder pre-sanitized names
 *
 * Reference: mlx-audio Model.sanitize() in voxtral_realtime.py
 */

import Foundation
import MLX
import MLXNN

// MARK: - Model Loading

public func loadVoxtralRealtimeModel(
    from modelDirectory: URL,
    progressCallback: ((Float, String) -> Void)? = nil
) throws -> VoxtralRealtimeModel {

    // Step 1: Load configuration
    progressCallback?(0.1, "Loading configuration...")
    let config = try loadRealtimeConfig(from: modelDirectory)

    // Step 2: Create model
    progressCallback?(0.2, "Creating model structure...")
    let model = VoxtralRealtimeModel(config: config)

    // Step 3: Apply quantization if config specifies it
    if let quantConfig = config.quantization {
        progressCallback?(0.25, "Applying quantization (bits=\(quantConfig.bits))...")
        // Skip quantization on norms, embeddings, conv layers, and adapter projections
        MLXNN.quantize(
            model: model,
            groupSize: quantConfig.groupSize,
            bits: quantConfig.bits
        ) { path, module in
            let skipPatterns = ["norm", "ada_rms_norm", "tok_embeddings", "conv_layers", "audio_language_projection"]
            return module is Linear && !skipPatterns.contains(where: { path.contains($0) })
        }
    }

    // Step 4: Load weights
    progressCallback?(0.3, "Loading weights...")
    let rawWeights = try loadAllRealtimeWeights(from: modelDirectory)
    progressCallback?(0.6, "Mapping weight names...")

    // Step 5: Sanitize
    let sanitizedWeights = sanitizeRealtimeWeights(rawWeights)
    progressCallback?(0.7, "Applying weights to model...")

    // Step 6: Apply
    let parameters = ModuleParameters.unflattened(sanitizedWeights)
    try model.update(parameters: parameters, verify: .none)

    progressCallback?(1.0, "Model loaded successfully")
    return model
}

// MARK: - Config Loading

private func loadRealtimeConfig(from directory: URL) throws -> VoxtralRealtimeConfiguration {
    // Try config.json first (mlx-community format)
    let configURL = directory.appendingPathComponent("config.json")
    if FileManager.default.fileExists(atPath: configURL.path) {
        return try VoxtralRealtimeConfiguration.load(from: configURL)
    }

    // Try params.json (Mistral format)
    let paramsURL = directory.appendingPathComponent("params.json")
    if FileManager.default.fileExists(atPath: paramsURL.path) {
        return try VoxtralRealtimeConfiguration.load(from: paramsURL)
    }

    throw VoxtralRealtimeError.fileNotFound("Neither config.json nor params.json found in \(directory.path)")
}

// MARK: - Weight Loading

private func loadAllRealtimeWeights(from directory: URL) throws -> [String: MLXArray] {
    // Check for sharded model
    let indexFile = directory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexFile.path) {
        let indexData = try Data(contentsOf: indexFile)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: indexData)
        let shardFiles = Set(index.weightMap.values)

        var weights: [String: MLXArray] = [:]
        for shard in shardFiles {
            let shardURL = directory.appendingPathComponent(shard)
            let shardWeights = try MLX.loadArrays(url: shardURL)
            for (k, v) in shardWeights { weights[k] = v }
        }
        return weights
    }

    // Single model.safetensors
    let modelFile = directory.appendingPathComponent("model.safetensors")
    if FileManager.default.fileExists(atPath: modelFile.path) {
        return try MLX.loadArrays(url: modelFile)
    }

    // consolidated.safetensors (Mistral format)
    let consolidated = directory.appendingPathComponent("consolidated.safetensors")
    if FileManager.default.fileExists(atPath: consolidated.path) {
        return try MLX.loadArrays(url: consolidated)
    }

    throw VoxtralRealtimeError.fileNotFound("No safetensors files found in \(directory.path)")
}

private struct SafetensorsIndex: Codable {
    let weightMap: [String: String]
    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

// MARK: - Weight Sanitization

/// Auto-detect format and sanitize weight names.
func sanitizeRealtimeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let isMistralFormat = weights.keys.contains(where: { $0.hasPrefix("mm_streams_embeddings.") })

    var sanitized: [String: MLXArray] = [:]

    let encPrefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
    let adapterPrefix = "mm_streams_embeddings.embedding_module"
    let tokEmbKey = "mm_streams_embeddings.embedding_module.tok_embeddings.weight"

    for (key, var value) in weights {
        if key.contains("rotary_emb") || key.contains("position_ids") { continue }

        var newKey: String?

        if isMistralFormat {
            // Format A: Mistral consolidated.safetensors
            if key == tokEmbKey {
                newKey = "decoder.tok_embeddings.weight"

            } else if key == "norm.weight" {
                newKey = "decoder.norm.weight"

            } else if key.hasPrefix("\(encPrefix).conv_layers.") {
                // e.g., ...conv_layers.0.conv.weight → encoder.conv_layers_0_conv.conv.weight
                let rest = String(key.dropFirst("\(encPrefix).conv_layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 2)
                let layerIdx = parts[0]
                let param = parts[2]  // "weight" or "bias"
                newKey = "encoder.conv_layers_\(layerIdx)_conv.conv.\(param)"

                // Transpose conv weights from PyTorch [out, in, K] to MLX [out, K, in]
                if param == "weight" && value.ndim == 3 {
                    value = value.transposed(0, 2, 1)
                }

            } else if key.hasPrefix("\(encPrefix).transformer.layers.") {
                let rest = String(key.dropFirst("\(encPrefix).transformer.layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 1)
                let layerIdx = parts[0]
                var paramPath = String(parts[1])

                // Map FFN names
                paramPath = paramPath.replacingOccurrences(of: "feed_forward.w1.", with: "feed_forward_w1.")
                paramPath = paramPath.replacingOccurrences(of: "feed_forward.w2.", with: "feed_forward_w2.")
                paramPath = paramPath.replacingOccurrences(of: "feed_forward.w3.", with: "feed_forward_w3.")

                newKey = "encoder.transformer_layers.\(layerIdx).\(paramPath)"

            } else if key.hasPrefix("\(encPrefix).transformer.norm.") {
                let rest = String(key.dropFirst("\(encPrefix).transformer.norm.".count))
                newKey = "encoder.transformer_norm.\(rest)"

            } else if key.hasPrefix("\(adapterPrefix).audio_language_projection.") {
                let rest = String(key.dropFirst("\(adapterPrefix).audio_language_projection.".count))
                let parts = rest.split(separator: ".", maxSplits: 1)
                let idx = parts[0]
                let param = parts[1]
                newKey = "encoder.audio_language_projection_\(idx).\(param)"

            } else if key.hasPrefix("layers.") {
                let rest = String(key.dropFirst("layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 1)
                let layerIdx = parts[0]
                var paramPath = String(parts[1])

                // Map FFN names
                paramPath = paramPath.replacingOccurrences(of: "feed_forward.w1.", with: "feed_forward_w1.")
                paramPath = paramPath.replacingOccurrences(of: "feed_forward.w2.", with: "feed_forward_w2.")
                paramPath = paramPath.replacingOccurrences(of: "feed_forward.w3.", with: "feed_forward_w3.")
                // Map Ada RMS-Norm: .0. → .ada_down., .2. → .ada_up.
                paramPath = paramPath.replacingOccurrences(of: "ada_rms_norm_t_cond.0.", with: "ada_rms_norm_t_cond.ada_down.")
                paramPath = paramPath.replacingOccurrences(of: "ada_rms_norm_t_cond.2.", with: "ada_rms_norm_t_cond.ada_up.")

                newKey = "decoder.layers.\(layerIdx).\(paramPath)"
            }

        } else {
            // Format B: mlx-community pre-sanitized
            // Map decoder attention keys: wq→q_proj, wk→k_proj, wv→v_proj, wo→o_proj
            // (LlamaAttention uses HuggingFace naming, weights use Mistral naming)
            var mapped = key
            if mapped.contains("decoder.layers.") && mapped.contains(".attention.w") {
                mapped = mapped.replacingOccurrences(of: ".attention.wq.", with: ".attention.q_proj.")
                mapped = mapped.replacingOccurrences(of: ".attention.wk.", with: ".attention.k_proj.")
                mapped = mapped.replacingOccurrences(of: ".attention.wv.", with: ".attention.v_proj.")
                mapped = mapped.replacingOccurrences(of: ".attention.wo.", with: ".attention.o_proj.")
            }
            // Map decoder FFN keys: feed_forward_w1→mlp.gate_proj, etc.
            // Actually LlamaAttention's layer uses @ModuleInfo keys — but our RealtimeDecoderLayer
            // has feed_forward_w1/w2/w3 directly, so those already match.
            newKey = mapped
        }

        if let newKey {
            sanitized[newKey] = value
        } else {
            sanitized[key] = value
        }
    }

    return sanitized
}

// MARK: - Errors

public enum VoxtralRealtimeError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidConfiguration(String)
    case modelLoadingFailed(String)
    case transcriptionError(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let msg): return "File not found: \(msg)"
        case .invalidConfiguration(let msg): return "Invalid configuration: \(msg)"
        case .modelLoadingFailed(let msg): return "Model loading failed: \(msg)"
        case .transcriptionError(let msg): return "Transcription error: \(msg)"
        }
    }
}
