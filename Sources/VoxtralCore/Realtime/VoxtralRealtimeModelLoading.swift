/**
 * VoxtralRealtimeModelLoading - Weight loading and sanitization
 *
 * Supports three formats, detected from the weight names (K-91):
 * - Format A: consolidated.safetensors (Mistral original) with mm_streams_embeddings prefix
 * - Format B: mlx-community quantized with encoder/decoder pre-sanitized names (mlx-audio)
 * - Format C: voxmlx (`adapter.*`, `encoder.layers.*`, `language_model.*`), params.json-style config.json with
 *   `quantization`; the quantized modules are the ones whose `.scales` the pack holds (tied head and Ada MLP included)
 * A pack whose keys do not cover the model, or hold keys it does not have, throws `VoxtralRealtimePackError`.
 *
 * Reference: mlx-audio Model.sanitize() in voxtral_realtime.py
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Model Loading

func loadVoxtralRealtimeModel(
    from modelDirectory: URL,
    progressCallback: ((Float, String) -> Void)? = nil
) throws -> VoxtralRealtimeModel {

    // Step 1: Load configuration
    progressCallback?(0.1, "Loading configuration...")
    let config = try loadRealtimeConfig(from: modelDirectory)

    // Step 2: Create model
    progressCallback?(0.2, "Creating model structure...")
    let model = VoxtralRealtimeModel(config: config)

    // Step 3: Load weights and detect their format (K-91)
    progressCallback?(0.25, "Loading weights...")
    let rawWeights = try loadAllRealtimeWeights(from: modelDirectory)
    guard let format = RealtimePackFormat.detect(rawWeights.keys) else {
        throw VoxtralRealtimePackError(
            format: "unknown", source: modelDirectory.path, missing: [], unexpected: Array(rawWeights.keys.sorted().prefix(20)))
    }
    let quantization = try loadRealtimeQuantization(from: modelDirectory, config: config, format: format)
    if quantization != nil {
        try PackQuantization.checkSupported(
            weightKeys: rawWeights.keys, source: modelDirectory.path, makeError: VoxtralRealtimeError.invalidConfiguration)
    }

    // Step 4: Sanitize
    progressCallback?(0.6, "Mapping weight names...")
    let sanitizedWeights = format == .voxmlx ? sanitizeVoxmlxWeights(rawWeights) : sanitizeRealtimeWeights(rawWeights)

    // Step 5: Apply quantization if config specifies it: per layer and mode, decoded like MLXLMCommon (K-8)
    if let quantization, let defaults = quantization.defaults {
        progressCallback?(0.65, "Applying quantization (bits=\(defaults.bits), \(defaults.mode.rawValue))...")
        if format == .voxmlx {
            // voxmlx quantizes every module it can, tied head and Ada MLP included: follow the pack's scales
            MLXNN.quantize(model: model) { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                guard module is Linear || module is Embedding, sanitizedWeights["\(path).scales"] != nil else { return nil }
                if case .quantize(let layer) = quantization.explicitOption(layer: path) { return layer.asTuple }
                return defaults.asTuple
            }
        } else {
            // Skip quantization on norms, embeddings, conv layers, and adapter projections
            let skipPatterns = ["norm", "ada_rms_norm", "tok_embeddings", "conv_layers", "audio_language_projection"]
            MLXNN.quantize(model: model) { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                guard module is Linear, !skipPatterns.contains(where: { path.contains($0) }) else { return nil }
                switch quantization.explicitOption(layer: path) {
                case .skip: return nil
                case .quantize(let layer): return layer.asTuple
                case nil: return defaults.asTuple
                }
            }
        }
    }
    progressCallback?(0.7, "Applying weights to model...")
    try checkRealtimeWeightCoverage(model, sanitizedWeights, format: format, source: modelDirectory.path)

    // Step 6: Apply
    let parameters = ModuleParameters.unflattened(sanitizedWeights)
    try model.updateVerified(parameters: parameters)
    castRealtimeParameters(model, to: realtimeComputeDType(model))
    if let bits = realtimeHeadBits() {
        quantizeRealtimeHead(model.decoder, bits: bits)
    }

    progressCallback?(1.0, "Model loaded successfully")
    return model
}

// MARK: - Compute dtype (K-38)

/// The dtype the Realtime path computes in: the tied token table's. The mlx-audio and Mistral packs never quantize it
/// (it is also the LM head, `tok_embeddings` is skipped by the quantizer); a voxmlx pack does (K-91) and K-46 may after
/// load: then its scales carry the dtype. It holds the 16-bit dtype of every pack's matrices. Not `norm` or
/// `ada_down`: the mlx-community packs store norms, biases and the Ada MLP in fp32 (300 tensors).
func realtimeComputeDType(_ model: VoxtralRealtimeModel) -> DType { realtimeComputeDType(model.decoder) }

func realtimeComputeDType(_ decoder: VoxtralRealtimeDecoder) -> DType {
    // A quantized table (K-46) holds packed uint32 weights: its scales keep the 16-bit dtype
    if let quantized = decoder.tokEmbeddings as? QuantizedEmbedding { return quantized.scales.dtype }
    return decoder.tokEmbeddings.weight.dtype
}

// MARK: - Quantized tied head (K-46)

/// Bits of the tied head quantized at load time: `VOXTRAL_REALTIME_HEAD_BITS` = 8 or 4; unset (default) keeps the
/// pack's 16-bit table. Internal switch for the measurement (P-61 step 2), not a public option.
func realtimeHeadBits(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> Int? {
    guard let value = environment["VOXTRAL_REALTIME_HEAD_BITS"], let bits = Int(value), [4, 8].contains(bits) else {
        return nil
    }
    return bits
}

/// Replace the tied token table by a `QuantizedEmbedding` (affine, group size 64, as mlx-audio): the logits go
/// through `quantizedMM` and a token embedding de-quantizes one row; the 16-bit table is released. Run after
/// `castRealtimeParameters`, which reads the compute dtype on this table.
func quantizeRealtimeHead(_ decoder: VoxtralRealtimeDecoder, bits: Int, groupSize: Int = 64) {
    let table = decoder.tokEmbeddings
    guard !(table is QuantizedEmbedding) else { return }
    let quantized = QuantizedEmbedding(table, groupSize: groupSize, bits: bits)
    decoder.update(modules: ModuleChildren.unflattened(["tok_embeddings": quantized]))
    MLX.eval(quantized)
}

/// Cast the fp32 parameters (norms, biases, Ada MLP) to `dtype`, a 16-bit float: each one would otherwise promote the
/// activations, every Linear (a fp32 copy of its weights per call), the KV cache and the logits to fp32
/// (`rms_norm` and `matmul` return `result_type`, `fast.cpp:82`). A fp32 model is left as it is.
func castRealtimeParameters(_ model: Module, to dtype: DType) {
    guard dtype == .float16 || dtype == .bfloat16 else { return }
    let cast = model.parameters().flattened().compactMap { key, value in
        value.dtype == .float32 ? (key, value.asType(dtype)) : nil
    }
    guard !cast.isEmpty else { return }
    model.update(parameters: ModuleParameters.unflattened(cast))
    MLX.eval(model.parameters())
}

// MARK: - Config Loading

/// config.json when it has the mlx-community shape; otherwise params.json (Mistral format). The original Mistral
/// repository also ships a transformers config.json, which this loader cannot read: it is skipped (K-9)
func loadRealtimeConfig(from directory: URL) throws -> VoxtralRealtimeConfiguration {
    let configURL = directory.appendingPathComponent("config.json")
    let paramsURL = directory.appendingPathComponent("params.json")
    if FileManager.default.fileExists(atPath: configURL.path) {
        // An mlx config.json (or an unreadable one) is read or reported, never replaced by params.json (K-87);
        // another schema (transformers in the original Mistral repo, K-9) gives way to params.json when there is one
        if try VoxtralRealtimeConfiguration.isMLXSchema(at: configURL) || !FileManager.default.fileExists(atPath: paramsURL.path) {
            return try VoxtralRealtimeConfiguration.load(from: configURL)
        }
    }
    if FileManager.default.fileExists(atPath: paramsURL.path) {
        return try VoxtralRealtimeConfiguration.load(from: paramsURL)
    }

    throw VoxtralRealtimeError.fileNotFound("Neither config.json nor params.json found in \(directory.path)")
}

/// The quantization of an mlx-community config.json, or of a voxmlx one (params.json keys plus `quantization`, K-91);
/// the Mistral params.json has none; nil when unquantized
func loadRealtimeQuantization(
    from directory: URL, config: VoxtralRealtimeConfiguration, format: RealtimePackFormat
) throws -> PackQuantization? {
    guard config.quantization != nil || format == .voxmlx else { return nil }
    let configURL = directory.appendingPathComponent("config.json")
    return try PackQuantization.decode(
        configData: Data(contentsOf: configURL), source: configURL.path,
        makeError: VoxtralRealtimeError.invalidConfiguration)
}

// MARK: - Weight Loading

func loadAllRealtimeWeights(from directory: URL) throws -> [String: MLXArray] {
    // The original Mistral checkpoint first: its repository also has a transformers model.safetensors (K-9)
    let mistralConsolidated = directory.appendingPathComponent("consolidated.safetensors")
    if FileManager.default.fileExists(atPath: mistralConsolidated.path) {
        return try MLX.loadArrays(url: mistralConsolidated)
    }
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

// MARK: - Pack format (K-91)

/// Weight layout of a Realtime pack, read from its key names
enum RealtimePackFormat: String, Sendable {
    /// Mistral `consolidated.safetensors` (format A)
    case mistral
    /// mlx-community / mlx-audio, pre-sanitized names (format B)
    case mlxAudio = "mlx-audio"
    /// voxmlx (format C): `adapter.*`, `encoder.layers.*`, `language_model.*`
    case voxmlx

    static func detect(_ keys: some Sequence<String>) -> RealtimePackFormat? {
        var format: RealtimePackFormat?
        for key in keys {
            if key.hasPrefix("mm_streams_embeddings.") { return .mistral }
            if key.hasPrefix("language_model.") || key.hasPrefix("adapter.") { format = .voxmlx }
            if format == nil, key.hasPrefix("decoder.") || key.hasPrefix("encoder.transformer_layers.") { format = .mlxAudio }
        }
        return format
    }
}

/// voxmlx names (format C) → the model's: encoder attention `{q,k,v,o}_proj` → `w{q,k,v,o}`, `mlp.{gate,down,up}_proj`
/// → `feed_forward_w{1,2,3}`, `attn_norm` → `attention_norm`, `adapter.w_{in,out}` → `audio_language_projection_{0,2}`,
/// `language_model.*` → `decoder.*` (`embed_tokens` → `tok_embeddings`, `ada_norm.linear_{in,out}` →
/// `ada_rms_norm_t_cond.ada_{down,up}`). Conv weights are already in MLX layout ([out, K, in]). Unknown keys are kept
/// as they are, so the coverage check names them.
func sanitizeVoxmlxWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let common: [(String, String)] = [
        (".attn_norm.", ".attention_norm."),
        (".mlp.gate_proj.", ".feed_forward_w1."), (".mlp.down_proj.", ".feed_forward_w2."),
        (".mlp.up_proj.", ".feed_forward_w3."),
    ]
    let encoderAttention: [(String, String)] = [
        (".attention.q_proj.", ".attention.wq."), (".attention.k_proj.", ".attention.wk."),
        (".attention.v_proj.", ".attention.wv."), (".attention.o_proj.", ".attention.wo."),
    ]
    let decoderAda: [(String, String)] = [
        (".ada_norm.linear_in.", ".ada_rms_norm_t_cond.ada_down."), (".ada_norm.linear_out.", ".ada_rms_norm_t_cond.ada_up."),
    ]
    func replacing(_ key: String, _ pairs: [(String, String)]) -> String {
        pairs.reduce(key) { $0.replacingOccurrences(of: $1.0, with: $1.1) }
    }
    func replacePrefix(_ key: String, _ prefix: String, _ with: String) -> String? {
        key.hasPrefix(prefix) ? with + key.dropFirst(prefix.count) : nil
    }

    var sanitized: [String: MLXArray] = [:]
    for (key, value) in weights {
        var newKey = key
        if let k = replacePrefix(key, "encoder.layers.", "encoder.transformer_layers.") {
            newKey = replacing(replacing(k, common), encoderAttention)
        } else if let k = replacePrefix(key, "encoder.conv1.", "encoder.conv_layers_0_conv.conv.")
            ?? replacePrefix(key, "encoder.conv2.", "encoder.conv_layers_1_conv.conv.")
            ?? replacePrefix(key, "encoder.norm.", "encoder.transformer_norm.")
            ?? replacePrefix(key, "adapter.w_in.", "encoder.audio_language_projection_0.")
            ?? replacePrefix(key, "adapter.w_out.", "encoder.audio_language_projection_2.")
            ?? replacePrefix(key, "language_model.embed_tokens.", "decoder.tok_embeddings.")
            ?? replacePrefix(key, "language_model.norm.", "decoder.norm.") {
            newKey = k
        } else if let k = replacePrefix(key, "language_model.layers.", "decoder.layers.") {
            newKey = replacing(replacing(k, common), decoderAda)
        }
        sanitized[newKey] = value
    }
    return sanitized
}

/// Every model parameter covered and no key the model lacks, else `VoxtralRealtimePackError` (format, first keys
/// apart); then the weights are applied with shapes checked (K-91; before it, uncovered parameters threw an untyped
/// error and extra keys were ignored)
func checkRealtimeWeightCoverage(
    _ model: Module, _ weights: [String: MLXArray], format: RealtimePackFormat, source: String
) throws {
    let expected = Set(model.parameters().flattened().map { $0.0 })
    let provided = Set(weights.keys)
    let missing = expected.subtracting(provided).sorted()
    let unexpected = provided.subtracting(expected).sorted()
    guard missing.isEmpty, unexpected.isEmpty else {
        throw VoxtralRealtimePackError(format: format.rawValue, source: source, missing: missing, unexpected: unexpected)
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
                // Map decoder attention names like Format B (Mistral wq/wk/wv/wo → q/k/v/o_proj);
                // without it the attention stayed randomly initialized (found by verified loading, K-7)
                paramPath = paramPath.replacingOccurrences(of: "attention.wq.", with: "attention.q_proj.")
                paramPath = paramPath.replacingOccurrences(of: "attention.wk.", with: "attention.k_proj.")
                paramPath = paramPath.replacingOccurrences(of: "attention.wv.", with: "attention.v_proj.")
                paramPath = paramPath.replacingOccurrences(of: "attention.wo.", with: "attention.o_proj.")
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

/// A Realtime pack the loader cannot read whole: unknown layout, model parameters it does not cover, or keys the model
/// does not have. Thrown instead of loading part of the weights (K-91).
public struct VoxtralRealtimePackError: Error, LocalizedError, Sendable, Equatable {
    /// Detected layout: `mistral`, `mlx-audio`, `voxmlx`, or `unknown`
    public let format: String
    /// The pack directory
    public let source: String
    /// Model parameters the pack does not cover (sorted)
    public let missing: [String]
    /// Pack keys, after name mapping, that the model does not have (sorted)
    public let unexpected: [String]

    public init(format: String, source: String, missing: [String], unexpected: [String]) {
        self.format = format; self.source = source; self.missing = missing; self.unexpected = unexpected
    }

    public var errorDescription: String? {
        func first(_ keys: [String]) -> String {
            keys.prefix(5).joined(separator: ", ") + (keys.count > 5 ? ", … (\(keys.count))" : "")
        }
        var parts = ["Realtime pack not loadable (format: \(format)) at \(source)"]
        if !missing.isEmpty { parts.append("missing \(missing.count): \(first(missing))") }
        if !unexpected.isEmpty { parts.append("unexpected \(unexpected.count): \(first(unexpected))") }
        return parts.joined(separator: "; ")
    }
}

public enum VoxtralRealtimeError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidConfiguration(String)
    case modelLoadingFailed(String)
    case transcriptionError(String)
    /// Another operation (transcription, loading) holds the pipeline (K-11)
    case busy(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let msg): return "File not found: \(msg)"
        case .invalidConfiguration(let msg): return "Invalid configuration: \(msg)"
        case .modelLoadingFailed(let msg): return "Model loading failed: \(msg)"
        case .transcriptionError(let msg): return "Transcription error: \(msg)"
        case .busy(let msg): return "Pipeline busy: \(msg)"
        }
    }
}
