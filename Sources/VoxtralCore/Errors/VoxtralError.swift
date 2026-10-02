/**
 * VoxtralError - errors of the STT loaders and pipeline. Moved out of the legacy `VoxtralModelLoading.swift` (K-23).
 */

import Foundation

/**
 * Error types for model loading
 */
public enum VoxtralError: Error {
    case fileNotFound(String)
    case invalidConfiguration(String)
    case loadingFailed(String)
    // From VoxtralGenerator.swift
    case modelNotLoaded
    case processorNotLoaded
    case audioProcessingFailed(String)
    case generationFailed(String)
    case tokenizerNotAvailable
    case invalidTokenFormat
    // From VoxtralProcessor.swift
    case invalidInput(String)
    case tokenizerRequired(String)
    case languageNotSupported(String)
    // From MLXLMBridge.swift
    case configurationNotFound
    /// A public entry point that cannot do what it is asked (e.g. the legacy `downloadModel(modelId:)`)
    case unsupported(String)
    /// An MLX error (shape, dtype, mask…) caught at a public entry point instead of terminating the host
    case mlx(String)
    /// Model parameters absent from the weight files (they would stay randomly initialized)
    case missingWeights([String])
    /// `tekken.json` unreadable or unusable (the loader no longer falls back to a demo tokenizer)
    case invalidTokenizer(String)
    /// Prompt + `maxTokens` exceed an explicit KV cache limit (`maxKVCacheSize` / `contextSize`)
    case contextTooLong(prompt: Int, maxTokens: Int, limit: Int)
}
