/**
 * VoxtralMLXLMLoader - Use MLXLMCommon.loadWeights instead of our custom logic
 * 
 * Replaces our entire VoxtralModelLoading.swift with MLXLMCommon framework calls
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/**
 * Load Voxtral model using MLXLMCommon.loadWeights()
 * Replaces our custom loadVoxtralModel() completely
 */
public func loadVoxtralModelWithMLXLM(
    modelPath: String,
    dtype: MLX.DType = .float16
) throws -> (VoxtralForConditionalGeneration, VoxtralConfig) {
    
    writeDebugToDump("🚀 MLXLM LOADER: Using MLXLMCommon.loadWeights() instead of custom logic\n")
    
    // Step 1: Load configuration
    let modelURL = URL(fileURLWithPath: modelPath)
    let configDict = try loadConfigHelper(modelPath: modelURL)
    let pythonConfig = try PythonVoxtralConfig.fromDictionary(configDict)
    let config = VoxtralConfig(from: pythonConfig)
    
    writeDebugToDump("✅ MLXLM: Loaded config\n")
    
    // Step 2: Initialize empty model
    let model = VoxtralForConditionalGeneration(config: config)
    
    // Step 3: Use MLXLMCommon.loadWeights() - THE OFFICIAL WAY
    writeDebugToDump("🎯 MLXLM: Calling MLXLMCommon.loadWeights() - official framework\n")
    
    try MLXLMCommon.loadWeights(
        modelDirectory: modelURL,
        model: model,
        quantization: nil,  // No quantization for now
        perLayerQuantization: nil
    )
    
    writeDebugToDump("✅ MLXLM: Successfully loaded weights with official MLXLMCommon framework\n")
    
    return (model, config)
}

/**
 * Temporary: Keep our config loading until we migrate it
 */
private func loadConfigHelper(modelPath: URL) throws -> [String: Any] {
    let configURL = modelPath.appendingPathComponent("config.json")
    let data = try Data(contentsOf: configURL)
    let json = try JSONSerialization.jsonObject(with: data, options: [])
    guard let configDict = json as? [String: Any] else {
        throw VoxtralError.invalidConfiguration("Invalid config.json format")
    }
    return configDict
}