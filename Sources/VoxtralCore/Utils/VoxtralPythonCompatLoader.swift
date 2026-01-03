/**
 * VoxtralPythonCompatLoader - Python-compatible wrapper for our validated standard loader
 *
 * Uses our existing loadVoxtralStandardModel (which already does the Python sequence)
 * and wraps it to return VoxtralForConditionalGeneration like Python does
 */

import Foundation
import MLX
import MLXNN

/**
 * Python-compatible load_voxtral_model equivalent
 * Uses our validated loadVoxtralStandardModel internally
 */
public func loadVoxtralModel(
    modelPath: String,
    dtype: DType = .float16
) throws -> (VoxtralForConditionalGeneration, VoxtralStandardConfiguration) {

    // Use our validated standard loader (does the Python sequence 1-4)
    let (standardModel, config) = try loadVoxtralStandardModel(modelPath: modelPath, dtype: dtype)

    // Wrap in VoxtralForConditionalGeneration to get generate() methods
    let conditionalModel = VoxtralForConditionalGeneration(standardModel: standardModel)

    return (conditionalModel, config)
}