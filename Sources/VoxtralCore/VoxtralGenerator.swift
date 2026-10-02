/**
 * VoxtralGenerator - Swift equivalent of mlx.voxtral/scripts/generate.py
 * 
 * A command-line interface for transcribing audio using MLX-Voxtral models.
 * Pure conversion of the Python generate.py script functionality.
 */

import Foundation
import MLX
import MLXNN

/**
 * Generation Parameters - Equivalent to Python argparse arguments
 */
public struct VoxtralGenerationParameters {
    /// Model name or path (default: "mistralai/Voxtral-Mini-3B-2507")
    public let model: String
    
    /// Maximum number of tokens to generate (default: 1024)
    public let maxTokens: Int
    
    /// Sampling temperature, 0.0 for deterministic output (default: 0.0)
    public let temperature: Float
    
    /// Path to audio file or URL
    public let audioPath: String
    
    /// Nucleus sampling parameter (default: 0.95)
    public let topP: Float
    
    /// Model dtype (default: bfloat16)
    public let dtype: DType
    
    /// Show detailed output including performance metrics
    public let verbose: Bool
    
    /// Language code for transcription (default: "en")
    public let language: String
    
    /// Enable streaming output (generates text token by token)
    public let stream: Bool
    
    public init(
        model: String = "mistralai/Voxtral-Mini-3B-2507",
        maxTokens: Int = 1024,
        temperature: Float = 0.0,
        audioPath: String,
        topP: Float = 0.95,
        dtype: DType = .bfloat16,
        verbose: Bool = false,
        language: String = "en",
        stream: Bool = false
    ) {
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.audioPath = audioPath
        self.topP = topP
        self.dtype = dtype
        self.verbose = verbose
        self.language = language
        self.stream = stream
    }
}

/**
 * VoxtralGenerator - Main class equivalent to Python generate.py main() function
 */
public class VoxtralGenerator {
    
    internal let parameters: VoxtralGenerationParameters
    internal var model: VoxtralModel?
    internal var processor: VoxtralProcessor?
    
    public init(parameters: VoxtralGenerationParameters) {
        self.parameters = parameters
    }
    
    /**
     * Main generation function - equivalent to Python main()
     */
    public func generate() throws -> String {
        if parameters.verbose {
            VoxtralDebug.log("Loading model: \(parameters.model)")
        }
        
        let loadStart = Date()
        
        // Python: model, config = load_voxtral_model(args.model, dtype=dtype)
        try loadModelFromPath()
        
        // Python: processor = VoxtralProcessor.from_pretrained(args.model)
        try loadProcessorFromPath()
        
        if parameters.verbose {
            let loadTime = Date().timeIntervalSince(loadStart)
            VoxtralDebug.log("Model loaded in \(String(format: "%.2f", loadTime)) seconds")
            VoxtralDebug.log("Model dtype: \(dtypeString(parameters.dtype))")
        }
        
        if parameters.verbose {
            VoxtralDebug.log("\nProcessing audio: \(parameters.audioPath)")
        }
        
        // Python: inputs = processor.apply_transcrition_request(audio=args.audio, language=args.language)
        let inputs = try processAudioWithExistingPipeline()
        
        if parameters.verbose {
            VoxtralDebug.log("\nGenerating transcription...")
            if parameters.stream {
                VoxtralDebug.log("(Streaming mode enabled)")
            }
        }
        
        let generateStart = Date()
        
        if parameters.stream {
            return try streamingGenerateWithExistingModel(inputs: inputs, startTime: generateStart)
        } else {
            return try batchGenerateWithExistingModel(inputs: inputs, startTime: generateStart)
        }
    }
    
    
    
    
    
    
    /**
     * Helper function to convert DType to string - equivalent to Python dtype mapping
     */
    internal func dtypeString(_ dtype: DType) -> String {
        switch dtype {
        case .float32:
            return "float32"
        case .float16:
            return "float16"
        case .bfloat16:
            return "bfloat16"
        default:
            return "unknown"
        }
    }
}

/**
 * Supporting types and errors
 */
public struct ProcessedInputs {
    public let inputIds: MLXArray
    public let inputFeatures: MLXArray
    
    public init(inputIds: MLXArray, inputFeatures: MLXArray) {
        self.inputIds = inputIds
        self.inputFeatures = inputFeatures
    }
}

// VoxtralError moved to VoxtralModelLoading.swift to avoid duplication

/**
 * Command line interface - equivalent to Python if __name__ == "__main__":
 */
public class VoxtralCLI {
    
    /**
     * Parse command line arguments - equivalent to Python argparse
     */
    public static func parseArguments(from args: [String]) throws -> VoxtralGenerationParameters {
        // Simple argument parsing - in a real implementation, you might use SwiftArgumentParser
        // Returns the default parameters with the required audio path
        
        guard args.count >= 2 else {
            throw VoxtralError.audioProcessingFailed("Audio path required")
        }
        
        let audioPath = args[1] // First argument after program name
        
        return VoxtralGenerationParameters(
            audioPath: audioPath,
            verbose: args.contains("--verbose")
        )
    }
    
    /**
     * Main CLI entry point - equivalent to Python main()
     */
    public static func main() {
        do {
            let args = CommandLine.arguments
            let parameters = try parseArguments(from: args)
            
            let generator = VoxtralGenerator(parameters: parameters)
            let result = try generator.generate()
            
            if !parameters.verbose {
                VoxtralDebug.log(result)
            }
            
        } catch {
            VoxtralDebug.log("Error: \(error)")
            exit(1)
        }
    }
}

/**
 * Usage examples - equivalent to Python script examples:
 * 
 * // Basic usage
 * let params = VoxtralGenerationParameters(audioPath: "audio.mp3")
 * let generator = VoxtralGenerator(parameters: params)
 * let transcription = try generator.generate()
 * 
 * // Advanced usage with custom parameters
 * let advancedParams = VoxtralGenerationParameters(
 *     model: "mistralai/Voxtral-Mini-3B-2507",
 *     maxTokens: 2048,
 *     temperature: 0.1,
 *     audioPath: "audio.mp3",
 *     verbose: true,
 *     stream: true
 * )
 * let advancedGenerator = VoxtralGenerator(parameters: advancedParams)
 * let result = try advancedGenerator.generate()
 */