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
            print("Loading model: \(parameters.model)")
        }
        
        let loadStart = Date()
        
        // Python: model, config = load_voxtral_model(args.model, dtype=dtype)
        try loadModelFromPath()
        
        // Python: processor = VoxtralProcessor.from_pretrained(args.model)
        try loadProcessorFromPath()
        
        if parameters.verbose {
            let loadTime = Date().timeIntervalSince(loadStart)
            print("Model loaded in \(String(format: "%.2f", loadTime)) seconds")
            print("Model dtype: \(dtypeString(parameters.dtype))")
        }
        
        if parameters.verbose {
            print("\nProcessing audio: \(parameters.audioPath)")
        }
        
        // Python: inputs = processor.apply_transcrition_request(audio=args.audio, language=args.language)
        let inputs = try processAudioWithExistingPipeline()
        
        if parameters.verbose {
            print("\nGenerating transcription...")
            if parameters.stream {
                print("(Streaming mode enabled)")
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
     * Load model - equivalent to Python load_voxtral_model()
     */
    private func loadModel() throws {
        // Python: model, config = load_voxtral_model(args.model, dtype=dtype)
        let dtype = parameters.dtype
        
        if parameters.verbose {
            print("Loading model: \(parameters.model)")
        }
        
        // Use the loadVoxtralModel function from VoxtralModelLoading.swift
        let (loadedModel, _) = try loadVoxtralModel(
            modelPath: parameters.model,
            dtype: dtype,
            lazy: true
        )
        
        model = loadedModel
        
        if parameters.verbose {
            print("Model dtype: \(dtypeString(parameters.dtype))")
        }
    }
    
    /**
     * Load processor - equivalent to Python VoxtralProcessor.from_pretrained()
     */
    private func loadProcessor() throws {
        // Python: processor = VoxtralProcessor.from_pretrained(args.model)
        processor = try VoxtralProcessor.fromPretrained(parameters.model)
    }
    
    /**
     * Process audio input - equivalent to Python apply_transcrition_request()
     */
    private func processAudio() throws -> ProcessedInputs {
        guard let processor = processor else {
            throw VoxtralError.processorNotLoaded
        }
        
        // Python: conversation = [{"role": "user", "content": [{"type": "text", "text": "décrit ce fichier audio"}, {"type": "audio", "audio": args.audio}]}]
        //         inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="mlx")
        if parameters.verbose {
            print("Processing audio: \(parameters.audioPath)")
        }
        
        let conversation: [[String: Any]] = [
            [
                "role": "user",
                "content": [
                    ["type": "text", "text": "décrit ce fichier audio"],
                    ["type": "audio", "audio": parameters.audioPath]
                ]
            ]
        ]
        let chatResult = try processor.applyChatTemplate(
            conversation: conversation,
            tokenize: true,
            returnTensors: "mlx"
        ) as! [String: MLXArray]
        
        let inputs = ProcessedInputs(
            inputIds: chatResult["input_ids"]!,
            inputFeatures: chatResult["input_features"]!
        )
        
        return inputs
    }
    
    /**
     * Generate with streaming - equivalent to Python streaming mode
     */
    private func generateStreaming(inputs: ProcessedInputs, startTime: Date) throws -> String {
        guard let model = model, let processor = processor else {
            throw VoxtralError.modelNotLoaded
        }
        
        if parameters.verbose {
            print("\nGenerating transcription...")
            print("(Streaming mode enabled)")
            print("\n" + String(repeating: "=", count: 50))
            print("TRANSCRIPTION:")
            print(String(repeating: "=", count: 50))
        }
        
        var generatedTokens: [Int] = []
        var numTokens = 0
        
        // Python: for token, _ in model.generate_stream(**mlx_inputs, max_new_tokens=args.max_token, temperature=args.temperature, top_p=args.top_p):
        let streamResults = try model.generateStream(
            inputIds: inputs.inputIds,
            inputFeatures: inputs.inputFeatures,
            maxNewTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP
        )
        
        for (token, _) in streamResults {
            // Python: token_id = token.item()
            let tokenId = token.item(Int.self)
            generatedTokens.append(tokenId)
            
            // Python: text = processor.decode([token_id], skip_special_tokens=False)
            let text = try processor.decode([tokenId], skipSpecialTokens: false)
            
            // Python: if token_id not in [processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id]:
            if tokenId != processor.tokenizer?.eosTokenIdValue && tokenId != processor.tokenizer?.padTokenIdValue {
                // Python: print(text, end='', flush=True)
                print(text, terminator: "")
                fflush(stdout)
            }
            
            numTokens += 1
        }
        
        print() // New line after streaming output
        
        if parameters.verbose {
            let generationTime = Date().timeIntervalSince(startTime)
            let tokensPerSecond = Double(numTokens) / generationTime
            print(String(repeating: "=", count: 50))
            print(String(format: "\nGenerated %d tokens in %.2f seconds (%.2f tokens/s)", 
                         numTokens, generationTime, tokensPerSecond))
        }
        
        // Decode the full transcription for return
        let transcription = try processor.decode(generatedTokens, skipSpecialTokens: true)
        return transcription
    }
    
    /**
     * Generate in batch mode - equivalent to Python non-streaming mode
     */
    private func generateBatch(inputs: ProcessedInputs, startTime: Date) throws -> String {
        guard let model = model, let processor = processor else {
            throw VoxtralError.modelNotLoaded
        }
        
        // Python: output_ids = model.generate(**mlx_inputs, max_new_tokens=args.max_token, temperature=args.temperature, top_p=args.top_p)
        let outputIds = try model.generate(
            inputIds: inputs.inputIds,
            inputFeatures: inputs.inputFeatures,
            maxNewTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP
        )
        
        if parameters.verbose {
            let generationTime = Date().timeIntervalSince(startTime)
            // Python: num_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
            let numTokens = outputIds.shape[1] - inputs.inputIds.shape[1]
            let tokensPerSecond = Double(numTokens) / generationTime
            print(String(format: "\nGenerated %d tokens in %.2f seconds (%.2f tokens/s)", 
                         numTokens, generationTime, tokensPerSecond))
        }
        
        // Python: generated_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        let generatedTokens = outputIds[0, inputs.inputIds.shape[1]...]
        
        // Python: transcription = processor.decode(generated_tokens, skip_special_tokens=True)
        let transcription = try processor.decode(generatedTokens, skipSpecialTokens: true)
        
        if parameters.verbose {
            print("\n" + String(repeating: "=", count: 50))
            print("TRANSCRIPTION:")
            print(String(repeating: "=", count: 50))
        }
        
        print(transcription)
        
        if parameters.verbose {
            print(String(repeating: "=", count: 50))
        }
        
        return transcription
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
        // For now, return default parameters with required audio path
        
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
                print(result)
            }
            
        } catch {
            print("Error: \(error)")
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