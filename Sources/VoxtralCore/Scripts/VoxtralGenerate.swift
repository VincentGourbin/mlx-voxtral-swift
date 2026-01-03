/**
 * VoxtralGenerate - Swift equivalent of mlx.voxtral/scripts/generate.py
 * 
 * Exact conversion of Python CLI script for audio transcription.
 * Direct line-by-line translation following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

import Foundation
import MLX
import MLXNN
import ArgumentParser

/**
 * Direct Python equivalent: def main()
 */
struct VoxtralGenerate: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate transcriptions from audio using Voxtral models"
    )
    
    // Python: parser.add_argument("--model", default="mistralai/Voxtral-Mini-3B-2507")
    @Option(name: .long, help: "Model name or path")
    var model: String = "mistralai/Voxtral-Mini-3B-2507"
    
    // Python: parser.add_argument("--max-token", type=int, default=1024)
    @Option(name: .long, help: "Maximum number of tokens to generate")
    var maxToken: Int = 1024
    
    // Python: parser.add_argument("--temperature", type=float, default=0.0)
    @Option(name: .long, help: "Sampling temperature")
    var temperature: Float = 0.0
    
    // Python: parser.add_argument("--audio", required=True)
    @Argument(help: "Path to audio file or URL")
    var audio: String
    
    // Python: parser.add_argument("--top-p", type=float, default=0.95)
    @Option(name: .long, help: "Nucleus sampling parameter")
    var topP: Float = 0.95
    
    // Python: parser.add_argument("--dtype", default="bfloat16")
    @Option(name: .long, help: "Model dtype")
    var dtype: String = "bfloat16"
    
    // Python: parser.add_argument("--verbose", action="store_true")
    @Flag(name: .long, help: "Enable verbose output")
    var verbose: Bool = false
    
    // Python: parser.add_argument("--language", default="en")
    @Option(name: .long, help: "Language code")
    var language: String = "en"
    
    // Python: parser.add_argument("--stream", action="store_true")
    @Flag(name: .long, help: "Enable streaming output")
    var stream: Bool = false
    
    /**
     * Direct Python equivalent: def main()
     */
    func run() throws {
        if verbose {
            print("Loading model: \(model)")
        }
        
        let startTime = Date()
        
        // Python: model, config = load_voxtral_model(args.model, dtype=dtype)
        let voxtralModel = try loadVoxtralModel(model, dtype: parseDType(dtype))
        
        // Python: processor = VoxtralProcessor.from_pretrained(args.model)
        let processor = try VoxtralProcessor.fromPretrained(model)
        
        if verbose {
            let loadTime = Date().timeIntervalSince(startTime)
            print("Model loaded in \(String(format: "%.2f", loadTime)) seconds")
            print("Model dtype: \(dtype)")
        }
        
        if verbose {
            print("\nProcessing audio: \(audio)")
        }
        
        // Python: conversation = [{"role": "user", "content": [{"type": "text", "text": "décrit ce fichier audio"}, {"type": "audio", "audio": args.audio}]}]
        //         inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="mlx")
        let conversation: [[String: Any]] = [
            [
                "role": "user",
                "content": [
                    ["type": "text", "text": "décrit ce fichier audio"],
                    ["type": "audio", "audio": audio]
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
        
        if verbose {
            print("\nGenerating transcription...")
            if stream {
                print("(Streaming mode enabled)")
            }
        }
        
        let generateStart = Date()
        
        let transcription: String
        if stream {
            // Python: for token, _ in model.generate_stream(...)
            transcription = try generateStreaming(
                model: voxtralModel,
                processor: processor,
                inputs: inputs,
                maxTokens: maxToken,
                temperature: temperature,
                topP: topP,
                verbose: verbose
            )
        } else {
            // Python: output_ids = model.generate(...)
            transcription = try generateBatch(
                model: voxtralModel,
                inputs: inputs,
                maxTokens: maxToken,
                temperature: temperature,
                topP: topP,
                processor: processor,
                verbose: verbose
            )
        }
        
        if verbose {
            let generationTime = Date().timeIntervalSince(generateStart)
            let tokensPerSecond = Float(maxToken) / Float(generationTime)
            print("\nGeneration completed in \(String(format: "%.2f", generationTime)) seconds (\(String(format: "%.2f", tokensPerSecond)) tokens/s)")
        }
        
        if !verbose {
            print(transcription)
        }
    }
}

/**
 * Direct Python equivalent: load_voxtral_model function
 */
private func loadVoxtralModel(_ modelPath: String, dtype: MLX.DType) throws -> VoxtralForConditionalGeneration {
    // Python: model = VoxtralForConditionalGeneration.from_pretrained(model_path)
    return try VoxtralForConditionalGeneration.fromPretrained(modelPath)
}

/**
 * Direct Python equivalent: streaming generation
 */
private func generateStreaming(
    model: VoxtralForConditionalGeneration,
    processor: VoxtralProcessor,
    inputs: ProcessedInputs,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    verbose: Bool
) throws -> String {
    // Python: mlx_inputs = {"input_ids": inputs.input_ids, "input_features": inputs.input_features}
    let mlxInputs = [
        "input_ids": inputs.inputIds,
        "input_features": inputs.inputFeatures
    ]
    
    if verbose {
        print("\n" + String(repeating: "=", count: 50))
        print("TRANSCRIPTION:")
        print(String(repeating: "=", count: 50))
    }
    
    // Python: generated_tokens = []
    var generatedTokens: [Int] = []
    // Python: num_tokens = 0
    var numTokens = 0
    
    // Python: for token, _ in model.generate_stream(**mlx_inputs, max_new_tokens=args.max_token, temperature=args.temperature, top_p=args.top_p)
    for (token, _) in try model.generateStream(
        inputIds: mlxInputs["input_ids"]!,
        inputFeatures: mlxInputs["input_features"]!,
        maxNewTokens: maxTokens,
        temperature: temperature,
        topP: topP
    ) {
        // Python: token_id = token.item()
        let tokenId: Int = token.item()
        // Python: generated_tokens.append(token_id)
        generatedTokens.append(tokenId)
        
        // Python: text = processor.decode([token_id], skip_special_tokens=False)
        let text = try processor.decode([tokenId], skipSpecialTokens: false)
        
        // Python: if token_id not in [processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id]:
        if let tokenizer = processor.tokenizer,
           tokenId != tokenizer.eosTokenIdValue && tokenId != tokenizer.padTokenIdValue {
            // Python: print(text, end='', flush=True)
            print(text, terminator: "")
            fflush(stdout)
        }
        
        // Python: num_tokens += 1
        numTokens += 1
    }
    
    // Python: print()
    print()
    
    if verbose {
        // Python: generation_time = time.time() - start_time
        // Python: tokens_per_second = num_tokens / generation_time
        // Python: print("="*50)
        // Python: print(f"\nGenerated {num_tokens} tokens in {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/s)")
        print(String(repeating: "=", count: 50))
        print("\nGenerated \(numTokens) tokens")
    }
    
    // Return empty string since we printed directly (like Python)
    return ""
}

/**
 * Direct Python equivalent: batch generation
 */
private func generateBatch(
    model: VoxtralForConditionalGeneration,
    inputs: ProcessedInputs,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    processor: VoxtralProcessor,
    verbose: Bool
) throws -> String {
    // Python: mlx_inputs = {"input_ids": inputs.input_ids, "input_features": inputs.input_features}
    let mlxInputs = [
        "input_ids": inputs.inputIds,
        "input_features": inputs.inputFeatures
    ]
    
    // Python: output_ids = model.generate(**mlx_inputs, max_new_tokens=args.max_token, temperature=args.temperature, top_p=args.top_p)
    let outputIds = try model.generate(
        inputIds: mlxInputs["input_ids"]!,
        inputFeatures: mlxInputs["input_features"]!,
        maxNewTokens: maxTokens,
        temperature: temperature,
        topP: topP
    )
    
    if verbose {
        // Python: generation_time = time.time() - start_time
        // Python: num_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]  
        let numTokens = outputIds.shape[1] - inputs.inputIds.shape[1]
        // Python: tokens_per_second = num_tokens / generation_time
        // Python: print(f"\nGenerated {num_tokens} tokens in {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/s)")
        print("\nGenerated \(numTokens) tokens")
    }
    
    // Python: generated_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    let generatedTokens = outputIds[0, inputs.inputIds.shape[1]...]
    
    // Python: transcription = processor.decode(generated_tokens, skip_special_tokens=True)
    let transcription = try processor.decode(generatedTokens, skipSpecialTokens: true)
    
    if verbose {
        print("\n" + String(repeating: "=", count: 50))
        print("TRANSCRIPTION:")
        print(String(repeating: "=", count: 50))
    }
    
    // Python: print(transcription)
    print(transcription)
    
    if verbose {
        print(String(repeating: "=", count: 50))
    }
    
    return transcription
}

/**
 * Helper function to parse dtype string to MLX.DType
 */
private func parseDType(_ dtype: String) -> MLX.DType {
    switch dtype.lowercased() {
    case "float32", "float":
        return .float32
    case "float16":
        return .float16
    case "bfloat16":
        return .bfloat16
    default:
        return .bfloat16
    }
}