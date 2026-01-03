/**
 * VoxtralTranscriptionTest - E2E Audio Transcription Test
 * Tests the complete audio-to-text pipeline with a real audio file
 */

import Foundation
import MLX
import VoxtralCore
import MLXLMCommon

// Configuration
let modelPath = "/Users/vincent/Developpements/convertvoxtral/voxtral_models/voxtral-mini-3b-8bit"
let audioPath = "/Users/vincent/Developpements/convertvoxtral/podcast_45s.mp3"

func main() throws {
    print("=" * 60)
    print("VOXTRAL E2E TRANSCRIPTION TEST")
    print("=" * 60)

    // Step 1: Load model
    print("\n[1/4] Loading model...")
    let startLoad = Date()
    let (standardModel, config) = try loadVoxtralStandardModel(modelPath: modelPath, dtype: .float16)
    let model = VoxtralForConditionalGeneration(standardModel: standardModel)
    let processor = try VoxtralProcessor.fromPretrained(modelPath)
    let loadTime = Date().timeIntervalSince(startLoad)
    print("  Model loaded in \(String(format: "%.2f", loadTime))s")
    print("  Model type: \(config.modelType)")
    print("  Audio encoder: \(config.audioConfig.hiddenLayers) layers, \(config.audioConfig.hiddenSize) hidden")

    // Step 2: Load and process audio
    print("\n[2/4] Processing audio...")
    guard FileManager.default.fileExists(atPath: audioPath) else {
        print("  ERROR: Audio file not found at \(audioPath)")
        return
    }

    let startAudio = Date()
    let inputs = try processor.applyTranscritionRequest(
        audio: audioPath,
        language: "en",
        samplingRate: 16000
    )
    let audioTime = Date().timeIntervalSince(startAudio)

    print("  Audio processed in \(String(format: "%.2f", audioTime))s")
    print("  Input IDs shape: \(inputs.inputIds.shape)")
    print("  Input features shape: \(inputs.inputFeatures.shape)")

    // Debug: Print first 20 input IDs
    var first20Ids: [Int] = []
    for i in 0..<min(20, inputs.inputIds.shape[1]) {
        first20Ids.append(inputs.inputIds[0, i].item(Int.self))
    }
    print("  First 20 input_ids: \(first20Ids)")
    print("  Python ref: [1, 3, 25, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]")

    let numChunks = inputs.inputFeatures.shape[0]
    let audioDuration = Float(numChunks) * 30.0  // 30 seconds per chunk
    print("  Audio duration: ~\(Int(audioDuration))s (\(numChunks) chunks)")

    // Step 3: Generate transcription
    print("\n[3/4] Generating transcription...")
    let startGen = Date()

    var generatedTokens: [Int] = []
    var fullText = ""

    do {
        let streamResults = try model.generateStream(
            inputIds: inputs.inputIds,
            inputFeatures: inputs.inputFeatures,
            attentionMask: nil,
            maxNewTokens: 200,  // Quick test
            temperature: 0.0,   // Greedy for transcription
            topP: 1.0,
            repetitionPenalty: 1.1
        )

        print("  Generating tokens...")

        for (i, (token, _)) in streamResults.enumerated() {
            let tokenId = token.item(Int.self)
            generatedTokens.append(tokenId)

            // Decode incrementally
            let tokenText = (try? processor.decode([tokenId])) ?? ""
            fullText += tokenText

            // Progress indicator every 100 tokens
            if (i + 1) % 100 == 0 {
                print("  ... \(i + 1) tokens generated")
            }
        }

    } catch {
        print("  ERROR during generation: \(error)")
        return
    }

    let genTime = Date().timeIntervalSince(startGen)

    // Step 4: Display results
    print("\n[4/4] Transcription complete!")
    print("=" * 60)
    print("\nTRANSCRIPTION:")
    print("-" * 40)
    print(fullText)
    print("-" * 40)

    // Statistics
    let tokensPerSecond = Double(generatedTokens.count) / genTime
    print("\nSTATISTICS:")
    print("  Total tokens: \(generatedTokens.count)")
    print("  Generation time: \(String(format: "%.2f", genTime))s")
    print("  Speed: \(String(format: "%.1f", tokensPerSecond)) tokens/s")
    print("  Audio duration: ~\(Int(audioDuration))s")
    print("  Real-time factor: \(String(format: "%.2fx", audioDuration / Float(genTime)))")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
}

// String multiplication helper
extension String {
    static func * (string: String, count: Int) -> String {
        return String(repeating: string, count: count)
    }
}

// Run the test
print("🚀🚀🚀 TOP LEVEL CODE STARTS 🚀🚀🚀")
do {
    try main()
} catch {
    print("FATAL ERROR: \(error)")
}
