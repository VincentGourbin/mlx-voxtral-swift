# MLX Voxtral Swift

A native Swift implementation of [Voxtral](https://huggingface.co/mistralai/Voxtral-mini-3B-2507) speech-to-text model, running on Apple Silicon with [MLX](https://github.com/ml-explore/mlx-swift).

This is a Swift port of the excellent Python implementation by [@mzbac](https://github.com/mzbac): **[mlx.voxtral](https://github.com/mzbac/mlx.voxtral)**

## Features

- **Native Swift** - Pure Swift implementation, no Python dependencies at runtime
- **MLX Acceleration** - Leverages Apple's MLX framework for optimal Apple Silicon performance
- **Quantized Models** - Supports 4-bit and 8-bit quantized models for reduced memory usage
- **Streaming Output** - Real-time token-by-token transcription
- **SwiftUI App** - Ready-to-use macOS application with drag-and-drop interface
- **Library Integration** - Import `VoxtralCore` into your own Swift projects
- **Chat Mode** - Ask questions about audio content (like the Python `voxtral_chat.py`)

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon Mac (M1/M2/M3/M4)
- Xcode 15.0 or later
- Swift 5.9 or later

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/VincentGourbin/mlx-voxtral-swift", branch: "main")
]
```

Then add `VoxtralCore` to your target dependencies:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "VoxtralCore", package: "mlx-voxtral-swift")
    ]
)
```

### Clone and Build

```bash
git clone https://github.com/VincentGourbin/mlx-voxtral-swift.git
cd mlx-voxtral-swift
swift build
```

## Model Download

Download a quantized Voxtral model from HuggingFace:

```bash
# Recommended: 8-bit quantized (best quality/size balance)
huggingface-cli download mzbac/voxtral-mini-3b-8bit --local-dir ./voxtral_models/voxtral-mini-3b-8bit

# Alternative: 4-bit quantized (smaller, slightly lower quality)
huggingface-cli download mzbac/voxtral-mini-3b-4bit --local-dir ./voxtral_models/voxtral-mini-3b-4bit
```

## Usage

### SwiftUI Application

Build and run the macOS app:

```bash
swift build --product VoxtralApp
./create_app_bundle.sh
open Voxtral.app
```

The app provides:
- Drag-and-drop audio file selection
- Transcription and Chat modes
- Real-time streaming output
- Configurable max tokens and temperature

### Command Line Interface

```bash
swift build --product VoxtralCLI
./.build/debug/VoxtralCLI
```

### Library Integration

```swift
import VoxtralCore
import MLX

// Load model
let (model, config) = try loadVoxtralStandardModel(
    modelPath: "/path/to/voxtral-mini-3b-8bit",
    dtype: .float16
)

// Create wrapper and processor
let voxtral = VoxtralForConditionalGeneration(standardModel: model)
let processor = try VoxtralProcessor.fromPretrained("/path/to/voxtral-mini-3b-8bit")

// Transcribe audio
let inputs = try processor.applyTranscritionRequest(
    audio: "/path/to/audio.mp3",
    language: "en",
    samplingRate: 16000
)

// Generate with streaming
let results = try voxtral.generateStream(
    inputIds: inputs.inputIds,
    inputFeatures: inputs.inputFeatures,
    attentionMask: nil,
    maxNewTokens: 500,
    temperature: 0.0,
    topP: 1.0,
    repetitionPenalty: 1.1
)

// Process tokens as they arrive
var transcription = ""
for (token, _) in results {
    let tokenId = token.item(Int.self)
    if let text = try? processor.decode([tokenId]) {
        transcription += text
        print(text, terminator: "")
    }
}
```

### Chat Mode (Ask Questions About Audio)

```swift
// Build conversation with audio and text prompt
let conversation: [[String: Any]] = [
    [
        "role": "user",
        "content": [
            ["type": "audio", "audio": "/path/to/audio.mp3"],
            ["type": "text", "text": "What language is being spoken?"]
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

// Generate response...
```

## Architecture

```
mlx-voxtral-swift/
├── Sources/
│   ├── VoxtralCore/           # Core library
│   │   ├── VoxtralModeling.swift      # Main model architecture
│   │   ├── VoxtralProcessor.swift     # Audio & text processing
│   │   ├── VoxtralFeatureExtractor.swift  # Mel-spectrogram extraction
│   │   ├── VoxtralComponents.swift    # Tokenizer, encoder components
│   │   ├── VoxtralConfiguration.swift # Model config parsing
│   │   ├── Models/                    # LLM model definitions
│   │   └── Utils/                     # Loading & utility functions
│   ├── VoxtralApp/            # SwiftUI macOS application
│   │   ├── VoxtralAppMain.swift
│   │   ├── ContentView.swift
│   │   └── TranscriptionManager.swift
│   └── VoxtralTranscriptionTest/  # CLI example
└── Tests/
    └── VoxtralCoreTests/      # Unit tests
```

## Key Components

| Component | Description |
|-----------|-------------|
| `VoxtralForConditionalGeneration` | Main model class combining audio encoder and language model |
| `VoxtralProcessor` | Handles audio loading, mel-spectrogram, and tokenization |
| `VoxtralEncoder` | Whisper-style audio encoder (conv layers + transformer) |
| `MultiModalProjector` | Projects audio embeddings to LLM hidden dimension |
| `TekkenTokenizer` | Mistral's tiktoken-based tokenizer |

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- And any format supported by AVFoundation

## Performance

On Apple Silicon (M1/M2/M3):
- **Model loading**: ~5-10 seconds (8-bit quantized)
- **Transcription speed**: ~15-25 tokens/second
- **Memory usage**: ~4GB (8-bit) / ~2GB (4-bit)

## Acknowledgments

This project is a Swift port of the Python implementation:

- **[mlx.voxtral](https://github.com/mzbac/mlx.voxtral)** by [@mzbac](https://github.com/mzbac) - The original MLX Python implementation that made this port possible. Thank you for the excellent reference implementation!

Built with:
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's machine learning framework
- [Swift Transformers](https://github.com/huggingface/swift-transformers) - HuggingFace tokenizers
- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples) - LLM implementations

## License

MIT License - See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Related Projects

- [Voxtral (HuggingFace)](https://huggingface.co/mistralai/Voxtral-mini-3B-2507) - Original Mistral model
- [mlx.voxtral (Python)](https://github.com/mzbac/mlx.voxtral) - Python MLX implementation
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Swift
