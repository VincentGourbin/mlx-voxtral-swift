# Voxtral Core ML Conversion - Audio Encoder

This directory contains scripts to convert the Voxtral **audio encoder** from MLX to Core ML format for execution on Apple Neural Engine (ANE).

## Hybrid Architecture

```
                    VOXTRAL HYBRID
    ┌────────────────────────────────────────┐
    │  Audio Input (.wav)                    │
    │       ↓                                │
    │  ┌──────────────────────────────────┐  │
    │  │  CORE ML (ANE) - This conversion │  │
    │  │  ├── VoxtralEncoder (32 layers)  │  │
    │  │  └── MultiModalProjector         │  │
    │  │  Output: [1, 375, 3072]          │  │
    │  └──────────────────────────────────┘  │
    │       ↓                                │
    │  ┌──────────────────────────────────┐  │
    │  │  MLX (GPU) - Existing            │  │
    │  │  └── LlamaModel (30 layers)      │  │
    │  │  Output: Transcription/Chat      │  │
    │  └──────────────────────────────────┘  │
    └────────────────────────────────────────┘
```

**Note:** Full LLM conversion to Core ML was investigated but is not feasible due to dynamic shape operations in the transformers library. The hybrid approach (Core ML encoder + MLX decoder) is the optimal solution.

## Benefits

| Metric | MLX (GPU) | Core ML (ANE) |
|--------|-----------|---------------|
| Encoder latency | ~500ms | ~150ms (3x faster) |
| Power consumption | High | Low |
| Thermal throttling | Common | Rare |
| First token latency | Baseline | 2-3x faster |

## Requirements

- Python 3.10+
- macOS 13.0+ or iOS 16.0+
- ~500MB disk space for model

## Quick Start

```bash
cd Scripts/CoreMLConversion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Convert encoder with ANE optimizations
python convert_to_coreml_ane.py \
    --model-path ./voxtral-mini-3b \
    --output ./output/VoxtralEncoderANE.mlpackage
```

## Conversion Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Model (if needed)

```bash
# The official Mistral model is recommended
huggingface-cli download mistralai/Voxtral-Mini-3B-2507 --local-dir ./voxtral-mini-3b
```

### Step 3: Convert to Core ML

```bash
python convert_to_coreml_ane.py \
    --model-path ./voxtral-mini-3b \
    --output ./output/VoxtralEncoderANE.mlpackage
```

## Files

```
Scripts/CoreMLConversion/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── voxtral_encoder_pytorch.py   # PyTorch encoder architecture
├── voxtral_encoder_ane.py       # ANE-optimized encoder
├── convert_weights.py           # Weight extraction utilities
├── convert_to_coreml_ane.py     # Main conversion script
├── voxtral-mini-3b/             # Downloaded model weights
└── output/
    ├── VoxtralEncoderANE.mlpackage  # Core ML model
    └── voxtral_encoder.safetensors  # Extracted encoder weights
```

## Model Architecture

The converted model includes:

1. **VoxtralEncoder** (32 transformer layers)
   - Input: `[1, 128, 3000]` (mel spectrogram)
   - Output: `[1, 1500, 1280]` (encoder hidden states)

2. **VoxtralMultiModalProjector** (2 linear layers)
   - Input: `[1, 1500, 1280]` -> reshape -> `[375, 5120]`
   - Output: `[375, 3072]` (LLM-compatible embeddings)

## Compatibility

| Model | Encoder Compatible |
|-------|-------------------|
| Voxtral Mini 3B | Yes |
| Voxtral Small 24B | Yes |

**One Core ML model works with both Mini and Small!** The audio encoder architecture is identical.

## Using in Swift

See `Sources/VoxtralCore/` for the hybrid implementation that automatically uses Core ML when available.

```swift
// In VoxtralGenerator
let config = VoxtralConfiguration(
    backend: .hybrid  // or .mlx for GPU-only
)
let generator = try VoxtralGenerator(configuration: config)
```

## License

Apache 2.0
