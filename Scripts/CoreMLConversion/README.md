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

<sub>Figures published with the hybrid mode (commit `1944576`, 2026-01-06); hardware, precision and
method not recorded, not measured to the repository protocol (audit 2026-09-27, A-12 / F-12). The
MLX vs Core ML measurement is fiche K-42.</sub>

## Requirements

- Python 3.11-3.13 (see `requirements.txt`: every dependency is pinned; torch is held at 2.7.0,
  the last version tested by coremltools 9.0)
- macOS 13.0+ or iOS 16.0+
- Disk: the Mini download holds 9 356 474 312 bytes of shards plus `consolidated.safetensors`
  (9 348 806 528 bytes) (Hub listing, `docs/Weights.md` row `mini-3b`), then the extracted weights and
  the Core ML package

## Quick Start

```bash
cd Scripts/CoreMLConversion
./convert.sh
```

`convert.sh` creates `.venv`, installs the pinned requirements, downloads
`mistralai/Voxtral-Mini-3B-2507` at a pinned revision, extracts the encoder weights, converts
them (Mini) and compiles `output/VoxtralEncoderFull.mlmodelc`. The steps below do the same by hand.

## Conversion Steps

### Step 1: Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download Model (pinned revision)

```bash
# Official Mistral model; revision = file listing identical to `main` on 2026-09-28
hf download mistralai/Voxtral-Mini-3B-2507 \
    --revision 3060fe34b35ba5d44202ce9ff3c097642914f8f3 \
    --local-dir ./voxtral-mini-3b
```

### Step 3: Extract the Encoder Weights

```bash
python convert_weights.py \
    --model-path ./voxtral-mini-3b \
    --variant mini \
    --output ./output/voxtral_encoder.pt
```

### Step 4: Convert to Core ML

The ANE model always includes the multimodal projector; `--variant` sets its output width.

```bash
python convert_to_coreml_ane.py \
    --weights ./output/voxtral_encoder.pt \
    --variant mini \
    --output ./output/VoxtralEncoderMini.mlpackage
```

### Step 5: Compile

```bash
xcrun coremlcompiler compile ./output/VoxtralEncoderMini.mlpackage ./output/
```

The library looks for `VoxtralEncoderMini.mlmodelc` / `VoxtralEncoderSmall.mlmodelc` first, then the
legacy `VoxtralEncoderFull.mlmodelc` written by `convert.sh`
(`Sources/VoxtralCore/CoreML/VoxtralCoreMLEncoder.swift:73-77`, `:239-247`). The script's closing hint
`VoxtralCLI benchmark-coreml` refers to a subcommand that does not exist; parity and timing of the
converted model are measured by fiche K-42.

## Files

```
Scripts/CoreMLConversion/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── voxtral_encoder_pytorch.py   # PyTorch encoder architecture
├── voxtral_encoder_ane.py       # ANE-optimized encoder
├── convert_weights.py           # Weight extraction utilities
├── convert_to_coreml_ane.py     # Main conversion script
├── convert.sh                   # End-to-end conversion (Mini)
├── voxtral-mini-3b/             # Downloaded model weights
└── output/
    ├── voxtral_encoder.pt           # Extracted encoder weights (+ .safetensors copy)
    ├── VoxtralEncoderFull.mlpackage # Core ML model (convert.sh; VoxtralEncoderMini.* by hand)
    └── VoxtralEncoderFull.mlmodelc  # Compiled model
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

| Model | `--variant` | Output | Published Core ML model |
|-------|-------------|--------|-------------------------|
| Voxtral Mini 3B | `mini` | `[1, 375, 3072]` | `VincentGOURBIN/voxtral-encoder-coreml-mini` |
| Voxtral Small 24B | `small` | `[1, 375, 5120]` | `VincentGOURBIN/voxtral-encoder-coreml-small` |

The audio encoder is the same, but the converted model includes the projector, whose output width is
the LLM hidden size: **one Core ML model per variant** (`voxtral_encoder_ane.py:50-57`,
`Sources/VoxtralCore/CoreML/VoxtralCoreMLEncoder.swift:52-77`). To convert Small (download: shards
48 527 546 144 bytes plus `consolidated.safetensors` 48 519 877 672 bytes, Hub listing 2026-09-28):

```bash
# revision = file listing identical to `main` on 2026-09-28
hf download mistralai/Voxtral-Small-24B-2507 \
    --revision da5b42409f279fdd92febee0511a6c32828569c1 \
    --local-dir ./voxtral-small-24b
python convert_weights.py --model-path ./voxtral-small-24b --variant small \
    --output ./output/voxtral_encoder_small.pt
python convert_to_coreml_ane.py --weights ./output/voxtral_encoder_small.pt --variant small \
    --output ./output/VoxtralEncoderSmall.mlpackage
xcrun coremlcompiler compile ./output/VoxtralEncoderSmall.mlpackage ./output/
```

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
