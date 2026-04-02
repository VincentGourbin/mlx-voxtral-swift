# Voxtral TTS Benchmark

Benchmark comparing Voxtral TTS 4B generation speed across three quantization levels on Apple Silicon.

## Setup

- **Hardware:** Apple M3 Max, 96 GB unified memory
- **Model:** Voxtral-4B-TTS-2603 (mlx-community variants)
- **Voice:** `neutral_male`
- **Config:** cfgAlpha=1.2, flowSteps=8, temperature=0.0
- **Date:** 2026-04-02

### Model variants

| Variant | HuggingFace repo | Size on disk | Quantization |
|---|---|---|---|
| bf16 | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | ~8 GB | bfloat16 (none) |
| 6-bit | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | ~3.5 GB | 6-bit affine, group_size=64 |
| 4-bit | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | ~2.5 GB | 4-bit affine, group_size=64 |

> Note: The audio tokenizer (codec decoder) is **not** quantized in any variant -- it always runs in bf16.

## Test texts

**Long text (150 words):**

> Voice: the original UI. Voice was humanity's first interface -- long before writing or typing, it let us share ideas, coordinate work, and build relationships. As digital systems become more capable, voice is returning as our most natural form of human-computer interaction. Yet today's systems remain limited -- unreliable, proprietary, and too brittle for real-world use. Closing this gap demands tools with exceptional transcription, deep understanding, multilingual fluency, and open, flexible deployment. We release the Voxtral models to accelerate this future. These state-of-the-art speech understanding models are available in two sizes -- a 24B variant for production-scale applications and a 3B variant for local and edge deployments. Both versions are released under the Apache 2.0 license, and are also available on our API. The API routes transcription queries to a transcribe-optimized version of Voxtral Mini (Voxtral Mini Transcribe) that delivers unparalleled cost and latency-efficiency. For a comprehensive understanding of the research and development behind Voxtral, please refer to our detailed research paper, available for download here.

**Short text (13 words):**

> Open-source ASR systems with high word error rates and limited semantic understanding

## Results

### Long text (150 words)

| Metric | bf16 | 6-bit | 4-bit |
|---|---|---|---|
| Model load time | 0.06s | 0.12s | 0.12s |
| Generation time | 246.24s | 50.49s | **25.27s** |
| Frames generated | 861 | 914 | 801 |
| Audio duration | 68.88s | 73.12s | 64.08s |
| WAV file size | 3.2 MB | 3.3 MB | 2.9 MB |
| Real-time factor (RTF) | 3.57x | 0.69x | **0.39x** |
| Frames/sec | 3.5 | 18.1 | **31.7** |
| Speedup vs bf16 | 1.0x | 4.9x | **9.7x** |

### Short text (13 words)

| Metric | bf16 | 6-bit | 4-bit |
|---|---|---|---|
| Model load time | 0.06s | 0.14s | 0.12s |
| Generation time | 27.41s | 4.92s | **2.37s** |
| Frames generated | 71 | 79 | 70 |
| Audio duration | 5.68s | 6.32s | 5.60s |
| WAV file size | 266 KB | 296 KB | 263 KB |
| Real-time factor (RTF) | 4.82x | 0.78x | **0.42x** |
| Frames/sec | 2.6 | 16.0 | **29.5** |
| Speedup vs bf16 | 1.0x | 5.6x | **11.6x** |

### RTF comparison (lower is better, <1.0 = faster than real-time)

```
bf16   Long  ████████████████████████████████████  3.57x
bf16   Short ████████████████████████████████████████████████  4.82x

6-bit  Long  ███████  0.69x
6-bit  Short ████████  0.78x

4-bit  Long  ████  0.39x
4-bit  Short ████  0.42x
                                              ↑ real-time (1.0x)
```

## Key findings

1. **4-bit is ~10x faster than bf16** and runs well under real-time (RTF 0.39-0.42x), meaning audio is generated 2.5x faster than it can be played back.

2. **6-bit is ~5x faster than bf16** and also runs under real-time (RTF 0.69-0.78x), offering a middle ground between speed and quality.

3. **All three variants produce similar audio durations** for the same input text, suggesting quantization does not fundamentally alter the decoding behavior.

4. **The speedup comes from memory bandwidth**: quantized models are smaller and fit better in the GPU's cache hierarchy, dramatically reducing memory-bound latency during autoregressive generation.

5. **Model load time is negligible** (~0.1s) for all variants since models are cached on disk. The quantization graph setup adds ~0.06s overhead.

6. **Short texts have higher RTF** across all variants due to fixed per-generation overhead (prefill, voice embedding injection).

## Audio samples

Generated WAV files (24kHz mono, 16-bit) are available in [`docs/examples/`](examples/):

| File | Variant | Text |
|---|---|---|
| [`tts_bench_long_bf16.wav`](examples/tts_bench_long_bf16.wav) | bf16 | Long |
| [`tts_bench_long_6bit.wav`](examples/tts_bench_long_6bit.wav) | 6-bit | Long |
| [`tts_bench_long_4bit.wav`](examples/tts_bench_long_4bit.wav) | 4-bit | Long |
| [`tts_bench_short_bf16.wav`](examples/tts_bench_short_bf16.wav) | bf16 | Short |
| [`tts_bench_short_6bit.wav`](examples/tts_bench_short_6bit.wav) | 6-bit | Short |
| [`tts_bench_short_4bit.wav`](examples/tts_bench_short_4bit.wav) | 4-bit | Short |

## Reproducing

```bash
# Build
xcodebuild -scheme VoxtralCLI -configuration Release \
  -derivedDataPath .build/xcode -destination 'platform=macOS' build

# Download models
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-mlx
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-4bit
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-6bit

# Run benchmark
.build/xcode/Build/Products/Release/VoxtralCLI tts "Your text here" \
  -o output.wav --voice neutral_male --model tts-4b-4bit
```
