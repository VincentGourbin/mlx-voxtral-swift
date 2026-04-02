# Voxtral TTS Benchmark

Benchmark comparing Voxtral TTS 4B generation speed across three quantization levels on Apple Silicon, in English and French.

## Setup

- **Hardware:** Apple M3 Max, 96 GB unified memory
- **Model:** Voxtral-4B-TTS-2603 (mlx-community variants)
- **Voices:** `neutral_male` (EN), `fr_male` (FR)
- **Config:** cfgAlpha=1.2, flowSteps=8, temperature=0.0
- **Features:** Text sanitization + lead-in silence trimming enabled
- **Date:** 2026-04-02

### Model variants

| Variant | HuggingFace repo | Size on disk | Quantization |
|---|---|---|---|
| bf16 | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | ~8 GB | bfloat16 (none) |
| 6-bit | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | ~3.5 GB | 6-bit affine, group_size=64 |
| 4-bit | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | ~2.5 GB | 4-bit affine, group_size=64 |

> Note: The audio tokenizer (codec decoder) is **not** quantized in any variant -- it always runs in bf16.

## Test texts

**Short EN:** "FORGE YOUR IDEA -- COMPLETELY REDESIGNED."

**Short FR:** "FORGEZ VOTRE IDEE -- ENTIEREMENT REPENSE."

**Long EN / Long FR (~200 words):** Full feature changelog text covering video generation, animation, canvas extension, new model, interface redesign, and quality improvements. See [Reproducing](#reproducing) for the full text.

## Results

### Short text (1 sentence)

| Text | Model | Voice | Load | Audio | GenTime | RTF | FPS |
|---|---|---|---|---|---|---|---|
| Short EN | **4-bit** | neutral_male | 0.12s | 7.68s | **3.17s** | **0.41x** | 30.3 |
| Short EN | 6-bit | neutral_male | 0.12s | 3.20s | 2.03s | 0.63x | 19.7 |
| Short EN | bf16 | neutral_male | 0.06s | 3.12s | 10.60s | 3.40x | 3.8 |
| Short FR | **4-bit** | fr_male | 0.11s | 5.84s | **2.58s** | **0.44x** | 30.3 |
| Short FR | 6-bit | fr_male | 0.11s | 4.24s | 2.77s | 0.65x | 20.9 |
| Short FR | bf16 | fr_male | 0.05s | 4.96s | 17.50s | 3.53x | 3.8 |

### Long text (~200 words)

| Text | Model | Voice | Load | Audio | GenTime | RTF | FPS |
|---|---|---|---|---|---|---|---|
| Long EN | **4-bit** | neutral_male | 0.12s | 89.20s | **40.76s** | **0.46x** | 27.6 |
| Long EN | 6-bit | neutral_male | 0.18s | 96.80s | 73.86s | 0.76x | 16.4 |
| Long EN | bf16 | neutral_male | 0.06s | 90.96s | 542.10s | 5.96x | 2.1 |
| Long FR | 4-bit | fr_male | 0.12s | 200.00s* | 113.80s | 0.57x | 22.0 |
| Long FR | **6-bit** | fr_male | 0.13s | 113.04s | **143.95s** | **1.27x** | 9.8 |
| Long FR | bf16 | fr_male | 0.06s | 116.24s | 551.76s | 4.75x | 2.6 |

> \* 4-bit FR Long hit maxFrames (2500) without finding EOA -- the 4-bit quantization is too aggressive for long French sequences with `fr_male`. Use 6-bit for French.

### RTF comparison (lower is better, <1.0 = faster than real-time)

```
                           EN Short    EN Long    FR Short    FR Long
4-bit                       0.41x      0.46x      0.44x      0.57x*
6-bit                       0.63x      0.76x      0.65x      1.27x
bf16                        3.40x      5.96x      3.53x      4.75x
                                                         ↑ real-time
* 4-bit FR Long: hit maxFrames, duration unreliable
```

### Speedup vs bf16

| | EN Short | EN Long | FR Short | FR Long |
|---|---|---|---|---|
| **4-bit** | **8.3x** | **13.3x** | **8.0x** | 4.8x* |
| **6-bit** | 5.4x | 7.3x | 6.3x | **3.8x** |

## Key findings

### Performance

1. **4-bit is 8-13x faster than bf16** in English, running well under real-time (RTF 0.41-0.46x). Audio is generated 2x faster than it can be played back.

2. **6-bit is 4-7x faster than bf16** and the best choice for French -- reliable EOA detection with sub-real-time generation (RTF 0.65-1.27x).

3. **bf16 is 3.4-6.0x slower than real-time** -- unusable for interactive/conversational applications.

### Language-specific observations

4. **4-bit degrades EOA detection for long French text**: the 4-bit model hit maxFrames (2500) on the 200-word French test, producing 200s of audio instead of ~113s. The 6-bit and bf16 models handled it correctly.

5. **French voices have shorter embeddings** (97 frames vs 169 for English), which makes the model more sensitive to quantization artifacts.

6. **Recommendation: use 6-bit for French, 4-bit for English.** Both run in real-time on M3 Max.

### Audio quality improvements

7. **Text sanitization** (auto-append terminal punctuation) is critical -- without it, the model fails to predict `end_audio` and loops indefinitely, especially for non-English text.

8. **Lead-in silence trimming** removes 2-7 transition frames (160-560ms) that the model generates before speech starts. French voices benefit most (up to 560ms of near-silence removed).

## Audio samples

Generated WAV files (24kHz mono, 16-bit) are available in [`docs/examples/`](examples/).

## Reproducing

```bash
# Build
xcodebuild -scheme VoxtralCLI -configuration Release \
  -derivedDataPath .build/xcode -destination 'platform=macOS' build

# Download models
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-mlx
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-4bit
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-6bit

# Run benchmark (English)
.build/xcode/Build/Products/Release/VoxtralCLI tts \
  "FORGE YOUR IDEA — COMPLETELY REDESIGNED." \
  -o output.wav --voice neutral_male --model tts-4b-4bit

# Run benchmark (French)
.build/xcode/Build/Products/Release/VoxtralCLI tts \
  "FORGEZ VOTRE IDÉE — ENTIÈREMENT REPENSÉ." \
  -o output.wav --voice fr_male --model tts-4b-6bit
```

### Full long text (EN)

> FORGE YOUR IDEA — COMPLETELY REDESIGNED. The creation screen becomes a full creative workshop with branching history. Every variation, animation, or edit creates a branch — navigate freely via an interactive timeline with action badges. Projects are auto-saved and accessible from the library. New input modes: text, image, audio with Voxtral, or audio plus image combined. Six generation scenarios with automatic GPU memory management. VIDEO GENERATION. New LTX 2.3 engine to create videos from text, images, or other videos. Distilled fast and Dev quality variants. Built-in audio generation. Retake mode to regenerate video segments with new prompts. ANIMATION. Animate your images from the creation tree. Combinable presets: camera, action, nature, expression, style. Automatic prompt enrichment via Gemma 3. CANVAS EXTENSION. Outpainting: extend your images beyond their borders. AI completes the scene automatically. NEW MODEL: KLEIN 9B KV. Model optimized for creative iteration, auto-selected in Forge. REDESIGNED INTERFACE. Unified image and animation panel with multi-select. Lag-free prompt input. Liquid Glass styling on macOS 26. Improved image and video previews. Per-category cache management. Multi-image upscale. QUALITY AND STABILITY. Aspect ratio preservation, no more forced square cropping. sRGB conversion for iPhone photos. Built-in help with full documentation. Over 1300 automated tests.

### Full long text (FR)

> FORGEZ VOTRE IDÉE — ENTIÈREMENT REPENSÉ. L'écran de création devient un véritable atelier créatif avec historique en branches. Chaque variation, animation ou modification crée une branche — naviguez librement via une timeline interactive avec badges d'action. Les projets sont sauvegardés automatiquement et accessibles depuis la bibliothèque. Nouveaux modes d'entrée : texte, image, audio avec Voxtral, ou audio plus image combinés. Six scénarios de génération avec gestion automatique de la mémoire GPU. GÉNÉRATION VIDÉO. Nouveau moteur LTX 2.3 pour créer des vidéos à partir de texte, d'images ou d'autres vidéos. Variantes distillée rapide et Dev qualité. Génération audio intégrée. Mode reprise pour régénérer des segments vidéo avec de nouveaux prompts. ANIMATION. Animez vos images depuis l'arbre de création. Préréglages combinables : caméra, action, nature, expression, style. Enrichissement automatique du prompt via Gemma 3. EXTENSION DE CANVAS. Outpainting : étendez vos images au-delà de leurs bords. L'IA complète la scène automatiquement. NOUVEAU MODÈLE : KLEIN 9B KV. Modèle optimisé pour l'itération créative, auto-sélectionné dans Forge. INTERFACE REPENSÉE. Panneau unifié image et animation avec sélection multiple. Saisie de prompt sans latence. Style Liquid Glass sur macOS 26. Aperçus d'images et de vidéos améliorés. Gestion du cache par catégorie. Upscale multi-images. QUALITÉ ET STABILITÉ. Préservation du ratio d'aspect, plus de recadrage carré forcé. Conversion sRGB pour les photos iPhone. Aide intégrée avec documentation complète. Plus de 1300 tests automatisés.
