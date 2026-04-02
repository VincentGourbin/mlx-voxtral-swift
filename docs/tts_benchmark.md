# Voxtral TTS Benchmark

Bilingual (EN/FR) benchmark of Voxtral TTS 4B across three quantization levels on Apple Silicon, with prosody-aware text sanitization.

## Setup

- **Hardware:** Apple M3 Max, 96 GB unified memory
- **Model:** Voxtral-4B-TTS-2603 (mlx-community variants)
- **Voices:** `neutral_male` (EN), `fr_female` (FR)
- **Config:** cfgAlpha=1.2, flowSteps=8, temperature=0.0
- **Features:** Prosody-aware sanitization, lead-in silence trimming, TTFT measurement
- **Date:** 2026-04-02

### Model variants

| Variant | HuggingFace repo | Size on disk | Quantization |
|---|---|---|---|
| bf16 | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | ~8 GB | bfloat16 (none) |
| 6-bit | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | ~3.5 GB | 6-bit affine, group_size=64 |
| 4-bit | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | ~2.5 GB | 4-bit affine, group_size=64 |

> The audio tokenizer (codec decoder) is **not** quantized — it always runs in bf16.

## Test texts

**Short EN:** "Fluxforge Studio turns your Mac into a complete AI creative studio."

**Short FR:** "Fluxforge Studio transforme votre Mac en un studio de creation IA complet."

**Long (~350 words):** Full Fluxforge app description with 11 sections (Forge, Image Generation, Video, Animation, Canvas, LoRA, Training, Background Removal, Library, Privacy, Requirements). See [full text below](#full-test-texts).

## Results

### Short text (1 sentence)

| Text | Model | Voice | TTFT | Frames | Audio | GenTime | RTF |
|---|---|---|---|---|---|---|---|
| EN | **4-bit** | neutral_male | **400ms** | 72 | 5.68s | 6.63s | **1.17x** |
| EN | 6-bit | neutral_male | 616ms | 63 | 5.04s | 9.46s | 1.88x |
| EN | bf16 | neutral_male | 1454ms | 76 | 5.60s | 38.43s | 6.86x |
| FR | **4-bit** | fr_female | **224ms** | 53 | 4.16s | 3.54s | **0.85x** |
| FR | 6-bit | fr_female | 338ms | 65 | 4.80s | 6.69s | 1.39x |
| FR | bf16 | fr_female | 864ms | 63 | 4.96s | 25.81s | 5.20x |

### Long text (Fluxforge full description, ~350 words)

| Text | Model | Voice | TTFT | Frames | Audio | GenTime | RTF |
|---|---|---|---|---|---|---|---|
| EN | **4-bit** | neutral_male | **909ms** | 2266 | 181.28s | 120.61s | **0.67x** |
| EN | 6-bit | neutral_male | 795ms | 2101 | 166.96s | 155.46s | 0.93x |
| EN | bf16 | neutral_male | 1523ms | 2314 | 185.04s | 902.26s | 4.88x |
| FR | 4-bit* | fr_female | 1412ms | 2500* | 200.00s* | 158.75s | 0.79x |
| FR | **6-bit** | fr_female | **1132ms** | 2175 | 173.76s | 194.41s | **1.12x** |
| FR | bf16* | fr_female | 1696ms | 2500* | 200.00s* | 971.31s | 4.86x |

> \* 4-bit and bf16 FR hit maxFrames (2500) on this very long text. **6-bit is the reliable choice for long French content.**

### TTFT for conversational use (<500ms target)

```
                    Short EN    Short FR    Long EN    Long FR
4-bit                400ms      224ms       909ms      1412ms
6-bit                616ms      338ms       795ms      1132ms
bf16                1454ms      864ms      1523ms      1696ms
```

4-bit FR short achieves **224ms TTFT** — well within the 500ms conversational threshold.

## Key findings

1. **4-bit delivers 224ms TTFT on French short text** — viable for real-time conversation on Apple Silicon.

2. **6-bit is the best choice for long French text** — reliable EOA detection (no maxFrames overflow), sub-real-time generation (1.12x RTF).

3. **4-bit EN is fastest overall** — 0.67x RTF on long text (audio generated 1.5x faster than playback).

4. **bf16 is 5-7x slower than real-time** — unsuitable for interactive use, but serves as quality baseline.

5. **Prosody-aware sanitization** converts structural formatting (paragraphs, headers, bullets) into punctuation that produces natural pauses. Without it, multi-section text reads robotically.

6. **Text sanitization is critical** — ALL-CAPS text, missing terminal punctuation, and em-dashes all degrade quality. The sanitizer handles all of these automatically (disable with `--no-sanitize`).

## Audio samples

Generated WAV files (24kHz mono, 16-bit) are in [`docs/examples/`](examples/):

### Short text samples

| File | Model | Language |
|---|---|---|
| [`fluxforge_short_en_4bit.wav`](examples/fluxforge_short_en_4bit.wav) | 4-bit | EN |
| [`fluxforge_short_en_6bit.wav`](examples/fluxforge_short_en_6bit.wav) | 6-bit | EN |
| [`fluxforge_short_en_bf16.wav`](examples/fluxforge_short_en_bf16.wav) | bf16 | EN |
| [`fluxforge_short_fr_4bit.wav`](examples/fluxforge_short_fr_4bit.wav) | 4-bit | FR |
| [`fluxforge_short_fr_6bit.wav`](examples/fluxforge_short_fr_6bit.wav) | 6-bit | FR |
| [`fluxforge_short_fr_bf16.wav`](examples/fluxforge_short_fr_bf16.wav) | bf16 | FR |

### Long text samples

| File | Model | Language |
|---|---|---|
| [`fluxforge_long_en_4bit.wav`](examples/fluxforge_long_en_4bit.wav) | 4-bit | EN |
| [`fluxforge_long_en_6bit.wav`](examples/fluxforge_long_en_6bit.wav) | 6-bit | EN |
| [`fluxforge_long_en_bf16.wav`](examples/fluxforge_long_en_bf16.wav) | bf16 | EN |
| [`fluxforge_long_fr_4bit.wav`](examples/fluxforge_long_fr_4bit.wav) | 4-bit | FR |
| [`fluxforge_long_fr_6bit.wav`](examples/fluxforge_long_fr_6bit.wav) | 6-bit | FR |
| [`fluxforge_long_fr_bf16.wav`](examples/fluxforge_long_fr_bf16.wav) | bf16 | FR |

## Reproducing

```bash
# Build
xcodebuild -scheme VoxtralCLI -configuration Release \
  -derivedDataPath .build/xcode -destination 'platform=macOS' build

# Download models
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-mlx
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-4bit
.build/xcode/Build/Products/Release/VoxtralCLI download tts-4b-6bit

# Short text
.build/xcode/Build/Products/Release/VoxtralCLI tts \
  "Fluxforge Studio transforme votre Mac en un studio de création IA complet." \
  -o output.wav --voice fr_female --model tts-4b-6bit

# Long text (pass the full multi-paragraph text)
.build/xcode/Build/Products/Release/VoxtralCLI tts "$(cat long_text.txt)" \
  -o output.wav --voice fr_female --model tts-4b-6bit

# Disable sanitization for raw text control
.build/xcode/Build/Products/Release/VoxtralCLI tts "YOUR TEXT" \
  -o output.wav --voice fr_female --model tts-4b-6bit --no-sanitize
```

## Full test texts

### Long FR

> Fluxforge Studio transforme votre Mac en un studio de creation IA complet. Generez des images et des videos de haute qualite a partir de texte, entrainez vos propres modeles personnalises, et gerez votre bibliotheque creative, le tout en local sur votre Apple Silicon, sans cloud ni abonnement.
>
> FORGE TON IDEE. Un atelier creatif complet pour explorer vos idees visuelles. Decrivez votre concept en texte, importez une image ou un audio, puis iterez librement : variations, changements de style, animations video. Chaque etape est sauvegardee dans un arbre de branches facon Git, rien ne se perd, tout se retrouve.
>
> GENERATION D'IMAGES AVANCEE. Quatre modeles Flux 2 au choix selon vos besoins. GENERATION VIDEO. Creez des videos a partir de texte ou d'images grace au modele LTX-2.3. ANIMATION. Animez n'importe quelle image generee en un clic. EXTENSION DE CANVAS. Etendez vos images au-dela de leurs bordures grace a l'outpainting. ADAPTATEURS LoRA. Importez des adaptateurs LoRA depuis HuggingFace en un clic. ENTRAINEMENT LoRA. Creez vos propres adaptateurs personnalises directement dans l'app. SUPPRESSION D'ARRIERE-PLAN. Retirez l'arriere-plan de vos images en un clic. BIBLIOTHEQUE ET EXPORT. Retrouvez toutes vos creations dans une galerie organisee. 100% LOCAL ET PRIVE. Aucun compte requis. Aucune donnee envoyee dans le cloud.

### Long EN

> Fluxforge Studio turns your Mac into a complete AI creative studio. Generate high-quality images and videos from text, train your own custom models, and manage your creative library, all locally on your Apple Silicon, with no cloud or subscription required.
>
> FORGE YOUR IDEA. A full creative workshop for exploring your visual ideas. Every step is saved in a Git-style branch tree, nothing is lost, everything is recoverable.
>
> ADVANCED IMAGE GENERATION. Four Flux 2 models to choose from. VIDEO GENERATION. Create videos from text or images using the LTX-2.3 model. ANIMATION. Animate any generated image in one click. CANVAS EXTENSION. Extend your images beyond their borders with outpainting. LoRA ADAPTERS. Import LoRA adapters from HuggingFace in one click. LoRA TRAINING. Create your own custom adapters directly in the app. BACKGROUND REMOVAL. Remove the background from your images in one click. LIBRARY AND EXPORT. Find all your creations in an organized gallery. 100% LOCAL AND PRIVATE. No account required. No data sent to the cloud.
