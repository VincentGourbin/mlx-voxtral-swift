# Voice Cloning

Clone a voice from a short reference recording and use it to synthesize
any text, entirely in Swift on Apple Silicon — no Python at runtime.

Voxtral TTS ships 20 fixed voice presets. Voice cloning lets you add your
own: you "enroll" a voice once from a reference clip, which produces a
voice embedding you can reuse forever with the normal TTS pipeline.

> **Why this exists.** Mistral never published the Voxtral codec *encoder*
> weights, so there is no official way to turn reference audio into a voice
> embedding. This feature recovers the voice codes by optimizing them
> directly through the frozen *decoder* (gradient descent + straight-through
> estimators). See [Background](#background) below.

## Quick start

```bash
# 1. Enroll a voice from a reference recording (offline, one time)
voxtral enroll my_voice.wav -o my_voice.safetensors --model tts-4b --duration 16

# 2. Make that voice say anything
voxtral tts "Hello, this is my cloned voice." \
    -o hello.wav --model tts-4b --voice-embedding my_voice.safetensors
```

The enrolled `.safetensors` is a `[T+1, 3072]` voice embedding. It is
interchangeable with the built-in presets everywhere the pipeline accepts
a voice embedding (`--voice-embedding`, or
`VoxtralTTSPipeline.synthesize(text:voiceEmbedding:)` in code).

## The reference recording

Quality depends far more on the reference than on anything else.

| Reference length | Speaker similarity* | Verdict |
|---|---:|---|
| 4 s  | 0.67 | usable minimum |
| 8 s  | 0.69 | ok |
| **16 s** | **0.72** | **recommended sweet spot** |
| 24 s | 0.72 | no further gain |

<sub>*ECAPA speaker-embedding cosine of synthesized unseen text vs. the
reference speaker. Same source, same text, 2000 epochs, only length
varies. Cross-speaker baseline ≈ 0.05; official presets ≈ 0.84.</sub>

Practical guidance:

- **Aim for 10–16 s** of clean speech. Below 16 s quality climbs steadily;
  beyond it, returns flatten (the code capacity is `duration × 12.5`
  frames, and ~200 frames is enough to capture a voice).
- **One speaker, clean audio.** No background music, minimal noise, no
  overlapping voices.
- **Any format** (wav/mp3/m4a/flac) — it is resampled to 24 kHz mono.
- The reference must be **at least** `--duration` seconds long. Enrollment
  automatically trims to the last natural pause, fades, and pads with
  silence, so a reference that ends mid-word is handled gracefully — but a
  clip that *starts* clean (no long intro) gives the best result.

## `voxtral enroll` options

| Option | Default | Meaning |
|---|---|---|
| `<reference>` | — | Reference audio file (required) |
| `-o, --output` | `voice.safetensors` | Output voice embedding path (must end in `.safetensors`) |
| `-m, --model` | `tts-4b-mlx` | TTS model (matches the `tts` command's default so the voice is synthesized through the same weights) |
| `-e, --epochs` | `5000` | Optimization epochs (5000 good, more helps slightly) |
| `--duration` | `16.0` | Reference seconds to use (min 2 s; ~16 s is the sweet spot) |

> Use the **same `--model`** for `enroll` and `tts` — a voice embedding is
> tied to the decoder/embedding table it was optimized against. The defaults
> already match (`tts-4b-mlx`).

Enrollment is offline and one-time per voice. On an unloaded M-series GPU
it runs at roughly 15× the speed of the original PyTorch reference.

## Using a cloned voice in code

```swift
let pipeline = VoxtralTTSPipeline()
try await pipeline.loadModel(modelInfo: VoxtralTTSRegistry.model(withId: "tts-4b")!)

// Enroll (offline)
try pipeline.enrollVoice(
    referenceURL: URL(fileURLWithPath: "my_voice.wav"),
    outputURL: URL(fileURLWithPath: "my_voice.safetensors"),
    config: { var c = VoxtralVoiceEnrollment.Config(); c.numFrames = 200; return c }()
)

// Later: synthesize with it
let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: "my_voice.safetensors"))
let result = try await pipeline.synthesize(text: "Any text.", voiceEmbedding: arrays["embedding"]!)
```

## Examples (public-domain voices)

Two voices cloned from LibriVox recordings (US public domain), each made to
speak a sentence it never uttered. Audio in [`docs/examples/`](examples/).

| Voice source | Reference | Test phrase | Fidelity* | Output |
|---|---|---|---:|---|
| **English** — Annie Coleman reading *Pride and Prejudice* ([LibriVox](https://archive.org/details/solo_pride_librivox)) | 16 s | "Voxtral is a text to speech model, and we've just added the ability to clone a voice." | 0.67 | [`clone_en.wav`](examples/clone_en.wav) |
| **French** — Isabelle Brasme reading *La Princesse de Clèves* ([LibriVox](https://archive.org/details/la_princesse_de_cleves_1402_librivox)) | 16 s | "Voxtral est un modèle de synthèse vocale, et nous venons d'ajouter la possibilité de cloner une voix." | 0.74 | [`clone_fr.wav`](examples/clone_fr.wav) |

<sub>*ECAPA speaker cosine of the synthesized phrase vs. the reference
speaker (3000 epochs). Cross-checks are near zero — each clone matches its
own target speaker, not the other.</sub>

Reproduce:

```bash
# Download a reference (public domain), take a clean 16s+ segment, then:
voxtral enroll en_ref.wav -o en_voice.safetensors --model tts-4b --duration 16 --epochs 3000
voxtral tts "Voxtral is a text to speech model, and we've just added the ability to clone a voice." \
    -o clone_en.wav --model tts-4b --voice-embedding en_voice.safetensors
```

## Background

The enrollment loop is a native Swift/MLX port of the community workaround
([MarvinRomson/voxtral-tts-codes-for-audio](https://github.com/MarvinRomson/voxtral-tts-codes-for-audio)):

1. Relax the discrete codec codes `[T, 37]` into learnable parameters
   (Gumbel-Softmax for the semantic code, tanh/FSQ for the 36 acoustic
   codes), with straight-through estimators so the forward pass stays
   discrete while gradients flow.
2. Decode them to a waveform through the **frozen** codec decoder and
   minimize reconstruction losses against the reference (L1 + multi-
   resolution STFT + log-mel).
3. Convert the trained codes to a voice embedding (per-codebook table
   lookup + sum, plus the required `END_AUDIO` terminator frame).

Implementation: `Sources/VoxtralCore/TTS/VoiceCloning/`.

**Quality ceiling.** Cloned voices reach ~0.72 speaker similarity on
unseen text vs. ~0.84 for the official presets. They are clearly
recognizable but slightly "hazier": gradient-descent codes reproduce the
reference but sit a little off-manifold as LLM conditioning compared to
true encoder outputs. This is inherent to the workaround, not the port.

**Notes.**
- The MLX losses compute correct gradients across the whole signal; the
  original PyTorch pipeline silently corrupted them past ~2.5 s on MPS
  (a `torch.stft` backward bug), which the Swift port sidesteps.
- Voxtral TTS weights are **CC BY-NC 4.0** (non-commercial). Only clone
  voices you have the right to use, and with the speaker's consent.

## Research annex (Python)

The original Python proof-of-concept and the empirical notes that led to
this feature live in
[`Scripts/VoiceCloningResearch/`](../Scripts/VoiceCloningResearch/). It is
kept for reference and reproducibility only — **the supported path is the
native Swift `voxtral enroll` command above.** The Python scripts require a
separate toolchain (PyTorch, speechbrain) and are not needed to use voice
cloning.
