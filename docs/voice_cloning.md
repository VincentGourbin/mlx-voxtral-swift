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
- **Record at a healthy level.** The voice continues the reference's
  loudness, and the normalization below is peak-guarded, so a very quiet take
  cannot be fully rescued. Measured on the same speaker: a −30 dB take gave
  −27 dB syntheses, a −23 dB take gave −22 dB (preset level is ≈ −24 dB).
  Aim for peaks around −6 dBFS with no clipping.
- **The reference is cleaned before optimization**, in this order:
  1. a **70 Hz high-pass** (zero-phase Butterworth) removes rumble and DC;
  2. **loudness normalization** brings active speech to
     `--reference-target-rms-db` (−20 dBFS), capped so no sample exceeds 0.98;
  3. a **noise gate** attenuates windows more than 30 dB below the loudest
     20 ms window by 24 dB — it *attenuates*, it does not zero them.

  Whatever is in the reference — noise floor included — is learned as part of
  the voice. The gate deliberately leaves a low floor rather than digital
  silence: a hard-zeroed reference teaches the voice that chopped-up style and
  reproduces it as micro-gaps in every synthesis. To keep the low end or the
  ambience as recorded, disable with `--high-pass-hz 0` / `--no-gate` (or
  `Config.referenceHighPassHz = nil` / `Config.gateReference = false`).

  > **Low-pitched voices: consider `--high-pass-hz 50`.** The 70 Hz default
  > still costs 2.7 dB at 90 Hz, and a deep male voice sits right there —
  > measured on one speaker, preparation alone took the fundamental from
  > +0.5 dB to −2.1 dB (relative to the second harmonic). Dropping the corner
  > to 50 Hz costs only 0.8 dB at 90 Hz, in exchange for 19 dB of rumble
  > rejection at 30 Hz instead of 30 dB — worth it if the room is quiet.
  > Leave the default for ordinary and higher-pitched voices.

  > The high-pass used to be a 64-tap complementary FIR, which at 24 kHz
  > cannot resolve a 70 Hz corner: it attenuated a male fundamental
  > (100–120 Hz) by 24–27 dB, and since the embedding is a prefix the model
  > continues, every synthesis came out thin. Voices enrolled before this fix
  > are worth re-enrolling — measured on one speaker, re-enrolling the *same*
  > recording recovered 6.4 dB of fundamental and moved the synthesized pitch
  > from a harmonic (144 Hz) back onto the speaker's own (88 Hz vs 97 Hz real).

## `voxtral enroll` options

| Option | Default | Meaning |
|---|---|---|
| `<reference>` | — | Reference audio file (required) |
| `-o, --output` | `voice.safetensors` | Output voice embedding path (must end in `.safetensors`) |
| `-m, --model` | `tts-4b-mlx` | TTS model (matches the `tts` command's default so the voice is synthesized through the same weights) |
| `-e, --epochs` | `5000` | Optimization epochs (5000 good, more helps slightly) |
| `--duration` | `16.0` | Reference seconds to use (min 2 s; ~16 s is the sweet spot) |
| `--no-gate` | off | Keep the reference's noise floor (disable the silence gate) |
| `--gate-threshold-db` | `-30` | Gate threshold in dB relative to the loudest 20 ms window |
| `--high-pass-hz` | `70` | Reference high-pass cutoff in Hz (`0` disables) |
| `--reference-target-rms-db` | `-20` | Target active-speech RMS for the reference, in dBFS (`0` disables normalization) |

> Negative values need `=`: `--reference-target-rms-db=-26`. Without it the
> parser reads `-26` as another option.

> Use the **same `--model`** for `enroll` and `tts` — a voice embedding is
> tied to the decoder/embedding table it was optimized against. The defaults
> already match (`tts-4b-mlx`).

### Quantized models with a cloned voice — evaluate for yourself

The defaults are bf16 because it is the safe reference, not because quantized
models were found wanting. If synthesis speed or memory matters to you, measure
the quantized variants **on your own voice and language** before deciding —
results here are a single data point, not a recommendation:

| observed on one speaker | `tts-4b-6bit` | `tts-4b-mlx` (bf16) |
|---|---|---|
| word coverage (ASR-scored) | 99.4% | 96.5% |
| real-time factor | 1.47 | 3.44 |
| warm-up vocalise leaked | 2–3 / 15 | 0 / 15 |

*Sample: one enrolled French male voice (~90 Hz fundamental), 3 ordinary
sentences × 5 seeds per model, transcribed with `mini-3b-8bit` and scored on
word overlap. Small, single-speaker, single-language — quantization interacts
with the voice and the text, so it may not transfer to yours.*

What this does rule out is the earlier assumption that a quantized model drops
words with enrolled voices: on this sample q6 dropped **fewer** than bf16. That
assumption came from an observation made before the enrollment high-pass was
fixed, when embeddings were measurably thinner and plausibly more fragile.

Reproduce it on your own voice with `TTSQuantizationCampaignTests`:

```bash
TEST_RUNNER_VOXTRAL_TTS_CAMPAIGN=1 \
TEST_RUNNER_VOXTRAL_TTS_REPRO_EMB=/path/to/your_voice.safetensors \
xcodebuild test-without-building -scheme MLXVoxtralSwift-Package \
  -destination 'platform=macOS' \
  -only-testing:VoxtralCoreTests/TTSQuantizationCampaignTests
```

Enrollment is offline and one-time per voice. On an unloaded M-series GPU
it runs at roughly 15× the speed of the original PyTorch reference.

Enrollment refuses to save a voice whose optimization diverged to a
non-finite loss: it falls back to the best finite step, and if there was none
it throws rather than write a NaN embedding (a NaN prefix is continued as
runaway babble to the frame cap).

## Synthesizing with a cloned voice: seed and warm-up

Two `voxtral tts` options matter for enrolled voices specifically.

| Option | Default | Meaning |
|---|---|---|
| `--seed` | random | RNG seed. **Without it, output varies run to run** |
| `--warm-up` | off | Prepend a throwaway vocalise, then trim it back off |

**`--seed` — reproducibility.** The acoustic step samples flow-matching noise,
and those samples feed back into the autoregressive state, so the same text and
voice give a different take (different wording quality, pacing, even length)
on every call — at temperature 0. Pass a seed whenever you need to compare two
things, or to be able to reproduce a take you liked.

**`--warm-up` — first-word quality.** Enrolled-voice codes are optimized to
reconstruct audio, not to be a plausible autoregressive context, so the opening
of a generation is unstable. Prepending a short vocalise and cutting it back
off measurably improves the first word: over three seeds on one voice, the
first word came out `"Flux Forge"` with the warm-up versus `"Sorche"` /
`"Loxforge"` without.

```bash
voxtral tts "Fluxforge Studio transforme votre Mac." \
    -o out.wav --model tts-4b-mlx --voice-embedding my_voice.safetensors \
    --seed 7 --warm-up
```

> **Check the output.** The carrier cut is a heuristic — the pause the model
> leaves after the vocalise has no fixed depth (measured −55 dB, −65 dB and
> −126 dB across three seeds on one voice), so roughly one generation in eight
> either leaks the vocalise or clips the opening. Enrolled voices also
> occasionally hallucinate a preamble. For unattended use, transcribe the
> result and check it against the input text — `voxtral transcribe out.wav
> --model mini-3b-8bit --language fr` — and regenerate with another seed if it
> does not match. That check is what filtered 7 bad takes out of 12 in testing.

## Demo app

The `VoxtralTTSStreamingDemo` app (macOS) has a **Voice Cloning** panel:
pick a reference recording, name it, set the reference length/epochs, and
click **Enroll** — progress (epoch/loss) streams live while the run happens
off the main thread. The enrolled voice then appears in the voice picker
alongside the presets and streams like any other voice. Enrolled voices are
saved under `Application Support/VoxtralClonedVoices/` and reloaded on launch.

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
