# Voxtral TTS Streaming Demo — App Guide

`VoxtralTTSStreamingDemo` is a small macOS SwiftUI app that drives the
Voxtral TTS pipeline: streaming synthesis with live metrics, plus an
end-to-end **voice cloning** workflow (enroll a voice, then speak with it).

Build & run:

```bash
xcodebuild -scheme VoxtralTTSStreamingDemo -configuration Release \
  -derivedDataPath .build/xcode -destination 'platform=macOS' build
.build/xcode/Build/Products/Release/VoxtralTTSStreamingDemo
```

## Top controls

| Control | What it does |
|---|---|
| **Model** | TTS model to load: `4-bit (2.5 GB)`, `6-bit (3.5 GB)`, `bf16 original (8 GB)` (mistralai), `bf16 MLX (8 GB)` (mlx-community). |
| **Load** | Downloads (if needed) and loads the selected model. Complete models load instantly from `~/Library/Caches/models`. |
| **Voice** | The 20 built-in presets **plus** any enrolled (cloned) voices, shown with a 🎙️ prefix. |
| **Sanitize** | Prosody-aware text normalization (ALL-CAPS expansion, auto-punctuation) for more natural pauses. |
| **Presets** | Quick-fill the text box with short/long EN/FR samples. |
| **Text editor** | The text to synthesize. |
| **Play Streaming / Stop** | Streams synthesized audio chunk-by-chunk to the speakers as they are generated. |

### Metrics row

Measured for each synthesis: **TTFT** (time to first token/audio), **Total**
time, **Audio** duration, **RTF** (real-time factor — <1 is faster than
real time), **FPS** (frames/s), **Frames**, **Chunks**. A console below logs
every step (and the enrollment loss when cloning).

## Voice Cloning panel

Clone a voice from a reference, then use it like any preset. Three ways to
provide the reference:

| Button | Source |
|---|---|
| **Audio…** | An existing audio file (wav/mp3/m4a/…). |
| **Build from video…** | Extract from a video/audio file — including `.webm`, which AVFoundation can't read — via ffmpeg (see below). |
| **Record…** | Record yourself reading a prompt from the microphone (see below). |

Then set:

- **Name** — the cloned voice's name (and output filename).
- **Ref (s)** — reference length to use (4–24 s; ~16 s is the sweet spot).
- **Epochs** — optimization epochs (default 3000; more helps quality, ~10–25 min).
- **Enroll** — runs the optimization off the main thread with live
  `epoch/loss` progress. On completion the voice is saved and appears in the
  voice picker.

Enrolled voices are written to
`~/Library/Application Support/VoxtralClonedVoices/<name>.safetensors` and
reloaded on launch. The enrollment uses the **currently loaded model**, so a
cloned voice is always synthesized with the decoder it was optimized against.

### Build from video (ffmpeg)

Opens a builder to assemble a reference from one or more extracts:

1. **Choose video/audio…** — pick the source (any format ffmpeg reads).
2. **Start / End sliders** — select an extract; the length is shown live.
3. **Preview** — play the selected range.
4. **Add extract** — add it to the list. Repeat to combine several extracts.
5. The running total is shown against the target length; when it reaches the
   target, **Use as reference** concatenates the extracts into one 24 kHz
   mono reference.

Requires a local `ffmpeg` (`brew install ffmpeg`). macOS-only; the
enrollment core itself stays pure Swift.

### Record from microphone

1. Pick a **prompt** (EN/FR, sized to ~the target length) and read it aloud.
2. **Record** — a live timer shows elapsed time vs. the target.
3. **Stop** (enabled once you pass the target) — the recording (24 kHz mono)
   becomes the reference.

Requires microphone permission (macOS will prompt on first use). Record in a
quiet room at a natural pace, aiming slightly past the target length — the
reference is trimmed to end on a natural pause.

## Notes

- Cloned-voice quality: clearly recognizable but slightly hazier than the
  official presets (~0.6–0.7 speaker similarity vs. ~0.84 for presets) — an
  inherent limit of the encoder-less workaround. See
  [voice_cloning.md](voice_cloning.md).
- TTS weights are CC BY-NC 4.0 (non-commercial); clone only voices you have
  the right to use, with the speaker's consent.
