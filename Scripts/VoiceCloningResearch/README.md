# Voice Cloning Research — Voxtral TTS (Python annex)

> **This is the research annex.** The supported, production path is the
> native Swift command `VoxtralCLI enroll` — see
> [`docs/voice_cloning.md`](../../docs/voice_cloning.md). These Python
> scripts are the original proof-of-concept, kept for reference and
> reproducibility; they require a separate PyTorch/speechbrain toolchain
> and are **not** needed to use voice cloning.

Offline voice enrollment for Voxtral TTS, working around the fact that
Mistral never published the codec **encoder** weights of
`mistralai/Voxtral-4B-TTS-2603` (decoder only — voice cloning from
reference audio is officially unsupported, [HF discussion #17](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/discussions/17)).

The workaround (from [MarvinRomson/voxtral-tts-codes-for-audio](https://github.com/MarvinRomson/voxtral-tts-codes-for-audio)):
optimize the discrete codec codes `[T, 37]` directly by gradient descent
through the **frozen decoder** (Gumbel-Softmax + straight-through
estimators, multi-resolution STFT + mel + MFCC + speaker losses), then
convert the codes to a voice embedding `[T+1, 3072]` compatible with the
preset system. The embedding plugs into the existing Swift pipeline via
`VoxtralTTSPipeline.synthesize(text:voiceEmbedding:)` / `VoxtralCLI tts
--voice-embedding`.

## Setup

Requires macOS 14 or later on Apple Silicon (the pinned torch 2.12.1 ships `macosx_14_0_arm64` wheels
only), Python 3.12-3.14 and FFmpeg.

```bash
python3 -m venv .venv                      # Python 3.12-3.14, see requirements.txt
.venv/bin/pip install -r requirements.txt  # every version pinned
brew install ffmpeg                        # torchaudio.load -> torchcodec -> system FFmpeg

git clone https://github.com/MarvinRomson/voxtral-tts-codes-for-audio.git upstream
git -C upstream checkout ac3e3f3c17244e4a6811c0168fc2b80bd4b3b332   # pinned, checked by enroll_voice.py
git -C upstream apply ../patches/upstream_fixes.patch   # REQUIRED, see below

.venv/bin/hf download mistralai/Voxtral-4B-TTS-2603 \
    --revision b81be46c3777f88621676791b512bb01dc1cb970 --local-dir voxtral-tts-weights
```

What is pinned, and why:

- **Upstream commit** `ac3e3f3` (2026-04-06, head of `main` on 2026-09-28): the patch was written
  against it (its blob ids `50b812c` and `2ee2720` are those of `audio_tokenizer.py` and
  `training_script.py` at that commit; `git apply --check` passes). `enroll_voice.py` refuses to run
  when `upstream/` is at another commit (`UPSTREAM_COMMIT`, `check_workspace`).
- **TTS weights**: revision `b81be46` of `mistralai/Voxtral-4B-TTS-2603`, whose file listing is
  identical to `main` on 2026-09-28 (Hub, via the Hugging Face connector).
- **Python packages**: `requirements.txt`, one exact version per package, with the reason for each
  pin in the file (transitive dependencies are not locked).

## Enroll a voice

```bash
.venv/bin/python enroll_voice.py --reference my_recording.wav --name my_voice
# ~30 min (5000 epochs) on an M-series Mac; --epochs 15000 for better quality

# Then synthesize any text with it:
VoxtralCLI tts "Any text you want." -o out.wav --model tts-4b \
    --voice-embedding voices/my_voice.safetensors
```

## Findings (June–July 2026)

Three non-obvious pitfalls, all handled by `enroll_voice.py`:

1. **PyTorch MPS `torch.stft` backward is silently wrong on long signals**
   (verified torch 2.12): forward values are correct, but gradients are
   garbage beyond ~2.5 s of a 192 k-sample signal (per-second grad cosine
   vs CPU: `[1.0, 1.0, 0.73, 0, 0, 0, 0, 0]`). All spectral losses were
   noise during optimization → speaker timbre captured but garbled
   content after 2.5 s. `patches/upstream_fixes.patch` computes the
   spectral losses on CPU (autograd carries gradients across devices).
   Result: reconstruction mel distance 3.10 → 0.69.

2. **The END_AUDIO terminator frame is required.** All 20 official
   presets end with the exact END_AUDIO embedding frame (cosine 1.0000).
   Without it (`codes_to_embeddings.py` default!), generation starts as
   if mid-stream: first seconds of every synthesis are degraded
   (speaker similarity 0.43 on the first half vs 0.65 on the second).
   Always pass `--add-end-token`.

3. **The reference must end on a pause.** A reference cut mid-speech at
   full volume conditions the model on an interrupted utterance.
   `enroll_voice.py` cuts at the quietest window near the end, fades,
   and pads with silence.

**Quality ceiling of this Python path (June–July 2026):** speaker
similarity (ECAPA cosine) on unseen text ≈ 0.56–0.59 vs the reference,
with the default 100-frame (8 s) reference, where official presets reach
0.837 on theirs and cross-speaker baseline is 0.045. The voice is
clearly recognizable but noticeably "hazier" than presets: the
gradient-descent codes reproduce the reference waveform but are slightly
off-manifold as LLM conditioning compared to true encoder outputs.
Levers not tried on this path: longer reference (presets go up to 218
frames ≈ 17 s vs our 100), 15000 epochs, code regularization toward
preset statistics. The Swift path's figures are in the next section.
In session: the number of speakers, the epochs and the code revision of
these runs were not recorded.

## Python and Swift paths

The native Swift command `VoxtralCLI enroll` (`VoxtralVoiceEnrollment.swift`,
commit `bd59931`) ports this loop, but not with the same objective:

| | Python annex (`enroll_voice.py` → `upstream/training_script.py` at `ac3e3f3`) | Swift (`VoxtralCLI enroll`) |
|---|---|---|
| Loss | 0.5 L1 + 1 multi-resolution STFT + 1 log-mel + 1 MFCC + 0.5 speaker loss 1 − cos(ECAPA), gradients through ECAPA (`enroll_voice.py:171-172`; `training_script.py:80-82`, `:613-633`) | 0.5 L1 + 1 multi-resolution STFT + 1 log-mel (`VoxtralVoiceEnrollment.swift:43-45`, `:556-568`) |
| Gumbel temperature | 1.0, ×0.995 per epoch, floor 0.5 (`training_script.py:770-775`) | 2.0, ×0.99, floor 0.3 (`VoxtralVoiceEnrollment.swift:46-48`) |
| Spectral gradients | on CPU (MPS `torch.stft` backward bug, patch) | MLX, whole signal |
| Published similarity | 0.56–0.59 at 8 s (above) | 0.69 at 8 s, 0.72 at 16 s, 2000 epochs (`docs/voice_cloning.md:41-53`, commit `ca49c7a`, in session) |

These figures do not rank the two loss sets: the runs differ in reference
duration, epochs, temperature schedule and code revision (audit
2026-09-27, A-21, which rejects "0.72 vs 0.56–0.59" as a confounded
comparison). A comparison needs the same reference, the same duration and
the same epochs on both paths, scored with the command below.

## Speaker similarity (ECAPA)

The similarities above are ECAPA speaker-embedding cosines between the
reference and a synthesis of unseen text; the script that produced them
was not committed. To score a synthesis the same way (SpeechBrain's
`SpeakerRecognition.verify_files`: cosine of the two
`speechbrain/spkrec-ecapa-voxceleb` embeddings, audio resampled to
16 kHz):

```bash
.venv/bin/python - my_recording.wav test.wav <<'PY'
import sys
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import FetchConfig

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="checkpoints/ecapa",
    fetch_config=FetchConfig(revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"),
)
score, _ = model.verify_files(sys.argv[1], sys.argv[2])
print(f"ECAPA cosine: {score.item():.3f}")
PY
```

Revision `0f99f2d` of the ECAPA model: file listing identical to `main`
on 2026-09-28 (Hub). Score the reference against itself (≈ 1) and against
another speaker (low) before reading a result.

## Repository hygiene

Everything heavy or voice-derived stays local: `.gitignore` here blocks
weights, checkpoints, embeddings (`*.pt`, `*.safetensors`) and audio
(`*.wav`). Only scripts, patches and this README are committed.

**License note:** the TTS weights are CC BY-NC 4.0 (non-commercial).
Voice cloning requires the speaker's consent — enroll only voices you
have the right to use.

## Swift/MLX port (done)

The enrollment loop was ported to Swift/MLX in `bd59931` (merged with
PR #34): differentiable decoder forward and losses in MLX, which also
sidesteps the PyTorch MPS bug, so enrollment runs without a Python
toolchain. Use it through `VoxtralCLI enroll`: see
[`docs/voice_cloning.md`](../../docs/voice_cloning.md). Its objective
differs from this annex: see "Python and Swift paths" above.
