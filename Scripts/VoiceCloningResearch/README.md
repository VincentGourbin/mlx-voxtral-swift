# Voice Cloning Research — Voxtral TTS (Python annex)

> **This is the research annex.** The supported, production path is the
> native Swift command `voxtral enroll` — see
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

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

git clone https://github.com/MarvinRomson/voxtral-tts-codes-for-audio.git upstream
git -C upstream apply ../patches/upstream_fixes.patch   # REQUIRED, see below

.venv/bin/hf download mistralai/Voxtral-4B-TTS-2603 --local-dir voxtral-tts-weights
```

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

**Quality ceiling (current):** speaker similarity (ECAPA cosine) on
unseen text ≈ 0.56–0.59 vs the reference, where official presets reach
0.837 on theirs and cross-speaker baseline is 0.045. The voice is
clearly recognizable but noticeably "hazier" than presets: the
gradient-descent codes reproduce the reference waveform but are slightly
off-manifold as LLM conditioning compared to true encoder outputs.
Untested levers: longer reference (presets go up to 218 frames ≈ 17 s vs
our 100), 15000 epochs, code regularization toward preset statistics.

## Repository hygiene

Everything heavy or voice-derived stays local: `.gitignore` here blocks
weights, checkpoints, embeddings (`*.pt`, `*.safetensors`) and audio
(`*.wav`). Only scripts, patches and this README are committed.

**License note:** the TTS weights are CC BY-NC 4.0 (non-commercial).
Voice cloning requires the speaker's consent — enroll only voices you
have the right to use.

## Next step

Swift/MLX port of the enrollment loop (differentiable decoder forward +
losses in MLX, which also sidesteps the PyTorch MPS bug) so enrollment
can run in-app without a Python toolchain.
