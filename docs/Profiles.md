# Reference profiles (v0, provisional)

A profile fixes **every** setting that matters for one pipeline: weights, backend, memory policy, decoding settings.
Same profile + same seed = same output, comparable time on comparable hardware. Naming follows the Gemma4 standard:
`<family>/<bits>bit-<fast|lean>`.

- `fast`: everything resident, generous MLX buffer cache, no clearing between calls.
- `lean`: tight buffer cache, cleared after each call, smallest pack that keeps quality; for 16–32 GB Macs and for
  chains of models (FluxForge: Voxtral then LTX).

**Status (2026-10-02).** The profiles are **not in the code yet**: the `Voxtral…ReferenceProfile` types come with fiche
K-76 (lot 5), the full matrix is measured by K-77 to K-79 (lot 6). Until then, a profile is applied by hand with the
existing settings below. Values marked *measured* come from one Release `VoxtralCLI bench` pass on an **M3 Max 96 GB**
(commit `32c400c6`, mlx-swift 0.31.6, mlx-swift-lm `main@604fae7`, raw `BENCH` lines in [`BENCHMARKS.md`](../BENCHMARKS.md),
2026-10-02, background load present): they are **not** the reference baselines (K-33 to K-37, A/A ≤ 3 %), which wait
for the cloud verification of lots 1–2. Quality is compared between packs (same text or not), not scored yet (WER: K-33).

Glossary: **RTF** = processing time ÷ audio duration (< 1 = faster than real time). **TTFA** = time to the first
audio. **Peak** = highest process footprint, MLX buffer cache included.

## Recommended now

| Mac | Transcription | Live transcription | Speech synthesis | Voice cloning |
|---|---|---|---|---|
| 16 GB | `mini/4bit-lean` | `realtime/4bit-lean` | `tts/4bit-lean` | `tts/6bit-lean` (8 s reference) |
| 32 GB | `mini/8bit-fast` | `realtime/4bit-fast` | `tts/6bit-fast` | `tts/6bit-fast` |
| ≥ 48 GB | `mini/8bit-fast` (`small/8bit-*` if K-34 shows a WER gain) | `realtime/4bit-fast` | `tts/6bit-fast` | `tts/6bit-fast` |

The 16-bit profiles are **not recommended** today: the 16-bit packs compute in fp32 (STT 4.8 × slower than 8-bit for
the same text; Realtime and TTS slower than real time). Fiches K-38 and K-40 address it. The TTS default of the
registry is still bf16 (`tts-4b-mlx`): switching it to 6-bit is ASK-5.

## Profiles

### `mini` — STT Voxtral Mini 3B 2507 (Apache-2.0)

Measured on C-moyen EN (167 s of audio), backend `.mlx`, `maxTokens` nil (sized from the duration).

| Profile | Weights (Hub · GB) | Total | RTF | First token | ms/token | Peak | Text vs bf16 |
|---|---|---|---|---|---|---|---|
| `mini/4bit-fast` | `mzbac/voxtral-mini-3b-4bit-mixed` · 3.20 | 12.8 s | 0.077 | 4.7 s | 15.8 | 9.3 GB | differs (−5 % chars) |
| `mini/4bit-lean` | same | to measure | | | | | |
| `mini/8bit-fast` | `mzbac/voxtral-mini-3b-8bit` · 5.40 | 16.2 s | 0.097 | 4.7 s | 22.0 | 10.9 GB | **identical** |
| `mini/8bit-lean` | same | to measure | | | | | |
| `mini/16bit-fast` | `mistralai/Voxtral-Mini-3B-2507` · 9.36 | 77.1 s | 0.46 | 4.8 s | 139.1 | 16.8 GB | reference |
| `mini/16bit-lean` | same | to measure | | | | | |

- Long audio (C-long, 11 min 21 s, `mini/8bit`): without a cache limit the peak reaches **37.6 GB**; with
  `cacheLimitBytes` = 2 GB, **10.3 GB**, same text, no time cost (K-52, measured).
- Candidate weights: `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` (6.02 GB, bf16 encoder) loads since
  K-8 and transcribes C-court EN exactly like `mlx-voxtral` (Python); external WER 4.27 % vs 4.74 % for the uniform
  8-bit. Using a third-party repository in a profile is ASK-16.
- Chat (C-court EN, one question, greedy): `mini/8bit` 0.93 s to first token, 49 tok/s, 7.7 GB; `mini/4bit` 0.86 s,
  71 tok/s, 6.0 GB.

### `small` — STT Voxtral Small 24B 2507 (Apache-2.0)

Not measured on this machine (packs not downloaded); K-34 gives the verdict for 32 GB Macs.

| Profile | Weights (Hub · GB) | Machine | Status |
|---|---|---|---|
| `small/4bit-*` | `VincentGOURBIN/voxtral-small-4bit-mixed` · 14.86 (or Markus 4-bit dense encoder · 15.02, loads since K-8) | 32 GB (`lean`, to confirm: issue #21 saw a 22 GB peak) | to measure |
| `small/8bit-*` | `VincentGOURBIN/voxtral-small-8bit` · 26.50 (ASK-15) | ≥ 48 GB | to measure |
| `small/16bit-*` | `mistralai/Voxtral-Small-24B-2507` · 48.53 | ≥ 64 GB | to measure |

### `realtime` — Voxtral Mini 4B Realtime 2602 (Apache-2.0)

Measured on C-moyen EN (167 s), transcription delay 480 ms (Mistral's recommendation). Budget per frame: 80 ms.

| Profile | Weights (Hub · GB) | Total | RTF | ms per frame | Peak | Status |
|---|---|---|---|---|---|---|
| `realtime/4bit-fast` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` · 3.13 | 61.7 s | 0.37 | 27.8 | 9.8 GB | measured |
| `realtime/4bit-lean` | same | to measure | | | | |
| `realtime/8bit-*` | **none loadable** → pack PK-1 to publish (4.73 GB) | | | | | ASK-20, ASK-22 |
| `realtime/16bit-fast` | `mlx-community/…-2602-fp16` · 8.87 (or `mistralai/…` bf16 · 8.86, loads since K-9, same text) | 293.6 s | 1.76 | 137.9 | 16.4 GB | **slower than real time** (fp32, K-60) |
| `realtime/16bit-lean` | same | to measure | | | | |

72 % of the decode steps emit no text (`pad_fraction`): the room fiche K-73 would use.

### `tts` — Voxtral 4B TTS 2603 (**CC BY-NC 4.0**)

Non-commercial use only. FluxForge is free and will stay free (Vincent, ASK-18 = A, 2026-10-02), so these profiles are
recommended for it; a paid or monetized host needs a commercial licence from Mistral.

Measured with `neutral_female`, seed 42; short = 13 words, long = 163 words. The TTS exists in 4, 6 and 16 bits;
6-bit is declared as an intermediate width (ASK-19).

| Profile | Weights (Hub · GB) | Text | Audio | RTF | TTFA | Peak |
|---|---|---|---|---|---|---|
| `tts/4bit-fast` | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` · 2.51 | short / long | 7.4 / 74.1 s | 0.63 / 0.46 | 0.23 / 0.38 s | 4.3 / 12.9 GB |
| `tts/6bit-fast` | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` · 3.47 | short / long | 7.8 / 77.3 s | 0.65 / 0.54 | 0.25 / 0.40 s | 5.2 / 14.1 GB |
| `tts/8bit-*` | **none valid** → pack PK-2 to publish (4.37 GB) | | | | | ASK-19, ASK-22 |
| `tts/16bit-fast` | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` · 8.00 (registry default) | short / long | 6.5 / 83.9 s | **1.81 / 1.85** | 0.39 / 0.51 s | 10.2 / 20.6 GB |
| `tts/*-lean` | same packs | to measure | | | | |

- Streaming a long text is much slower than batch today (each chunk re-decodes the audio so far: 495 s for 182 s of
  audio in 4-bit, K-43): stream short texts only.
- Voice cloning (`tts/6bit`): 5 000 epochs ≈ 7.7 min for an 8 s reference; reproducible with a seed and resumable from
  a checkpoint (K-26).

## What each setting does (existing settings, v0)

| Setting | `fast` | `lean` | Status |
|---|---|---|---|
| STT backend | `.mlx` | `.mlx` | measured: Core ML (`.auto`) is 1.1–1.9 × slower on these clips (K-32b); the rule of K-42 decides (ASK-6) |
| `cacheLimitBytes` (STT, TTS, Realtime) | 4 GB | 2 GB | 2 GB measured on STT C-long (K-52); 4 GB to measure |
| Clear the MLX cache after each call | no | yes (`unload()` between models of a chain) | to measure |
| STT `maxTokens` | nil (6 tokens/s of audio + 64, ≥ 500) | same | measured (K-5) |
| STT `temperature` / `repetitionPenalty` | 0 / 1.2 | 0 / 1.2 | 1.0 to evaluate with WER (K-33; external: 1.2 drops 27 % of commas on a 10 min podcast) |
| Realtime `transcriptionDelayMs` | 480 | 480 | Mistral's value (FLEURS WER 8.72 %) |
| TTS `temperature`, `cfgAlpha`, `flowSteps` | 0, 1.2, 8 | same | defaults; effect of `flowSteps` to verify |
| TTS frame cap | 70 + 10.4 × text tokens | same | measured (K-14) |

Settings that do not exist yet and will join the profiles fiche by fiche (lot 4): bf16 compute (K-38, K-40, K-60),
8-bit KV, prefill chunk, Realtime tied head in 8 bits, TTS codec by windows.

## Packs to publish on Hugging Face

Nothing is published without a decision (ASK-22); the upload stays a manual step by Vincent (K-80).

| # | Pack | Size | Needed for | Prerequisites | Decision |
|---|---|---|---|---|---|
| PK-1 | Realtime 8-bit, mlx-audio format (encoder + decoder 8 b, `tok_embeddings` 8 b) | 4.73 GB | `realtime/8bit-*` (no loadable 8-bit pack exists) | K-8 ✓ | ASK-20, ASK-22 |
| PK-2 | TTS 8-bit (LLM + FM 8 b, codec bf16) | 4.37 GB | `tts/8bit-*` | K-8 ✓; licence CC BY-NC kept (ASK-18 = A: non-commercial use) | ASK-19, ASK-22 |
| PK-3 | Mini « LM 4 b, `lm_head` 6 b, encoder 8 b or bf16 » | 3.05 / 3.67 GB | a better `mini/4bit` (today's 4-bit text differs from bf16) | K-42 keeps `.mlx` | ASK-6, ASK-22 |
| PK-4 | Small 4/8-bit with dense encoder | 15.02 / 27.14 GB | `small/*` quality | K-8 ✓ (Markus packs load) | **no upload if ASK-16 = A**: reference the Markus repositories, pinned by revision + SHA-256 |

## Not measured yet

`lean` variants, Small 24B, Macs with 8/16/32 GB, quality scores (WER, K-33), the 16-bit profiles after the bf16
compute fiches, and the published packs.
