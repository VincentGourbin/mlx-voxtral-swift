# Model profiles (provisional)

Measured on 2026-10-02 on an **M3 Max, 96 GB**, Release `VoxtralCLI bench` at commit `32c400c6` (clean tree), mlx-swift
0.31.6, mlx-swift-lm `main@604fae7`. One measured pass after one warm-up, per configuration; the raw `BENCH` lines are
in [`BENCHMARKS.md`](../BENCHMARKS.md) (section of 2026-10-02).

**Provisional**: these are not the reference baselines of the audit plan (K-34 to K-37, A/A ≤ 3 %, quality measured
with the K-33 tool). Background load during the run (Spotlight indexing, `mediaanalysisd` up to 270 % CPU) makes the
times somewhat pessimistic: the 8-bit STT on 11 min took 82 s here and 52 s on a quiet machine the day before.
Quality is only compared between packs (identical output or not), not scored.

Glossary: **RTF** = processing time ÷ audio duration (below 1 = faster than real time). **TTFA** = time to the first
audio. **Peak** = highest process memory (`phys_footprint`), MLX buffer cache included.

## Recommended profiles

| Use | Model | Settings | Why |
|---|---|---|---|
| Transcription (default) | `mini-3b-8bit` | backend `.mlx`, `maxTokens` nil | Same text as the bf16 model, 4.8 × faster, 10.9 GB peak |
| Transcription, small memory | `mini-3b-4bit` | backend `.mlx` | 21 % faster than 8-bit and 1.7 GB lighter, but its text differs (−5 % characters on C-moyen) |
| Long audio (10 min and more) | `mini-3b-8bit` | `cacheLimitBytes` = 2 GB | Peak 10.3 GB instead of 37.6 GB on 11 min, same text |
| Question about an audio (chat) | `mini-3b-4bit` or `-8bit` | greedy | 70 / 49 tokens/s, first token under 1 s |
| Live transcription | `realtime-4b-4bit` | delay 480 ms | 28 ms per 80 ms frame: real time with margin |
| Speech synthesis | `tts-4b-4bit` (or `-6bit`) | seed fixed for reproducibility | 2.2 × / 1.9 × faster than real time; first audio in 0.25–0.4 s |
| Voice cloning | `tts-4b-6bit` | `seed`, `checkpointURL` | 5 000 epochs ≈ 7.7 min (8 s reference, K-26) |

Avoid for now: `realtime-4b-fp16` (138 ms per 80 ms frame: **slower than real time**) and `tts-4b-mlx` (bf16, the
current registry default: RTF 1.8, **slower than real time**, 20.6 GB peak); `mini-3b` (bf16) is 4.8 × slower than
the 8-bit for the same text. The 16-bit packs are slowed by fp32 conversions the audit already lists (K-38, K-40).

## Speech-to-text (C-moyen EN, 167 s of audio)

| Model | Total | RTF | First token | ms/token | Peak | Output vs bf16 |
|---|---|---|---|---|---|---|
| `mini-3b-8bit` | 16.2 s | 0.097 | 4.7 s | 22.0 | 10.9 GB | identical |
| `mini-3b-4bit` | 12.8 s | 0.077 | 4.7 s | 15.8 | 9.3 GB | different (2 555 vs 2 683 chars) |
| `mini-3b` (bf16) | 77.1 s | 0.46 | 4.8 s | 139.1 | 16.8 GB | reference |

Long audio, `mini-3b-8bit`, C-long (11 min 21 s, EN/FR): default 82.0 s, peak **37.6 GB** (the MLX buffer cache
keeps growing); with `cacheLimitBytes` 2 GB 61.5 s, peak **10.3 GB**, same text (K-52).

## Chat (C-court EN, one question, greedy)

| Model | First token | tokens/s | Peak |
|---|---|---|---|
| `mini-3b-8bit` | 0.93 s | 49 | 7.7 GB |
| `mini-3b-4bit` | 0.86 s | 71 | 6.0 GB |

## Realtime (C-moyen EN, 167 s, delay 480 ms)

| Model | Total | RTF | ms per frame (budget 80) | Silent steps | Peak |
|---|---|---|---|---|---|
| `realtime-4b-4bit` | 61.7 s | 0.37 | 27.8 | 72 % | 9.8 GB |
| `realtime-4b-fp16` | 293.6 s | 1.76 | 137.9 | 73 % | 16.4 GB |

The two packs give different texts (2 755 vs 2 748 chars). 72 % of the decode steps emit no text (`pad_fraction`):
the room K-73 would use.

## Text-to-speech (`neutral_female`, seed 42)

| Model | Text | Audio | Total | RTF | TTFA | Peak |
|---|---|---|---|---|---|---|
| `tts-4b-4bit` | short (13 words) | 7.4 s | 4.7 s | 0.63 | 0.23 s | 4.3 GB |
| `tts-4b-4bit` | long (163 words) | 74.1 s | 33.8 s | 0.46 | 0.38 s | 12.9 GB |
| `tts-4b-6bit` | short | 7.8 s | 5.1 s | 0.65 | 0.25 s | 5.2 GB |
| `tts-4b-6bit` | long | 77.3 s | 41.6 s | 0.54 | 0.40 s | 14.1 GB |
| `tts-4b-mlx` (bf16) | short | 6.5 s | 11.7 s | 1.81 | 0.39 s | 10.2 GB |
| `tts-4b-mlx` (bf16) | long | 83.9 s | 155.0 s | 1.85 | 0.51 s | 20.6 GB |

Streaming a long text is much slower than batch today (each chunk re-decodes all the audio so far: 495 s for 182 s
of audio with `tts-4b-4bit`, K-43): use streaming for short texts.

## Not measured yet

Small 24B packs (not downloaded here), `realtime-4b` (official Mistral, fixed by K-9), the Core ML backend (`.auto`,
1.1–1.9 × slower than `.mlx` on these clips, K-32b), memory-constrained Macs (8 / 16 / 32 GB), and quality scores
(WER, K-33).
