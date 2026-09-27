# The reference configurations

> **Status (2026-09-27): sourced skeleton, nothing measured yet.** Built by audit fiche
> [K-81](audit/2026-09-27/fiches/K-81.md) from the audit's profile matrix
> ([`profils.md`](audit/2026-09-27/profils.md) §0-§8) and weight inventory
> ([`modeles-2026-09.md`](audit/2026-09-27/modeles-2026-09.md)). Every value cell is either **"to measure (fiche)"**
> or a value with its source; a figure marked **"in session"** was published before the audit (old revision, cold
> pass, no A/B/B/A) and is not a reference ([PLAN.md](audit/2026-09-27/PLAN.md) §0). Measurements come from
> K-77 (STT), K-78 (Realtime), K-79 (TTS) and K-64 (enrollment); K-82 copies them here from
> [`BENCHMARKS.md`](../BENCHMARKS.md), each value citing its line.
>
> **Not wired yet**: the profile types, `voxtral references` and `--reference` are created by
> [K-76](audit/2026-09-27/fiches/K-76.md) (v0 = existing knobs only). Settings whose knob does not exist yet carry the
> fiche that creates it; they enter a profile only once created **and** measured (`profils.md:9-11`).
>
> Sources: `profils.md:N` and `modeles:N` = line N of `docs/audit/2026-09-27/profils.md` and
> `docs/audit/2026-09-27/modeles-2026-09.md`. Sizes: Hub, 2026-09-27, 1 GB = 10⁹ bytes; exact bytes, licences and
> loadability in [Weights.md](Weights.md).

`voxtral references` lists them, `voxtral <transcribe|realtime|tts|enroll|bench> --reference <id>` applies one,
`Voxtral{STT,Realtime,TTS,Enrollment}ReferenceProfile.all` exposes them to an app (all three from K-76). Each one pins
every setting that matters (weights, encoder backend, compute precision, KV cache, token budget, memory limits,
residency). Same seed + same profile = same output, comparable time on comparable hardware.

Source (after K-76): `Sources/VoxtralCore/Configuration/ReferenceProfiles.swift` (`profils.md:243`). Measurements:
[Benchmarks.md](Benchmarks.md) (protocol, corpus, metric glossary) and [`BENCHMARKS.md`](../BENCHMARKS.md) (raw lines).
Weights: [Weights.md](Weights.md).

**28 profiles** (`profils.md:22-28`): STT Mini 6, STT Small 6, Realtime 6 (of which 2 not available), TTS 8 (6-bit
declared outside the 4/8/16 standard, ASK-19; 2 not available), enrollment 2.

## Conditions of every measurement

- Protocol: Release binary, `machine-check.sh` with no `KO`, 120 s cool-down, A/B/B/A with one warm-up request
  excluded, one `BENCH` line per measurement ([Benchmarks.md](Benchmarks.md) §2; PLAN.md §0).
- Corpus: C-court `fluxforge_short_{en,fr}_6bit.wav` (5.0 / 4.8 s), C-moyen `fluxforge_long_{en,fr}_6bit.wav`
  (167.0 / 173.8 s), C-long (≈ 11 min 22 s concatenation) ([Benchmarks.md](Benchmarks.md) §3; PLAN.md:313, :318).
- Machine and resolved revisions of mlx-swift, mlx-swift-lm (`branch: "main"`) and swift-mlx-profiler: recorded on
  each `BENCH` line (to record by K-77…K-79, K-64). At the audit: mlx-swift 0.31.6 (`0bb916c`), mlx-swift-lm
  `main@ee673d6` (PLAN.md §1).
- Metrics: RTF = generation ÷ audio; TTFT-frame ≠ TTFA ([Benchmarks.md](Benchmarks.md) §4).

## Speech-to-text (Voxtral Mini 3B 2507, Small 24B 2507)

| Id | Weights (HF) | Encoder backend | Compute | Prefill tok/s | Decode tok/s | TTFT | Peak process | WER EN/FR | Who it is for | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| mini `4bit-fast` | `mzbac/voxtral-mini-3b-4bit-mixed` (3.20 GB) | to decide (K-42) | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to decide (K-77) | `profils.md:59`, `:68`, `:75`, `:101` |
| mini `4bit-lean` | same | `.hybrid` | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to decide (K-77) | `profils.md:59`, `:68`, `:75`, `:102` |
| mini `8bit-fast` | `mzbac/voxtral-mini-3b-8bit` (5.40 GB) | to decide (K-42) | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | registry default | `profils.md:60`, `:68`, `:103` ; `modeles:79` |
| mini `8bit-lean` | same | `.hybrid` | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to decide (K-77) | `profils.md:60`, `:68`, `:104` |
| mini `16bit-fast` | `mistralai/Voxtral-Mini-3B-2507` (9.36 GB of shards; `consolidated` also downloaded until K-24) | to decide (K-42) | bf16 (K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | quality reference (gate below) | `profils.md:61`, `:105`, `:378` |
| mini `16bit-lean` | same | `.hybrid` | bf16 (K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to decide (K-77) | `profils.md:61`, `:106` |
| small `4bit-fast` | `VincentGOURBIN/voxtral-small-4bit-mixed` (14.86 GB) | to decide (K-42) | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to decide (K-77) | `profils.md:112`, `:125` |
| small `4bit-lean` | same | `.hybrid` | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | 32 GB Macs, to verify (K-34; ASK-7: peak ≤ 24 GB at 10 min of audio) | `profils.md:112`, `:126` |
| small `8bit-fast` | to decide (ASK-15): `VincentGOURBIN/voxtral-small-8bit` (26.50 GB), `mzbac/Voxtral-Small-24B-2507-8bit` (28.06 GB) or `MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder` (27.14 GB, after K-8) | to decide (K-42) | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | 48 GB+ Macs (indicative) | `profils.md:113`, `:127` |
| small `8bit-lean` | same | `.hybrid` | bf16 (K-40) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | to measure (K-77) | 48 GB+ Macs (indicative) | `profils.md:113`, `:128` |
| small `16bit-fast` | `mistralai/Voxtral-Small-24B-2507` (48.53 GB of shards; `consolidated` also downloaded until K-24) | to decide (K-42) | bf16 (K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | 64 GB+ Macs (indicative; README GPU peak ≈ 56 GB, in session) | `profils.md:114`, `:129` |
| small `16bit-lean` | same | `.hybrid` | bf16 (K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | to measure (K-77, after K-40) | 64 GB+ Macs (indicative) | `profils.md:114`, `:130` |

- **Backend and pack go together**: in `.auto` (library default, `VoxtralPipeline.swift:196`) and `.hybrid`, the
  encoder and projector come from the Core ML fp16 model, so the pack's encoder bits play no role; they only count
  in `.mlx` (M-07, `modeles:343-359`). K-42 picks the `fast` backend (`profils.md:68`).
- **Compute**: fp32 on the whole STT path today (P-01); bf16 needs the knob of K-40, then a measurement
  (`profils.md:75`). The `16bit-*` rows are unusable for performance before K-40 (fp32 copy of each weight, P-05,
  `profils.md:61`).
- **`fast` vs `lean`** (existing knobs): `lean` = `.hybrid` backend (or `.mlx` + audio tower release once K-63
  exists) and `evalFrequency 8`, `clearCacheOnEval false`, `resetPeakMemory false`; `fast` =
  `MemoryOptimizationConfig` `.disabled`; `maxKVCacheSize: nil` in both (`profils.md:68-70`). Small `lean` adds
  8-bit KV (K-55) and encoder release (K-63) once those knobs exist and are measured (`profils.md:126`).
- Earlier figures (README, issues #13-#21; all in session, not references): `profils.md:101-105`, `:120-122`.

## Realtime (Voxtral Mini 4B Realtime 2602)

| Id | Weights (HF) | Compute | Tied head | ms/step p50 / p90 (budget 80) | First text token | Peak process | WER EN/FR | Who it is for | Source |
|---|---|---|---|---|---|---|---|---|---|
| `4bit-fast` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` (3.13 GB) | bf16 (K-38) | 8-bit (K-46) | to measure (K-78) | to measure (K-78) | to measure (K-78) | to measure (K-78) | registry default | `profils.md:139`, `:149-150`, `:167` ; `modeles:84` |
| `4bit-lean` | same | bf16 (K-38) | 4-bit if parity holds (WER ≤ +0.5 pt, K-46) | to measure (K-78) | to measure (K-78) | to measure (K-78) | to measure (K-78) | to decide (K-78) | `profils.md:139`, `:150`, `:167` |
| `8bit-fast` | not available: no loadable 8-bit pack (P-70); PK-1 or voxmlx sanitizer per ASK-20 → K-80 | — | — | — | — | — | — | — | `profils.md:26`, `:140`, `:168` |
| `8bit-lean` | not available (same) | — | — | — | — | — | — | — | `profils.md:26`, `:140`, `:168` |
| `16bit-fast` | `mistralai/Voxtral-Mini-4B-Realtime-2602` `consolidated.safetensors` (8.86 GB, bf16) after K-9; `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` (8.87 GB, fp16) until then | bf16 (K-38) | 16-bit | to measure (K-78, after K-38) | to measure (K-78, after K-38) | to measure (K-78, after K-38) | to measure (K-78, after K-38) | quality reference | `profils.md:141`, `:169`, `:338` |
| `16bit-lean` | same | bf16 (K-38) | 8-bit (K-46) | to measure (K-78, after K-38) | to measure (K-78, after K-38) | to measure (K-78, after K-38) | to measure (K-78, after K-38) | to decide (K-78) | `profils.md:141`, `:169`, `:339` |

- Transcription delay 480 ms in every profile (Mistral's value; external: FLEURS WER 8.72 % at 480 ms, model card),
  temperature 0 (`profils.md:146-147`).
- `lean` also uses adaptive `cacheLimit` (K-52) and encoder-only extraction (K-59) once those knobs exist and are
  measured (`profils.md:152-153`).
- The original `realtime-4b` entry does not load today (M-01, `modeles:191-212`); K-9 fixes it.
- Earlier figures (issues #23-#25; in session, contested instrument P-73): `profils.md:161-163`.

## Text-to-speech (Voxtral 4B TTS 2603, CC BY-NC 4.0)

Commercial use of these weights is a legal question (ASK-18, `profils.md:176-177`).

| Id | Weights (HF) | fps | TTFA (streaming) | RTF (gen / audio) | Peak process | ASR coverage | Who it is for | Source |
|---|---|---|---|---|---|---|---|---|
| `4bit-fast` | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` (2.51 GB) | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | English, short texts (in session: misses end of audio on long French, one take, no seed) | `profils.md:181`, `:215` |
| `4bit-lean` | same | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | English, short texts (same source) | `profils.md:181`, `:215` |
| `6bit-fast` | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` (3.47 GB) | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | cloned voices, long French; the only pack FluxForge ships (in session: coverage 99.4 % vs 96.5 % bf16, n = 15, one voice) | `profils.md:182`, `:216` |
| `6bit-lean` | same | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | to measure (K-79) | cloned voices, long French (same source) | `profils.md:182`, `:216` |
| `8bit-fast` | not available: no valid 8-bit pack (`majentik` rejected); PK-2 per ASK-18, ASK-19 → K-80 | — | — | — | — | — | — | `profils.md:27`, `:183`, `:217` |
| `8bit-lean` | not available (same) | — | — | — | — | — | — | `profils.md:27`, `:183`, `:217` |
| `16bit-fast` | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` (8.00 GB) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | quality reference; registry default (ASK-5) | `profils.md:184`, `:218` |
| `16bit-lean` | same | to measure (K-79, after K-39) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | to measure (K-79, after K-39) | quality reference (same source) | `profils.md:184`, `:218` |

- **`fast` vs `lean`** share the weights; `lean` differs by the voice prefix cache (1 entry vs LRU 2-4, K-50) and the
  memory policy (`clearCache` after decoding and at `unload()`, adaptive `cacheLimit`, K-52) once those knobs exist
  and are measured (`profils.md:200`, `:202`).
- Temperature 0, `cfgAlpha` 1.2, 8 flow-matching steps: these fields are not read today (P-35), K-48 wires them
  (`profils.md:190-192`). Seed fixed for every measurement; warm-up for cloned voices (`profils.md:194-195`).
- Earlier figures (bench `6ad4e56`, `0be05af`, campaign `e83778a`; in session): `profils.md:207-211`.

## Voice enrollment

| Id | Weights | Reference length | Epochs | s/epoch | Peak process | ECAPA similarity | Who it is for | Source |
|---|---|---|---|---|---|---|---|---|
| `enroll-fast` | TTS pack of the target synthesis (bf16 = CLI default `tts-4b-mlx`) | 16 s (200 frames) | 5,000 | to measure (K-64) | to measure (K-64) | to measure (K-64); in session: 0.72 at 16 s, 2,000 epochs ([voice_cloning.md](voice_cloning.md):36-45) | to decide (K-64) | `profils.md:224-226` |
| `enroll-lean` | 6-bit TTS pack | 16 s (200 frames) | 5,000 (3,000 to measure) | to measure (K-64) | to measure (K-64) | to measure (K-64) | to decide (K-64) | `profils.md:224-226` |

- Library default reference length is 8 s (`Config.numFrames` 100); moving it to 16 s is ASK-9 (`profils.md:225`).
- Quality gate: similarity ≥ 0.70 (`fast`); `lean` within ±0.01 of `fast`, peak ≤ 60 % of `fast`, time ±5 %
  (`profils.md:233`).

## What each setting does

| Setting | Values | Effect (measured) |
|---|---|---|
| Encoder backend (STT) | `.mlx` / `.hybrid` / `.auto` | to measure (K-42); `.auto` and `.hybrid` ignore the pack's encoder precision (M-07, `modeles:343-359`) |
| `maxKVCacheSize` (STT) | always `nil` | every memory preset sets a window (2,048 / 4,096 / 6,144 / 8,192 positions) that stops the process at prefill from 6 / 11 / 17 / 22 audio windows of 30 s, i.e. beyond 2 min 30 to 10 min 30 of audio, until K-2/K-3 (PLAN.md §1; `profils.md:48-49`) |
| Token budget (STT) | tokens per audio second | to measure (K-5); the default 500 truncates beyond ≈ 3 min of speech (PLAN.md §1; `profils.md:71`) |
| Repetition penalty (STT) | 1.0 / 1.2 (today) | to measure (K-61); external: 1.2 loses 27 % of commas over 10 min (`profils.md:72`) |
| Compute precision | fp32 today → bf16 | to measure (K-40 STT, K-38 Realtime, K-39 / K-58 TTS) (`profils.md:75`, `:149`, `:201`) |
| Tied head (Realtime) | 16 / 8 / 4 bits | to measure (K-46) (`profils.md:150`) |
| Flow-matching steps (TTS) | 8 / 6 / 5 / 4 (sweep of [K-48](audit/2026-09-27/fiches/K-48.md)) | to measure (K-48); ignored by the code today (P-35) (`profils.md:192`) |

## Choosing

- **Mac, 32 GB and up**: to write after measurement (K-77…K-79).
- **Mac, 16-24 GB**: to write after measurement (K-77…K-79).
- **8 GB Mac, iPhone**: pending the iOS decision (ASK-2).
- **Even faster, memory no object**: to write after measurement (K-77…K-79).

## Adding or changing a profile

1. Add the entry to `.all` (every field is an existing knob, nothing new to wire) (`profils.md:9-11`).
2. Measure with `voxtral bench <pipeline> --reference <id> --passes 2 --warmup 1 --cooldown 120` (A/B/B/A, protocol
   in [Benchmarks.md](Benchmarks.md); `bench` from K-32, `--reference` from K-76).
3. Quality gate: WER (STT, Realtime) or ASR coverage + blind listening (TTS) against the `16bit-fast` profile, same
   seed (`profils.md:378`).
4. Add the row to [`BENCHMARKS.md`](../BENCHMARKS.md) and the decision to
   `docs/knowledge/decisions/reference-profiles.md` (created by K-77).

## Command-line equivalents

- After K-76: `--reference 8bit-fast` (mini) = `-m mini-3b-8bit -b <backend> --max-tokens <ceil(duration × rate) + 64>`
  plus the memory setting `.disabled` (`profils.md:383`).
- Today there is no exact equivalent: `voxtral transcribe` takes `-m`, `--max-tokens`, `-l` and `-b mlx|hybrid`
  only (`Sources/VoxtralTranscriptionTest/VoxtralCLI.swift:177-187`, `.auto` unreachable: `:210`), and always runs
  `VoxtralPipeline.Configuration.default` (`:215`), whose `memoryOptimization: .recommended()` is a preset with a KV
  window (`Sources/VoxtralCore/Pipeline/VoxtralPipeline.swift:107-113`;
  `Sources/VoxtralCore/Configuration/MemoryOptimizationConfig.swift:41-90`).
