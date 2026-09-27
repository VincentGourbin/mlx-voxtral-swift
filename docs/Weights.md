# Model weights

> **Status (2026-09-27): inventory from the Hub, nothing downloaded.** Built by audit fiche
> [K-81](audit/2026-09-27/fiches/K-81.md) from [`modeles-2026-09.md`](audit/2026-09-27/modeles-2026-09.md) §2.1
> (what the code loads) and §3.2 (MLX variants). Bytes are the exact sizes of the weight files listed on the Hub on
> **2026-09-27** (`modeles:73`), re-listed the same day by K-81 (Hugging Face connector, `hf_fs ls`): identical for
> every repository of the report; the two `aufklarer` rows marked † were not inspected by the report (`modeles:135`)
> and come from K-81's listing only. Licences (Hub `license:` tags) and "Hub update" dates were read by K-81 on
> 2026-09-27 (`hub_repo_details`); "no tag" means the repository declares none.
>
> **SHA-256 and revisions: to record (K-82).** The audit's connector does not expose them and the tree API was
> blocked from the audit session (`modeles:18`). K-82 records them with
> `curl -s https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=true` (`lfs.oid`) for every retained file.
>
> Units: B = bytes (exact); GB = 10⁹ bytes. `modeles:N` = line N of `docs/audit/2026-09-27/modeles-2026-09.md`;
> Swift file names in the Source columns are under `Sources/VoxtralCore/` (`Utils/`, `Realtime/`, `TTS/`,
> `Pipeline/`, `CoreML/`).
> Which pack each reference profile uses: [References.md](References.md).

## 1. Registry entries (13 ids)

The ids a caller can pass today. ★ = default of its registry (`modeles:92`). Declared sizes and precisions in the
registries are wrong for several entries (M-03, `modeles:272-297`); K-24 corrects them.

| Registry · id | Repository | Weight files | Bytes (Hub, 2026-09-27) | Hub update | Licence (Hub tag) | Format · precision | Loads today? | SHA-256 | Source |
|---|---|---|---|---|---|---|---|---|---|
| STT · `mini-3b` | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | `model-0000{1,2}-of-00002.safetensors` (+ `consolidated.safetensors` 9 348 806 528 B, also downloaded until K-24, S-07) | 9 356 474 312 | 2025-07-28 | apache-2.0 | HF transformers shards · bf16 (declared "float16") | yes; 16-bit unusable for performance before K-40 (P-01/P-05) | to record (K-82) | `ModelRegistry.swift:49-56` ; `modeles:77` |
| STT · `small-24b` | [`mistralai/Voxtral-Small-24B-2507`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | 11 shards `model-000NN-of-00011.safetensors` (+ `consolidated.safetensors` 48 519 877 672 B, also downloaded until K-24, S-07) | 48 527 546 144 | 2025-12-20 | apache-2.0 | HF transformers shards · bf16 (declared "float16") | yes (same limit) | to record (K-82) | `ModelRegistry.swift:58-65` ; `modeles:78` |
| STT · `mini-3b-8bit` ★ | [`mzbac/voxtral-mini-3b-8bit`](https://huggingface.co/mzbac/voxtral-mini-3b-8bit) | 2 shards | 5 404 054 476 | 2025-07-24 | no tag (base: Apache-2.0) | MLX · affine 8 b g64 uniform, encoder included; `embed_tokens`/`lm_head` 8 b | yes | to record (K-82) | `ModelRegistry.swift:68-77` ; `modeles:79` |
| STT · `mini-3b-4bit` | [`mzbac/voxtral-mini-3b-4bit-mixed`](https://huggingface.co/mzbac/voxtral-mini-3b-4bit-mixed) | `model.safetensors` | 3 195 753 212 | 2025-07-24 | no tag (base: Apache-2.0) | MLX · LM 4 b g64, MLP of layers 0-1/28-29 6 b, encoder + projector 6 b, `embed_tokens` 4 b, `lm_head` 6 b g128 | yes | to record (K-82) | `ModelRegistry.swift:78-86` ; `modeles:80` |
| STT · `small-24b-8bit` | [`VincentGOURBIN/voxtral-small-8bit`](https://huggingface.co/VincentGOURBIN/voxtral-small-8bit) | 5 shards | 26 499 134 369 | 2025-07-31 | apache-2.0 | MLX · 8 b uniform (card titled "mixed", config uniform, `modeles:155`) | yes | to record (K-82) | `ModelRegistry.swift:89-97` ; `modeles:81` |
| STT · `small-4bit` | [`VincentGOURBIN/voxtral-small-4bit-mixed`](https://huggingface.co/VincentGOURBIN/voxtral-small-4bit-mixed) | 3 shards | 14 857 318 962 | 2025-07-31 | apache-2.0 | MLX · same predicate as the Mini 4-bit mixed (encoder 6 b, `lm_head` 6 b g128) | yes | to record (K-82) | `ModelRegistry.swift:98-106` ; `modeles:82` |
| Realtime · `realtime-4b-4bit` ★ | [`mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) | `model.safetensors` | 3 133 798 126 | 2026-02-10 | apache-2.0 | mlx-audio · affine 4 b g64; `tok_embeddings` (tied head) and adapter not quantized | yes | to record (K-82) | `VoxtralRealtimeRegistry.swift:29-38` ; `modeles:84` |
| Realtime · `realtime-4b-fp16` | [`mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16`](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16) | `model.safetensors` | 8 870 608 794 | 2026-02-09 | apache-2.0 | mlx-audio · fp16 (≠ bf16 of training) | yes | to record (K-82) | `VoxtralRealtimeRegistry.swift:39-47` ; `modeles:85` |
| Realtime · `realtime-4b` | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | `consolidated.safetensors` 8 859 462 744 B + `model.safetensors` 8 859 446 848 B (both downloaded: 17.72 GB, M-01) | 8 859 462 744 (`consolidated`) | 2026-03-11 | apache-2.0 | Mistral `consolidated` + transformers `config.json`/`model.safetensors` added 2026-03-11 · bf16 | **no** (M-01, fixed by K-9) | to record (K-82) | `VoxtralRealtimeRegistry.swift:48-56` ; `modeles:86`, `:191-212` |
| TTS · `tts-4b-mlx` ★ | [`mlx-community/Voxtral-4B-TTS-2603-mlx-bf16`](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-bf16) | 2 shards | 8 004 759 170 | 2026-03-27 | cc-by-nc-4.0 | mlx-audio · bf16 | yes (slowest default, P-34) | to record (K-82) | `VoxtralTTSRegistry.swift:29-38` ; `modeles:87` |
| TTS · `tts-4b` | [`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | `consolidated.safetensors` (+ 20 voice `.pt` files, not counted) | 8 004 752 248 | 2026-03-31 | cc-by-nc-4.0 | Mistral `consolidated` · bf16 | yes | to record (K-82) | `VoxtralTTSRegistry.swift:39-47` ; `modeles:88` |
| TTS · `tts-4b-4bit` | [`mlx-community/Voxtral-4B-TTS-2603-mlx-4bit`](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-4bit) | `model.safetensors` | 2 509 879 373 | 2026-03-27 | cc-by-nc-4.0 | mlx-audio · affine 4 b g64; codec (116 tensors) bf16 | yes | to record (K-82) | `VoxtralTTSRegistry.swift:48-56` ; `modeles:89` |
| TTS · `tts-4b-6bit` | [`mlx-community/Voxtral-4B-TTS-2603-mlx-6bit`](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-6bit) | `model.safetensors` | 3 465 520 393 | 2026-03-27 | cc-by-nc-4.0 | mlx-audio · affine 6 b g64 | yes | to record (K-82) | `VoxtralTTSRegistry.swift:57-65` ; `modeles:90` |

Licence: the TTS weights are CC BY-NC 4.0 (non-commercial); their use in a commercial app is a legal question
(ASK-18, `modeles:594-596`).

## 2. Also downloaded by the code (not registry ids)

| Used by | Repository | Weight files | Bytes (Hub, 2026-09-27) | Hub update | Licence (Hub tag) | Format · precision | Loads today? | SHA-256 | Source |
|---|---|---|---|---|---|---|---|---|---|
| enum `VoxtralPipeline.Model.small24b8bit` (the registry id `small-24b-8bit` points elsewhere, S-06, ASK-15) | [`mzbac/Voxtral-Small-24B-2507-8bit`](https://huggingface.co/mzbac/Voxtral-Small-24B-2507-8bit) | 6 shards | 28 056 927 031 | 2025-08-20 | no tag (base: Apache-2.0) | MLX · 8 b uniform; `config.json` identical byte for byte to `VincentGOURBIN/voxtral-small-8bit` (80 122 B), size gap unexplained | yes | to record (K-82) | `VoxtralPipeline.swift:47-48` ; `modeles:81` |
| Core ML encoder + projector, Mini (backends `.auto`/`.hybrid`) | [`VincentGOURBIN/voxtral-encoder-coreml-mini`](https://huggingface.co/VincentGOURBIN/voxtral-encoder-coreml-mini) | `VoxtralEncoderMini.mlmodelc/weights/weight.bin` | 1 324 309 760 | 2026-01-29 | no tag (base: Apache-2.0) | Core ML · fp16, converted from `mistralai/…` | yes | to record (K-82) | `VoxtralCoreMLEncoder.swift:65-69` ; `modeles:83` |
| Core ML encoder + projector, Small | [`VincentGOURBIN/voxtral-encoder-coreml-small`](https://huggingface.co/VincentGOURBIN/voxtral-encoder-coreml-small) | `VoxtralEncoderSmall.mlmodelc/weights/weight.bin` | 1 378 839 808 | 2026-01-29 | no tag (base: Apache-2.0) | Core ML · fp16 | yes | to record (K-82) | `VoxtralCoreMLEncoder.swift:65-69` ; `modeles:83` |

## 3. Candidates (not in a registry)

MLX variants that matter for the choice of a pack (`modeles:129-144`). "Loads today?" is the audit's reading of the
loaders at `9392ed1`; a pack marked "no" becomes a candidate only after the fix named.

| Repository | Model · format | Weight files | Bytes (Hub, 2026-09-27) | Hub update | Licence (Hub tag) | Converter | Loads today? | SHA-256 | Source |
|---|---|---|---|---|---|---|---|---|---|
| [`mlx-community/Voxtral-Mini-3B-2507-bf16`](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16) | Mini · bf16, same 2 shards as `mistralai/…`, no `consolidated` | 2 shards | 9 356 474 312 | 2026-01-13 | apache-2.0 | mlx-audio | yes (avoids the `consolidated` download, S-07) | to record (K-82) | `modeles:133` |
| [`aufklarer/Voxtral-Mini-3B-2507-MLX-8bit`](https://huggingface.co/aufklarer/Voxtral-Mini-3B-2507-MLX-8bit) | Mini · 8 b g64 uniform, `"mode": "affine"` | 2 shards | 5 566 018 003 | 2026-07-23 | apache-2.0 | unknown | no (M-02; K-8) | to record (K-82) | `modeles:134` |
| [`aufklarer/Voxtral-Mini-3B-2507-MLX-5bit`](https://huggingface.co/aufklarer/Voxtral-Mini-3B-2507-MLX-5bit) † | Mini · 5 b (Hub tag), config not inspected | `model.safetensors` | 4 051 350 065 | 2026-07-23 | apache-2.0 | unknown | not inspected | to record (K-82) | `modeles:135` ; K-81 listing |
| [`aufklarer/Voxtral-Mini-3B-2507-MLX-FP16`](https://huggingface.co/aufklarer/Voxtral-Mini-3B-2507-MLX-FP16) † | Mini · FP16 (name), config not inspected | 2 shards | 9 352 633 693 | 2026-07-23 | apache-2.0 | unknown | not inspected | to record (K-82) | `modeles:135` ; K-81 listing |
| [`MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder`](https://huggingface.co/MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder) | Mini · LM + `lm_head` 8 b g64, encoder + projector bf16, `"mode": "affine"` | 2 shards | 6 017 427 099 | 2026-07-29 | apache-2.0 | noScribe `tools/quantize_voxtral.py` | no (M-02; K-8) | to record (K-82) | `modeles:136` |
| [`MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder`](https://huggingface.co/MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder) | Small · LM + `lm_head` 4 b g64, encoder + projector bf16, `"mode"` | 3 shards | 15 016 527 526 | 2026-09-23 | apache-2.0 | noScribe | no (M-02; K-8) | to record (K-82) | `modeles:137` |
| [`MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder`](https://huggingface.co/MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder) | Small · 8 b, encoder bf16, `"mode"` | 6 shards | 27 138 066 384 | 2026-09-23 | apache-2.0 | noScribe | no (M-02; K-8) | to record (K-82) | `modeles:138` |
| [`mlx-community/Voxtral-Mini-4B-Realtime-6bit`](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-6bit) | Realtime · 6 b, voxmlx format | `model.safetensors` | 3 609 304 614 | 2026-02-08 | apache-2.0 | voxmlx | no: loaded wrong without error (P-70) | to record (K-82) | `modeles:139` |
| [`ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx`](https://huggingface.co/ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx) | Realtime · 8 b, voxmlx format | `model.safetensors` | 4 714 618 595 | 2026-02-20 | apache-2.0 | voxmlx | no: loaded wrong without error (P-70) | to record (K-82) | `modeles:140` |
| [`T0mSIlver/Voxtral-Mini-4B-Realtime-2602-4bit-qhead`](https://huggingface.co/T0mSIlver/Voxtral-Mini-4B-Realtime-2602-4bit-qhead) | Realtime · 4 b with quantized tied head, mlx-audio format | `model.safetensors` | 2 554 984 165 | 2026-07-20 | apache-2.0 | mlx-audio | no: `tok_embeddings` excluded from quantization (`VoxtralRealtimeModelLoading.swift:39`), loaded without error (to measure) | to record (K-82) | `modeles:141` |
| [`shreyask/voxtral-mini-4b-realtime-mlx-mixed-4-6`](https://huggingface.co/shreyask/voxtral-mini-4b-realtime-mlx-mixed-4-6) | Realtime · mixed 4/6 per layer, `"mode"` | `model.safetensors` | 3 279 615 774 | 2026-02-07 | no tag | mlx-audio 0.3.2 | no: per-layer quantization ignored | to record (K-82) | `modeles:142` |
| [`majentik/Voxtral-4B-TTS-2603-TurboQuant-MLX-8bit`](https://huggingface.co/majentik/Voxtral-4B-TTS-2603-TurboQuant-MLX-8bit) | TTS · 8 b, only `quantization_config` | 2 shards | 4 267 207 958 | 2026-07-20 | apache-2.0 on a CC BY-NC 4.0 base: **rejected** | unknown | no: packed weights loaded as plain `Linear` without error (to measure) | to record (K-82) | `modeles:143` |
| [`jburtoft/Voxtral-Mini-3B-2507-draft-4layer`](https://huggingface.co/jburtoft/Voxtral-Mini-3B-2507-draft-4layer) | Mini · 4-layer drafter for speculative decoding, transformers format | `model.safetensors` | 3 794 476 992 | 2026-07-31 | apache-2.0 | distillation | no (no speculative decoding in Voxtral) | to record (K-82) | `modeles:144` |

Packs proposed for publication (PK-1…PK-4: Realtime 8-bit, TTS 8-bit, Mini with 8-bit or bf16 encoder, Small with
dense encoder) are not on the Hub; their sizes are estimates (`modeles:552-566`), handled by K-80.

## 4. Refreshing this page

- **Relist every repository at each audit** and re-read its `config.json`: an official repository can gain files
  of another format after the fact (`mistralai/Voxtral-Mini-4B-Realtime-2602`, M-01), which breaks a loader that
  reads `config.json` first and doubles a glob download (`modeles:617-620`).
- A third-party pack enters a profile only with its licence, revision, SHA-256 and a reproducible recipe
  (`modeles:632-634`).
- K-82 fills the SHA-256 and revision columns; K-24 aligns the registries' declared sizes with the bytes above
  (M-03: each entry within ± 5 %, `modeles:294-297`); K-80 adds the published packs.
