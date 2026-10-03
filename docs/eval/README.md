# Evaluation corpus (K-33)

`voxtral eval stt|realtime` transcribes the clips of [`corpus.json`](corpus.json) and prints one `EVAL {json}` line per
clip: normalized WER (NUL removed, Unicode NFKD without accents, lowercase, every non-alphanumeric character → space;
word-level Levenshtein, `Sources/VoxtralTranscriptionTest/WER.swift`, checked by `Scripts/check-wer.sh`), word counts,
`length_ratio`, `last_sentence_present`, `must_contain_present`, `out_sha256`. Each clip is checked against the
SHA-256 of its audio and reference before anything runs.

## Clips

| Id | Audio | Reference | Origin |
|---|---|---|---|
| `c_court_en`, `c_court_fr` | `docs/examples/fluxforge_short_{en,fr}_6bit.wav` | `refs/c_court_{en,fr}.txt` | short texts of `docs/tts_benchmark.md` |
| `c_moyen_en`, `c_moyen_fr` | `clips/c_moyen_{en,fr}.wav` | `refs/c_moyen_{en,fr}.txt` | **exact-text** C-moyen (K-33): `tts-4b-6bit`, `neutral_female` / `fr_female`, seed 5 |
| `c_20s_en`, `c_20s_fr` | `clips/c_20s_{en,fr}.wav` | `refs/c_20s_{en,fr}.txt` | `tts-4b-6bit`, `neutral_female` / `fr_female`, seed 7 |
| `es_1`, `es_2`, `es_3` | `clips/es_{1,2,3}.wav` | `refs/es_{1,2,3}.txt` | `tts-4b-6bit`, `es_male` / `es_female` / `es_male`, seed 11 |
| `c_long_exact` | `.local-runs/corpus/c_long_exact.wav` (not versioned) | `refs/c_long_exact.txt` | 2 × (C-moyen EN + FR), 16 kHz mono |
| `rt_ref_c_moyen_en`, `rt_ref_c_moyen_fr` | `docs/examples/fluxforge_long_{en,fr}_6bit.wav` (old C-moyen) | `realtime-reference/mlx-audio_c-moyen_{en,fr}.txt` | Realtime against the frozen mlx-audio output (K-13) |

Why new C-moyen clips: the "Full test texts" of `docs/tts_benchmark.md` (163 words EN / 202 FR) are a condensed
version of what the old C-moyen clips say (167 / 174 s; a faithful transcription is about 2.6 × longer, K-5). The old
clips stay the performance witnesses and the Realtime reference clips; quality is measured on the exact-text clips.

Generation (Release `VoxtralCLI`, `--seed` applies to preset voices since K-33):

```bash
$CLI tts "$(cat docs/eval/refs/c_moyen_en.txt)" -m tts-4b-6bit -v neutral_female --seed 5 --max-frames 4000 -o docs/eval/clips/c_moyen_en.wav
$CLI tts "$(cat docs/eval/refs/c_moyen_fr.txt)" -m tts-4b-6bit -v fr_female --seed 5 --max-frames 4000 -o docs/eval/clips/c_moyen_fr.wav
$CLI tts "$(cat docs/eval/refs/c_20s_en.txt)" -m tts-4b-6bit -v neutral_female --seed 7 -o docs/eval/clips/c_20s_en.wav
$CLI tts "$(cat docs/eval/refs/c_20s_fr.txt)" -m tts-4b-6bit -v fr_female --seed 7 -o docs/eval/clips/c_20s_fr.wav
$CLI tts "$(cat docs/eval/refs/es_1.txt)" -m tts-4b-6bit -v es_male --seed 11 -o docs/eval/clips/es_1.wav     # es_2: es_female, es_3: es_male
printf "file '%s'\n" $PWD/docs/eval/clips/c_moyen_{en,fr}.wav $PWD/docs/eval/clips/c_moyen_{en,fr}.wav > .local-runs/corpus/list_exact_2.txt
ffmpeg -y -f concat -safe 0 -i .local-runs/corpus/list_exact_2.txt -ar 16000 -ac 1 .local-runs/corpus/c_long_exact.wav
cat docs/eval/refs/c_moyen_{en,fr}.txt docs/eval/refs/c_moyen_{en,fr}.txt > docs/eval/refs/c_long_exact.txt
```

Usage:

```bash
$CLI eval stt --model mini-3b-8bit --backend mlx                       # every clip, the clip's language
$CLI eval stt --model mini-3b-8bit --language-mode auto --clips es_1,es_2,es_3
$CLI eval realtime --model realtime-4b-4bit --clips c_court_en,c_20s_en,rt_ref_c_moyen_en
```
