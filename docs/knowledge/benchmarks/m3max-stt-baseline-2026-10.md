# Baseline STT — Apple M3 Max 96 Go — 2026-10-05 (K-34)

Machine : M3 Max (GPU 40 cœurs), 96 Go, macOS 27.0.0, secteur. Dépendances résolues : mlx-swift 0.31.6 (`0bb916c`),
mlx-swift-lm `main@604fae7`, swift-mlx-profiler 1.5.1 (`bfe71d8`). Binaires Release, arbre propre (`3a0cf13b`,
`db8e19eb`, `9685d556`). Lignes brutes : `BENCHMARKS.md` §« 2026-10-05 — K-34 ». Protocole : `ASK.md` §Dérogations
(2026-10-05). C-moyen = C-moyen exact (146,1 / 130,9 s), C-long = C-long exact (9 min 14 s, `--cache-limit-mb
2048`, sans amorçage). Valeurs de la passe p1. WER : `voxtral eval stt --language-mode auto` (K-33), même sortie
que la passe (`out_sha256` égal, 32 sur 32).

| Modèle | Backend | Clip | ms/pas p50 | TTFT ms | RTF | pic Mo | WER % |
|---|---|---|---|---|---|---|---|
| mini-3b-8bit | .mlx | C-court EN | 19,51 | 748 | 0,208 | 7 761 | 9,09 |
| mini-3b-8bit | .mlx | C-moyen EN | 21,32 | 3 866 | 0,088 | 9 587 | 0,79 |
| mini-3b-8bit | .mlx | C-moyen FR | 21,35 | 3 856 | 0,113 | 10 176 | 1,97 |
| mini-3b-8bit | .mlx | C-long | 26,79 | 18 612 | 0,086 | 9 811 | 53,69 |
| mini-3b-8bit | .auto | C-court EN | 19,33 | 1 035 | 0,264 | 5 676 | 9,09 |
| mini-3b-8bit | .auto | C-moyen EN | 21,27 | 5 208 | 0,097 | 8 392 | 0,79 |
| mini-3b-8bit | .auto | C-moyen FR | 21,39 | 5 209 | 0,124 | 8 977 | 1,97 |
| mini-3b-8bit | .auto | C-long | 27,10 | 22 971 | 0,094 | 9 385 | 53,75 |
| mini-3b-4bit | .mlx | C-court EN | 13,35 | 743 | 0,189 | 6 016 | 9,09 |
| mini-3b-4bit | .mlx | C-moyen EN | 15,43 | 3 869 | 0,070 | 7 880 | 2,11 |
| mini-3b-4bit | .mlx | C-moyen FR | 15,47 | 3 873 | 0,091 | 8 471 | 2,22 |
| mini-3b-4bit | .mlx | C-long | 21,17 | 19 961 | 0,066 | 8 036 | 55,73 |
| mini-3b-4bit | .auto | C-court EN | 13,50 | 1 027 | 0,245 | 4 140 | 9,09 |
| mini-3b-4bit | .auto | C-moyen EN | 15,37 | 5 255 | 0,080 | 6 844 | 1,58 |
| mini-3b-4bit | .auto | C-moyen FR | 15,44 | 5 237 | 0,101 | 7 431 | 2,22 |
| mini-3b-4bit | .auto | C-long | 20,46 | 22 186 | 0,069 | 7 838 | 55,73 |
| mini-3b | .mlx | C-court EN | 132,10 | 813 | 0,557 | 14 065 | 9,09 |
| mini-3b | .mlx | C-moyen EN | 137,56 | 3 974 | 0,413 | 15 496 | 1,05 |
| mini-3b | .mlx | C-moyen FR | 138,99 | 4 117 | 0,570 | 16 051 | 1,97 |
| mini-3b | .mlx | C-long | 150,06 | 29 432 | 0,475 | 15 532 | 52,16 |
| mini-3b | .auto | C-court EN | 133,63 | 1 108 | 0,625 | 11 243 | 9,09 |
| mini-3b | .auto | C-moyen EN | 136,65 | 5 608 | 0,432 | 13 503 | 0,79 |
| mini-3b | .auto | C-moyen FR | 138,79 | 5 526 | 0,580 | 14 088 | 1,97 |
| mini-3b | .auto | C-long | 150,15 | 25 102 | 0,472 | 14 599 | 52,54 |
| small-4bit | .mlx | C-court EN | 70,53 | 2 737 | 0,762 | 16 922 | 9,09 |
| small-4bit | .mlx | C-moyen EN | 79,19 | 18 194 | 0,350 | 19 301 | 1,58 |
| small-4bit | .mlx | C-moyen FR | 75,41 | 19 857 | 0,445 | 20 022 | 1,23 |
| small-4bit | .mlx | C-long | 84,19 | 67 912 | 0,402 | 19 593 | 1,72 |
| small-4bit | .auto | C-court EN | 71,53 | 3 020 | 0,823 | 15 035 | 9,09 |
| small-4bit | .auto | C-moyen EN | 75,0 à 85,4 ¹ | 17 057 à 19 576 | 0,335 à 0,383 | 18 935 à 19 059 | 1,32 |
| small-4bit | .auto | C-moyen FR | 74,77 | 18 624 | 0,433 | 19 045 | 1,23 |
| small-4bit | .auto | C-long | 83,03 | 65 950 | 0,316 | 19 387 | 27,23 |


¹ Pas d'A/A : régime bimodal sur 11 passes (6 à 75,0–75,7 ms, 5 à 80,6–85,4), cause non isolée. Décision de Vincent
(ASK-32 = A, 2026-10-06) : la cellule est gardée sans A/A, avec pour référence la médiane des passes rapides,
**75,13 ms**. Une comparaison sur cette cellule doit montrer les deux régimes (A/B/B/A). Cause : fiche K-85.

## Lecture

- **Le pas de décodage ne dépend pas du backend** : `.auto` (encodeur Core ML sur le Neural Engine) et `.mlx`
  (encodeur sur le GPU) décodent au même rythme (écarts ≤ 1,5 %, sauf Mini 4 bits C-long 3,4 % et Small C-moyen
  EN ¹). `.auto` coûte en TTFT sur Mini (+ 35 % sur C-moyen : 5,2 s contre 3,9 s) et économise de la mémoire (pic
  − 0,2 à − 2,8 Go selon le modèle et le clip).
- **Mini bf16 (`mini-3b`) est 5,6 à 6,8 fois plus lent que le 8 bits** (132 à 150 ms par pas, contre 19,5 à 26,8) : au-delà
  de ce que pèsent les poids (× 2). Même signature que le Realtime fp16 (K-36) : à traiter par K-38 et K-46.
- **C-long (EN/FR alterné), jugé par langue** (K-33, `SEGWER` dans `BENCHMARKS.md`) :

  | Modèle | Backend | WER EN | WER FR | segments EN1 / FR1 / EN2 / FR2 |
  |---|---|---|---|---|
  | mini-3b-8bit | .mlx / .auto | 8,95 / 8,95 | 95,57 / 95,69 | 1,84 / 99,75 / 16,05 / 91,4 à 91,6 |
  | mini-3b-4bit | .mlx / .auto | 8,68 / 8,68 | 99,75 / 99,75 | 2,63 / 99,75 / 14,74 / 99,75 |
  | mini-3b | .mlx / .auto | 38,42 / 37,89 | 65,02 / 66,26 | 1,58 / 92,36 / 75,26–74,21 / 37,68–40,15 |
  | small-4bit | .mlx | 1,97 | 1,48 | 1,58 / 1,48 / 2,37 / 1,48 |
  | small-4bit | .auto | 2,11 | 50,74 | 1,58 / 1,72 / 2,63 / 99,75 |

  Mini traduit les segments français en anglais, une limite du modèle (K-33) ; en bf16, il dérive aussi sur le 2e
  segment anglais. Small `.mlx` transcrit tout. Small `.auto` traduit le dernier segment français : seul écart entre
  backends du corpus, propre à l'encodeur Core ML de Small sur l'audio long (fiche de suivi).
- **Pas plus lent en audio long** : + 25 % (Mini 8 bits, 21,3 → 26,8 ms) à + 9 % (Mini bf16) entre C-moyen et C-long :
  le contexte grandit.

## Occupation GPU du préfill (C-moyen EN complet, `bench --trace --metal-trace`)

| Modèle | Phase | Profiler (IOReport) | Metal System Trace | `ioreg` | Écart max |
|---|---|---|---|---|---|
| mini-3b-8bit `.mlx` | Prefill (2,2 s) | 99,9 % | 99,0 % | 99,2 % | 0,9 pt |
| small-4bit `.mlx` | Prefill (16,0 s) | 100 % | 99,8 % | 99,9 % | 0,2 pt |
| mini-3b-8bit `.mlx` | Generation (9,1 s) | 100 % | 92,8 % | 94,4 % | 7,2 pts |
| small-4bit `.mlx` | Generation (32,9 s) | 99,8 % | 96,6 % | 97,6 % | 3,2 pts |

Le préfill sature le GPU : le « 49 % » des issues #13 et #14 n'était pas une occupation (même artefact que #24,
`decisions/realtime-diagnostics-23-25.md`). Le profiler lit l'état « non idle » du GPU (résidence IOReport), il
sur-estime légèrement le décodage, où des trous de quelques µs séparent les noyaux.

## Chat (mini-3b-8bit, `fluxforge_long_en_6bit.wav`, 4 questions, greedy)

| Backend | TTFT q1 / q2 / q3 / q4 (ms) | ms/pas p50 | A/A |
|---|---|---|---|
| `.mlx` | 4 696 / 4 892 / 5 285 / 5 993 | 21,4 | pas ≤ 0,23 %, total ≤ 2,81 % |
| `.auto` | 6 315 / 6 310 / 6 317 / 6 401 | 21,3 à 21,5 | pas ≤ 0,23 %, total ≤ 0,43 % |

Chaque question réencode l'audio : le TTFT est payé à chaque tour (référence de K-49 et K-70).

## Small 4 bits, 8 min d'audio

Sans plafond, pic `phys_footprint` de 37 313 Mo ; avec `--cache-limit-mb 2048`, 19 013 Mo. Verdict :
`decisions/small-32gb.md`.
