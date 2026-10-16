# Baseline STT — Apple M3 Max 96 Go — 2026-10-05 (K-34)

Machine : M3 Max (GPU 40 cœurs), 96 Go, macOS 27.0.0, secteur. Dépendances résolues : mlx-swift 0.31.6 (`0bb916c`),
mlx-swift-lm `main@604fae7`, swift-mlx-profiler 1.5.1 (`bfe71d8`). Binaires Release, arbre propre (`3a0cf13b`,
`db8e19eb`, `9685d556`). Lignes brutes : `BENCHMARKS.md` §« 2026-10-05 — K-34 ». Protocole : `ASK.md` §Dérogations
(2026-10-05). C-moyen = C-moyen exact (146,1 / 130,9 s), C-long = C-long exact (9 min 14 s, `--cache-limit-mb
2048`, sans amorçage). Valeurs de la passe p1. WER : `voxtral eval stt --language-mode auto` (K-33), même sortie
que la passe (`out_sha256` égal, 32 sur 32). ² : cellule reprise par K-96 (2026-10-13), voir ci-dessous.

| Modèle | Backend | Clip | ms/pas p50 | TTFT ms | RTF | pic Mo | WER % |
|---|---|---|---|---|---|---|---|
| mini-3b-8bit | .mlx | C-court EN | 19,51 | 748 | 0,208 | 7 761 | 9,09 |
| mini-3b-8bit | .mlx | C-moyen EN ² | 21,25 | 3 869 | 0,088 | 9 587 | 0,79 |
| mini-3b-8bit | .mlx | C-moyen FR ² | 21,28 | 3 870 | 0,113 | 10 176 | 1,97 |
| mini-3b-8bit | .mlx | C-long ² | 26,36 | 16 722 | 0,082 | 9 692 | 53,69 |
| mini-3b-8bit | .auto | C-court EN | 19,33 | 1 035 | 0,264 | 5 676 | 9,09 |
| mini-3b-8bit | .auto | C-moyen EN ² | 21,22 | 5 270 | 0,098 | 8 392 | 0,79 |
| mini-3b-8bit | .auto | C-moyen FR ² | 21,26 | 5 248 | 0,124 | 8 977 | 1,97 |
| mini-3b-8bit | .auto | C-long ² | 26,38 | 21 762 | 0,091 | 9 386 | 53,75 |
| mini-3b-4bit | .mlx | C-court EN | 13,35 | 743 | 0,189 | 6 016 | 9,09 |
| mini-3b-4bit | .mlx | C-moyen EN ² | 15,41 | 3 877 | 0,070 | 7 873 | 2,11 |
| mini-3b-4bit | .mlx | C-moyen FR ² | 15,40 | 3 875 | 0,091 | 8 471 | 2,22 |
| mini-3b-4bit | .mlx | C-long ² | 20,41 | 16 809 | 0,059 | 8 120 | 55,73 |
| mini-3b-4bit | .auto | C-court EN | 13,50 | 1 027 | 0,245 | 4 140 | 9,09 |
| mini-3b-4bit | .auto | C-moyen EN ² | 15,31 | 5 247 | 0,080 | 6 841 | 1,58 |
| mini-3b-4bit | .auto | C-moyen FR ² | 15,32 | 5 245 | 0,101 | 7 428 | 2,22 |
| mini-3b-4bit | .auto | C-long ² | 20,50 | 21 726 | 0,068 | 7 836 | 55,73 |
| mini-3b | .mlx | C-court EN | 132,10 | 813 | 0,557 | 14 065 | 9,09 |
| mini-3b | .mlx | C-moyen EN ² | 135,98 | 3 955 | 0,409 | 15 498 | 1,05 |
| mini-3b | .mlx | C-moyen FR ² | 136,26 | 3 963 | 0,559 | 16 087 | 1,97 |
| mini-3b | .mlx | C-long | 150,06 | 29 432 | 0,475 | 15 532 | 52,16 |
| mini-3b | .auto | C-court EN | 133,63 | 1 108 | 0,625 | 11 243 | 9,09 |
| mini-3b | .auto | C-moyen EN ² | 136,02 | 5 531 | 0,430 | 13 503 | 0,79 |
| mini-3b | .auto | C-moyen FR ² | 136,37 | 5 537 | 0,572 | 14 089 | 1,97 |
| mini-3b | .auto | C-long | 150,15 | 25 102 | 0,472 | 14 599 | 52,54 |
| small-4bit | .mlx | C-court EN | 70,53 | 2 737 | 0,762 | 16 922 | 9,09 |
| small-4bit | .mlx | C-moyen EN ² | 72,80 | 13 989 | 0,305 | 19 299 | 1,58 |
| small-4bit | .mlx | C-moyen FR ² | 72,89 | 13 867 | 0,392 | 20 021 | 1,23 |
| small-4bit | .mlx | C-long | 84,19 | 67 912 | 0,402 | 19 593 | 1,72 |
| small-4bit | .auto | C-court EN | 71,53 | 3 020 | 0,823 | 15 035 | 9,09 |
| small-4bit | .auto | C-moyen EN | 75,0 à 85,4 ¹ | 17 051 à 19 575 | 0,335 à 0,383 | 18 862 à 19 059 | 1,32 |
| small-4bit | .auto | C-moyen FR ² | 72,77 | 15 169 | 0,401 | 19 041 | 1,23 |
| small-4bit | .auto | C-long | 83,03 | 65 950 | 0,316 | 19 387 | 27,23 |


¹ Pas d'A/A : régime bimodal sur 11 passes (6 à 75,0–75,7 ms, 5 à 80,6–85,4 ; plages sur les 11 passes), cause non
isolée. Décision de Vincent
(ASK-32 = A, 2026-10-06) : la cellule est gardée sans A/A, avec pour référence la médiane des passes rapides,
**75,13 ms**. Une comparaison sur cette cellule doit montrer les deux régimes (A/B/B/A). Cause : fiche K-85.

² Repris par K-96 (2026-10-13) : la passe d'origine avait suivi l'amorçage sans repos (`--warmup 1 --cooldown 0`).
Remesuré au protocole actuel (`--warmup 1 --cooldown 120` ; C-long `--warmup 0`), binaire `44713fd3`, même commande ;
valeurs de la première passe de la paire retenue (p3 pour mini-3b-8bit et mini-3b-4bit `.mlx` C-long, dont la 1re
paire a échoué à l'A/A). `out_sha256` identique à l'ancienne passe : WER inchangés. Lignes : `BENCHMARKS.md`
§« 2026-10-13 — K-96 », qui remplace pour ces cellules les lignes de K-34. Écarts notables (moyenne des deux passes) :
small-4bit C-moyen `.mlx` EN 78,57 → 72,78 ms/pas (− 7,4 %), `.mlx` FR 76,55 → 72,92 (− 4,7 %), TTFT − 17,7 à
− 25,3 % sur les trois cellules Small ; C-long `.mlx` TTFT − 11,7 % (8 bits) et − 16,2 % (4 bits), pas 21,20 →
20,40 en 4 bits (− 3,8 %). Les autres cellules STT reprises bougent de moins de 3 % sur le pas.

## Lecture

- **Le pas de décodage ne dépend pas du backend** : `.auto` (encodeur Core ML sur le Neural Engine) et `.mlx`
  (encodeur sur le GPU) décodent au même rythme (écarts ≤ 1,5 %, sauf Small C-moyen EN ¹ ; l'écart de 3,4 % du Mini
  4 bits C-long a disparu à la reprise de K-96, 0,4 %). `.auto` coûte en TTFT sur Mini (+ 35 % sur C-moyen : 5,2 s contre 3,9 s) et économise de la mémoire (pic
  − 0,2 à − 2,8 Go selon le modèle et le clip).
- **Mini bf16 (`mini-3b`) est 5,7 à 6,9 fois plus lent que le 8 bits** (`.mlx` 5,7 à 6,8 ; `.auto` 5,7 à 6,9 ; 132 à
  150 ms par pas, contre 19,3 à 26,4 ; valeurs de K-96 pour les cellules reprises) : au-delà
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
- **Pas plus lent en audio long** : de + 10 % (Mini bf16, Small `.auto`) à + 34 % (Mini 4 bits `.auto`, 15,31 →
  20,50 ms) entre C-moyen EN et C-long : le contexte grandit (valeurs de K-96 pour les cellules reprises ; avant
  K-96 : + 6 % à + 37 %, Small `.mlx` C-moyen EN mesuré sans repos).
- **Réserve de sélection (contradicteur, 2026-10-06)** : sur mini-3b `.auto` × C-moyen EN, les 8 passes des
  journaux vont de 136,0 à 144,9 ms par pas ; la paire retenue (136,65 / 137,14) est en bas de la plage. Pour le bf16,
  la dispersion réelle d'un processus à l'autre (≈ 6 %) dépasse l'A/A retenu (0,36 %). Small `.mlx` × C-moyen se
  place aussi entre les deux régimes de `.auto` (77,95 / 79,19 EN ; 75,41 / 77,68 FR) : K-85 couvre les deux
  backends. Reprise de K-96, avec repos après l'amorçage : mini-3b `.auto` C-moyen EN 136,02 / 135,98 ; Small `.mlx`
  C-moyen 72,80 / 72,76 (EN) et 72,89 / 72,95 (FR), sous les deux régimes de `.auto` C-moyen EN (cellule non reprise,
  hors inventaire).

## Occupation GPU du préfill (C-moyen EN complet, `bench --trace --metal-trace`)

Traces d'origine (amorçage puis `--cooldown 0`). Refaites par K-96 avec repos (`BENCHMARKS.md` §« 2026-10-13 —
K-96 ») : mini-3b-8bit inchangé à 0,2 pt près (Prefill 99,1 %, Generation 92,8 % / 94,4 % en xctrace) ; small-4bit
inchangé à 0,2 pt près (Prefill 99,7 %, Generation 96,8 % / 97,5 % en xctrace ; `ioreg` 99,8 / 97,6 / 98,2 %), refaite
le 2026-10-16 avec un délai d'arrêt `xctrace` de 600 s passé par `bench` (quatre essais avaient échoué avec les 120 s
par défaut). Le tableau garde les chiffres d'origine.

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

| Backend | TTFT q1 / q2 / q3 / q4 (ms), 1re passe | A/A du TTFT | tok/s | ms/pas p50 | A/A pas / total |
|---|---|---|---|---|---|
| `.mlx` ² | 4 684 / 4 692 / 4 688 / 4 685 | 0,54 / 0,20 / 1,14 / 1,11 % | 45,4 à 46,4 | 21,2 à 21,8 | ≤ 2,04 % / ≤ 1,74 % |
| `.auto` ² | 6 337 / 6 352 / 6 363 / 6 350 | ≤ 0,27 % | 45,7 à 46,5 | 21,3 à 21,7 | ≤ 2,05 % / ≤ 0,23 % |

² Repris par K-96 (2026-10-13), un amorçage par question puis 120 s de repos ; `.mlx` p1 / p2, `.auto` p3 / p4 (1re
paire en échec à l'A/A, `step_ms_p90` de q4) ; mêmes réponses (`out_sha256`) que K-34. Avant K-96 (`.mlx`, p1) :
4 696 / 4 892 / 5 285 / 5 993 ms, A/A du TTFT jusqu'à 3,98 %.

Le TTFT est consigné, pas jugé (dérogation A/A). Avec repos, le TTFT `.mlx` vaut ≈ 4,7 s aux quatre questions : la
hausse de q2 à q4 et la dispersion de q3 / q4 de K-34 venaient de la passe sans repos. Pour K-49 et K-70, le TTFT du
chat se compare en A/B/B/A. Le chat `.auto` avait été refait le 2026-10-06 : sa 1re paire avait tourné avec
Microsoft Teams à 603 % de CPU.

Chaque question réencode l'audio : le TTFT est payé à chaque tour (référence de K-49 et K-70).

## Small 4 bits, 8 min d'audio

Sans plafond, pic `phys_footprint` de 37 313 Mo ; avec `--cache-limit-mb 2048`, 19 013 Mo. Verdict :
`decisions/small-32gb.md`.
