# Baseline TTS — Apple M3 Max 96 Go — 2026-10-09 (K-35)

Machine : M3 Max (GPU 40 cœurs), 96 Go, secteur. Dépendances résolues : mlx-swift 0.31.6 (`0bb916c`), mlx-swift-lm
`main@604fae7`, swift-mlx-profiler 1.5.1 (`bfe71d8`). Binaires Release, arbre propre (60 cellules sur 60
`"dirty":false`). Lignes brutes : `BENCHMARKS.md` §« 2026-10-09 — K-35 ». Protocole : `ASK.md` §Dérogations
(2026-10-05 et 2026-10-09). Textes : `docs/eval/tts/` (court 11 mots, moyen 163, long 326 ; SHA-256 dans le
README). Voix : `neutral_female` (prédéfinie) et `clone_fr` (enrôlée sur `docs/examples/clone_fr.wav`, 2 000
époques, graine 7). Matrice réduite (Vincent, 2026-10-06) : graines 1 à 3 sur le texte court, graine 1 sur le moyen
et le long.

Sources des lignes, par cellule :

- **Texte court (36 cellules)** : `--passes 2` dans un processus, `--cooldown 120`, commit `d6129d02` / `73079515`,
  macOS 27.0.0, sans plafond de cache. Contrôle sous macOS 27.0.1 : § « Contrôle macOS ».
- **Batch moyen et long (12 cellules) et ligne consommateur** : profil par défaut (sans plafond de cache, le chemin
  de FluxForge), un processus par passe, commit `599d622a`, macOS 27.0.1.
- **Streaming moyen et long (12 cellules)** : `--cache-limit-mb 2048`, un processus par passe, commit `097fd145`,
  macOS 27.0.1. Sans plafond, le pic `phys_footprint` atteignait 74 à 75,9 Go sur 96 (swap non exclu).

Le TTFT est consigné, pas jugé. Sur le texte long, mesuré sans amorçage, il est erratique : 5 768 ms pour mlx
prédéfinie en batch contre 619 en clonée, 3 995 ms de TTFA pour 6 bits clonée en streaming.

| Pack | Texte | Voix | Mode | ms/pas ou ms/frame p50 | TTFT ou TTFA ms | RTF | frames | pic MLX Go | A/A |
|---|---|---|---|---|---|---|---|---|---|
| tts-4b-6bit | court | neutral_female | batch | 38,15 à 38,18 | 252 à 258 | 0,522 à 0,559 | 71 à 99 | 4,1 à 4,3 | PASS (3 graines) |
| tts-4b-6bit | court | neutral_female | streaming | — ¹ | 343 à 346 | 0,547 à 0,557 | 71 à 99 | 4,2 à 4,3 | PASS (3 graines) |
| tts-4b-6bit | court | clone_fr | batch | 38,01 à 38,06 | 153 à 160 | 0,527 à 0,556 | 62 à 81 | 4,1 à 4,2 | PASS (3 graines) |
| tts-4b-6bit | court | clone_fr | streaming | — ¹ | 246 à 256 | 0,539 à 0,542 | 62 à 81 | 4,2 | PASS (3 graines) |
| tts-4b-6bit | moyen | neutral_female | batch | 46,19 | 462 | 0,613 | 998 | 8,7 | PASS |
| tts-4b-6bit | moyen | neutral_female | streaming | 71,74 | 628 | 0,918 | 998 | 9,1 | sans A/A (ASK-34) |
| tts-4b-6bit | moyen | clone_fr | batch | 44,32 | 336 | 0,590 | 858 | 7,3 | PASS |
| tts-4b-6bit | moyen | clone_fr | streaming | 67,48 | 540 | 0,884 | 858 | 7,7 | sans A/A (ASK-34) |
| tts-4b-6bit | long | neutral_female | batch | 38,96 | 590 | 0,550 | 1 903 | 21,6 | PASS |
| tts-4b-6bit | long | neutral_female | streaming | 107,89 | 740 | 1,721 ² | 1 903 | 23,1 | sans A/A (ASK-34) |
| tts-4b-6bit | long | clone_fr | batch | 38,94 | 525 | 0,548 | 1 868 | 20,9 | PASS |
| tts-4b-6bit | long | clone_fr | streaming | 95,86 | 3 995 | 1,557 | 1 868 | 22,5 | PASS |
| tts-4b-4bit | court | neutral_female | batch | 24,80 à 24,95 | 232 à 236 | 0,361 à 0,379 | 78 à 111 | 3,3 à 3,5 | PASS (3 graines) |
| tts-4b-4bit | court | neutral_female | streaming | — ¹ | 301 à 303 | 0,376 à 0,388 | 78 à 111 | 3,3 à 3,5 | PASS (3 graines) |
| tts-4b-4bit | court | clone_fr | batch | 24,86 à 24,87 | 139 à 146 | 0,361 à 0,365 | 59 à 96 | 3,2 à 3,4 | PASS (3 graines) |
| tts-4b-4bit | court | clone_fr | streaming | — ¹ | 210 à 212 | 0,371 à 0,379 | 59 à 96 | 3,2 à 3,4 | PASS (3 graines) |
| tts-4b-4bit | moyen | neutral_female | batch | 26,46 | 359 | 0,364 | 1 123 | 9,1 | PASS |
| tts-4b-4bit | moyen | neutral_female | streaming | 55,43 | 495 | 0,776 ² | 1 123 | 9,6 | sans A/A (ASK-34) |
| tts-4b-4bit | moyen | clone_fr | batch | 29,90 | 350 | 0,406 | 1 806 | 19,0 | PASS |
| tts-4b-4bit | moyen | clone_fr | streaming | 85,61 | 395 | 1,625 | 1 806 | 20,4 | sans A/A (ASK-34) |
| tts-4b-4bit | long | neutral_female | batch | 25,85 | 586 | 0,365 | 1 932 | 21,2 | PASS |
| tts-4b-4bit | long | neutral_female | streaming | 84,72 | 646 | 1,452 | 1 932 | 22,8 | sans A/A (ASK-34) |
| tts-4b-4bit | long | clone_fr | batch | 25,84 | 496 | 0,370 | 2 149 | 25,5 | PASS |
| tts-4b-4bit | long | clone_fr | streaming | 108,66 | 578 | 1,712 | 2 149 | 27,3 | sans A/A (ASK-34) |
| tts-4b-mlx | court | neutral_female | batch | 129,18 à 129,38 | 366 à 373 | 1,674 à 1,833 | 73 à 111 | 8,7 | PASS (3 graines) |
| tts-4b-mlx | court | neutral_female | streaming | — ¹ | 647 à 649 | 1,702 à 1,719 ² | 73 à 111 | 8,7 | PASS (3 graines) |
| tts-4b-mlx | court | clone_fr | batch | 128,96 à 129,43 | 282 à 287 | 1,676 à 1,776 | 61 à 75 | 8,6 | PASS (3 graines) |
| tts-4b-mlx | court | clone_fr | streaming | — ¹ | 555 à 560 | 1,704 à 1,711 | 61 à 75 | 8,6 | PASS (3 graines) |
| tts-4b-mlx | moyen | neutral_female | batch | 137,78 | 507 | 1,782 | 1 033 | 13,3 | PASS |
| tts-4b-mlx | moyen | neutral_female | streaming | 166,37 | 809 | 2,197 ² | 1 033 | 13,8 | PASS |
| tts-4b-mlx | moyen | clone_fr | batch | 134,48 | 443 | 1,743 | 919 | 12,1 | PASS |
| tts-4b-mlx | moyen | clone_fr | streaming | 164,50 | 776 | 2,170 | 919 | 12,5 | PASS |
| tts-4b-mlx | long | neutral_female | batch | 136,98 | 5 768 | 1,808 | 2 049 | 28,5 | PASS |
| tts-4b-mlx | long | neutral_female | streaming | 213,57 | 991 | 3,007 | 2 049 | 30,5 | sans A/A (ASK-34) |
| tts-4b-mlx | long | clone_fr | batch | 134,48 | 619 | 1,762 | 1 735 | 22,8 | PASS |
| tts-4b-mlx | long | clone_fr | streaming | 207,21 | 908 | 3,105 | 1 735 | 24,2 | sans A/A (ASK-34) |

¹ Le streaming court a été mesuré avant `stream_frame_ms_p50` (`0e7588a9`) ; A/A jugé sur `total_ms` (passe de moins
de 60 s).
² Cellule dont le flux portait deux fois l'audio (K-92, corrigé par `599d622a`) : `audio_s` et `rtf` de la ligne
BENCH sont faux. Le RTF affiché est celui de la ligne × 2, car `audio_s` y vaut exactement 2 × frames × 80 ms.
L'`out_sha256` de ces cellules n'est pas une référence. Le temps par frame n'est pas touché : la tranche doublée
est une copie du waveform déjà décodé.

Ligne consommateur (FluxForge : tts-4b-6bit, `clone_fr`, `--warm-up`, batch, texte moyen) : **43,70 ms/pas**
(p90 47,11), TTFT 336 ms, RTF 0,582, 931 frames dont **7 frames de porteur** (dernière synthèse d'amorçage), pic
MLX 8,0 Go, `phys_footprint` 13,1 Go ; A/A 0,97 %.

## Lecture

- **Le 4 bits décode 1,5 fois plus vite que le 6 bits** (24,8 à 26,5 contre 38,0 à 46,2 ms/pas), le bf16 (`mlx`)
  3,4 fois plus lentement (129 à 138 ms) : RTF 0,36 à 0,41 (4 bits), 0,52 à 0,61 (6 bits), 1,67 à 1,83 (bf16, plus lent que
  le temps réel en batch). Même signature bf16 que le STT et le Realtime (K-34, K-36) : K-38, K-46.
- **La voix clonée démarre plus vite** : TTFT court 139 à 160 ms contre 232 à 258 (4 et 6 bits), 282 à 287 contre 366
  à 373 (bf16). Cause non vérifiée.
- **Le streaming coûte cher dès le texte moyen** : ms/frame 55 à 86 (4 bits), 67 à 72 (6 bits), 165 (bf16) sur le
  moyen ; 85 à 109, 96 à 108 et 207 à 214 sur le long. RTF du streaming long 1,45 à 3,1 : plus lent que le temps
  réel pour les trois packs. Cause : chaque morceau re-décode tout l'accumulé (ASK-34). Référence « avant » de K-43.
- **TTFA du streaming court** : 210 à 303 ms (4 bits), 246 à 346 (6 bits), 555 à 649 (bf16).
- **Le pic mémoire suit la longueur, par le codec** : le codec décode toute la séquence d'un bloc. Pic MLX de la phase
  `codec` 21 à 29 Go sur le texte long en batch (`phys_footprint` 32 à 42 Go), contre 2,6 à 3,8 Go pour le décodage
  LLM en 4 et 6 bits (8,8 Go en bf16). Même plafonné, le texte long garde 21 à 29 Go de `phys_footprint` : sur
  un appareil de 32 Go, la phase codec fixe la limite.
- **Le plafond de cache (`--cache-limit-mb 2048`) ne change pas la vitesse du batch et économise 3 à 12 Go** : sur
  les 12 cellules batch moyen et long, l'écart de `step_ms_p50` entre profil par défaut et profil plafonné va de
  − 5,0 % à + 4,1 %, dans les deux sens, sous le seuil de 5 %. Le `phys_footprint` baisse de 3,0 à 4,9 Go sur le
  moyen et de 9,3 à 12,3 Go sur le long (41,3 → 29,0 Go pour bf16 prédéfinie). Le bloc de comparaison est dans
  `BENCHMARKS.md`. Le défaut public (pas de plafond) est une décision de Vincent. Un premier relevé, avec 8 à 19 %
  de surcoût, comparait aux cellules de la matrice d'origine, mesurées avec un autre protocole et un autre macOS :
  il ne tient pas.
- **Constats sans cause isolée** (à reprendre, pas de cause écrite) :
  - Le 6 bits est plus lent sur le texte moyen (44,3 et 46,2 ms/pas) que sur le court et le long (38,0 à 39,0). Le
    profil plafonné donne le même écart : ce n'est pas du bruit de processus.
  - Le 4 bits avec la voix clonée produit beaucoup plus de frames pour le même texte : 1 806 contre 1 123 sur le
    moyen, 2 149 contre 1 932 sur le long. Il décode aussi plus lentement sur le moyen (29,9 contre 26,5 ms/pas).
    L'aller-retour ASR n'est fait qu'en voix prédéfinie : la qualité de ces sorties n'est pas mesurée.

## Aller-retour ASR (TTS → STT, `voxtral eval tts-roundtrip`, batch, `neutral_female`, mini-3b-8bit `.mlx`)

| Pack | court (graines 1 / 2 / 3) | moyen | long |
|---|---|---|---|
| tts-4b-6bit | 0,909 / 0,909 / 0,909 | 0,940 / 0,946 / 0,964 | 0,952 / 0,958 / 0,946 |
| tts-4b-4bit | 0,909 / 0,909 / 0,909 | 0,976 / 0,952 / 0,952 | 0,955 (graine 1) |
| tts-4b-mlx | 0,909 / 0,909 / 0,909 | 0,952 (graine 1) | 0,949 (graine 1) |

Couverture = part des mots du texte retrouvés dans l'ordre. Sur le court, 10 mots sur 11 pour tous les packs et
toutes les graines. Référence des portes qualité de K-39, K-48, K-58 et K-79.

## Variance d'un processus à l'autre

Le 2026-10-08 au soir, 5 cellules batch sur 14 du profil par défaut ont échoué l'A/A (3,8 à 69,8 % sur
`step_ms_p50`). Pour 3 d'entre elles, l'échantillonneur a relevé de la charge pendant la passe lente : compilations
Xcode et Podcasts. Les 2 autres n'en montrent aucune. Refaites le 2026-10-09 après-midi, machine calme : 5 sur 5 à
0,03 à 0,97 %. Même régime que K-85 (bimodalité par processus, cause non isolée) : une cellule qui échoue sans
charge relevée se refait, sans cause écrite.

## Contrôle macOS

Les 36 cellules courtes ont tourné sous macOS 27.0.0, deux passes dans un processus ; les autres sous 27.0.1, un
processus par passe. Quatre cellules courtes (graine 1, voix prédéfinie : 6 bits, 4 bits et bf16 en batch, 6 bits en
streaming) ont été remesurées sous 27.0.1, un processus par passe. Écart à la matrice : − 0,2 %, − 0,2 %, + 0,3 % sur
`step_ms_p50` et − 0,6 % sur `total_ms` (streaming), A/A 0,08 à 0,19 %, sortie identique bit à bit dans les quatre
cas. Ni la mise à jour du système ni le régime à un processus ne déplacent les cellules courtes.
