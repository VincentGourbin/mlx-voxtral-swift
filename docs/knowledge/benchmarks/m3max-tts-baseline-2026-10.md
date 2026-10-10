# Baseline TTS — Apple M3 Max 96 Go — 2026-10-09, texte moyen repris les 2026-10-09 et 10 (K-35)

Machine : M3 Max (GPU 40 cœurs), 96 Go, secteur. Dépendances résolues : mlx-swift 0.31.6 (`0bb916c`), mlx-swift-lm
`main@604fae7`, swift-mlx-profiler 1.5.1 (`bfe71d8`). Binaires Release, arbre propre (60 cellules sur 60
`"dirty":false`). Lignes brutes : `BENCHMARKS.md` §« 2026-10-09 — K-35 » et, pour le texte moyen et la ligne
consommateur, §« 2026-10-10 — K-35, reprise ». Protocole : `ASK.md` §Dérogations
(2026-10-05, 2026-10-09 et 2026-10-10). Textes : `docs/eval/tts/` (court 11 mots, moyen 163, long 326 ; SHA-256 dans le
README). Voix : `neutral_female` (prédéfinie) et `clone_fr` (`enroll docs/examples/clone_fr.wav -m tts-4b-6bit
--epochs 2000 --duration 8 --seed 7`, SHA-256 `43ff457c…645b` ; `--duration 8` car le clip dure 8,6 s, `ASK.md`
§Dérogations). Go = Mo ÷ 1024 (champs `*_mb` des lignes). Matrice réduite (Vincent, 2026-10-06) : graines 1 à 3 sur le texte court, graine 1 sur le moyen
et le long.

Sources des lignes, par cellule :

- **Texte court (36 cellules)** : `--passes 2` dans un processus, `--cooldown 120`, commit `d6129d02` / `73079515`,
  macOS 27.0.0, sans plafond de cache. Contrôle sous macOS 27.0.1 : § « Contrôle macOS ».
- **Texte moyen (12 cellules) et ligne consommateur** : un processus par passe, amorçage puis `--cooldown 120`
  avant la passe mesurée, commit `f6efa48d` (binaire de `599d622a`), macOS 27.0.1, 2026-10-09 au soir et nuit du 10. Batch en profil par
  défaut (sans plafond de cache, le chemin de FluxForge), streaming avec `--cache-limit-mb 2048`.
- **Batch long (6 cellules)** : profil par défaut, un processus par passe, sans amorçage, commit `599d622a`, macOS
  27.0.1.
- **Streaming long (6 cellules)** : `--cache-limit-mb 2048`, un processus par passe, sans amorçage, commit
  `097fd145`, macOS 27.0.1 ; 6 bits prédéfinie et bf16 clonée refaites le 2026-10-10 (`eed131bc`), leurs premières
  passes ayant tourné pendant des compilations Xcode et un simulateur démarré.
- Sans plafond, le pic `phys_footprint` plafonnait à 74,1 Go (75 899 Mo) sur 96 dans les 12 cellules streaming
  moyen et long de la matrice réduite, et dans 18 cellules en comptant les graines 2 et 3 (swap non exclu).

**Repos après l'amorçage.** Les mesures du texte moyen faites jusqu'au 2026-10-09 enchaînaient la passe mesurée
juste après l'amorçage (`--cooldown 0`). Le GPU encore chaud les ralentissait de 3 à 17 % (moyenne des deux passes). A/B/B/A sur 6 bits
moyen : 41,40 / 44,70 ms/pas sans repos, 38,52 / 38,50 avec 120 s, même sortie. Toutes ces cellules ont été
refaites avec 120 s de repos.

Le TTFT (batch) et le TTFA (streaming) sont consignés, pas jugés. Ils ne servent de référence que dans les 47
cellules où leurs deux passes sont à 3 % ou moins ; les 13 autres sont listées dans `BENCHMARKS.md` (reprise du
2026-10-10) et marquées ⁴ dans le tableau. Sur le texte long, mesuré sans amorçage, ils sont erratiques : 619 puis 6 096 ms pour bf16
clonée en batch, 3 995 puis 647 ms de TTFA pour 6 bits clonée en streaming. K-66 (porte TTFT) refait sa propre
référence en A/B/B/A.

| Pack | Texte | Voix | Mode | ms/pas ou ms/frame p50 | TTFT ou TTFA ms | RTF | frames | pic MLX Go | A/A |
|---|---|---|---|---|---|---|---|---|---|
| tts-4b-6bit | court | neutral_female | batch | 38,15 à 38,18 | 252 à 258 | 0,522 à 0,559 | 71 à 99 | 4,1 à 4,3 | PASS (3 graines) |
| tts-4b-6bit | court | neutral_female | streaming | — ¹ | 343 à 346 | 0,548 à 0,557 | 71 à 99 | 4,2 à 4,3 | PASS (3 graines) |
| tts-4b-6bit | court | clone_fr | batch | 38,01 à 38,06 | 153 à 160 ⁴ | 0,527 à 0,556 | 62 à 81 | 4,1 à 4,2 | PASS (3 graines) |
| tts-4b-6bit | court | clone_fr | streaming | — ¹ | 247 à 256 | 0,539 à 0,542 | 62 à 81 | 4,2 | PASS (3 graines) |
| tts-4b-6bit | moyen | neutral_female | batch | 38,50 | 395 | 0,516 | 998 | 8,7 | PASS |
| tts-4b-6bit | moyen | neutral_female | streaming | 58,09 | 491 | 0,829 | 998 | 9,1 | PASS |
| tts-4b-6bit | moyen | clone_fr | batch | 38,45 | 327 | 0,507 | 858 | 7,3 | PASS |
| tts-4b-6bit | moyen | clone_fr | streaming | 54,60 | 426 | 0,751 | 858 | 7,7 | PASS |
| tts-4b-6bit | long | neutral_female | batch | 38,96 | 590 | 0,550 | 1 903 | 21,6 | PASS |
| tts-4b-6bit | long | neutral_female | streaming | 100,56 | 3 997 ⁴ | 1,650 | 1 903 | 23,1 | PASS |
| tts-4b-6bit | long | clone_fr | batch | 38,94 | 525 | 0,548 | 1 868 | 20,9 | PASS |
| tts-4b-6bit | long | clone_fr | streaming | 95,86 | 3 995 ⁴ | 1,557 | 1 868 | 22,5 | PASS |
| tts-4b-4bit | court | neutral_female | batch | 24,80 à 24,95 | 232 à 236 | 0,361 à 0,379 | 78 à 111 | 3,3 à 3,5 | PASS (3 graines) |
| tts-4b-4bit | court | neutral_female | streaming | — ¹ | 301 à 303 ⁴ | 0,376 à 0,388 | 78 à 111 | 3,3 à 3,5 | PASS (3 graines) |
| tts-4b-4bit | court | clone_fr | batch | 24,86 à 24,87 | 139 à 146 ⁴ | 0,361 à 0,365 | 59 à 96 | 3,2 à 3,4 | PASS (3 graines) |
| tts-4b-4bit | court | clone_fr | streaming | — ¹ | 210 à 212 ⁴ | 0,371 à 0,379 | 59 à 96 | 3,2 à 3,4 | PASS (3 graines) |
| tts-4b-4bit | moyen | neutral_female | batch | 25,15 | 372 | 0,339 | 1 123 | 9,1 | PASS |
| tts-4b-4bit | moyen | neutral_female | streaming | 49,14 | 440 | 0,721 | 1 123 | 9,6 | PASS |
| tts-4b-4bit | moyen | clone_fr | batch | 25,34 | 301 | 0,355 | 1 806 | 19,0 | PASS |
| tts-4b-4bit | moyen | clone_fr | streaming | 79,81 | 380 ⁴ | 1,342 | 1 806 | 20,4 | sans A/A (ASK-34) |
| tts-4b-4bit | long | neutral_female | batch | 25,85 | 586 ⁴ | 0,365 | 1 932 | 21,2 | PASS |
| tts-4b-4bit | long | neutral_female | streaming | 84,72 | 646 | 1,452 | 1 932 | 22,8 | sans A/A (ASK-34) |
| tts-4b-4bit | long | clone_fr | batch | 25,84 | 496 | 0,370 | 2 149 | 25,5 | PASS |
| tts-4b-4bit | long | clone_fr | streaming | 108,66 | 578 | 1,712 | 2 149 | 27,3 | sans A/A (ASK-34) |
| tts-4b-mlx | court | neutral_female | batch | 129,18 à 129,38 | 366 à 373 | 1,674 à 1,833 | 73 à 111 | 8,7 | PASS (3 graines) |
| tts-4b-mlx | court | neutral_female | streaming | — ¹ | 647 à 649 | 1,702 à 1,719 ² | 73 à 111 | 8,7 | PASS (3 graines) |
| tts-4b-mlx | court | clone_fr | batch | 128,96 à 129,43 | 282 à 287 ⁴ | 1,676 à 1,776 | 61 à 75 | 8,6 | PASS (3 graines) |
| tts-4b-mlx | court | clone_fr | streaming | — ¹ | 555 à 560 | 1,704 à 1,711 | 61 à 75 | 8,6 | PASS (3 graines) |
| tts-4b-mlx | moyen | neutral_female | batch | 131,51 | 500 | 1,713 | 1 033 | 13,3 | PASS |
| tts-4b-mlx | moyen | neutral_female | streaming | 160,10 | 799 | 2,087 | 1 033 | 13,7 | PASS |
| tts-4b-mlx | moyen | clone_fr | batch | 131,83 | 457 | 1,758 | 919 | 12,1 | PASS |
| tts-4b-mlx | moyen | clone_fr | streaming | 157,89 | 762 | 2,034 | 919 | 12,5 | sans A/A (ASK-34) |
| tts-4b-mlx | long | neutral_female | batch | 136,98 | 5 768 ⁴ | 1,808 | 2 049 | 28,5 | PASS |
| tts-4b-mlx | long | neutral_female | streaming | 213,57 | 991 | 3,007 | 2 049 | 30,5 | sans A/A (ASK-34) |
| tts-4b-mlx | long | clone_fr | batch | 134,48 | 619 ⁴ | 1,762 | 1 735 | 22,8 | PASS |
| tts-4b-mlx | long | clone_fr | streaming | 201,61 | 7 840 ⁴ | 3,007 | 1 735 | 24,2 | sans A/A (ASK-34) |

¹ Le streaming court a été mesuré avant `stream_frame_ms_p50` (`0e7588a9`) ; A/A jugé sur `total_ms` (passe de moins
de 60 s).
² Cellule dont le flux portait deux fois l'audio (K-92, corrigé par `599d622a`) : `audio_s` et `rtf` de la ligne
BENCH sont faux. Le RTF affiché est celui de la ligne × 2, car `audio_s` y vaut exactement 2 × frames × 80 ms.
L'`out_sha256` de ces cellules n'est pas une référence. Le temps par frame n'est pas touché : la tranche doublée
est une copie du waveform déjà décodé.
⁴ TTFT ou TTFA hors référence : ses deux passes s'écartent de plus de 3 % (pour une plage de graines, au moins une
graine). Liste : `BENCHMARKS.md`, reprise du 2026-10-10.

Ligne consommateur (FluxForge : tts-4b-6bit, `clone_fr`, `--warm-up`, batch, texte moyen) : **38,47 ms/pas**
(p90 41,11), TTFT 325 ms, RTF 0,513, 931 frames dont **7 frames de porteur** (dernière synthèse d'amorçage), pic
MLX 8,0 Go, `phys_footprint` 13,0 Go ; A/A 0,05 % (38,47 / 38,49). La paire de la matrice d'origine (deux passes
dans un processus, `--cooldown 120`) donnait 38,47 / 38,47 et la même sortie. Les paires mesurées sans repos
après l'amorçage allaient de 40,4 à 46,6 ms/pas : c'était la chaleur, pas une dispersion propre à la ligne.

## Lecture

- **Le 4 bits décode 1,5 fois plus vite que le 6 bits** (24,8 à 25,9 contre 38,0 à 39,0 ms/pas), le bf16 (`mlx`)
  3,4 à 3,5 fois plus lentement (129 à 137 ms) : RTF 0,34 à 0,38 (4 bits), 0,51 à 0,56 (6 bits), 1,67 à 1,83 (bf16,
  plus lent que le temps réel en batch). Le temps par pas varie peu avec la longueur du texte : + 2 à 6 % du court
  au long selon le pack. Même signature bf16 que le STT et le Realtime (K-34, K-36) : K-38, K-46.
- **La voix clonée démarre plus vite** : TTFT court 139 à 160 ms contre 232 à 258 (4 et 6 bits), 282 à 287 contre 366
  à 373 (bf16). Cause non vérifiée.
- **Le streaming coûte cher dès le texte moyen** : ms/frame 49 à 80 (4 bits), 55 à 58 (6 bits), 158 à 160 (bf16)
  sur le moyen ; 85 à 109, 96 à 101 et 202 à 214 sur le long. RTF du streaming : 0,72 à 2,09 sur le moyen, 1,45 à
  3,0 sur le long, plus lent que le temps réel dès le moyen pour le bf16 et le 4 bits cloné, sur le long pour les trois packs.
  Cause : chaque morceau re-décode tout l'accumulé (ASK-34). Référence « avant » de K-43.
- **TTFA du streaming court** : 210 à 303 ms (4 bits), 247 à 346 (6 bits), 555 à 649 (bf16).
- **Le pic mémoire suit la longueur, par le codec** : le codec décode toute la séquence d'un bloc. Pic MLX de la phase
  `codec` 21 à 29 Go sur le texte long en batch (`phys_footprint` 31 à 41 Go), contre 2,5 à 3,7 Go pour le décodage
  LLM en 4 et 6 bits (8,6 Go en bf16). Même plafonné, le texte long garde 21 à 29 Go de `phys_footprint` : sur
  un appareil de 32 Go, la phase codec fixe la limite.
- **Le plafond de cache (`--cache-limit-mb 2048`) ne change pas la vitesse du batch long de façon mesurable et
  retire 9 à 12 Go** : sur les 6 cellules batch long, mesurées sans amorçage dans les deux profils, l'écart de
  `step_ms_p50` va de − 1,0 % à + 2,9 % (moyenne des deux passes, paire plafonnée qui passe l'A/A), sous le seuil de
  5 %. Le `phys_footprint` baisse de 9,3 à 12,3 Go (41,3 → 29,0 Go pour bf16 prédéfinie). Le texte moyen et la
  ligne consommateur ne sont pas comparés : leurs paires plafonnées ont été mesurées sans repos après l'amorçage.
  Le bloc de comparaison est dans `BENCHMARKS.md` (reprise du 2026-10-10). Le défaut public (pas de plafond) est
  une décision de Vincent. Deux premiers relevés ne tiennent pas : 8 à 19 % de surcoût (comparaison à la matrice
  d'origine, autre protocole et autre macOS), puis − 3,9 à + 4,8 % sur 11 cellules (texte moyen mesuré chaud).
- **Constat sans cause isolée** (à reprendre, pas de cause écrite) : le 4 bits avec la voix clonée produit beaucoup
  plus de frames pour le même texte (1 806 contre 1 123 sur le moyen, 2 149 contre 1 932 sur le long) et son pic
  MLX double sur le moyen (19,0 contre 9,1 Go). Le temps par pas est le même (25,34 contre 25,15 ms). L'aller-retour
  ASR n'est fait qu'en voix prédéfinie : la qualité de ces sorties n'est pas mesurée.
- **Retiré le 2026-10-10** : « le 6 bits est plus lent sur le texte moyen » (44,3 et 46,2 ms/pas). Reposées, ces
  cellules sont à 38,45 et 38,50, comme le court et le long ; l'écart venait de la chaleur.

## Aller-retour ASR (TTS → STT, `voxtral eval tts-roundtrip`, batch, `neutral_female`, mini-3b-8bit `.mlx`)

| Pack | court (graines 1 / 2 / 3) | moyen | long |
|---|---|---|---|
| tts-4b-6bit | 0,909 / 0,909 / 0,909 | 0,940 / 0,946 / 0,964 | 0,952 / 0,958 / 0,946 |
| tts-4b-4bit | 0,909 / 0,909 / 0,909 | 0,976 / 0,952 / 0,952 | 0,955 (graine 1) |
| tts-4b-mlx ³ | 0,909 / 0,909 / 0,909 | 0,952 (graine 1) | 0,955 (graine 1) |

Couverture = part des mots du texte retrouvés dans l'ordre. Sur le court, 10 mots sur 11 pour tous les packs et
toutes les graines. Référence des portes qualité de K-39, K-48, K-58 et K-79.

³ EVAL bf16 refaits le 2026-10-09 par `--voice-embedding`, le chemin des BENCH (même `out_sha256`, 5 sur 5). Par
`synthesize(voice:)`, le bf16 sort un autre audio à même graine (4 textes sur 5 ; long 0,949) ; les 4 et 6 bits
sont identiques par les deux chemins. Fiche de suite.

## Variance d'un processus à l'autre

Le 2026-10-08 au soir, 5 cellules sur 9 mesurées ce soir-là dans le profil par défaut ont échoué l'A/A (3,8 à
69,8 % sur `step_ms_p50`) ; les cellules bf16 n'ont tourné que le lendemain. Pour 3 d'entre elles,
l'échantillonneur a relevé de la charge pendant la passe lente : compilations Xcode et Podcasts. Refaites le
2026-10-09 après-midi, 5 sur 5 sont passées. Une partie de ces échecs touchait le texte moyen, alors mesuré sans
repos après l'amorçage.

Avec 120 s de repos (2026-10-10), 2 des 6 passes de la cellule bf16 prédéfinie batch tournent 12 et 25 % plus
lentement (147,80 et 165,95 contre 131,5 à 132,4 ms/pas), sans charge bloquante relevée, et deux cellules streaming moyen restent hors A/A (4 bits et bf16 clonée).
Même régime que K-85 (bimodalité par processus, cause non isolée) : une cellule qui échoue sans charge relevée se
refait, sans cause écrite.

## Contrôle macOS

Les 36 cellules courtes ont tourné sous macOS 27.0.0, deux passes dans un processus ; toutes les autres sous 27.0.1,
un processus par passe. Quatre cellules courtes (graine 1, voix prédéfinie : 6 bits, 4 bits et bf16 en batch, 6 bits en
streaming) ont été remesurées sous 27.0.1, un processus par passe. Écart à la matrice : − 0,2 %, − 0,2 %, + 0,3 % sur
`step_ms_p50` et − 0,6 % sur `total_ms` (streaming), A/A 0,08 à 0,19 %, sortie identique bit à bit dans les quatre
cas. Ni la mise à jour du système ni le régime à un processus ne déplacent les cellules courtes.
