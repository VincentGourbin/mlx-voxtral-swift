# Décision — les conclusions des issues #23, #24 et #25 (Realtime) sont caduques

**Contexte** (2026-09-27, audit `mlx-swift-audit` à `9392ed1`) : les issues #23, #24 et #25 (profil
`realtime-4b-4bit` du 2026-04-11, machine et durée d'audio non notées) ont été fermées le même jour sur trois
conclusions :
- #23 : « measurement artifact — IOKit sampling … 21 tok/s is reasonable » ;
- #24 : « systemic 49 % GPU cap … root cause is MLX/Metal memory allocation pattern » ;
- #25 : « acceptable — short prompt doesn't saturate GPU ».

Source : audit-performance-realtime-instruments.md §2 (F-R1, F-R2).

**Constat** (P-73, statut VÉRIFIÉ en lecture de code ; aucun chiffre nouveau) :
- **Phases imbriquées** : « Realtime Generation » contient « Audio Encoding » et « Prefill »
  (`Sources/VoxtralCore/Realtime/Pipeline/VoxtralRealtimePipeline.swift:134`,
  `Sources/VoxtralCore/Realtime/VoxtralRealtimeModel.swift:73`, `:106`). Avec swift-mlx-profiler ≤ 1.4, les pas ne
  sont plus attribués à la phase englobante : son GPU % est la moyenne de deux lectures de bord, d'où « 0 % ». Les
  « 21 tok/s » divisent par une durée qui inclut encodage et préfill : 501 × 33,7 ms = 16,9 s de décodage, soit
  29,7 pas/s (§3 bis).
- **Lecture instantanée** : le « 49 % » de #24 (comme celui de #13 et #14) est une moyenne entière de lectures
  instantanées de « Device Utilization % » prises aux bords d'une phase, pas une occupation mesurée. L'hypothèse
  (≈ 0 + ≈ 98) / 2 ≈ 49 est compatible, non établie (F-R4).
- **Mel et poids paresseux** : la phase « Mel Spectrogram » ne mesure que la lecture audio ; le STFT et la lecture
  disque des poids sont facturés à l'encodage et au préfill (P-66).
- **Run probablement tronqué** : 501 pas = plafond + 1 (très probable, non prouvé ; P-64).
- **Défaut masqué** : 33,7 ms/pas n'est pas « raisonnable » : la tête liée est recopiée en fp32 à chaque pas
  (P-61, ≈ 5,8 Go lus ou écrits par pas, calcul). Voir le piège
  [tête liée recopiée en fp32](../pitfalls/tied-head-fp32-copy.md).
- **« Peak MLX Active » n'est pas un pic** : c'est le maximum des instantanés `activeMemory` (profiler ≤ 1.4).

**Décision** :
- Les conclusions de #23, #24 et #25 sont **caduques**. Aucun chiffre de ces issues n'est une référence ; aucune
  cause « allocation Metal » n'est retenue.
- On ne rouvre pas les issues depuis l'audit (P-73, « Correction »). La re-mesure passe par l'instrument
  `VoxtralCLI bench` (K-32), puis la fiche **K-36** : baseline Realtime et occupation GPU du décodage rapportée de
  deux façons (swift-mlx-profiler 1.5.x `.fineGrained` avec `.ioReportResidency`, et Metal System Trace ; écart
  ≤ 10 pts).
- Le défaut principal est traité par les leviers K-38 (dtype du modèle) et K-46 (tête quantifiée).

**Mesure** (K-36, 2026-10-05, M3 Max 96 Go, `BENCHMARKS.md` §« 2026-10-05 — K-36 », A/A ≤ 3 % sur `step_ms_p50`
pour les 8 cellules) :

| Modèle | Clip | ms/pas p50 / p90 | TTFT ms | encodage ms par s d'audio | RTF | pic `phys_footprint` Mo | pas |
|---|---|---|---|---|---|---|---|
| realtime-4b-4bit | C-court EN | 25,45 / 25,53 | 193 | 23,19 | 0,40 | 6 206 | 72 |
| realtime-4b-4bit | C-moyen EN | 27,25 / 28,37 | 2 958 | 19,58 | 0,36 | 10 590 | 1 835 |
| realtime-4b-4bit | C-moyen FR | 27,22 / 28,74 | 2 689 | 19,76 | 0,37 | 10 431 | 1 645 |
| realtime-4b-4bit | C-long | 29,26 / 33,53 | 14 521 | 20,85 | 0,40 | 7 908 | 6 933 |
| realtime-4b-fp16 | C-court EN | 125,88 / 126,35 | 410 | 26,18 | 1,88 | 12 930 | 72 |
| realtime-4b-fp16 | C-moyen EN | 135,97 / 141,79 | 3 438 | 21,38 | 1,74 | 16 538 | 1 835 |
| realtime-4b-fp16 | C-moyen FR | 136,16 / 142,19 | 3 075 | 21,11 | 1,75 | 16 379 | 1 645 |
| realtime-4b-fp16 | C-long | 148,29 / 161,11 | 13 038 | 22,74 | 1,92 | 14 296 | 6 933 |

(passe p1 de chaque cellule ; C-moyen = C-moyen exact, 146,1 / 130,9 s ; C-long = C-long exact, 9 min 14 s (K-33,
`ASK.md` §Dérogations) ; C-long sans amorçage, `--cache-limit-mb 2048`.)

- **Occupation GPU du décodage** (« Realtime Generation », `bench --trace --metal-trace` sur `c_20s_en`) : 4 bits
  profiler 99,3 % · xctrace 92,3 % · `ioreg` 95,3 % (écart profiler/xctrace 7,0 pts) ; fp16 99,9 % · 98,4 % ·
  98,5 % (1,5 pt). Le GPU est occupé pendant tout le décodage : le « 49 % » de #24 n'était pas une occupation.
  Recoupement sur C-moyen EN complet (4 bits, trace de la 1re série) : xctrace 93,0 %, hors encodage et préfill.
- **Coût d'encodage** (amendement du 2026-10-03, remesuré à froid) : 19,6 à 23,2 ms par seconde d'audio (4 bits) et
  21,1 à 26,2 (fp16) en p1. Dispersion p1/p2 notable sur C-long 4 bits : 20,85 → 27,16 (+26 %), alors que
  `step_ms_p50` ne varie que de 2,6 %. La valeur haute de K-13 (27,76 sur 12 min) se reproduit donc d'une passe à
  l'autre sur l'audio long. L'encodage C-long n'est pas une référence stable, à remesurer par K-38 et K-46 avant
  tout gain annoncé.
- **WER** (`voxtral eval realtime`, clips à texte exact) : C-court 9,09 %, C-moyen EN 1,05 %, C-moyen FR 1,97 % en
  4 bits comme en fp16 ; C-long 1,15 % (4 bits) / 1,40 % (fp16). Contre la référence Realtime (`rt_ref_*`, K-33) :
  EN 5,15 / 7,96 %, FR 2,44 / 4,07 % (dernière phrase couverte à 0,625 / 0,375 : le juge Realtime n'est pas fiable
  sur du français, K-33).
- **`pad_fraction`** (tout pas sans texte) : 0,82 / 0,76 / 0,69 / 0,72 ; **`[STREAMING_PAD]` seul**
  (`streaming_pad_fraction`, entrée de K-73) : 0,70 / 0,60 / 0,52 / 0,56 (C-court, C-moyen EN, C-moyen FR, C-long).
  `[STREAMING_PAD]` est le rang 32 de `tekken.json` (`mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`).
- **fp16 n'est pas temps réel** (RTF 1,7 à 1,9 ; 126 à 148 ms par pas de 80 ms) : 5 fois le 4 bits, cohérent avec la
  tête liée recopiée en fp32 (piège ci-dessus), à traiter par K-38 et K-46.
- **Invite** (constat, hors périmètre) : `streamingPadTokenId` vaut 11, `<pad>` dans `tekken.json` ; `[STREAMING_PAD]`
  est 32 et la référence mlx-audio utilise 32 avec 32 jetons de remplissage à gauche (nous : 1). Fiche de suivi K-84.

**Non retenu** :
- Rouvrir #23-#25 : l'audit n'ouvre pas d'issue (P-73).
- Garder le « 49 % » comme plafond systémique : il ne mesure pas l'occupation (voir le piège
  [même % GPU](../pitfalls/same-gpu-percent-instrument-artifact.md)).

**Ouvert** : « 49 % » du préfill STT et de Core ML (#13, #14 ; K-34, K-42). L'occupation du décodage Realtime est
mesurée (K-36, ci-dessus).

Source : [audit-performance-realtime-instruments.md](../../audit/2026-09-27/audit-performance-realtime-instruments.md)
§2, §3 bis et P-73 ; [faits-et-actions.md](../../audit/2026-09-27/faits-et-actions.md) ACT-27.
