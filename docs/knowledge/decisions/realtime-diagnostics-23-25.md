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

**Mesure** : aucune à ce jour. K-36 complète cette décision avec ses lignes de `BENCHMARKS.md`.

**Non retenu** :
- Rouvrir #23-#25 : l'audit n'ouvre pas d'issue (P-73).
- Garder le « 49 % » comme plafond systémique : il ne mesure pas l'occupation (voir le piège
  [même % GPU](../pitfalls/same-gpu-percent-instrument-artifact.md)).

**Ouvert** : occupation GPU réelle de l'encodage et du décodage Realtime (À MESURER, K-36) ; même question pour le
« 49 % » du préfill STT et de Core ML (#13, #14 ; K-34, K-42).

Source : [audit-performance-realtime-instruments.md](../../audit/2026-09-27/audit-performance-realtime-instruments.md)
§2, §3 bis et P-73 ; [faits-et-actions.md](../../audit/2026-09-27/faits-et-actions.md) ACT-27.
