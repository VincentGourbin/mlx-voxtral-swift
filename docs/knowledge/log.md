# Journal de la base de connaissances

Une entrée par mesure, hypothèse réfutée, décision ou correctif, dans l'ordre d'arrivée
(`- AAAA-MM-JJ — **titre** : …`). Source de vérité des chiffres cités ailleurs.

Rappels de discipline : `machine-check.sh` sans ligne `KO` avant toute mesure ; binaire Release ; révisions résolues
de mlx-swift, mlx-swift-lm et swift-mlx-profiler notées ; une ligne `BENCH` recopiée dans `BENCHMARKS.md`, jamais
éditée (voir [`CLAUDE.md`](../../CLAUDE.md), [`docs/Benchmarks.md`](../Benchmarks.md)).

- 2026-09-27 — **audit** : audit `mlx-swift-audit` (phases 0 à 4) à `9392ed1` (= tag `v2.2.2`), session cloud
  Linux sans Mac : aucun build, aucun test, aucune mesure. Livrables dans
  [`docs/audit/2026-09-27/`](../audit/2026-09-27/README.md) : 7 rapports, profils, plan de 82 fiches, 31 décisions
  ASK, `tasks.yaml` (76 tâches `macos-gpu`) (PLAN.md §7). Aucun chiffre publié avant cette date n'est une référence
  (faits-et-actions.md §2.1, §2.8).
- 2026-09-27 — **mémoire du projet (K-17)** : création de `CLAUDE.md`, `BENCHMARKS.md` (en-tête seul),
  `docs/Benchmarks.md` (protocole, corpus, glossaire), de la décision
  [#23-#25 caduques](decisions/realtime-diagnostics-23-25.md) et de six pièges (index : [`index.md`](index.md)).
  Aucune mesure.
- 2026-09-28 — **définition du TTFT-frame corrigée (revue des fiches cloud)** : le TTFT TTS n'exclut le préfill
  des trames de voix que si le préfixe vient du cache par voix (voix prédéfinies ; streaming avec `voiceKey` ;
  `Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:210`, `:512`), cache introduit par `f4fd21c`
  (2026-07-10). Les TTFT publiés avant (banc `6ad4e56`) l'incluent ; une voix clonée en batch l'inclut encore
  (`:333-340`). Glossaire : [`docs/Benchmarks.md`](../Benchmarks.md) §4. Aucune mesure.
- 2026-09-28 — **catalogue de patterns renuméroté** : claude-skills a publié MLX-016 (experts MoE) en 0.5.0 ; les
  patterns issus de cet audit sont MLX-017…025 dans claude-skills 0.6.0 (mlx-swift 0.4.0, claude-skills#1).
  Correspondance provisoire → définitif en tête de `docs/audit/2026-09-27/patterns-verdicts.md`. MLX-016 sur ce
  dépôt : 3 filtres `Linear || Embedding`, sans objet (aucun module MoE).
