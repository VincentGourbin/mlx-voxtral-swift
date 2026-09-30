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
- 2026-09-30 — **K-6 : complétude des téléchargements prouvée** : un modèle n'est « téléchargé » que si
  `.voxtral-complete.json` (écrit en dernier par `downloadRepoDirect`, SHA-256 `lfs.oid` vérifié par fichier) est
  présent et ses tailles justes ; les dossiers anciens (index + shards + `tekken.json` + voix) reçoivent leur
  manifeste au premier contrôle, les dépôts sans index (originaux Mistral) repassent une fois en ligne. Coupure à
  51 % puis relance : reprise complète. Le détecteur MLX-012 ne voit pas ces variantes (0/0).
- 2026-09-30 — **K-16 : état partagé protégé** : plus aucun `nonisolated(unsafe)` dans `VoxtralCore` (boîte `Locked`),
  la configuration mémoire est portée par chaque pipeline (créer une pipeline écrasait celle des autres), et les
  types `@unchecked Sendable` porteurs d'`MLXArray` évaluent dans leur `init`. Avant : 11 courses TSan dans
  VoxtralCore et un plantage du test concurrent ; après : 0. **TSan signale toujours 2 courses dans MLX**
  (`MetalAllocator`, `active_memory_` lu sans verrou, jusqu'à mlx main) : bruit amont, à ignorer dans les portes TSan.
- 2026-09-30 — **K-25 : encodeur Core ML sous `customModelsDirectory`** : téléchargé par `downloadRepoDirect` avec
  le manifeste K-6 (`<racine>/<org>/<repo>/<nom>.mlmodelc`), rechargé hors ligne, plus rien dans
  `~/.cache/huggingface`. Un encodeur mini sous configuration small, ou l'encodeur MLX aléatoire par défaut, lève
  une erreur. Astuce de test : `sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip "*:*"))'`
  coupe le réseau d'un seul processus sans toucher au Wi-Fi.
- 2026-09-30 — **rôles consignés (#599)** : seule la session Voxtral du Mac committe ici ; planification et
  vérification dans action-plans par une session cloud ; ASK et fusions à Vincent. `machine-check.sh` doit recevoir
  `--procs 'Voxtral.*|FluxForge.*'` : FluxForge Studio charge MLX sur le même GPU.
- 2026-09-30 — **K-1 : erreurs MLX converties en `VoxtralError.mlx`** aux points d'entrée publics (STT, TTS, Realtime)
  : le déclencheur P-03 (préfill de 2 600 positions dans un cache glissant de 2 048) ne tue plus le processus.
  Leçon : une erreur capturée laisse des tableaux vides et la couche suivante piège en Swift en lisant leur forme ;
  il faut s'arrêter entre les couches (`MLXErrorScope.hasError`). Surcoût : +0,06 % (bruit). Mesure : amorcer
  **chaque** binaire avant A/B/B/A (le 1ᵉʳ lancement d'un binaire neuf coûte ≈ 2 s de cache Metal).
- 2026-09-30 — **K-4 : jetons d'arrêt STT = ceux du tokenizer** (`</s>`, `[/INST]`), plus l'id 32000 (« ␣Capital ») :
  le clip de test passe de « Capital A's and Capital » à la phrase complète ; greedy identique sur C-court EN/FR et
  C-moyen EN. En Tekken, un id ≥ 1 000 est toujours un mot (id = rang + 1 000).
- 2026-09-30 — **K-7 : poids vérifiés au chargement, tokenizer strict**. La vérification a trouvé que
  `realtime-4b-fp16` chargeait toute l'attention du décodeur au hasard (noms `wq/wk/wv/wo` non traduits) : il
  transcrivait « .. » ; corrigé. Une constante calculée stockée en `MLXArray` sur un `Module` compte comme paramètre :
  la préfixer par `_`. Le CLI ignore `--seed` avec `-v <voix>` : pour une parité TTS, passer `--voice-embedding`.
- 2026-09-30 — **K-11 : une opération GPU à la fois par pipeline** (`PipelineGate`) : l'interblocage ABBA compile ×
  vjp de mlx-swift 0.31.6 est **reproduit** (enrôlement ∥ streaming : blocage dès le 1er essai) et évité (20/20, refus
  `busy` immédiat). `unload()` pendant une opération : la Task périmée ne réécrit plus l'état (jeton de génération).
