# Plan — mlx-voxtral-swift : stabilité, performance et profils de référence — 2026-09-27

> Produit par le skill `mlx-swift-audit` (bêta, phases 3 et 4). Révision auditée : `9392ed1` (= tag `v2.2.2`),
> branche `claude/action-plan-skills-beta-wifgmu`. Rapports d'audit : `docs/audit/2026-09-27/` (index :
> [`README.md`](README.md)). Profils : [`profils.md`](profils.md). Décisions : [`ASK.md`](ASK.md). Fiches :
> [`fiches/`](fiches/). Tâches Mac : [`tasks.yaml`](tasks.yaml) (format task-dispatch).
>
> **82 fiches** : 6 exécutables en cloud (documentation, scripts Python, tracker), 76 sur `macos-gpu` (build, tests,
> mesure ou écoute). Aucune n'était exécutée à la fin de la phase 4 (2026-09-27) ; état courant : État (§3), journal §7.

## 0. Règles de mesure

Binaire **Release** (`xcodebuild -scheme VoxtralCLI -configuration Release`) · `machine-check.sh` vert
(`~/.claude/skills/mlx-swift-audit/scripts/machine-check.sh $CLI --cooldown 120 --procs 'Voxtral.*|FluxForge.*'`) ·
refroidissement 120 s · **un levier par comparaison** · A/B/B/A (deux passes par variante, une requête d'amorçage
exclue) · une différence n'est lue que si elle dépasse la dispersion A/A · **gain < 5 % = bruit, le levier est
retiré** · **une ligne JSON par mesure** (`BENCH {…}`, instrument K-32) recopiée telle quelle dans `BENCHMARKS.md`
avec les révisions **résolues** de mlx-swift, mlx-swift-lm (dépendance sur `main`) et swift-mlx-profiler · parité
avant/après sur le **checkpoint réel**.

Précisions propres à Voxtral :
- **Tests en Debug, mesures en Release** : les 43 fichiers de tests font `@testable import VoxtralCore` ; une porte de
  test se prouve par `xcodebuild test` (Debug), une porte de mesure par `VoxtralCLI bench` (Release). Une campagne
  XCTest chronométrée (P-76) n'est pas une référence sauf build Release avec `ENABLE_TESTABILITY=YES`.
- **Tests sans parallélisme** (`-parallel-testing-enabled NO`) : jamais de gradient pendant une inférence (piège 20,
  A-01).
- **Parité par chemin** : STT et Realtime = transcription greedy identique ou WER normalisé (casse, ponctuation,
  accents : la référence FR est sans accents) dans la tolérance de la fiche (ASK-11) ; TTS = **bit-identique à graine
  fixée** pour les leviers exacts, **parité forcée par l'enseignant** + couverture ASR pour les leviers numériques
  (le FM est stochastique à rétroaction : une comparaison libre diverge par construction) ; enrôlement = codes
  identiques à graine fixée.
- **Deux conventions de RTF coexistent dans le dépôt** (faits-et-actions.md §2.1) : ici **RTF = génération / audio**
  (< 1 = plus vite que le temps réel) ; **TTFT-frame** (premier frame de codes interne) ≠ **TTFA** (premier
  échantillon audio reçu par le consommateur).
- Toute valeur antérieure à ce plan est **« en session »** : aucune n'est une référence (faits-et-actions.md §2.8).

## 1. Faits vérifiés (ne pas re-dériver)

| Fait | Preuve (fichier:ligne, commande, source) |
|---|---|
| `9392ed1` = tag `v2.2.2` ; 0 issue ouverte, 0 PR ouverte ; seule branche distante `main` | MCP GitHub `list_tags`/`list_issues`/`list_pull_requests`, `git ls-remote --heads` (2026-09-27) |
| Voxtral résout mlx-swift **0.31.6** (`0bb916c`, 2026-07-01, MLX C++ `ce45c52`) : `from: "0.31.6"` et mlx-swift-lm impose `.upToNextMinor(from: "0.31.6")` ; aucun tag > 0.31.6 | `Package.swift:43` ; mlx-swift-lm `Package.swift:64` ; `git ls-remote --tags` |
| mlx-swift-lm suivi sur `branch: "main"` (tête `ee673d6`, 2026-09-22 ; dernier tag 3.31.4) ; `Package.resolved` ignoré par git | `Package.swift:46-52` ; `.gitignore:27` |
| Le correctif du deadlock compile × vjp (`df9ae26`, #461) n'est dans **aucun** tag de mlx-swift | `git tag --contains df9ae26` vide (audit-annexes-serveur.md A-01) |
| Sans `withError`, une erreur MLX appelle `fatalError(message)` ; Voxtral n'appelle jamais `withError` | mlx-swift 0.31.6 `ErrorHandler.swift:337-347` ; `grep -rn withError Sources` = 0 |
| Tekken : id = rang + 1 000 ; **id 32000 = « ␣Capital »** ; la génération STT s'arrête sur `[2, 4, 32000]` | `VoxtralComponents.swift:155-176` ; `VoxtralModeling.swift:1124` ; `tekken.json` rang 31000 |
| Le LM Voxtral n'a pas de fenêtre glissante (`sliding_window: null`, 131 072 positions) ; 375 jetons audio par fenêtre de 30 s | `config.json` HF ; `VoxtralProcessor.swift:389-390` |
| Tous les préréglages mémoire posent `maxKVCacheSize` (2 048 / 4 096 / 6 144 / 8 192) ⇒ `RotatingKVCache` + masque `[T, offset+T]` ⇒ arrêt au préfill à partir de 6 / 11 / 17 / 22 fenêtres ; l'app impose 8 192 | `MemoryOptimizationConfig.swift:41-90` ; `VoxtralModeling.swift:1135-1147` ; `VoxtralStandardLoader.swift:450-481` ; simulation (audit-performance-stt.md annexe A) |
| `maxTokens` STT = 500 par défaut (≈ 3 min de parole) ; Realtime compte des trames (4 096 ≈ 5 min 27 s) | `VoxtralPipeline.swift:109` ; `VoxtralRealtimeModel.swift:115-124` |
| `update(parameters:)` non levant = `try! … verify: .none` ; les 3 chargeurs vivants ne vérifient ni clés ni formes | mlx-swift `Module.swift:401-408` ; `VoxtralStandardLoader.swift:1323`, `:1338` ; `VoxtralTTSModelLoading.swift:71` ; `VoxtralRealtimeModelLoading.swift:55` |
| « Téléchargé » = `config.json` / `params.json` présent ; sans index = « complet » ; aucun SHA-256 | `ModelDownloader.swift:206-210`, `:312-335`, `:537-592`, `:644-702` |
| `small-24b-8bit` : enum → `mzbac/…` (28 056 927 031 o), registre → `VincentGOURBIN/…` (26 499 134 369 o) | `VoxtralPipeline.swift:47-48` ; `ModelRegistry.swift:90-91` |
| Le glob `*.safetensors` télécharge aussi `consolidated.safetensors` (mini-3b 18,7 Go au lieu de 9,36) | `ModelDownloader.swift:361-365`, `:388-392` ; listings HF |
| Le « streaming » TTS génère tout dans la closure synchrone de `AsyncThrowingStream` ; `Task` sans `onTermination` | `VoxtralTTSModeling.swift:569-685` ; `VoxtralTTSPipeline.swift:556-671` |
| `enrollVoice` ne marque jamais la pipeline occupée ; `silu` (compilé par MLXNN) est traversé par le vjp ⇒ ABBA possible avec une synthèse | `VoxtralTTSPipeline.swift:425-452` ; `VoxtralCodecDecoder.swift:295-296` ; mlx-swift 0.31.6 `Transforms+Compile.swift:12,39-43,89` |
| STT : features mel fp32 jamais castées ⇒ préfill, KV (240 Kio/jeton Mini) et décodage en fp32 ; en bf16 chaque `Linear` recopie son poids en fp32 | `VoxtralFeatureExtractor.swift:298-335` ; `VoxtralModeling.swift:979` ; MLX `ops.cpp:3069-3082`, `dtype.cpp:45,48` |
| TTS : `decodeOneFrame` caste l'état caché en fp32 ⇒ FM en fp32 ; attention du codec en T×T fp32 (≈ 10,5 Go par tenseur à 2 266 frames, **calcul**, à mesurer) | `VoxtralFlowMatching.swift:289` ; `VoxtralCodecDecoder.swift:210-231` |
| Realtime : `logits = matmul(h, tokEmbeddings.weight.T)` avec `h` fp32 ⇒ copie fp32 de la table 131 072 × 3 072 (1,5 Gio par pas, **calcul** depuis la forme, à mesurer) ; `tok_embeddings` jamais quantifié | `VoxtralRealtimeDecoder.swift:187-189` ; `VoxtralRealtimeModelLoading.swift:39` |
| Realtime : fenêtres de la config (encodeur 750, décodeur 8 192) ignorées ; la référence mlx-audio les applique | `VoxtralRealtimeEncoder.swift:151-159` ; `VoxtralRealtimeDecoder.swift:191-194` ; mlx-audio `encoder.py:188-219`, `decoder.py:226-229` |
| `flowSteps`, `cfgAlpha`, `temperature` du TTS ne sont lus nulle part ; 8 pas et 1,2 codés en dur | `VoxtralFlowMatching.swift:195-196` ; `grep` |
| Aucune `Memory.cacheLimit` dans `VoxtralCore` ; seule pose `0` puis `Int.max` (qui n'est **pas** le défaut) dans une fonction morte de l'app | `TranscriptionManager.swift:285-297` ; MLX `allocator.cpp:52-54` |
| La bibliothèque remet le pic MLX à zéro tous les 2 à 16 jetons en STT | `VoxtralModeling.swift:1256-1257`, `:1433-1434` ; `MemoryOptimizationConfig.swift:44-68` |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` a gagné un `config.json` et un `model.safetensors` transformers : l'entrée `realtime-4b` ne se charge plus | listing HF (2026-09-27) ; `VoxtralRealtimeModelLoading.swift:63-77` |
| Les packs STT de 2026 (Markus, aufklarer) portent `"mode": "affine"` : refusés par le décodeur STT ; mode ignoré en TTS/Realtime | `VoxtralStandardLoader.swift:87-114` ; `VoxtralTTSModelLoading.swift:58` |
| Aucun nouveau checkpoint Voxtral chez Mistral depuis `4B-TTS-2603` ; TTS sous **CC BY-NC 4.0** ; pas d'encodeur de codec publié | Hub HF, MCP (2026-09-27) |
| Consommateurs connus : FluxForge Studio (App Store, chaîne LipDub/LTX : `VoxtralPipeline(.mini3b4bit)`, TTS 6 bits + voix enrôlées + warm-up, batch), SongAnalysisDb ; 0 usage connu de l'API legacy ni du Realtime | recherche de code GitHub (audit-stabilite.md §0 ; audit-performance-tts.md en-tête) |
| Aucune mesure publiée n'est une référence (froid, pas d'A/B/B/A, deux RTF, TTFT ≠ TTFA, instrument contesté) | faits-et-actions.md §2.1, §2.8 ; audit-performance-realtime-instruments.md P-73 |
| Le texte de référence « Full test texts » fait 163 mots EN / 202 mots FR pour 167 / 174 s d'audio (« ~350 words » annoncés) : peut-être abrégé | `docs/tts_benchmark.md:24-31`, `:135-151` ; `wc -w` (à contrôler par K-33) |
| Pas de `CLAUDE.md`, `AGENTS.md`, `BENCHMARKS.md`, `docs/knowledge/` ; pas de CI | `scan.md` §7 ; `.github/` absent |

## 2. Baseline (mesures de référence avant toute modification de performance)

Toutes les cellules sont **À MESURER** ; aucune valeur antérieure n'est reprise comme référence. Instrument : K-32
(`VoxtralCLI bench`, A/A ≤ 3 %) ; qualité : K-33 (`VoxtralCLI eval`). Les lignes produites vont dans `BENCHMARKS.md`.

| Chemin | Workload | Préfill tok/s | Décodage tok/s | TTFT | Pic phys_footprint | Ligne BENCHMARKS |
|---|---|---|---|---|---|---|
| STT Mini 4 / 8 / 16 bits, `.mlx` et `.auto` | C-court EN, C-moyen EN/FR, C-long | À MESURER | À MESURER | À MESURER | À MESURER | K-34 |
| STT Small 4 / 8 bits, `.mlx` et `.auto` | C-moyen EN, 8 min | À MESURER | À MESURER | À MESURER | À MESURER (≤ 24 Go ?) | K-34 |
| Realtime 4 bits / fp16 | C-court, C-moyen EN/FR, C-long | À MESURER (préfill ≤ 32 jetons) | À MESURER (`step_ms_p50`/`p90`, budget 80 ms) | À MESURER (1er jeton texte) | À MESURER | K-36 |

| Chemin | Workload | Débit | Latence | Autres | Pic phys_footprint | Ligne BENCHMARKS |
|---|---|---|---|---|---|---|
| TTS 4 / 6 / 16 bits × prédéfinie / clonée × batch / streaming | court, 60 s, long ; graines 1-3 ; chemin FluxForge (6 b + clonée + warm-up) | fr/s : À MESURER | TTFT-frame, TTFA : À MESURER | RTF (gén/audio), décodage codec, frames du porteur : À MESURER | À MESURER | K-35 |
| Enrôlement 6 / 16 bits | `clone_fr.wav`, 200 époques, graine 7 ; scénarios CLI et synthèse → enrôlement | s/époque : À MESURER | — | perte finale : À MESURER | À MESURER | K-37 |
| Chat (Q/R sur l'audio, `VoxtralPipeline.chat`) | mini-3b-8bit × `.mlx` / `.auto` × C-moyen EN × 4 questions fixes (`docs/eval/chat-questions.json`, greedy) | — | tok/s : À MESURER | `ttft_ms` par question : À MESURER | — | À MESURER | K-34 |
| Qualité | corpus `docs/eval/corpus.json` | — | — | WER STT et Realtime (lignes de K-34 et K-36), couverture ASR TTS (K-35), juge Realtime, auto-détection | — | K-33 |

## 3. Fiches

**Ordre imposé** (skill) : 1. stabilité bloquante (arrêt du processus, deadlock, sortie silencieusement fausse,
fuite) → 2. hygiène sans risque → 3. **baseline mesurée** de chaque chemin → 4. leviers perf par gain attendu
décroissant → 5. type de profils + CLI `references` / `--reference` → 6. mesures de la matrice et docs
(`References.md`, `Weights.md`, `docs/knowledge/`).

- **Cible** : `cloud` = documentation, scripts Python, tracker, sans build (exécutée dans la session cloud) ;
  `macos-gpu` = tout ce qui exige build, tests, mesure ou écoute (dispatchée via `tasks.yaml`). Une porte à test
  unitaire est `macos-gpu` (pas de toolchain Swift en cloud).
- **Porte** : toujours chiffrée ; gain < 5 % ⇒ code retiré. **⛔** : une décision de Vincent (ASK) est requise avant
  d'exécuter (ou de conclure) la fiche.
- Chaque fiche porte ses prérequis ; le graphe est acyclique et suit l'ordre de numérotation. **Contrôle mécanique du
  2026-09-27** (critique de complétude) : toute fiche des lots 4 à 6 atteint, par ses prérequis, la baseline de son
  chemin (K-34 STT et chat, K-35 TTS, K-36 Realtime, K-37 enrôlement) et, si sa porte cite un WER ou une couverture
  ASR, l'outil K-33 ; K-65 (+ K-36), K-74 (+ K-34), K-82 (+ K-64), K-35 et K-36 (+ K-33) ont été corrigés pour cela.
- Les fiches cloud **K-17…K-21 et K-81** n'ont aucune dépendance Mac : au 2026-09-27, K-17, K-18, K-19 et K-81 étaient
  exécutables, K-20 attendait ASK-30 et K-21 ASK-31 ; au 2026-09-28, toutes faites sauf K-20 (partielle, ASK-30).

### Lot 1 — stabilité bloquante

| Fiche | Objet | Source | Porte | Effort | Cible | Prérequis | État |
|---|---|---|---|---|---|---|---|
| K-1 | [Erreurs MLX levées au lieu de terminer le processus hôte (`withError` aux points d'entrée publics)](fiches/K-1.md) | MLX-021, S-02, P-03, P-17, MLX-019 | test `MLXErrorBoundaryTests` : invite synthétique de 2 600 positions préfillée avec `MemoryOptimizationConfig.ultra` (déclencheur P-03, présent tant que K-2 n'est pas livré) → `VoxtralError.mlx` levée, processus vivant (rouge sans correctif = plantage du runner de test, piège 38) ; suite complète verte ; durée totale de `VoxtralCLI transcribe` sur C-moyen EN inchangée à ±5 % (A/B/B/A, `/usr/bin/time`). | S | macos-gpu | — | verified |
| K-2 | [STT : cache KV sans fenêtre par défaut + garde-fou explicite (plus d'arrêt ni de perte du début sur audio long)](fiches/K-2.md) | S-02, P-03 | test `LongPromptKVCacheTests` (invite synthétique de 2 600 positions, préréglage `.ultra`) rouge → vert ; intégration `VOXTRAL_LONG_AUDIO=1` sur C-long (≈ 11 min 22 s) avec `recommended(forRAMGB: 8)` puis `(forRAMGB: 16)` et `maxTokens` 4 096 : 0 arrêt, première phrase EN « Fluxforge Studio turns your Mac into a complete AI creative studio. » et dernière phrase FR « Aucune donnee envoyee dans le cloud. » présentes (casse et accents ignorés) ; pic « peak memory footprint » consigné (ASK-7 au-delà de 12 Go). | S/M | macos-gpu | K-1 | verified |
| K-3 | [Masques d'attention construits par le cache (décodeur STT vivant et décodeur hérité)](fiches/K-3.md) | P-02, P-17, MLX-002, MLX-019 | logits du dernier jeton : écart relatif L2 ≤ 1e-3 contre l'actuel sur 3 clips (C-court EN/FR, C-moyen EN), greedy identique 3/3 ; test : un préfill à entrée bf16 ne lève plus « Mask type must promote to output type » ; test `RotatingKVCache` enroulé (préfill en 2 tranches, T ≥ 2) : 0 arrêt, masque `.bool` de la forme des clés ; invite de 600 positions via `loadVoxtralModel(modelPath:dtype:lazy:)` (chemin hérité) sans arrêt ; durée de transcription C-moyen EN ±5 % (A/B/B/A). | S | macos-gpu | K-2 | à faire |
| K-4 | [Jetons d'arrêt dérivés du tokenizer (fin de la troncature sur « ␣Capital »)](fiches/K-4.md) | S-01 | test `StopTokenTests` : chaque jeton d'arrêt est un id spécial (< 1 000), rouge avec la liste actuelle ; clip de test synthétisé « Capital Gains and Capital One are two different things. The capital of France is Paris. » transcrit au-delà du premier « ␣Capital » (« One » et « Paris » présents, 1/1) ; greedy identique avant/après sur C-court EN, C-court FR et C-moyen EN (3/3). | S | macos-gpu | — | verified |
| K-5 | [Plus de troncature silencieuse : `maxTokens` STT proportionnel à la durée, boucle Realtime bornée par l'audio](fiches/K-5.md) | P-11, P-64 | STT : C-moyen EN (167 s), C-moyen FR (174 s) et C-long : dernière phrase de la référence présente, longueur ≤ 1,5 × la référence ; Realtime : C-long transcrit en entier (dernière phrase présente) ; test unitaire « nombre de pas = nombre de trames » rouge sans le correctif ; un budget dépassé est signalé (champ `truncated` ou erreur). | S | macos-gpu | K-2 | verified |
| K-6 | [Téléchargements prouvés complets (manifeste + SHA-256), API de téléchargement factice neutralisée](fiches/K-6.md) | S-03, A-02, A-05, MLX-012 | 6 tests unitaires sur dossiers temporaires, rouges sans le correctif : 1 shard sur 5 sans index → non téléchargé ; index corrompu → non téléchargé ; TTS `params.json` seul → non téléchargé puis reprise effective ; Realtime `config.json` seul → idem ; SHA-256 faux → rejet ; `downloadModel` sur id inconnu → erreur, aucun dossier créé ; `ModelDownloaderSizeTests` et `ModelLoadingSymlinkedDirectoryTests` verts ; coupure réseau simulée au milieu d'un shard puis relance → modèle complet (1/1). | M | macos-gpu | — | verified |
| K-7 | [Chargement vérifié (`verify: [.allModelKeysSet, .shapeMismatch]`) et tokenizer strict](fiches/K-7.md) | S-04, S-05, MLX-018 | chargement sans erreur (0 clé manquante) de `mini-3b-4bit`, `mini-3b-8bit`, `mini-3b`, `tts-4b-4bit`, `tts-4b-6bit`, `tts-4b-mlx`, `tts-4b`, `realtime-4b-4bit`, `realtime-4b-fp16` (+ Small s'ils sont présents) ; dossier privé d'un shard → erreur nommant ≥ 1 clé (test rouge avant) ; dossier sans `tekken.json` → erreur typée pour STT, TTS et Realtime (3 tests) ; ids de jetons identiques sur 20 phrases FR/EN avec le vrai fichier ; sorties greedy (STT C-court) et audio TTS (graine 42) identiques avant/après. | S | macos-gpu | — | applied (preuve rouge à fournir) |
| K-8 | [Quantification lue comme l'amont (mode, per-layer, `quantization_config`, index TTS) et modes non affines refusés](fiches/K-8.md) | M-02, M-04, P-48 | 5/5 fixtures `config.json` décodées (mzbac 4 b mixte, VincentGOURBIN 8 b, aufklarer `mode`, Markus per-layer + `mode`, mxfp4 synthétique → erreur explicite) ; `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` chargé avec `verify: [.all]` : 0 clé manquante ou en trop ; transcription greedy identique à mlx-voxtral (Python, même pack) sur C-court EN ; TTS : fixture « 3 shards + index » chargée, fixture `quantization_config` seule → chargement correct ou erreur explicite. | S-M | macos-gpu | K-7 | à faire (⛔ ASK-21) |
| K-9 | [Realtime : entrée originale `realtime-4b` de nouveau chargeable ; id de modèle strict](fiches/K-9.md) | M-01, M-05 | 2 fixtures de décodage vertes (`config.json` transformers ignoré → config Mistral depuis `params.json`, `quantization == nil`) ; téléchargement de `realtime-4b` = 8,87 Go ± 1 % (au lieu de 17,72) ; chargement `verify: [.all]` : 0 clé manquante ou en trop ; transcription C-court EN identique au pack `realtime-4b-fp16` (ou WER ≤ +0,2 pt) ; test « id inconnu → erreur ; nil → défaut ». | S | macos-gpu | K-6, K-7 | à faire (⛔ ASK-17) |
| K-10 | [Une seule source pour les dépôts STT (`small-24b-8bit` sur deux dépôts)](fiches/K-10.md) | S-06 | test « chaque `VoxtralPipeline.Model` existe dans `ModelRegistry` avec le même repoId » vert (rouge avant) ; modèle Small 8 bits téléchargé par l'app puis `loadModel` en mode avion → chargé, 0 octet réseau (`nettop` ou Little Snitch) ; un seul dossier `small-8bit` sur disque. | S | macos-gpu | — | à faire (⛔ ASK-15) |
| K-11 | [Exclusion enrôlement / inférence et machine d'états atomique des pipelines (deadlock ABBA compile × vjp)](fiches/K-11.md) | A-01, S-10 | test d'intégration `EnrollInferenceExclusionTests` (`enrollVoice` 50 époques ∥ `synthesizeStreaming`) : 20/20 exécutions sans blocage (timeout 120 s), synthèse refusée `busy` en < 1 s ou exécutée après ; rouge sans le correctif (blocage ou chevauchement détecté) ; test de stress TSan (2 × `loadModel`, `unload` pendant un stream, synthèse pendant enrôlement) : 0 alerte, états finaux cohérents 10/10 ; suite complète verte. | M | macos-gpu | — | verified |
| K-12 | [Streaming TTS réel et annulable (production dans une Task, `onTermination`, `checkCancellation`)](fiches/K-12.md) | S-08, MLX-003 (variante synchrone) | texte long (≈ 350 mots, `tts-4b-4bit`), **sans warm-up**, graine 42 : `generateStreaming` rend en < 50 ms ; premier chunk ≤ 1,5 × `ttft` du batch ; annulation après 5 chunks → pipeline `.ready` en < 1 s et frames générées ≤ frames à l'annulation + 1 ; audio concaténé identique au batch (même graine). | M | macos-gpu | K-11 | à faire |
| K-13 | [Realtime : fenêtres glissantes appliquées (encodeur 750 par tranches, décodeur `RotatingKVCache(8192)`)](fiches/K-13.md) | P-62, P-63 | embeddings identiques (L2 relative < 1e-3) à `encodeFull` sur un clip ≤ 15 s ; C-moyen : WER ≤ valeur actuelle et comparaison à mlx-audio (commit épinglé) consignée ; texte identique sur C-moyen avec la fenêtre décodeur ; C-xlong (≈ 17 min) : pic stable après 8 192 pas (±5 %), ms/pas au-delà de 8 192 = ms/pas à 8 000 (±5 %) ; pic d'encodage indépendant de la durée (±10 % entre 3 et 12 min). | M | macos-gpu | K-5 | en cours (porte amendée le 2026-10-01) |
| K-14 | [TTS : plafond de frames proportionnel au texte (fin des emballements jusqu'à 200 s)](fiches/K-14.md) | P-41, FV-46, FV-53 | 0 troncature sur 12 textes × 3 graines × 3 packs (4 / 6 / bf16) ; le reproducteur #45 (voix dégénérée ou texte sans ponctuation finale, `--no-sanitize`) s'arrête sous 3 × la longueur attendue ; `a`, `b` et leur distribution frames/jeton consignés. | S | macos-gpu | — | à faire (⛔ ASK-9) |
| K-15 | [Annulation coopérative et calcul hors du pool coopératif (STT, TTS batch, Realtime, chargement)](fiches/K-15.md) | S-09 | annulation d'une transcription de C-long (≈ 11 min) → `CancellationError` en < 2 s, pipeline `.ready` ; idem TTS batch (texte long) et Realtime ; app `VoxtralApp` : 0 blocage > 250 ms du main thread pendant `loadModel` (Instruments, Hangs) ; tests rouges sans le correctif. | M | macos-gpu | K-2, K-11 | à faire |
| K-16 | [Données partagées sûres : globaux protégés, `MLXArray` évalués avant de traverser une frontière d'isolation](fiches/K-16.md) | S-11, S-12, MLX-004 | `grep -c 'nonisolated(unsafe)' Sources/VoxtralCore -r` : 8 → ≤ 2, chacune documentée ; test : deux pipelines de configurations mémoire différentes gardent chacune la leur ; TSan propre sur 4 extractions de features en parallèle ; synthèse hors MainActor consommée sur le MainActor 20/20 sans plantage, WAV identiques octet pour octet (fiche préventive : aucun plantage reproduit avant). | S/M | macos-gpu | — | verified |

### Lot 2 — hygiène sans risque

| Fiche | Objet | Source | Porte | Effort | Cible | Prérequis | État |
|---|---|---|---|---|---|---|---|
| K-17 | [Mémoire du projet : `CLAUDE.md`, `docs/knowledge/`, `BENCHMARKS.md`, protocole et glossaire des métriques](fiches/K-17.md) | ACT-20, P-73, FA-04 | fichiers présents : `CLAUDE.md`, `BENCHMARKS.md`, `docs/Benchmarks.md`, `docs/knowledge/index.md`, `docs/knowledge/log.md`, `docs/knowledge/decisions/realtime-diagnostics-23-25.md`, ≥ 6 fichiers `docs/knowledge/pitfalls/*.md` (V-P5 stream synchrone, V-P8 même % GPU, tête liée fp32, fenêtre ignorée, clé `mode`, jeton d'arrêt hérité) ; `CLAUDE.md` contient les 3 commandes exactes de PLAN.md §5 ; glossaire : 1 définition par métrique, citée par `docs/Benchmarks.md` ; 0 chiffre sans source (relecture) ; aucun fichier `.swift` modifié (`git diff --stat`). | S | cloud | — | fait |
| K-18 | [Docs utilisateur alignées sur le code (exigences, versions, llms.txt, README, réglages TTS, chiffres requalifiés)](fiches/K-18.md) | S-19, S-20, FA-02, FA-05, P-35, P-38, P-76, FV-54, ACT-52, P-31 | checklist S-20 (12 points) relue contre `9392ed1` : 0 écart ; `grep -nE 'Swift 6\.0\|Xcode 15\|macOS 14' README.md llms.txt` = 0 ; `grep -n 'true silence' README.md` = 0 ; llms.txt cite v2.2.x et les 3 pipelines ; chaque tableau de `README.md` et `docs/*benchmark*.md` / `docs/voice_cloning.md` porte définition + révision + « en session » ; FV-54 : les deux jeux de chiffres cités avec leur source ; `syntax_guard.py` : 0 erreur nouvelle sur les `.swift` touchés (commentaires seulement : `VoxtralTTSModeling.swift:480-482`, `:528`, `VoxtralVoiceEnrollment.swift:51`). | S | cloud | — | fait |
| K-19 | [Annexes Python reproductibles (conversion Core ML, recherche clonage)](fiches/K-19.md) | A-03, A-21, FA-06 | contrôle argparse par AST (sans torch) : 0 argument inconnu et 0 requis manquant pour `convert.sh` et les commandes du README ; `grep -E '>=' Scripts/*/requirements.txt` = 0 ; commit amont épinglé et contrôlé par `enroll_voice.py` ; `grep -n 'Next step' Scripts/VoiceCloningResearch/README.md` = 0 ; commande de similarité ECAPA documentée ; la validation Core ML sur Mac (parité L2 ≤ 1e-2) est portée par K-42. | S | cloud | — | fait |
| K-20 | [Hygiène git : fichiers suivis malgré `.gitignore` (cache `.serena`, WAV)](fiches/K-20.md) | S-25 | `git ls-files -ci --exclude-standard \| wc -l` = 0 (hors exceptions déclarées dans `.gitignore`) ; `.serena/` retiré (−1 790 543 o dans l'arbre suivi) ; si ASK-30 = C : −18 274 912 o (8 WAV) ; 0 lien de doc mort ; les 4 clips du corpus (`fluxforge_{short,long}_{en,fr}_6bit.wav`) toujours présents avec leur SHA-256 noté dans `docs/Benchmarks.md`. | S | cloud | — | à faire (⛔ ASK-30) |
| K-21 | [Tracker action-plans : solder #71, #307, #349 et créer le plan upstream-blocker mlx-swift-lm (> 3.31.4)](fiches/K-21.md) | FA-01, FA-02, ACT-02, ACT-03, ACT-04, ACT-07, ACT-39, ACT-41, S-18 | 0 plan `project:mlx-voxtral-swift` en `status:ready-to-act` ; #71, #307, #349 fermés `status:verified`, chacun avec le commentaire-preuve de faits-et-actions.md §3.9 ; 1 plan `kind:upstream-blocker` `github_release ml-explore/mlx-swift-lm semver_gt "3.31.4"` en `monitoring` — **après** avoir vérifié que l'amont publie des GitHub Releases (sinon : source `github_tag` ou plan `manual`, et retour skill) ; lien du plan ajouté au commentaire de `Package.swift:46-51` au prochain commit de code (K-22). | S | cloud | — | fait |
| K-22 | [Package : dépendances élaguées, résolution reproductible, plancher de toolchain, profiler épinglé](fiches/K-22.md) | S-15, S-18, S-19, P-74, FA-02 | `xcodebuild` Release de `VoxtralCLI`, `VoxtralApp`, `VoxtralBenchmark`, `VoxtralTTSStreamingDemo` : `** BUILD SUCCEEDED **` (4/4) et suite de tests verte ; `swift package show-dependencies` affiche `swift-mlx-profiler` 1.5.1 et la révision de `mlx-swift-lm` notée ; `Package.resolved` suivi, révision identique après `swift package resolve` sur deux clones ; `grep -rc '@available(macOS 1[34]' Sources` = 0 ; temps de build propre de `VoxtralCore` avant/après consigné ; FluxForge compile (si présent sur la machine). | S | macos-gpu | — | verified |
| K-23 | [Nettoyage sans risque d'API : code mort privé/interne, famille legacy silencieuse, chemins codés en dur, TODO, `print`](fiches/K-23.md) | S-13, S-14, S-23, S-24, S-29, A-17, MLX-010 (`Int.max`) | `BUILD SUCCEEDED` (4 schémas) et suite verte ; −≈ 900 lignes (`git diff --shortstat`) ; `grep -rn '/Users/' Sources Tests` = 0 ; `grep -rn 'print(' Sources/VoxtralCore \| grep -v VoxtralDebug` = 0 ; `grep -rnE 'TODO\|For now\|would integrate' Sources/VoxtralCore` = 0 (hors code public déprécié par K-30) ; 0 écriture dans `/tmp` pendant une génération par `VoxtralGenerator` (test) ; une synthèse CLI sans `--debug` n'écrit rien de la bibliothèque sur stdout. | M | macos-gpu | K-22 | à faire |
| K-24 | [Registres exacts : `consolidated.safetensors` exclu en STT, tailles et précisions réelles, variante Core ML par config](fiches/K-24.md) | S-07, M-03, M-06 | `mini-3b` téléchargé = 9,37 Go ± 1 % (`du -sb`) ; test d'exclusion `consolidated*` vert (STT) et `tts-4b` inchangé (consolidated présent) ; test « chaque entrée du registre à ± 5 % des octets de `docs/Weights.md` » vert ; test « dossier Small nommé `x/model` → `.small` » vert ; README sans « ~6 GB / ~3.5 GB / ~2 GB / ~12 GB ». | S | macos-gpu | K-6 | à faire |
| K-25 | [Encodeur Core ML : chemin unique sous `customModelsDirectory`, chargement hors ligne, erreurs explicites](fiches/K-25.md) | A-04, A-13 | test avec `customModelsDirectory = <tmp>/VoxtralModels` : encodeur trouvé sous ce dossier ; 2ᵉ chargement hybride réseau coupé → `Core ML available: true` ; 0 octet écrit sous `~/.cache/huggingface` (`du` avant/après) ; tests « Small + encodeur mini → erreur » et « encodeur MLX non initialisé → erreur » verts (rouges avant). | S-M | macos-gpu | K-6 | verified |
| K-26 | [Enrôlement reproductible : graine, point de contrôle et reprise, tests de la garde NaN](fiches/K-26.md) | A-07, A-08, A-09, T23 | 2 enrôlements de 200 époques, même graine → codes `[T, 37]` identiques ; graines différentes → différents ; arrêt à 2 500 / 5 000 puis reprise → codes finaux identiques bit à bit au run continu ; surcoût de sauvegarde ≤ 5 % du temps total (dans le bruit) ; 3 tests (NaN à l'époque 0 → `EnrollmentDivergedError` ; NaN à l'époque k → codes du meilleur pas ; annulation → `CancellationError`) rouges sans la garde, verts avec. | M | macos-gpu | K-11 | à faire |
| K-27 | [API honnête et erreurs explicites, sans cassure (souches, paramètres ignorés, `as!`, `precondition`)](fiches/K-27.md) | S-22, S-28, A-05 | `tokenCount` > 0 sur une transcription (test) ; `grep -rn 'as!' Sources/VoxtralCore` = 0 ; 0 `precondition`/`fatalError` atteignable depuis l'API publique (liste relue, un test par cas) ; build sans nouvel avertissement hors dépréciations voulues ; suite verte. | S | macos-gpu | K-6 | à faire |
| K-28 | [Racine, cibles annexes et app : build depuis un clone neuf, empaquetage, démo robuste, `RuntimeBeacon`](fiches/K-28.md) | S-26, A-14, A-18, A-19, A-20 | `git clone` dans un dossier neuf + `xcodebuild` Release des 4 schémas : `** BUILD SUCCEEDED **` sans « Invalid Resource » ; app empaquetée (script réécrit en `xcodebuild` Release + bundles) lancée : transcrit C-court EN ; test FFmpeg : 1 Mo sur stderr → processus terminé ; annulation → processus tué < 1 s ; nom `../x` refusé avant l'enrôlement, écrasement demandé explicitement ; 1 000 `update` concurrents d'un `end` → 0 manifeste résiduel. | M | macos-gpu | K-22 | à faire (⛔ ASK-27) |
| K-29 | [Filet de tests et CI macOS (tests non tautologiques, fixture Tekken, test symlink sur le vrai chargeur)](fiches/K-29.md) | S-27, A-22 | CI verte sur la branche (lien du run) ; `PerformanceOptimizationTests.swift:173-255` remplacés par des appels au code (chacun rouge si l'algorithme de production est cassé) ; `TekkenTokenizerTests` sur fixture réduite versionnée (ne dépend plus de `/Users/vincent`) ; `ModelLoadingSymlinkedDirectoryTests` couvre `loadWeights(from:)` ; suite locale verte. | M | macos-gpu | K-22 | à faire |
| K-30 | [Dépréciation de l'API legacy et du code mort public (version 2.3 ; suppression en 3.0 selon ASK-23)](fiches/K-30.md) | S-13, S-14, P-17, A-05, P-24 | liste S-13 lot 2 + S-14 entièrement annotée (grep) ; build de la bibliothèque sans avertissement interne (aucun appel interne à du déprécié) ; recherche GitHub `user:VincentGourbin` sur chaque symbole = 0 usage (relue) et, si FluxForge est présent, son build n'affiche aucun avertissement de dépréciation Voxtral ; CHANGELOG (section 2.3) listant chaque symbole. | S | macos-gpu | K-23, K-27 | à faire (⛔ ASK-23) |
| K-31 | [Revue d'API publique : façades, préfixes, résultats typés (version majeure)](fiches/K-31.md) | S-21 | liste validée par Vincent (ASK-25) appliquée ; `grep -c '^\s*public' -r Sources/VoxtralCore` ≤ 300 ; FluxForge compile sans ses typealias de contournement (`ModelManager.swift`, `LTXBeaconBridge.swift`) ; suite verte. | L | macos-gpu | K-30 | à faire (⛔ ASK-25) |

### Lot 3 — baseline mesurée

| Fiche | Objet | Source | Porte | Effort | Cible | Prérequis | État |
|---|---|---|---|---|---|---|---|
| K-32 | [Instrument de baseline `VoxtralCLI bench` (STT, TTS, Realtime, enrôlement ; JSON ; A/A ≤ 3 %)](fiches/K-32.md) | P-79, P-75, P-77, A-15, A-16, P-45, P-19, FA-04 | A/A sur la même commande, deux passes après 120 s : dispersion ≤ 3 % sur le total hors chargement et sur `step_ms_p50`, pour `stt` (mini-3b-8bit, C-moyen EN, `.mlx`), `tts` (tts-4b-6bit, texte court, graine 42) et `realtime` (realtime-4b-4bit, C-moyen EN) ; `out_sha256` identique entre passes ; lignes JSON valides contre `docs/bench.schema.json` ; binaire Debug refusé (ligne `REFUSED debug build`) ; `grep -rn resetPeakMemory Sources/VoxtralCore` = 0 ; `chat` (mini-3b-8bit, C-court EN, question fixe de `docs/eval/chat-questions.json`, greedy) : dispersion ≤ 3 % sur `ttft_ms` et tok/s. | M | macos-gpu | K-22, K-5 | applied (machine-check à fournir) |
| K-33 | [Éval reproductible : WER STT, aller-retour TTS → STT, juge ASR validé, auto-détection de langue](fiches/K-33.md) | A-22, P-78, FA-07, ACT-32 | texte de référence contrôlé contre l'audio (le texte « Full test texts » de `docs/tts_benchmark.md` fait 163 mots EN / 202 mots FR pour 167 / 174 s d'audio alors que « ~350 words » sont annoncés : s'il est abrégé, régénérer C-moyen depuis un texte exact avec `$CLI tts --seed` et versionner texte + SHA-256) ; deux exécutions à graine égale → scores identiques ; baseline WER `mini-3b-8bit` `.mlx` enregistrée sur C-court/C-moyen EN et FR ; WER du juge `realtime-4b-4bit` publié sur C-court, C-moyen et un clip de 20 s, version (pack + commit) figée dans `docs/zerovoice_benchmark.md` ; auto-détection : 3 langues (EN, FR, ES) × 3 clips (TTS `es_male`/`es_female`, graine fixée) : WER(`language: nil`) ≤ WER(langue explicite) + 2 pts. | M | macos-gpu | K-32, K-4, K-5 | à faire |
| K-34 | [Baseline STT (Mini et Small ; `.mlx` et `.auto`) + requalification du « 49 % GPU » et du support 32 Go](fiches/K-34.md) | P-19, FA-08, ACT-22, ACT-23 | A/A ≤ 3 % sur 2 passes pour chaque ligne ; une ligne `BENCHMARKS.md` par (modèle ∈ {mini-3b-4bit, mini-3b-8bit, mini-3b, small-4bit} × backend ∈ {.mlx, .auto} × clip ∈ {C-court EN, C-moyen EN, C-moyen FR, C-long}) avec la révision résolue et le WER ; occupation GPU du préfill Mini et Small consignée par les deux instruments (écart noté) ; small-4bit, 8 min d'audio : pic `phys_footprint` mesuré ≤ 24 Go (marge d'un Mac 32 Go), sinon verdict « non supporté sur 32 Go » écrit dans `docs/knowledge/decisions/small-32gb.md` ; chat : une ligne `BENCHMARKS.md` par backend ∈ {.mlx, .auto} pour mini-3b-8bit sur C-moyen EN × 4 questions fixes (greedy), avec `ttft_ms` de chaque question et tok/s (référence de K-49 et K-70). | M | macos-gpu | K-32, K-33, K-2, K-4, K-5 | à faire |
| K-35 | [Baseline TTS (prédéfinie / clonée, batch / streaming, 4 / 6 / 16 bits, chemin du consommateur)](fiches/K-35.md) | P-45, FA-04, P-76 | A/A ≤ 3 % ; baseline enregistrée pour {tts-4b-4bit, tts-4b-6bit, tts-4b-mlx} × {court, 60 s, long} × graines {1, 2, 3} × {neutral_female, voix clonée `docs/examples/clone_fr.wav` enrôlée} × {batch, streaming} ; ligne dédiée « consommateur » (6 bits + clonée + `--warm-up`, batch) ; frames du porteur / frames totales consignées ; couverture ASR (aller-retour TTS → STT, outil K-33) consignée par (pack × texte × graine) en batch : référence des portes qualité de K-39, K-48, K-58 et K-79. | M | macos-gpu | K-32, K-12, K-33 | à faire |
| K-36 | [Baseline Realtime et re-diagnostic des issues #23-#25 (occupation GPU réelle)](fiches/K-36.md) | P-73, P-75, ACT-27 | A/A ≤ 3 % ; lignes `BENCHMARKS.md` pour realtime-4b-4bit et realtime-4b-fp16 sur C-court, C-moyen EN/FR, C-long ; trace swift-mlx-profiler 1.5.x `.fineGrained` (`.ioReportResidency`) et Metal System Trace sur C-moyen complet : occupation GPU du décodage rapportée des deux façons, écart ≤ 10 pts ; décision « #23-#25 caducs » complétée par les chiffres ; `pad_fraction` consignée (entrée de K-73) ; WER (outil K-33) consigné pour chaque ligne : référence des portes de K-38, K-46, K-65 et K-78. | S-M | macos-gpu | K-32, K-5, K-33 | à faire |
| K-37 | [Baseline enrôlement (s/époque, pic, deux scénarios de résidence du LLM)](fiches/K-37.md) | A-06, A-15 | A/A ≤ 3 % sur `epoch_ms_p50` (200 époques, graine 7) ; pics (i) et (ii) consignés pour tts-4b-6bit et tts-4b-mlx ; décision écrite : si (ii) − (i) < 5 %, le volet résidence de K-64 est retiré sans code. | S | macos-gpu | K-32, K-26 | à faire |

### Lot 4 — leviers perf (gain attendu décroissant)

| Fiche | Objet | Source | Porte | Effort | Cible | Prérequis | État |
|---|---|---|---|---|---|---|---|
| K-38 | [Realtime en dtype du modèle (mel, tables RoPE, `tCond`/`adaScale`) — supprime la copie fp32 de la tête (1,5 Gio par pas)](fiches/K-38.md) | P-60, T17 | audit de dtype (`VOXTRAL_DTYPE_AUDIT=1`) : embedding, cache KV couche 0 et logits = dtype du modèle ; pack 4 bits, C-moyen EN : `step_ms_p50` ≥ −30 % et pic `phys_footprint` −≥ 1 Go (A/B/B/A) ; encodage ≥ −10 % ; `realtime-4b-fp16` : `step_ms_p50` ≥ −40 % ; parité : texte identique sur C-court ou WER ≤ réf + 0,2 pt sur C-moyen EN/FR (ASK-11). | S | macos-gpu | K-36, K-13 | à faire (⛔ ASK-11) |
| K-39 | [TTS : transformeur acoustique (FM) et tête sémantique dans le dtype des poids](fiches/K-39.md) | P-30, MLX-002, T17 | pas bf16 −40 % au moins et pas 4 bits −5 % au moins (A/B/B/A, graine fixée, Release) ; parité forcée par l'enseignant (mêmes états cachés du LLM passés aux deux variantes du FM) : codes acoustiques identiques ≥ 99 % avec écart ≤ 1 niveau, codes sémantiques identiques 100 % ; couverture ASR (`eval`, 3 textes × 5 graines) ≥ référence −0,5 pt ; 0 `maxFrames` atteint ; écoute à l'aveugle non inférieure si ASK-13 l'exige. | M | macos-gpu | K-35 | à faire (⛔ ASK-11, ASK-13) |
| K-40 | [STT : calcul bf16 de bout en bout (features et sortie Core ML castées, fusion au dtype du texte)](fiches/K-40.md) | P-01, P-05, T17 | test : dtype du cache KV après préfill = bf16 (rouge avant) ; encodage + préfill ≥ 5 % plus rapides **ou** pic MLX −≥ 10 % (A/B/B/A, Mini 8 bits et 4 bits mixte, `.mlx` et `.auto`, C-moyen et C-long) ; `mini-3b` bf16 : décodage ≥ +5 % et pic process −≥ 1 Go ; parité : greedy identique sur ≥ 90 % des clips et WER ≤ réf + 0,3 pt (FR et EN), sinon politique mixte (encodeur fp32, décodeur bf16) documentée et mesurée. | S | macos-gpu | K-3, K-34 | à faire |
| K-41 | [Codec TTS : attention par bandes (fenêtre ≤ 16) au lieu de T × T fp32](fiches/K-41.md) | P-32 | forme d'onde \|Δ\|max ≤ 1e-4 contre l'actuel sur 3 textes (court, 60 s, long) ; pic `phys_footprint` du décodage long ≤ poids + 1 Go (contre ≥ 10 Go attendus aujourd'hui) ; décodage court ±5 % (A/B/B/A). | M | macos-gpu | K-35 | à faire |
| K-42 | [Décider l'encodeur STT par la mesure : matrice MLX bf16 / Core ML GPU / ANE / `.all` × packs (backend par défaut)](fiches/K-42.md) | P-14, A-12, A-13, A-03, M-07, R14 | règle : Core ML reste le défaut `.auto` seulement s'il est ≥ 1,2× plus rapide **et** WER ≤ +0,3 pt **et** parité (L2 relative des embeddings ≤ 1e-2, transcriptions greedy identiques sur 5 fichiers), sinon `.mlx` par défaut (ASK-6) ; chaque cellule a une ligne `BENCHMARKS.md` ; pour chaque largeur, le pack retenu a un WER ≤ bf16 + 0,3 pt ; décision écrite dans `docs/knowledge/decisions/reference-profiles.md` ; si Core ML est gardé : reconversion `convert.sh` (K-19) rejouée de bout en bout. | M | macos-gpu | K-40, K-33, K-25 | à faire (⛔ ASK-6) |
| K-43 | [Streaming TTS : décodage incrémental exact (contexte gauche borné) au lieu du re-décodage de tout l'accumulé](fiches/K-43.md) | P-33 | forme d'onde concaténée du streaming = batch (\|Δ\|max ≤ 1e-4) sur 3 textes × 3 graines ; temps de décodage par chunk constant ±20 % du chunk 1 au chunk 200 ; RTF streaming du texte long ≤ RTF batch + 5 %. | M | macos-gpu | K-12, K-41 | à faire |
| K-44 | [TTS : pipelining `asyncEval` de la boucle AR (batch et streaming, itérateur de frames unique)](fiches/K-44.md) | P-31, T15 | fr/s +5 % au moins en 4 et en 6 bits (A/B/B/A) ; codes identiques bit à bit à graine fixe (batch et streaming) ; preuve que la branche asynchrone s'exécute (GPU % par frame en hausse, trace) ; streaming ≥ batch −3 %. | M | macos-gpu | K-39 | à faire |
| K-45 | [STT : pipelining `asyncEval` (décodage décalé d'un pas, `asyncEval(cache)` par tranche de préfill)](fiches/K-45.md) | P-07, P-25, T15 | `step_ms_p50` −5 % au moins (Mini 8 bits, C-moyen, A/B/B/A) ; sortie greedy identique 100 % ; preuve que la branche asynchrone est prise. | M | macos-gpu | K-40 | à faire |
| K-46 | [Realtime : tête liée quantifiée (8 bits en fast, 4 bits en lean) via `QuantizedEmbedding.asLinear`](fiches/K-46.md) | P-61, T12 | baseline = après K-38 : `step_ms_p50` −8 % au moins (tête 8 bits), `activeMemory` après chargement −≥ 300 Mo ; WER ≤ réf + 0,2 pt (8 bits) et ≤ réf + 0,5 pt (4 bits), sinon la tête 4 bits est rejetée et documentée. | M | macos-gpu | K-38 | à faire |
| K-47 | [Realtime : pipelining `asyncEval` du décodage (jeton en `MLXArray`, EOS lu avec un pas de retard)](fiches/K-47.md) | P-65, T15 | `step_ms_p50` −5 % au moins (A/B/B/A, après K-46) ; texte identique sur C-court et C-moyen ; branche asynchrone prouvée. | S | macos-gpu | K-46 | à faire |
| K-48 | [TTS : nombre de pas de flow matching et `cfgAlpha` réellement réglables + balayage 8 / 6 / 5 / 4](fiches/K-48.md) | P-35, T21 | câblage : défaut 8 bit-exact à graine fixe ; valeur < 8 retenue seulement si fr/s +10 % au moins, couverture ASR ≥ réf −0,5 pt (12 textes × 3 graines), 0 `maxFrames`, écoute à l'aveugle non inférieure (ASK-13) ; sinon rejet documenté. | S+M | macos-gpu | K-44 | à faire (⛔ ASK-13) |
| K-49 | [Chat : session audio réutilisable (embeddings audio et instantané KV avant la question)](fiches/K-49.md) | P-15, T6, T7 | 2ᵉ question : latence au premier jeton −50 % au moins ; réponses greedy identiques à un démarrage à froid sur 4 questions ; mémoire de la session consignée. | M-L | macos-gpu | K-45 | à faire |
| K-50 | [TTS : cache de préfixe pour toute voix (clonée, ZeroVoice, mélange) avec empreinte et LRU](fiches/K-50.md) | P-40, T6, T2 | 2ᵉ synthèse d'une voix clonée : préfill −30 % au moins (graine fixée, A/B/B/A) ; mêmes transcriptions ASR sur 3 textes × 3 graines ; test « clé réutilisée avec un autre embedding → détectée » rouge sans le correctif. | S | macos-gpu | K-11, K-35 | à faire |
| K-51 | [STT : nettoyage mémoire sorti de la boucle de décodage ; configuration par pipeline](fiches/K-51.md) | P-08, R13, T2 | `recommended(forRAMGB: 16)` sur le Mac de mesure : décodage +5 % au moins (A/B/B/A), pic `phys_footprint` ≤ +10 %, greedy identique. | S | macos-gpu | K-45 | à faire |
| K-52 | [Politique mémoire MLX opt-in par pipeline : `cacheLimit` par profil, `clearCache` en fin de réponse et dans `unload()`](fiches/K-52.md) | P-09, P-42, P-67, MLX-010 (dont `Int.max`), T1, T2, T3 | STT à 10 min d'audio : pic `phys_footprint` ≤ pic actif MLX + `cacheLimit` + empreinte Core ML (+5 %) ; temps fast ±5 % ; TTS : après `unload()`, footprint ≤ référence du process + 200 Mo, lean ≤ +5 % de temps pour un pic −20 % au moins sur le texte long ; Realtime : après `unload()`, footprint = avant chargement ±200 Mo, `step_ms_p50` ±3 %, max/médiane des pas ≤ 1,3. | S | macos-gpu | K-51 | partielle (avancée par Vincent, hors tracker ; reste → #590) |
| K-53 | [STT : cache KV pré-dimensionné (`step` = invite + `maxTokens`)](fiches/K-53.md) | P-10, T11, T1 | 30 min d'audio : préfill −5 % **ou** pic `phys_footprint` −10 % au moins (A/B/B/A) ; 10 min consigné ; sortie bit-exacte. | S | macos-gpu | K-40 | à faire |
| K-54 | [STT : dernier logit seulement au préfill](fiches/K-54.md) | P-06, T9 | préfill Mini −5 % **ou** pic −128 Mio au moins à 5 min d'audio (A/B/B/A) ; logits du dernier jeton : L2 relative ≤ 1e-5 (fp32) / ≤ 1e-3 (bf16) ; greedy identique 100 %. | S | macos-gpu | K-40 | à faire |
| K-55 | [STT lean : cache KV 8 bits (`QuantizedKVCache`, groupe 64) via l'aide amont](fiches/K-55.md) | P-16, T10 | Small 4 bits, 20 min : pic −1 Go au moins ; WER ≤ +0,3 pt ; décodage au pire −5 % (sinon réservé au lean). | M | macos-gpu | K-3, K-2, K-40 | à faire |
| K-56 | [STT : encodeur MLX par lots bornés de K fenêtres](fiches/K-56.md) | P-13, T16 | 30 min : pic MLX de l'encodage −30 % au moins ; temps d'encodage ±5 % ; embeddings bit-exacts attendus, à défaut L2 relative ≤ 1e-5 et greedy identique. | S | macos-gpu | K-40 | à faire |
| K-57 | [TTS : surcoût hôte du FM (invariants précalculés, SDPA fusionnée sans répétition GQA)](fiches/K-57.md) | P-36, P-37 | fr/s +5 % au moins en 4 bits (A/B/B/A) ; codes acoustiques forcés par l'enseignant identiques ≥ 99,9 % ; sinon retrait. | S | macos-gpu | K-44 | à faire |
| K-58 | [Codec TTS en bf16 (padding, scalaires, sorties d'étage) et invariants mis en cache](fiches/K-58.md) | P-38, P-39 | décodage −20 % au moins (A/B/B/A) ; SNR ≥ 40 dB contre fp32 **et** couverture ASR identique **et** A/B à l'aveugle non inférieur (ASK-13) ; test : sortie de chaque étage dans le dtype des poids ; sinon ne garder que les invariants (P-39). | S | macos-gpu | K-41 | à faire (⛔ ASK-13) |
| K-59 | [Matérialisation des poids résidents par voie au chargement (STT, TTS, Realtime) et modules factices neutralisés](fiches/K-59.md) | P-04, P-24, P-43, P-66 | 1er préfill = préfill à chaud ±5 % (A/A sur 2 requêtes) sur STT et Realtime ; `activeMemory` après chargement = poids résidents ±5 % ; en `.auto`, tour audio MLX non matérialisée ; aucun tenseur aléatoire dans `parameters()` du wrapper (test) ; TTS : temps jusqu'au 1er frame préfixe inclus de la 1re synthèse ≤ 1,1 × celui d'une 2e avec une autre voix (clonée et prédéfinie) ; Realtime `.encoderOnly` : `activeMemory` ≤ encodeur + 10 %. | S | macos-gpu | K-34, K-35, K-36 | à faire |
| K-60 | [Encodeurs dé-quantifiés en profil fast (STT et Realtime, T14/T20)](fiches/K-60.md) | P-27, P-69, T14, T20 | STT : encodage −5 % au moins, pic ≤ +0,8 Go, WER ≤ +0,1 pt ; Realtime : encodage −5 % au moins sur C-moyen (A/B/B/A), WER identique ; sinon retrait. | S | macos-gpu | K-40, K-38, K-13 | à faire |
| K-61 | [STT : pénalité de répétition tranchée par le WER (1,0 contre 1,2), version vectorisée si gardée](fiches/K-61.md) | P-12 | valeur au WER le plus bas retenue si l'écart ≥ 0,3 pt (sinon statu quo) ; version vectorisée : logits bit-exacts et `step_ms_p50` −5 % au moins, sinon non retenue. | S | macos-gpu | K-33, K-45 | à faire (⛔ ASK-9) |
| K-62 | [STT : tranche de préfill exposée (`prefillStepSize`) et balayée (256 / 512 / 1 024 / 2 048)](fiches/K-62.md) | P-22, T9 | A/B/B/A 256/512/1 024/2 048 à 5 et 30 min ; retenir la plus rapide dont le pic ≤ pic(512) + 5 % ; sortie bit-exacte après factorisation des deux boucles. | S | macos-gpu | K-54, K-53 | à faire |
| K-63 | [STT lean : tour audio MLX libérée après l'encodage (zéros non évalués), rechargée paresseusement](fiches/K-63.md) | P-23, T4 | pic de décodage −0,5 Go au moins (Mini 8 bits) ; temps par requête ≤ +5 % ; sortie identique. | M | macos-gpu | K-59 | à faire |
| K-64 | [Enrôlement lean : politique mémoire (cacheLimit, clearCache), résidence du sous-ensemble codec + table audio, profils enroll-fast\|lean mesurés](fiches/K-64.md) | A-06, A-10, A-11, T1, T3, T4 | scénario synthèse → enrôlement : `enroll-lean` −40 % de pic au moins contre `enroll-fast` (si le volet résidence est gardé), temps/époque ±5 % (A/B/B/A), codes identiques à graine fixe ou perte finale ±1 % ; A-10 retiré si < 5 % ; table des deux profils (temps, pic, similarité ECAPA) consignée ; CLI, démo et bibliothèque lisent la même table. | M | macos-gpu | K-37, K-52 | à faire |
| K-65 | [Realtime : RoPE de l'encodeur par le noyau fusionné (`RoPE(traditional: true)`)](fiches/K-65.md) | P-68 | embeddings L2 relative < 1e-2 contre l'actuel au même dtype ; WER identique sur C-moyen ; encodage −5 % au moins (A/B/B/A). | S | macos-gpu | K-13, K-36 | à faire |
| K-66 | [TTS : une seule passe LLM avant le premier frame (suffixe + jeton AUDIO)](fiches/K-66.md) | P-44 | TTFT −5 % au moins (A/B/B/A) ; 20 premiers codes sémantiques identiques à graine fixe ; sinon retrait. | S | macos-gpu | K-35 | à faire |
| K-67 | [TTS : coupes du porteur et des silences vectorisées (un seul transfert)](fiches/K-67.md) | P-46 | indices de coupe identiques sur `TrimSilenceTests`, `TrimLeadingCarrierTests`, `TTSWarmUpCarrierTrimTests` ; phase « Audio Post-processing » ÷ 5 au moins. | S | macos-gpu | K-35 | à faire |
| K-68 | [TTS : coût du porteur de warm-up (mesure, vocalise ou fenêtre d'attente plus courts)](fiches/K-68.md) | P-47, R3 | TTFA streaming des voix clonées −30 % au moins avec fuites du porteur ≤ la référence (0/10) sur 10 prises ; frames du porteur / frames totales rapportées en batch ; décision ASK-10 consignée. | S | macos-gpu | K-12, K-35 | à faire (⛔ ASK-10) |
| K-69 | [Extraction audio : mel en une passe et décodage / rééchantillonnage par blocs](fiches/K-69.md) | P-21, P-29 | phase mel −5 % au moins, features bit-exactes ; pic mémoire de l'extraction −50 % au moins (30 min, stéréo 48 kHz), échantillons : écart max ≤ 1e-6 hors des 20 dernières ms, mel ≤ 1e-5, greedy identique ; temps ±5 %. | S | macos-gpu | K-34 | à faire |
| K-70 | [Chat : top-p exact (nucleus sur les k meilleurs) mesuré contre l'actuel](fiches/K-70.md) | P-26, T5 | chat tok/s ≥ actuel −5 % ; test de distribution sur logits synthétiques (masse retenue = `topP` ± 1e-3). | S | macos-gpu | K-45 | à faire |
| K-71 | [Codec TTS : compile des chaînes élémentaires (SwiGLU, LayerScale), coupe-circuit, désactivé sous gradient](fiches/K-71.md) | P-49, T18, A-01 | décodage −5 % au moins, bit-exact, 0 blocage sur 20 exécutions « enrôlement + synthèse » (test de K-11) ; sinon retrait. | S | macos-gpu | K-41, K-43, K-11 | à faire |
| K-72 | [STT : préfill en flux par fenêtre (encoder la fenêtre k+1 pendant le préfill de k)](fiches/K-72.md) | P-28 | TTFT −10 % au moins à 10 min (A/B/B/A) ; greedy identique. | L | macos-gpu | K-42, K-56 | à faire |
| K-73 | [Realtime (R et D) : pas spéculatifs « remplissage » vérifiés en un forward](fiches/K-73.md) | P-72 | étape 1 : part de remplissage (`pad_fraction` de K-36) ≥ 50 % sur C-moyen, sinon abandon documenté ; étape 2 : passes de décodage −25 % au moins et texte identique sur C-court, C-moyen et C-long. | M-L | macos-gpu | K-47, K-36 | à faire |
| K-74 | [Un seul LLM typé et le chargeur amont (`MLXLMCommon.loadWeights` + `PerLayerQuantization`)](fiches/K-74.md) | S-16 | parité greedy 32/32 jetons sur 6 audios × 3 quantisations ; temps de chargement ≤ baseline ; 0 `fatalError` de dispatch (`grep`). | L | macos-gpu | K-7, K-8, K-30, K-34 | à faire (⛔ ASK-23) |
| K-75 | [Conformance `LanguageModel` : retrait, ou génération par `TokenIterator` avec un `prepare` conforme](fiches/K-75.md) | S-17, P-18, MLX-006 | (A) build vert contre mlx-swift-lm `main` courant et contre le tag 3.31.4, tests verts ; (B) greedy identique 100 % au code maison corrigé, tok/s ≥ boucle maison, build vert contre `main` et le prochain tag, prompt de 2 000 positions : `cache.offset` = 1 999 et `.tokens` d'un jeton. | L | macos-gpu | K-45, K-54, K-55 | à faire (⛔ ASK-24) |

### Lot 5 — type de profils + CLI

| Fiche | Objet | Source | Porte | Effort | Cible | Prérequis | État |
|---|---|---|---|---|---|---|---|
| K-76 | [Types de profils de référence par pipeline (v0, boutons existants) + CLI `references` / `--reference`](fiches/K-76.md) | P-20, K-M09, A-11, P-34 | `$CLI references` liste 6 profils STT Mini, 6 STT Small, 4 Realtime (4/16 × fast/lean), 6 TTS (4/6/16 × fast/lean, 6 bits déclaré hors standard) et 2 enrôlement, avec pour chacun `weights: <repo> (<octets>)` ; test : appliquer un profil produit exactement la configuration explicite équivalente (égalité des `Configuration`) ; `$CLI transcribe C-court EN --reference 8bit-fast` = même sortie greedy que les options explicites ; aucun profil v0 n'utilise un préréglage à fenêtre KV ; `VoxtralTTSSynthesisManager` charge le modèle demandé (test). | M | macos-gpu | K-2, K-5, K-10, K-24 | à faire (⛔ ASK-4) |

### Lot 6 — mesures de la matrice et docs

| Fiche | Objet | Source | Porte | Effort | Cible | Prérequis | État |
|---|---|---|---|---|---|---|---|
| K-77 | [Mesure de la matrice STT (Mini 3B, Small 24B) et décision `reference-profiles`](fiches/K-77.md) | P-20, M-07, FA-08 | 6 profils × 2 modèles mesurés (A/A ≤ 3 %, une ligne `BENCHMARKS.md` chacun) avec WER ; `16bit-*` ≥ WER de référence ; chaque profil `lean` a un pic ≤ son budget (Small 4 bits lean à 10 min d'audio ≤ 24 Go, valeur ASK-7) ; décision et points ouverts dans `docs/knowledge/decisions/reference-profiles.md`. | M | macos-gpu | K-76, K-42, K-34 | à faire |
| K-78 | [Mesure de la matrice Realtime (4 / 16 bits, 8 bits si un pack chargeable existe)](fiches/K-78.md) | P-70, P-60, P-67, M-01 | profils `4bit-fast\|lean`, `16bit-fast\|lean` mesurés (A/A ≤ 3 %, WER) ; `step_ms_p90` < 80 ms sur la machine de référence pour chaque profil `fast` ; 8 bits : pack chargé avec `verify: [.all]` (0 clé en écart) et WER 8 bits ≤ WER 4 bits sur C-moyen, sinon « non disponible » documenté ; une ligne `References.md` par profil (K-82). | M | macos-gpu | K-76, K-38, K-46, K-9 | à faire |
| K-79 | [Défaut TTS tranché par la mesure et matrice TTS (4 / 6 / 16 bits × fast / lean)](fiches/K-79.md) | FA-03, ACT-12, ACT-18, ACT-29, P-34, T15 | `TTSQuantizationCampaignTests` étendue, lancée en Release : si couverture ASR q6 ≥ bf16 − 1 pt **et** RTF (génération/audio) q6 ≤ 0,5 × bf16 → q6 par défaut sur les 4 surfaces, sinon bf16 documenté partout (selon ASK-5) ; test du détecteur : « là, là » détecté (rouge avant) ; 6 profils TTS mesurés (fps, TTFA, pic, couverture), une ligne `BENCHMARKS.md` chacun. | M | macos-gpu | K-76, K-35, K-33, K-39 | à faire (⛔ ASK-5, ASK-13, ASK-18) |
| K-80 | [Packs publiés : Realtime 8 bits, TTS 8 bits, Mini à encodeur 8 bits / bf16, Small à encodeur dense (SHA-256, cartes)](fiches/K-80.md) | K-M08, PK-1, PK-4, P-70, P-48 | pour chaque pack : SHA-256 publiés et vérifiés au téléchargement (test) ; parité : WER ≤ bf16 + 0,3 pt (STT, Realtime) ; TTS : couverture ASR ≥ 99 % et écoute contre bf16 (ASK-13) ; campagne de packs mixtes TTS : couverture ≥ 6 bits − 0,5 pt, 0 `maxFrames` sur le long FR (3 graines), fr/s ≥ 0,9 × le 4 bits, sinon non publiée ; carte sans exemple `transformers` erroné ; `docs/Weights.md` à jour. | L | macos-gpu | K-8, K-77, K-78, K-79 | à faire (⛔ ASK-16, ASK-18, ASK-19, ASK-20, ASK-22) |
| K-81 | [Squelette sourcé de `docs/References.md` et `docs/Weights.md` (sans mesure : valeurs « mesurée (source) » ou « à mesurer »)](fiches/K-81.md) | P-20, M-03 | `docs/References.md` : une ligne par profil de `profils.md` (28), chaque cellule « mesurée (source) » ou « à mesurer » (0 valeur non sourcée, relecture) ; `docs/Weights.md` : les 13 dépôts du registre + les candidats de `modeles-2026-09.md` §3.2 avec octets exacts et date ; aucun fichier `.swift` modifié. | S | cloud | — | fait |
| K-82 | [`docs/References.md`, `docs/Weights.md` et décisions remplis avec les mesures et les SHA-256 relevés](fiches/K-82.md) | P-20, M-03, K-M09 | 0 « à mesurer » restant dans la table de `References.md` pour les profils de `.all` (sauf largeur non disponible, marquée) ; chaque valeur cite sa ligne `BENCHMARKS.md` ; `Weights.md` : SHA-256 relevé par `curl -s https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=true` pour chaque fichier de poids retenu, révision notée ; `docs/knowledge/index.md` à jour. | S | macos-gpu | K-77, K-78, K-79, K-64 | à faire |

### Classement du lot 4 par gain attendu (aucun n'est « obtenu »)

| Rang | Fiche | Gain attendu (source) |
|---|---|---|
| 1 | K-38 | Realtime : ms/pas −30 % au moins, ≈ ×2,2 attendu (trafic 5,8 → 2,6 Go/pas, calcul) ; pic −1 Go |
| 2 | K-39 | TTS bf16 : pas ÷ 2 à 2,8 (32,1 → 11,3 Go/frame, calcul) ; 4/6 bits 5-15 % |
| 3 | K-40 | STT : KV ÷ 2 ; bf16 ≈ 5× moins de trafic par Linear |
| 4 | K-41 | TTS : transitoire du codec ≈ 2 × 10,5 Go → ≈ 10 Mo (texte long) |
| 5 | K-42 | STT : −1 min 09 à −2 min 25 au 1er lancement si `.mlx` gagne ; précision d'encodeur (CER ×1,74 en 6 bits, externe) |
| 6 | K-43 | TTS streaming : codec ÷ 33 sur le texte long |
| 7-8 | K-44, K-45 | `asyncEval` TTS / STT : −13 à −22 % par pas (T15) |
| 9-10 | K-46, K-47 | Realtime : tête 8 bits ≈ −15 % de trafic ; `asyncEval` ≥ −5 % |
| 11 | K-48 | TTS : 7 → 4 pas d'Euler, −15 à −20 % ms/frame (qualité à l'écoute) |
| 12 | K-49 | Chat : −3 à −6 s par question suivante |
| 13 | K-50 | TTS voix clonées : préfill −30 % (≈ −200 ms, chemin FluxForge) |
| 14-16 | K-51, K-52, K-53 | Mémoire STT/TTS/Realtime : décodage +5 % sur ≤ 31 Go ; pic ramené à actif + `cacheLimit` ; pic −10 % à 30 min |
| 17-20 | K-54, K-55, K-56, K-57 | Préfill −10 % ; KV −47 % en lean ; pic d'encodage −30 % ; FM +5 % |
| 21-37 | K-58 … K-75 | leviers faibles, UX, R&D et refontes (gain ≤ 5-20 % ou non chiffré) |



## 4. Pièges à cocher

Numéros de `references/techniques.md` §4 (skill `mlx-swift-audit`) applicables à ce dépôt, avec la fiche où ils
mordent :

| N° | Piège | Où il mord ici |
|---|---|---|
| 1 | Passe sur `parameters()` qui matérialise une voie non résidente | modules factices aléatoires (P-24) : K-59, K-58, K-63 |
| 2 / 30 | `QuantizedLinear.weight` est `uint32` ; `asType(layer.weight.dtype)` | dtype de calcul lu sur `norm`/`scales` : K-8, K-38, K-39, K-40, K-46 |
| 3 | Muter un tableau non évalué | K-12, K-44, K-45, K-53, K-72 |
| 4 | `eval` tenseur par tenseur au chargement | K-59 |
| 5 | Graphe différé trop gros | encodeur en un lot (P-13) : K-56 |
| 6 | `asyncEval` code mort | K-44, K-45, K-47 |
| 7 | Pas de `cacheLimit` = mémoire qui explose | K-51, K-52 |
| 8 | Autre app MLX sur la machine | toutes les mesures (K-32 refuse via `pgrep` + manifestes `RuntimeBeacon`) |
| 9 | Debug contre Release | tests en Debug, mesures en Release (§0) ; campagne XCTest (P-76) |
| 10 | Profiler par couche | K-32 (profiler désactivé pendant le chronométrage) |
| 11 | Première mesure faussée | requête d'amorçage exclue ; #29 était une mesure froide |
| 12 | Repli silencieux | tokenizer démo (K-7), id Realtime inconnu (K-9), pack absent |
| 13 / 14 | Jetons générés ≠ préfixe ; cache sans état positionnel | session de chat (K-49), cache de préfixe TTS (K-50) |
| 16 | `verify: [.all]` et clés en trop | `embedTokens.weight` dupliqué par `sanitize` : K-7, K-74 |
| 17 | `memoryLimit` n'est pas un plafond dur | K-51, K-52 |
| 18 | `activeMemory` immobile après un chargement paresseux | K-37, K-59 ; diagnostics #17/#28 |
| 19 | Spotlight, `mediaanalysisd`, Time Machine | sorties sous `.local-runs/bench.noindex/` |
| 20 | Deadlock ABBA compile × vjp | K-11 (enrôlement ∥ synthèse), K-71 ; mlx-swift 0.31.6 sans `df9ae26` |
| 21 | Branche de dépendance mouvante | mlx-swift-lm `main` : révision résolue dans chaque ligne (K-22, K-32) |
| 22 / 23 | iOS : GPU en arrière-plan, Simulator ; formes fixes Core ML | hors plan tant qu'ASK-2 n'est pas tranché ; K-42 |
| 24 | Tenseur transitoire qui double | K-41 (scores ×2 à 2 266 frames), K-53 |
| 25 | Estimation de gain trop optimiste | #13 prévoyait −75/−85 %, mesuré −20/−8 % (en session) : chaque fiche a sa porte |
| 26 | Constante scalaire fp32 qui promeut | `MLXArray(scale)` (TTS), tables RoPE (Realtime) : K-3, K-39, K-58 |
| 27 | `prepare` qui ignore la tranche de préfill | K-54, K-75 |
| 29 / 36 | `RotatingKVCache` : pas de retour arrière ; dimensionner à la longueur | K-2, K-13, K-73 |
| 31 | Stream sans `onTermination` | K-12 |
| 32 | « Téléchargé » = un fichier | K-6, K-9, K-24 |
| 33 | Instrument hors du chemin de la bibliothèque | `VoxtralBenchmark` (A-16) : K-32 |
| 37 | Modèle derrière un lien symbolique de dossier | tests #49 conservés (K-6, K-29) |
| 38 | Un test qui prouve un correctif échoue sans lui | toutes les fiches de stabilité (`{RED}`) |
| 39 | `tests | grep && git commit` | toutes les fiches |
| 40 | Poids aléatoires : graine + L2 relative | tests sur modèles réduits (K-1, K-2, K-3, K-41) |

Pièges **nouveaux** proposés au catalogue (numéros à attribuer par la phase 6) : jeton d'arrêt hérité d'un autre
tokenizer (S-01) ; `RotatingKVCache` comme plafond mémoire d'un LM sans fenêtre (S-02/P-03) ; `update(parameters:)`
non levant (MLX-018) ; tokenizer démo en repli (S-05) ; `AsyncThrowingStream` à production synchrone (MLX-003, variante synchrone) ;
compile × vjp sans aucun `compile()` dans le dépôt (A-01) ; complétude par `config.json`/`params.json` et index classé
après les shards (MLX-012) ; clé `"mode"` dans `quantization` (M-02) ; dérive d'un dépôt officiel (M-01) ; backend
externe qui masque la quantification (M-07) ; entrée fp32 non castée et fp16 × bf16 = fp32 (P-01) ; masque fp32 +
q bf16 ⇒ exception (P-02) ; tête liée par `matmul` brut (P-61) ; fenêtre déclarée mais ignorée (P-62/P-63) ;
`maxTokens` sur un modèle synchrone à la trame (P-64) ; phases imbriquées et GPU % instantané (P-73) ; attention
fenêtrée en T×T (P-32) ; streaming qui re-décode tout (P-33) ; champ public jamais lu (P-35) ; plafond fixe
indépendant de l'entrée (P-41) ; deux conventions de RTF (V-P4) ; tag dépendant d'une branche (V-P15) ; fichiers
suivis malgré `.gitignore` (S-25) ; bibliothèque qui remet le pic à zéro (P-77).

## 5. Commandes de référence

```bash
# Build (Release, toutes les mesures) — $CLI = .build/xcode/Build/Products/Release/VoxtralCLI
xcodebuild -scheme VoxtralCLI -configuration Release -derivedDataPath .build/xcode -destination 'platform=macOS' build
#   → ** BUILD SUCCEEDED **
# Autres schémas : VoxtralApp, VoxtralBenchmark, VoxtralTTSStreamingDemo (même commande, -scheme <nom>)

# Tests (Debug, @testable, sans parallélisme — piège 20) ; cibler avec -only-testing:VoxtralCoreTests/<Classe>
xcodebuild test -scheme MLXVoxtralSwift-Package -destination 'platform=macOS' \
  -derivedDataPath .build/xcode-test -parallel-testing-enabled NO
#   → Executed <N> tests, with 0 failures (0 unexpected) … ** TEST SUCCEEDED **
# Tests gardés par variable : préfixe TEST_RUNNER_ (ex. TEST_RUNNER_VOXTRAL_LONG_AUDIO=1 xcodebuild test …)
# Thread Sanitizer : ajouter -enableThreadSanitizer YES

# Machine prête à mesurer
~/.claude/skills/mlx-swift-audit/scripts/machine-check.sh $CLI --cooldown 120 --procs 'Voxtral.*|FluxForge.*'   # aucune ligne KO

# Corpus (PLAN §0) — sorties hors index Spotlight
mkdir -p .local-runs/corpus .local-runs/bench.noindex
# C-court : docs/examples/fluxforge_short_{en,fr}_6bit.wav (5,0 / 4,8 s) ; C-moyen : fluxforge_long_{en,fr}_6bit.wav (167,0 / 173,8 s)
for n in 2 3; do
  for i in $(seq $n); do echo "file '$PWD/docs/examples/fluxforge_long_en_6bit.wav'"; echo "file '$PWD/docs/examples/fluxforge_long_fr_6bit.wav'"; done \
    > .local-runs/corpus/list_$n.txt
done
ffmpeg -y -f concat -safe 0 -i .local-runs/corpus/list_2.txt -ar 16000 -ac 1 .local-runs/corpus/c_long.wav    # ≈ 11 min 22 s
ffmpeg -y -f concat -safe 0 -i .local-runs/corpus/list_3.txt -ar 16000 -ac 1 .local-runs/corpus/c_xlong.wav   # ≈ 17 min
ffmpeg -y -stream_loop 5 -i .local-runs/corpus/c_long.wav -t 1800 .local-runs/corpus/c_30min.wav               # 30 min
# Textes de référence (vérifier qu'ils couvrent l'audio : K-33)
awk 'f && /^> /{sub(/^> /,""); print >> (".local-runs/corpus/long_" f ".txt")} /^### Long FR/{f="fr"} /^### Long EN/{f="en"}' docs/tts_benchmark.md
# Dernières phrases (portes « rien n'est coupé ») : EN « No data sent to the cloud. » · FR « Aucune donnee envoyee dans le cloud. »

# Mesure et qualité (après K-32 / K-33)
$CLI bench stt --model mini-3b-8bit --backend mlx --input docs/examples/fluxforge_long_en_6bit.wav --language en \
  --passes 2 --warmup 1 --cooldown 120 --tag A
# A/B/B/A depuis deux worktrees : copier le Package.resolved de B dans A, construire les deux avec -onlyUsePackageVersionsFromResolvedFile, noter le rev-parse de mlx-swift, mlx-swift-lm et swift-mlx-profiler dans .build/xcode/SourcePackages/checkouts/ (identiques)
$CLI eval stt --model mini-3b-8bit --backend mlx --corpus docs/eval/corpus.json
VOXTRAL_DTYPE_AUDIT=1 $CLI bench realtime --model realtime-4b-4bit --input docs/examples/fluxforge_short_en_6bit.wav --passes 1
# Pic mémoire avant K-32 : /usr/bin/time -l $CLI … (ligne « peak memory footprint »)
# Occupation GPU réelle : xcrun xctrace record --template 'Metal System Trace' --launch -- $CLI bench … ; ioreg -r -c AGXAccelerator

# Garde syntaxique (cloud, sans toolchain Swift ; ce n'est pas une compilation)
python3 ~/.claude/skills/mlx-swift-patterns/scripts/syntax_guard.py .        # 0 erreur nouvelle

# Dispatch des fiches macos-gpu (planificateur)
python3 ~/.claude/skills/task-dispatch/scripts/dispatch.py docs/audit/2026-09-27/tasks.yaml                 # essai à blanc
python3 ~/.claude/skills/task-dispatch/scripts/dispatch.py docs/audit/2026-09-27/tasks.yaml --emit-json \
  --issue-map docs/audit/2026-09-27/map.json                                                                # sans gh (MCP)
```

## 6. Hors plan (noté pour plus tard)

| Élément | Source | Raison | Déclencheur / suite |
|---|---|---|---|
| Serveur d'inférence (HTTP OpenAI-compatible ou binaire MCP) | A-23, F-A12 | aucun consommateur ne le demande ; prérequis K-11, K-12, K-6 | ASK-1 = B ou C ⇒ fiche depuis `audit-annexes-serveur.md` §4 |
| Cible iOS réelle (STT, TTS, Realtime, enrôlement) | FA-09, T23, A-07 | compilée, jamais exécutée ; pics non sourcés | ASK-2 = oui ⇒ fiche appareil (profils `lean`, porte GPU, jetsam) |
| API de streaming Realtime | P-71 | périmètre produit | ASK-3 = oui ⇒ fiche L (encodeur par tranches de K-13 réutilisé) |
| Suppression de l'API legacy et morte | S-13 lot 2, S-14, S-17, S-21, P-24 | cassant | version 3.0, après K-30 et ASK-23 |
| Réécriture de l'historique git (WAV, `.serena`) | S-25 | cassant pour les clones ; gain réel seulement ainsi | ASK-30 « oui » explicite |
| Dither de sortie optionnel | ACT-14 (#45 item 5) | LTX tolère les zéros numériques du codec | un consommateur dont la détection de silence suppose un plancher naturel |
| Passe-haut 50 Hz par défaut | ACT-15, V-R6 | réjection 19 dB au lieu de 30 dB à 30 Hz pour tous | option `--high-pass-hz 50` gardée ; champ du profil d'enrôlement (K-64) |
| ≈ 5 dB de fondamentale perdus à la génération | ACT-17 | inhérent (pas d'encodeur de codec publié) | publication d'un encodeur de codec par Mistral |
| Actions côté FluxForge : réenrôler les voix antérieures à `f63e2a8`, retirer le déchargement après chaque aperçu, doc de stockage (#7 résolu en v2.2.1), ask #8 HubApi | ACT-40, ACT-34, ACT-38 | hors dépôt | plan manuel `project:fluxforge-studio-swift` (ASK-31) |
| Coût hôte d'`eval()` mlx-swift (#536, action-plans) | ACT-08 | suivi amont existant (`monitoring`) | cité par K-44, K-45, K-47 |
| Spéculatif TTS, « batch lookup » (#26) | ACT-28 | aucune mesure, rejeté a priori | après K-44 si le pas reste borné par le CPU |
| Drafter publié `jburtoft/Voxtral-Mini-3B-2507-draft-4layer` | modeles §3.2 | externe (L40S, anglais), pas de spéculatif dans Voxtral | après K-75 option B |
| KV TurboQuant / variance normalisée (mlx-swift-lm) | modeles §4 | aucune parité WER Voxtral | après K-55 |
| Modes mxfp4 / mxfp8 / nvfp4, `quantizedQuantizedMM` | M-02, Q-M8 | preuve externe négative, pas d'échelle globale en 0.31.6 | refus explicite livré par K-8 ; réexaminer au prochain tag |
| Relever `mlx-swift` au-delà de 0.31.6 (correctif `df9ae26`, mlx 0.32.2) | Q-M7, A-01 | aucun tag ne le contient ; mlx-swift-lm impose `upToNextMinor` | ASK-29 ; plan upstream-blocker à créer en même temps que celui de K-21 |
| Conversion Core ML du LLM complet ; cache binaire du tokenizer « 10-100× » ; WAV en bloc | faits §5.2 | aucune mesure | — |
| « Voxtral Mini Transcribe 2.0 » | modeles §1 | cité par la carte Realtime, sans dépôt sur le Hub | publication sur le Hub |
| Double synchronisation par époque d'enrôlement | A-10 | gain attendu < 1 % | passager de K-64, retiré si < 5 % |
| P-25 (masque concaténé) | audit STT annexe B | écarté : chemin jamais atteint | — |
| #18 (pas 1 lent), #29 (codec 4 bits) | ACT-26, ACT-31 | attendu (préfill inclus) ; faux positif de première mesure | capitalisés (piège 11, V-R10) |
| Capitalisation phase 6 (section « audio » du catalogue, pièges 41+, patterns MLX-017…025 livrés dans claude-skills mlx-swift 0.4.0) | tous les rapports | dépôt `claude-skills`, pas ce dépôt | retours skill de cet audit |

## Annexe A — Où atterrit chaque action inventoriée (« ne perdre aucune action »)

Source : `faits-et-actions.md` §3 (ACT-xx) et les listes d'actions des rapports frères.

| Action | Élément | Disposition |
|---|---|---|
| ACT-01 | Issues / PR / branches ouvertes | aucune (0/0, seule `main`) : rien à reprendre |
| ACT-02 | action-plans #71 (PR #34 fusionnée) | K-21 (fermer `verified`, ⛔ ASK-31) |
| ACT-03 | action-plans #307 (PR #41 fusionnée) | K-21 |
| ACT-04 | action-plans #349 (#45 fermée) | K-21 |
| ACT-05 | #66 (PR #33) : point du plan de test jamais coché | K-33 (auto-détection de langue) |
| ACT-06 | 11 plans `stale-branch` | déjà `verified` : rien |
| ACT-07 / ACT-39 / ACT-41 | « Revisit once … tag beyond 3.31.4 » sans plan | K-21 (plan upstream-blocker) ; K-22 (`Package.resolved`) ; ASK-28 |
| ACT-08 | #536 coût hôte d'`eval()` | hors plan (suivi amont), cité par K-44/K-45/K-47 |
| ACT-10 | #45 item 1 : garde NaN | fermé ; tests de la garde : K-26 |
| ACT-11 | #45 item 2 : coupe alignée sur les jetons | réfutée, fermée (V-R1…V-R4) |
| ACT-12 | #45 item 3 : q6 contre bf16 pour voix enrôlées | K-79 + ASK-5 |
| ACT-13 | #45 item 4 : niveau de sortie | fermé (documenté) |
| ACT-14 | #45 item 5 : zéros numériques, dither | hors plan (§6) |
| ACT-15 | `--high-pass-hz 50` | hors plan ; champ du profil d'enrôlement (K-64, K-76) |
| ACT-16 | réenrôlement depuis le micro brut | fait (`4fb44b7`, `a7045f5`) ; résidu FluxForge → §6 |
| ACT-17 | déficit de fondamentale | hors plan (§6) |
| ACT-18 | détecteur de fuite « Là, là » | K-79 |
| ACT-20 | consigne `xcodebuild` seulement dans une issue | K-17 (`CLAUDE.md`) |
| ACT-21 | #12 mel sur GPU | K-69 |
| ACT-22 | #13/#17/#19/#21 : « 49 % GPU », Small sur 32 Go | K-34 |
| ACT-23 / ACT-25 | #14, #16, #22 : Core ML à 48 %, compilation à froid | K-42 (+ K-25) |
| ACT-24 | #15/#20 chat lent, top-p approché | K-70 |
| ACT-26 | #18 pas 1 lent | attendu : rien |
| ACT-27 | #23-#25 Realtime | K-38, K-46 (P-61) ; K-17 et K-36 (re-diagnostic) |
| ACT-28 | #26 génération sémantique | K-44 ; spéculatif hors plan |
| ACT-29 | #27 bf16 lent, clôture sur affirmation fausse | K-79, K-76 (manager), K-18 (doc) |
| ACT-30 | #28 deux passes LLM | K-66 |
| ACT-31 | #29 codec 4 bits | fermé (faux positif, capitalisé) |
| ACT-32 | PR #33 contrôle non-anglais | K-33 |
| ACT-33 | PR #32 iOS | ASK-2 ; hors plan tant que non tranché |
| ACT-34 / ACT-38 / ACT-40 | FluxForge (déchargement, HubApi, réenrôlement, doc stockage) | plan manuel FluxForge (§6, ASK-31) |
| ACT-35 / ACT-36 / ACT-37 | PR #41, #44, #47 | soldés (#44, #45) ; #47 → K-79 |
| ACT-42 / ACT-43 | TODO `VoxtralComponents.swift:28`, `:566` | K-23 (privé) / K-30 (public) |
| ACT-44 / ACT-45 | « No quantization for now », workaround `embed_tokens` | K-30, K-74 |
| ACT-46 | `Memory.cacheLimit = Int.max` | K-23 (fonction morte supprimée), K-52 |
| ACT-47 | chemin `/Users/vincent/…` | K-23 |
| ACT-48 / ACT-50 | gate « true silence » (code, README) | K-18 |
| ACT-51 | « Next step: Swift/MLX port » | K-19 |
| ACT-52 | chiffres de réenrôlement divergents (FV-54) | K-18 (deux sources citées) |
| ACT-53 | llms.txt, exigences, architecture | K-18 |
| ACT-54 | métriques TTS (RTF, TTFT, TTFA) | K-17 (glossaire), K-18 (annotations), K-32 (TTFA mesuré) |
| ACT-55 | FluxForge et Voxtral passent ensemble à `from:` | ASK-28 (déclenché par le plan upstream-blocker de K-21) |
| stabilité — 5 ASK | dépôt Small 8 bits, cache KV, API legacy, épinglage, WAV | ASK-15, ASK-7, ASK-23, ASK-28, ASK-30 |
| annexes — F-A1…F-A12 | fiches du rapport annexes | K-11, K-6, K-25, K-19 + K-42, K-19, K-32, K-33, K-26, K-64, K-42, K-28, hors plan (serveur) |
| annexes — Q1…Q7 | serveur, iOS, backend, API factice, défaut d'enrôlement, `VoxtralBenchmark`, ressource d'app | ASK-1, ASK-2, ASK-6, ASK-23, ASK-9, ASK-26, ASK-27 |
| STT — ASK 1…6 | backend, cache KV, chemin hérité, modules factices, iOS, corpus / modèle de référence | ASK-6, ASK-7, ASK-23, ASK-23, ASK-2, ASK-14 |
| Realtime — Q1…Q8 | tolérance, référence, `maxTokens`, 8 bits, iOS, streaming, corpus, `VoxtralBenchmark` | ASK-11, ASK-12, ASK-8, ASK-20, ASK-2, ASK-3, ASK-14, ASK-26 |
| TTS — écoute, porteur, défaut | P-30/P-35/P-38, P-47, P-34 | ASK-13, ASK-10, ASK-5 |
| modèles — Q-M1…Q-M8 | Small 8 bits · packs tiers · `realtime-4b` · licence TTS · largeurs TTS · backend STT · mlx-swift · modes non affines | Q-M1 → ASK-15 · Q-M2 → ASK-16 · Q-M3 → ASK-17 · Q-M4 → ASK-18 · Q-M5 → ASK-19 · Q-M6 → ASK-6 · Q-M7 → ASK-29 · Q-M8 → ASK-21 |
| modèles — K-M01…K-M09 (pas de K-M04 : M-04 fusionné dans K-M02) | fiches proposées par `modeles-2026-09.md` §6 | K-M01 → K-9 · K-M02 → K-8 · K-M03 → K-24 (+ K-81, K-82 pour `Weights.md`) · K-M05 → K-9 · K-M06 → K-24 · K-M07 → K-42 · K-M08 → K-80 · K-M09 → K-76 (+ mesures K-77…K-79, K-82) |
| modèles — PK-1…PK-4 | packs à publier (Realtime 8 b, TTS 8 b, Mini à encodeur 8 b / bf16, Small à encodeur dense) | K-80, décisions : PK-1 → ASK-20, ASK-22 · PK-2 → ASK-18, ASK-19, ASK-22 · PK-3 → ASK-6, ASK-22 (utile seulement si K-42 retient `.mlx`) · PK-4 → ASK-16, ASK-22 |
| cadrage — 5 questions | profils, iOS, défaut TTS, hybride, serveur | ASK-4, ASK-2, ASK-5, ASK-6, ASK-1 |
| patterns — actions sur le skill (apply.py, détecteurs, MLX-017…025 livrés dans claude-skills mlx-swift 0.4.0) | dépôt `claude-skills` | hors plan de ce dépôt ; retours skill |
| patterns — hunk MLX-003 | dry-run sûr | K-12 |
| contrôle du 2026-09-27 (critique de complétude) | numéros ACT-09, ACT-19, ACT-49 jamais attribués par `faits-et-actions.md` (52 actions = ACT-01…ACT-55 moins ces trois) ; tracker relu par MCP : 15 plans `project:mlx-voxtral-swift` (3 ouverts #71, #307, #349 ; 12 `verified`), 0 issue et 0 PR ouvertes sur `mlx-voxtral-swift`, recherche `mlx-voxtral-swift` hors de ce label : 1 résultat (#540, `project:flux-2-swift-mlx`, sans lien avec Voxtral) | aucune action perdue ; rien à ajouter |

## 7. Journal

Les entrées s'ajoutent en fin de fichier, une par fiche exécutée (et une par ASK posée en cours d'exécution).
Gabarits :

```
## K-x — <titre> — <AAAA-MM-JJ> — validée|bloquée|retirée
- Fait : …
- Mesure : A <…> / B <…> / B <…> / A <…> (dispersion A : x %)
- Porte observée : <ligne recopiée>
- Parité : <ligne recopiée>
- Révisions : mlx-swift <…>, mlx-swift-lm <…>, swift-mlx-profiler <…>

## ASK — K-x — <AAAA-MM-JJ>
- Contexte : (2 lignes)
- Ce que j'ai essayé : (3 lignes)
- Question : (fermée)
- Options : A) … B) …
```

## 2026-09-27 — audit — phases 1 à 4 terminées
- Fait : scan (phase 1), 7 rapports vérifiés (phase 2, vérification croisée), profils (phase 3), plan de 82 fiches,
  31 décisions ASK, `tasks.yaml` (76 tâches `macos-gpu`, essai à blanc `dispatch.py` : 76 tâches valides).
- Mesure : aucune (session cloud Linux, sans Mac).
- Suite : exécution des fiches cloud K-17, K-18, K-19 et K-81 (K-20 et K-21 après ASK-30 et ASK-31) ; ASK à
  trancher ; dispatch des fiches Mac après accord.

## 2026-09-27 — audit — critique de complétude (phases 0-4 contre le SKILL.md)
- Fait : contrôle mécanique (script de lecture, aucun build) de l'ordre imposé, des prérequis, des portes et de la
  cohérence PLAN / fiches / `tasks.yaml` ; matrice points d'entrée × audits (`faits-et-actions.md` §1.2 bis) ;
  catalogue T1…T23 complété (`audit-performance.md` §2, §2.0 bis) ; matrice des profils complétée (`profils.md` §0).
- Corrections : prérequis K-35 et K-36 (+ K-33), K-65 (+ K-36), K-74 (+ K-34), K-82 (+ K-64) ; portes K-32, K-34
  (chat), K-35 (couverture ASR), K-36 (WER) étendues à l'identique dans PLAN, fiches et `tasks.yaml` ; méthode WER
  écrite dans K-9 et K-13 ; « calcul » et « en session » ajoutés à trois chiffres du §1 et du §4.
- Mesure : aucune. Validation : `dispatch.py tasks.yaml` → « 76 tâche(s) valides — rien créé » ; `dispatch.py PLAN.md
  --runs-on macos-gpu --project mlx-voxtral-swift` → « 82 tâche(s) valides » mais lot faux (README, section Exécution).
- Reste ouvert : voir le rapport de critique (pas de chiffre obtenu ; ASK inchangées).

## K-17 — Mémoire du projet : `CLAUDE.md`, `docs/knowledge/`, `BENCHMARKS.md`, protocole et glossaire — 2026-09-27 — validée
- Fait : `CLAUDE.md` (57 lignes : build `xcodebuild` Release, tests Debug sans parallélisme et `TEST_RUNNER_*`,
  `machine-check.sh`, A/B/B/A, seuil 5 %, `mlx-swift-lm` sur `main` avec la tête `ee673d6` notée, consommateurs
  FluxForge et SongAnalysisDb, commits) ; `BENCHMARKS.md` (règle « jamais éditée », colonnes de la ligne `BENCH` de
  K-32, aucune ligne) ; `docs/Benchmarks.md` (protocole §0, corpus §5, glossaire de 16 métriques, une définition
  chacune, sourcée) ; `docs/knowledge/` (index OKF, log daté « audit », décision `realtime-diagnostics-23-25.md`,
  6 pièges au format Symptôme · Cause · Correctif · Règle). Aucune mesure ; aucun `.swift` touché. Pièges cochés :
  21 (révision notée), 9 (Debug contre Release), 33 (instrument hors chemin : `VoxtralBenchmark`).
- Mesure : aucune (documentation, session cloud).
- Porte observée :
  `FILES CLAUDE.md BENCHMARKS.md docs/Benchmarks.md docs/knowledge/index.md docs/knowledge/log.md → présents (+ docs/knowledge/decisions/realtime-diagnostics-23-25.md)`
  · `PITFALLS docs/knowledge/pitfalls/*.md : 6 (≥ 6)`
  · `SWIFT git diff --stat -- '*.swift' : vide`
  · `LINKS 0 lien relatif mort (42 liens vérifiés)`
  · `CLAUDE.md commandes PLAN.md §5 (build :297, test :302-303, machine-check :309) : 4/4 lignes identiques ; 57 lignes (≤ 60)`
  · `GLOSSARY 16 métriques, 0 doublon` ; relecture : chaque chiffre porte sa source (fichier:ligne, commit, issue ou
  rapport d'audit) ou la mention « calcul » / « en session ».

## K-81 — Squelette sourcé de `docs/References.md` et `docs/Weights.md` — 2026-09-27 — validée
- Fait : `docs/References.md` (brouillon de `profils.md` §8 — la fiche dit « §7 », numérotation décalée — adapté :
  une ligne par profil, 28 ; colonne « Source » `profils.md:N` par ligne ; cellules de mesure « to measure (K-n) » ou
  « — » pour les 4 profils non disponibles ; valeurs du brouillon sans source remplacées par « to decide (K-n) » ;
  équivalents CLI corrigés d'après `VoxtralCLI.swift:177-187`, `:210`, `:215`) ; `docs/Weights.md` (13 ids du
  registre, 3 dépôts hors registre téléchargés par le code — enum `small24b8bit`, Core ML mini/small —, 13 candidats
  de `modeles-2026-09.md` §3.2 ; octets exacts, date de MAJ et licence du Hub, format, chargeable ou non,
  SHA-256 « to record (K-82) »). Octets relistés le 2026-09-27 par le connecteur Hugging Face (`hf_fs ls`) :
  identiques au rapport pour tous ses dépôts ; `aufklarer/…-MLX-5bit` et `…-MLX-FP16`, non inspectés par le rapport,
  relevés par K-81. Piège coché : dérive des dépôts officiels (Weights.md §4 « relister à chaque audit »). Aucun
  `.swift` touché.
- Mesure : aucune (documentation, session cloud).
- Porte observée :
  `REFERENCES 28 lignes, 0 valeur non sourcée`
  · `WEIGHTS 13 dépôts du registre + 13 candidats, octets exacts (Hub 2026-09-27)`
  · `REFERENCES 28 lignes (4 non disponibles), 0 valeur non sourcée, 0 ligne de tableau mal formée`
  · `LINKS 20 liens relatifs, 0 mort`
  · `WEIGHTS registre 13 dépôts + 3 hors registre + 13 candidats §3.2 ; 0 ligne sans octets sourcés`
  · `SWIFT git status -- '*.swift' : vide` ; relecture : chaque réglage et chaque valeur « in session » porte sa source
  (`profils.md:N`, `modeles:N`, fichier:ligne) ; octets datés « Hub, 2026-09-27 ».

## K-18 — Docs utilisateur alignées sur le code (exigences, versions, llms.txt, README, réglages TTS, chiffres requalifiés) — 2026-09-27 — validée
- Fait : `README.md` (exigences macOS 15+ / iOS 17+, Xcode 26+ / Swift 6.2 ; section « Depending on v2.2.x » :
  `revision:` ou `branch: "main"` tant que mlx-swift-lm n'a pas de tag > 3.31.4 ; Realtime dans l'intro, les
  Features, une section modèles/CLI/API ; tailles réelles d'après `docs/Weights.md` ; les deux dépôts de
  `small-24b-8bit` ; préparation de référence passe-haut 70 Hz → normalisation −20 dBFS → gate −24 dB ; réglages TTS
  `flowSteps`/`cfgAlpha`/`temperature` signalés sans effet (P-35, K-48) ; limite du streaming (S-08) ; section Hybrid
  réécrite (pas de mode auto en CLI, « ~660 MB » sourcé `1944576`/FV-11 et « en session ») ; arbre complet ; section
  Documentation liant `docs/References.md`, `docs/Weights.md` (K-81), `CLAUDE.md`, `docs/Benchmarks.md` (K-17)) ;
  `llms.txt` réécrit (v2.2.x, 3 pipelines, `tokenCount` toujours 0, `ModelDownloader` non thread-safe, section
  Concurrency, champs réels de `MemoryOptimizationConfig`, codec calculé en fp32) ; tableaux de mesure annotés
  (définition, révision, « en session ») dans `README.md`, `docs/tts_benchmark.md` (+ codec fp32, P-38),
  `docs/zerovoice_benchmark.md` (juge non validé, P-78), `docs/voice_cloning.md` (FV-54 : les deux jeux, `db34ca0`
  et `4fb44b7` ; RTF de la campagne, P-76) ; `docs/streaming_demo.md` (TTFT ≈ Total, FA-04) ; commentaires Swift
  seulement : `VoxtralTTSModeling.swift:480-482`, `:528` (P-31), `VoxtralVoiceEnrollment.swift:51` (FA-05),
  en-tête `Examples/ReferenceImplementation.swift`. Littéraux de version du code (`VoxtralCore.swift:45`,
  `VoxtralCLI.swift:25`) non modifiés (code exécutable) : signalés dans les docs. Pièges cochés : 21 (révision notée
  ou « non notée » par tableau ; dépendance de branche documentée), V-P4 (RTF = génération ÷ audio rappelé partout,
  « RT factor » de `profile` signalé inverse).
- Mesure : aucune (documentation, session cloud).
- Porte observée :
  `CHECKLIST S-20 : 12/12` (points de audit-stabilite.md:541-555 relus contre `9392ed1` : 1 version llms.txt →
  v2.2.x ; 2 TTS/clonage/Realtime → 3 pipelines ; 3 `ModelDownloader` « Thread-safe » → 0 ; 4 `tokenCount` →
  « always 0 » ; 5 trois versions → tag = version, 0.1.0 et 2.0.0 signalés ; 6 `small-24b-8bit` → deux dépôts ;
  7 tailles → octets Hub ; 8 `voxtral tts` → 0 ; 9 arbre → Realtime/, CoreML/, Pipeline/, VoiceCloning/,
  VoxtralBenchmark ; 10 Realtime dans les Features ; 11 section Hybrid `:263-267` → « Auto mode » 0, « 660 MB »
  sourcé ; 12 en-tête Examples → « Tested with: v1.0.8 » 0)
  · `GREP 'Swift 6.0|Xcode 15|macOS 14' : 0 · 'true silence' : 0`
  · `SYNTAX_GUARD 0 erreur nouvelle (3 fichiers .swift, commentaires)` (sortie : « 3 fichier(s) Swift, 0 erreur(s)
  d'analyse nouvelle(s) » ; `git diff -U0 -- '*.swift'` : 0 ligne modifiée hors commentaire)
  · `TABLES mesure annotées : README 6/6, tts_benchmark 3/3 + bloc TTFT, zerovoice 11/11 (bloc global), voice_cloning
  3/3` · `LLMS v2.2.x : 2 · VoxtralPipeline 16 · VoxtralRealtimePipeline 6 · VoxtralTTSPipeline 9`
  · `FV-54 : 6.4 dB/144→88 Hz (db34ca0) et −14.5→−3.4 dB/132.6→89.6 Hz (4fb44b7) cités`
  · `LINKS 56 liens relatifs, 0 mort`.
- Hors périmètre (note) : littéraux de version à générer depuis le tag et résolution `from: "2.2.2"` chez un projet
  témoin (FA-02, volet macos-gpu) ne sont portés par aucune fiche ; `docs/voice_cloning.md` emploie `voxtral …` pour
  `VoxtralCLI`.

## K-19 — Annexes Python reproductibles (conversion Core ML, recherche clonage) — 2026-09-27 — validée
- Fait : `Scripts/CoreMLConversion` : `convert.sh` sans `--include-projector`, `--variant mini` passé aux deux
  scripts, `pip install huggingface_hub` non épinglé retiré, `hf download --revision 3060fe3…` ; README (Quick Start =
  `convert.sh`, étapes 1-5 rejouables, Python 3.11-3.13, tableau « un modèle par variante » avec la commande Small
  `--revision da5b424…`, chiffres « ~150 ms / ~500 ms » annotés A-12/F-12 → K-42, octets du téléchargement d'après
  `docs/Weights.md`) ; `requirements.txt` épinglé (`torch==2.7.0` : `_TORCH_MAX_VERSION = "2.7.0"`,
  `coremltools/_deps/__init__.py:158` du wheel 9.0, avertissement dès 2.7.1 `:33-39` ; `coremltools==9.0` ;
  numpy 2.3.5 et safetensors 0.7.0 = dernières versions à `1944576` ; `huggingface_hub==1.22.0` car 1.2.4 ne démarre
  plus sur un venv neuf, « No module named 'click' », constaté en session). `Scripts/VoiceCloningResearch` :
  commit amont `ac3e3f3` (tête de `main`, blobs du patch `50b812c`/`2ee2720` identiques, `git apply --check` OK)
  épinglé dans le README et contrôlé par `enroll_voice.py` (`UPSTREAM_COMMIT`, `check_workspace`) ;
  `torch.load(weights_only=True)` (`codes_to_embeddings.py:234` sauve un tenseur nu) ; `hf download --revision
  b81be46…` ; `requirements.txt` = dernières versions PyPI à `f9d0ec9` (torch 2.12.1, cohérent avec « verified torch
  2.12 »), + `torchcodec==0.14.0` (`torchaudio.load` 2.11 passe par torchcodec, `torchaudio/__init__.py:9`, `:86`) et
  FFmpeg au README ; section « Python and Swift paths » (pertes : Python 0,5 L1 + STFT + log-mel + MFCC + 0,5
  locuteur ECAPA avec gradients, `training_script.py:613-633` ; Swift 0,5 L1 + STFT + log-mel,
  `VoxtralVoiceEnrollment.swift:43-45`, `:556-568` ; températures différentes ; 0,56-0,59 à 8 s contre 0,69/0,72 à
  8/16 s : aucune conclusion de supériorité, A-21) ; commande de similarité ECAPA (`SpeakerRecognition.verify_files`,
  `speechbrain/spkrec-ecapa-voxceleb` révision `0f99f2d…`) ; « Next step » remplacé par « Swift/MLX port (done) »
  (`bd59931`, PR #34). En-tête (commentaire) de `VoxtralVoiceEnrollment.swift` : pertes retirées et température
  confrontées. Révisions Hub : chaque SHA relevé dans des dépôts tiers épinglés (recherche de code GitHub), puis
  contrôlé par le connecteur Hugging Face : révision existante (une révision fictive rend `exists: false`) et listing
  identique à `main` (2026-09-28). Pièges cochés : V-P9 (la voie Python n'est pas une référence de gradients : ligne
  « Spectral gradients », aucun classement des deux voies).
- Mesure : aucune (scripts et documentation, session cloud). Parité Core ML sur Mac : K-42.
- Porte observée :
  `ARGPARSE convert.sh : 0 inconnu / 0 manquant · README : 0 / 0` (7 invocations : 2 de `convert.sh`, 4 du README
  Core ML dont Small, 1 du README recherche ; avant : `convert.sh : 1 inconnu / 0 manquant · README : 2 / 2`)
  · `PINNED requirements : 0 '>='` (`grep -E '>=' Scripts/*/requirements.txt` : 0 ligne ; 15 lignes, toutes `==`)
  · `GREP 'Next step' : 0`
  · `ARGPARSE enroll_voice.py → upstream : 0 inconnu / 0 manquant` (2 appels, contre l'amont à `ac3e3f3`)
  · `HF download : 4 commande(s), 0 option inconnue, 4/4 avec --revision` (options de `hf download --help`,
  huggingface_hub 1.22.0)
  · `CHECK_WORKSPACE` : amont à `f3070b4` → refus « not at the pinned commit » ; `ac3e3f3` sans patch → refus « lacks
  the MPS gradient fixes » ; `ac3e3f3` + patch → OK
  · `RESOLVE uv pip compile --python-platform aarch64-apple-darwin` : Core ML py3.11 et 3.13 OK (3.10, 3.14 : pas de
  solution) ; recherche (macOS 14) py3.12 et 3.14 OK (3.11 : pas de solution)
  · `SYNTAX_GUARD 1 fichier(s) Swift, 0 erreur(s) d'analyse nouvelle(s)` (`git diff -U0 -- '*.swift'` : 0 ligne
  modifiée hors commentaire)
  · ECAPA : snippet du README analysé (`ast.parse` OK), `SpeakerRecognition.verify_files` et
  `FetchConfig(revision=…)` présents dans le wheel speechbrain 1.1.0.
- Hors périmètre (note) : `convert_to_coreml_ane.py:314-316` affiche encore `VoxtralCLI benchmark-coreml` (signalé au
  README, fichier hors liste) ; section « Using in Swift » du README Core ML (`VoxtralConfiguration`,
  `VoxtralGenerator(configuration:)`) non vérifiée ; dépendances transitives non verrouillées ; `hf download`
  télécharge aussi `consolidated.safetensors` (≈ moitié du volume) ; métadonnées de version et SHA-256 du
  `weight.bin` produit (A-03) non portés par cette fiche.

## 2026-09-28 — revue adverse des fiches cloud K-17, K-81, K-18, K-19
- Fait : portes des quatre fiches rejouées avec des contrôles propres à la revue (scripts du scratchpad
  `rev3/` : liens, rejeu argparse par AST, versions PyPI à date, `check_workspace` dans trois états, clone neuf de
  l'amont) ; relecture de chaque document contre le code à `9392ed1` et à `HEAD`. Un commit `docs(review): …`.
- Portes rejouées : K-17 `FILES … → présents` · `PITFALLS 6 (≥ 6)` · `SWIFT git diff --stat -- '*.swift'` vide
  (22a117f..4170f36) · `LINKS 0 mort` · commandes de `CLAUDE.md` = PLAN.md :297, :302-303, :309 ; K-81
  `REFERENCES 28 lignes` (STT 12, Realtime 6, TTS 8, enrôlement 2) · `WEIGHTS 13 + 3 + 13` (octets recoupés avec
  `profils.md`/`modeles` ; `aufklarer` 5 bits et FP16 relistés : 4 051 350 065 et 9 352 633 693 o) ; K-18
  `CHECKLIST S-20 : 12/12` (greps rejoués) · `GREP 'Swift 6.0|Xcode 15|macOS 14' : 0 · 'true silence' : 0` ·
  `SYNTAX_GUARD 3 fichier(s) Swift, 0 erreur(s) d'analyse nouvelle(s)` (`--base 9392ed1`) ; K-19
  `ARGPARSE convert.sh : 0 inconnu / 0 manquant · README : 0 / 0` (avant : 1 / 0 et 2 / 2) · `PINNED
  requirements : 0 '>='` · `GREP 'Next step' : 0` · amont `ac3e3f3` = tête (`git ls-remote`), blobs `50b812c`/
  `2ee2720`, `git apply --check` OK · `check_workspace` : `f3070b4` refusé, `ac3e3f3` sans patch refusé, avec
  patch OK. `.swift` : commentaires seulement (`git diff 9392ed1 -- '*.swift'`).
- Défauts corrigés : (1) citations `VoxtralVoiceEnrollment.swift` et `VoxtralTTSModeling.swift` de README,
  llms.txt, `docs/Benchmarks.md`, `docs/tts_benchmark.md` décalées par les commentaires de K-18/K-19 (jusqu'à
  13 lignes) → numérotation de `HEAD` ; (2) « TTFT excludes the voice-prefix computation » faux pour les chiffres
  d'avril (cache de préfixe introduit par `f4fd21c`) et pour une voix clonée en batch → README, tts_benchmark,
  glossaire, log ; (3) llms.txt : `ModelDownloader.defaultModel` n'existe pas ; (4) `docs/References.md` :
  `voxtral …` → `VoxtralCLI …` (le README dit que `voxtral` n'est pas un binaire installé) ; (5) `voice_cloning.md`
  : note sur le nom `voxtral` ; `Scripts/VoiceCloningResearch/README.md` : `VoxtralCLI enroll` ; (6) README Core ML
  « Using in Swift » : `VoxtralConfiguration(backend:)` et `VoxtralGenerator(configuration:)` inexistants →
  `VoxtralPipeline(model:backend:)` ; (7) `convert.sh` conseillait `swift build` → `xcodebuild` ; (8)
  `docs/knowledge/index.md` : les 23,89 s sont la phase « Realtime Generation » (encodage et préfill compris), pas
  le décodage.
- Non corrigé (note) : chiffres du texte courant de `voice_cloning.md` sans source en ligne (« 15× », −55/−65/−126 dB,
  « one generation in eight ») ; « ~30 min (5000 epochs) » du README de recherche, « ~10GB » et « ~1.2GB » de `convert.sh` non
  sourcés ; « macOS 13.0+ or iOS 16.0+ » du README Core ML = cible de déploiement du modèle (`iOS16`), pas du
  paquet ; les pièges `docs/knowledge/pitfalls/` gardent la numérotation de `9392ed1` (datée dans leur en-tête) ;
  audit-stabilite.md §0 donne mlx-swift `@9019419` contre `0bb916c` au §1.

## K-21 — Tracker action-plans : solder #71, #307, #349, plan upstream-blocker — 2026-09-28 — validée
- Fait : ASK-31 tranchée par la demande de Vincent (« ne perds pas les actions sur Voxtral » : le tracker est le
  canal du test). Sources recontrôlées par le MCP GitHub (PR #34 fusionnée 2026-07-09T07:37:04Z, PR #41 fusionnée
  2026-07-20T14:06:41Z, #45 fermée « completed » 2026-07-27T08:16:47Z). Écritures faites par le mode sans gh du skill
  `track` (`update-plan.py --issue-json … --emit-json --close`, `new-plan.py --emit-json`) puis le MCP : corps
  identiques à l'octet près hors `status`, labels en phase (relus après écriture).
- Porte observée : `PLANS ready-to-act project:mlx-voxtral-swift : 0` · `CLOSED #71 #307 #349 status:verified
  (commentaire-preuve)` · `NEW upstream-blocker #556 monitoring (github_release ml-explore/mlx-swift-lm semver_gt
  3.31.4)` + source `calendar` (next_review 2026-11-02) en filet, car la présence de GitHub Releases amont n'a pas pu
  être vérifiée depuis la session (proxy : dépôt hors périmètre) ; repli sur les tags ajouté au plugin
  `github_release` (claude-skills, agent-tracker 0.2.0), actif sur le Jetson après #553/#554.
- En plus (ACT-40) : plan `manual` #557 `project:fluxforge` pour les suites côté consommateur (réenrôlement des voix
  d'avant `f63e2a8`, déchargement après aperçu, doc de stockage, ask #8 HubApi).
- Reste : lien du plan #556 à ajouter au commentaire de `Package.swift:46-51` au prochain commit de code (K-22).

## K-20 — Hygiène git — 2026-09-28 — partielle (⛔ ASK-30 pour les WAV)
- Fait : `git rm -r --cached .serena` (2 fichiers, 1 790 543 o retirés de l'arbre suivi ; les fichiers restent sur
  le disque, `.gitignore:119` les couvre déjà). Rien fait sur les WAV : ASK-30 sans réponse.
- Porte observée : `git ls-files -ci --exclude-standard | wc -l` : 24 → 22 (reste : les 22 WAV de ASK-30).

## Dispatch des lots 1 à 3 — 2026-09-28 — fait
- Fait : `dispatch.py tasks.yaml --only K-1…K-16,K-22…K-37 --emit-json --issue-map map.json` (task-dispatch de
  claude-skills 0.5.0, mode sans gh), 6 vagues créées par le MCP GitHub dans l'ordre des dépendances ; carte locale →
  issue dans [`map.json`](map.json) (32 tâches). Aller-retour vérifié sur #566 (corps identique à l'opération
  émise) ; titres relus (aucun `&gt;`).
- Porte observée : `32 tâche(s) valides` · `Vagues : 6` · `Naissent blocked (⛔) : 9` (K-5, K-9, K-10, K-13, K-22,
  K-28, K-30, K-31, K-32) · issues action-plans #558 à #589 · `Erreurs : aucune`.
- Lots 4 à 6 (44 fiches) volontairement **non créés** : leurs portes dépendent des baselines. Tâche de planification
  **#590** (`runs-on:linux`, `depends_on` #586-#589 avec `depends_on_status: verified`) : recaler, retirer ou
  dispatcher K-38…K-82 une fois K-34…K-37 vérifiées.
- Chemin critique : K-32 (instrument, ⛔ ASK-26) dépend de K-22 (⛔ ASK-28) ; sans ces deux réponses, aucune
  baseline (K-33…K-37) ne peut démarrer.


## Renumérotation du catalogue de patterns (claude-skills 0.6.0) — 2026-09-28 — fait
- Fait : `main` de claude-skills a publié sa 0.5.0 (tag `v0.5.0`, MLX-016 = experts MoE non quantifiés) pendant que
  le lot issu de cet audit était préparé sous ce numéro. Le lot sort donc en **0.6.0** (mlx-swift 0.4.0,
  claude-skills#1) et les patterns de l'audit sont décalés d'un rang : MLX-017…025. Fiches (ligne « Patterns du
  catalogue », sources de K-1, K-3, K-7), tableau des fiches (§3), pièges (§4), lignes de capitalisation et en-tête de
  `patterns-verdicts.md` recalés ; les ids
  provisoires du rapport de verdicts restent entre « ». Les mentions « claude-skills 0.5.0 » du journal ci-dessus
  désignent ce lot.
- Porte observée : `apply.py scan` (claude-skills `f0ebb9a`) sur ce dépôt : 25 patterns, identifiants uniques ;
  MLX-016 relève 3 filtres `Linear || Embedding` (`MLXLMBridge.swift:612`, `:706`, `VoxtralQuantization.swift:65`),
  sans objet ici : aucun `SwitchLinear` ni module MoE dans `Sources/`.

## K-6 — Téléchargements prouvés complets (manifeste + SHA-256), `downloadModel` factice neutralisé — 2026-09-30 — validée
- Fait : `downloadRepoDirect` lit `lfs.oid`, vérifie taille et SHA-256 (CryptoKit, lecture par blocs de 8 Mio) avant de
  placer chaque fichier, retire puis écrit en dernier `.voxtral-complete.json` (liste, tailles, SHA-256, révision) ;
  `isComplete(folder:requiresVoices:)` commun à STT, TTS et Realtime (manifeste vérifié ; sans manifeste : index +
  tous ses shards + `tekken.json` (+ voix en TTS) ⇒ manifeste écrit) ; `verifyShardedModel` ne dit plus « complet »
  sans index (sauf `model.safetensors` unique) ni sur index illisible ; `download()`, `downloadTTSModel`,
  `downloadRealtimeModel` lèvent si incomplet ; champ `revision` optionnel dans les 3 registres ;
  `downloadModel(modelId:)` lève `VoxtralError.unsupported` (nouveau cas) sans rien créer.
- Écarts à la fiche : `modelSize` et `deleteModel` localisent le dossier sans exiger la complétude (sinon
  `ModelDownloaderSizeTests` rougit, et un téléchargement partiel ne se supprimerait plus) ; la taille suit les
  liens symboliques (`stat`) ; coupure réseau simulée par `kill -9` du CLI (choix de Vincent) plutôt que Wi-Fi coupé ;
  CLI sans option de dossier : téléchargement dans `~/Library/Caches/models`, supprimé après.
- Catalogue : `apply.py scan . --pattern MLX-012` (claude-skills `04c888b`) : 0 avant, 0 après — le détecteur
  (`contains { $0.hasSuffix(".safetensors") }`) ne voit pas les variantes de ce dépôt.
- Porte observée :
  - `RED   DownloadCompletenessTests : Executed 6 tests, with 18 failures (0 unexpected)` (6/6 tests rouges, correctif stashé)
  - `GREEN Executed 6 tests, with 0 failures (0 unexpected)`
  - `GREEN ModelDownloaderSizeTests, ModelLoadingSymlinkedDirectoryTests : 0 failures` ; suite complète :
    `Executed 492 tests, with 12 tests skipped and 0 failures (0 unexpected)`
  - `RESUME tts-4b-4bit : coupure à 51.0 % → relance → manifeste écrit, SHA-256 OK (1/1)` (25 fichiers, 22 avec
    SHA-256 ; `model.safetensors` a62a28f0… = `shasum -a 256` ; `list -d` : absent après coupure, présent après)
- Mesure : aucune (fiche sans levier de performance).

## K-16 — Globaux protégés, `MLXArray` évalués avant transfert entre acteurs — 2026-09-30 — validée
- Fait : `Locked<Value>` (verrou, `Utils/Locked.swift`) remplace les 8 déclarations `nonisolated(unsafe)` : cache
  mel (filtres évalués avant mise en cache), `writeDebugToDump`, `customModelsDirectory`, `_hubApi`,
  `resourceBundle`, `VoxtralDebug.enabled`/`verboseGeneration` (propriétés publiques devenues calculées, même API) ;
  `_mergeCallCount` (jamais lu) supprimé. `VoxtralMemoryManager.config` et `evalCounter` sous verrou ; la pipeline
  n'écrit plus la configuration globale et passe la sienne à `generateStream*` et à
  `optimizeIfNeeded(tokenIndex:config:)` (surcharge additive). `TTSSynthesisResult`, `TTSStreamingChunk` et
  `GenerationChunk` évaluent leur tableau dans leur `init` (couvre tous les `return`/`yield`).
- Catalogue : `apply.py scan --pattern MLX-004` (claude-skills `04c888b`) : 0 avant, 0 après (le détecteur ne cherche
  que `nonisolated(unsafe) let` et `UncheckedTransfer(`).
- Porte observée :
  - `GREP nonisolated(unsafe) : 8 déclarations (13 lignes) → 0 déclaration (1 ligne : commentaire de Locked.swift)`
  - `RED   SharedStateTests : testTwoPipelinesKeepTheirOwnMemoryConfiguration échoue (evalFreq=0 ≠ evalFreq=8) ;
    testConcurrentSharedSettings plante le process de test` ; TSan avant : 14 avertissements (11 VoxtralCore, 3 MLX)
  - `GREEN SharedStateTests (TSan) : 0 avertissement VoxtralCore ; 2 dans MLX (MetalAllocator::malloc,
    allocator.cpp:165 / get_active_memory, allocator.h:28 : compteur lu sans verrou, présent jusqu'à mlx main et
    mlx-swift 0.32.2, hors de ce dépôt, sans objet pour la fiche — décision de Vincent)`
  - `CrossActorWaveformTests 20/20` (tts-4b-4bit sur /Volumes/Lexar/models, graine 42) ;
    `PARITY WAV identiques : sha256 b7ed7956… avant = après correctif (245 804 o)`
  - suite complète : `Executed 496 tests, with 13 tests skipped and 0 failures (0 unexpected)`
- Mesure : aucune (fiche de stabilité).

## K-25 — Encodeur Core ML : chemin unique sous `customModelsDirectory`, hors ligne, erreurs explicites — 2026-09-30 — validée
- Fait : `VoxtralCoreMLEncoder.downloadFromHuggingFace` passe par `ModelDownloader.downloadRepoDirect`
  (globs `<nom>/*`, `<nom>/*/*`) vers `modelsDirectory/<org>/<repo>/<nom>`, avec le manifeste K-6 ; une copie
  vérifiée (manifeste + `model.mil` + `weights/weight.bin`) se recharge sans réseau ; plus de `HubApi` (ni
  `import Hub`) dans l'encodeur. Forme de sortie vérifiée au chargement (largeur déclarée ≠ variante ⇒ erreur) et à
  chaque `encode` ; noms legacy (`VoxtralEncoderFull`…) réservés à `.mini` ; l'encodeur hybride transmet sa
  configuration de variante, ne charge plus de Core ML quand `.mlx` est demandé, et refuse d'encoder avec l'encodeur
  MLX aléatoire par défaut (un encodeur fourni par `setMLXEncoder` reste utilisable) ; `mlxAvailable` = poids
  chargés ; repli MLX et statut d'encodeur journalisés (`VoxtralDebug.always`).
- Écart : « réseau coupé » simulé par `URLProtocol` dans le test et par `sandbox-exec` (sortie IP interdite, vérifié :
  `curl` → 000) pour le CLI, qui n'a pas d'option de dossier (contrôle réel sous `~/Library/Caches/models`).
- Porte observée :
  - `RED   testMiniEncoderUnderSmallConfigThrows, testUninitializedMLXEncoderThrows : Executed 2 tests, with 2 failures`
  - `GREEN CoreMLEncoderPathTests : Executed 3 tests, with 0 failures (0 unexpected)` (lourds inclus,
    `customModelsDirectory = <tmp>/VoxtralModels`, encodeur trouvé sous ce dossier)
  - `OFFLINE 2e chargement hybride : Core ML available: true` (test : 0 requête ; CLI sous sandbox-exec :
    `Encoder: Core ML (Neural Engine), Core ML available: true`, transcription correcte)
  - `HFCACHE ~/.cache/huggingface : +0 octet` (test : 3 393 475 543 → 3 393 475 543 ; CLI : 3 314 216 Ko avant/après)
  - suite complète : `Executed 499 tests, with 15 tests skipped and 0 failures (0 unexpected)`
- Mesure : aucune.

## Rôles et amendements du planificateur — 2026-09-28 — fait
- Rôles (Vincent, 2026-09-28) : la session Voxtral du Mac est la seule à committer dans ce dépôt ; les autres agents
  passent par une tâche action-plans ; planification et vérification (applied → verified) par une session cloud,
  dans action-plans uniquement ; réponses aux ASK et fusions : Vincent ; claude-skills n'est écrit que par la
  session Mac. Consigné dans `CLAUDE.md` (section « Rôles », 60 lignes).
- Chemin critique corrigé : K-32 dépend aussi de K-5 ; il faut donc ASK-8, ASK-9, ASK-26 et ASK-28 pour ouvrir les
  baselines. K-14 (#564) et K-8 (#568) passées blocked sur ASK-9 et ASK-21 (⛔ dans le tableau §3, les fiches et
  `ASK.md`) ; K-33…K-37 (#585-#589) en `depends_on_status: verified`.
- Amendements recopiés depuis les corps des issues dans `tasks.yaml` : K-1 (#558) et K-32…K-37 (#584-#589) ; K-1.md
  étapes 2, 3 et 7 réécrites (test (b) sans `withMLXErrors`, ≥ 2 couches, vérification par couche si piège Swift ;
  rouge en ne retirant que l'enveloppement ; A/B/B/A avec `-onlyUsePackageVersionsFromResolvedFile` et rev-parse des
  dépendances). `--procs 'Voxtral.*|FluxForge.*'` partout (FluxForge Studio, app MLX, n'était pas détectée) ; #590
  réécrite (planification des lots 4 à 6). `.local-runs/` ignoré.
- Exécuté le 2026-09-30 (#599), après bascule de `~/Developpements/claude-skills` sur la branche de claude-skills#1
  (0.6.0, `f0ebb9a`, choix de Vincent).
- Porte observée : `wc -l < CLAUDE.md` = 60 ; `git grep -n "'Voxtral\.\*'" -- CLAUDE.md docs/audit/2026-09-27` → 0
  ligne ; `tasks.yaml` : `depends_on_status: verified` = 5, `Amendement du planificateur` = 7 ; `dispatch.py` (essai à
  blanc) : « Naissent blocked (⛔) : 23 », « Erreurs : aucune ».

## K-1 — Erreurs MLX levées au lieu de terminer le processus hôte — 2026-09-30 — validée
- Fait : `VoxtralError.mlx(String)` ; `Utils/MLXErrorBoundary.swift` : `withMLXErrors` (sync et async) autour de
  `withError`, qui convertit `MLXError.caught` et publie l'`ErrorBox` en `@TaskLocal` (`MLXErrorScope`). Enveloppés :
  `generateStream`, `generateStreamWithAudioEmbeds` (`try errors.check()` après chaque appel du modèle, chaque `eval`
  et avant `item`), `VoxtralPipeline.loadModel/transcribe/chat`, `VoxtralTTSPipeline.loadModel/synthesize(voice)/
  synthesize(voiceEmbedding)/enrollVoice` et le streaming **dans** sa `Task` productrice (`check()` après l'`eval` de
  chaque chunk), `VoxtralRealtimePipeline.loadModel/transcribe/extractAudioEmbeddings`. Amendement appliqué : avec le
  correctif, le test (b) s'arrêtait sur « Index out of range » (la couche suivante lit la forme d'un tableau vide) ;
  la boucle du décodeur (`LlamaStandardModel`) et `VoxtralRealtimeModel.generate` s'arrêtent dès qu'une erreur est
  enregistrée (`MLXErrorScope.hasError`), sans changer de signature publique.
- Catalogue : `apply.py scan --pattern MLX-021` (claude-skills `f0ebb9a`, 0.6.0) : 1 → 0.
- Porte observée :
  - `RED   MLXErrorBoundaryTests.testPrefillBeyondRotatingWindowThrows : crash du runner (fatalError) — correctif retiré`
    (`Fatal error: [broadcast_shapes] Shapes (512,2560) and (1,4,512,2559) cannot be broadcast`, fast.cpp:629)
  - `GREEN Executed 2 tests, with 0 failures (0 unexpected)`
  - `Executed 501 tests, with 15 tests skipped and 0 failures (0 unexpected)` (suite complète)
  - `OVERHEAD real A=15,76 B=15,75 B=15,75 A=15,72 → écart +0,06 % (≤ 5 %)` ; dispersion A/A 0,25 % ;
    transcriptions identiques (sha `d3a452…`, 4/4)
- Mesure : `/usr/bin/time -p $CLI transcribe docs/examples/fluxforge_long_en_6bit.wav -m mini-3b-8bit -b mlx -l en`
  (C-moyen EN, 167 s), A = `0742d08e` (worktree), B = correctif, `Package.resolved` de B copié dans A, les deux
  construits avec `-onlyUsePackageVersionsFromResolvedFile` ; `machine-check --procs 'Voxtral.*|FluxForge.*'` sans KO ;
  amorçage A puis B exclus ; 120 s entre points. Une 1ʳᵉ série avait amorcé B seul : A1 = 17,84 (1ᵉʳ lancement du
  binaire A, cache Metal froid), B = 15,71/15,74, A2 = 15,72 — non conclusive (dispersion A/A 13 %), refaite.
- Révisions : mlx-swift `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`, mlx-swift-lm
  `604fae710a4e3324346fc59e3845952350acd4b7`, swift-mlx-profiler `b2a83b36a24b2e252573369644259a648fbaf18a`
  (identiques A et B).

## K-4 — Jetons d'arrêt dérivés du tokenizer (fin de la troncature sur « ␣Capital ») — 2026-09-30 — validée
- Fait : `VoxtralForConditionalGeneration.stopTokenIds` (additif), une seule définition pour `generateStream` et
  `generateStreamWithAudioEmbeds`, par défaut `[2, 4]` (`</s>`, `[/INST]`) ; `VoxtralPipeline.loadModel` la pose depuis
  son tokenizer (`eosToken` de `generation_config.json`, `getControlToken("[/INST]")`). 32000 (« ␣Capital ») supprimé.
- Écart : la preuve rouge ne passe pas par `git stash` (le test lit `stopTokenIds`, absent avant) : la propriété a
  d'abord été introduite avec la liste historique `[2, 4, 32000]` (refonte sans changement de comportement), test
  rouge, puis la valeur corrigée. Porte « One » : l'ASR écrit « 1 » (normalisation des nombres, pas une troncature) ;
  jugée atteinte par Vincent.
- Catalogue : `apply.py scan --pattern MLX-017` : 2 → 1 (reste la valeur de repli `[2, 4]`, deux ids spéciaux ; le
  détecteur ne distingue pas un id spécial d'un id texte).
- Porte observée :
  - `RED   StopTokenTests.testStopTokensAreSpecialIds : 32000 ≥ 1000`
  - `GREEN Executed 2 tests, with 0 failures (0 unexpected)` ; suite complète :
    `Executed 503 tests, with 15 tests skipped and 0 failures (0 unexpected)`
  - `CAPITAL before: « Capital A's and Capital »  after: « Capital A's and Capital 1 are two different things. The
    capital of France is Paris, »` (clip `tts-4b-6bit`, `neutral_female`, graine 42 ; « One » rendu « 1 »)
  - `PARITY greedy 3/3 identiques (C-court EN, C-court FR, C-moyen EN)` (`mini-3b-8bit`, `-b mlx`, `cmp`)
- Mesure : aucune.

## K-7 — Chargement vérifié et tokenizer strict — 2026-09-30 — validée
- Fait : `Module.updateVerified(parameters:)` (`Utils/VerifiedWeights.swift`) dans les 3 chargeurs vivants (STT
  `VoxtralStandardLoader`, TTS, Realtime) : clés du modèle absentes ⇒ `VoxtralError.missingWeights([clés])`, puis
  `update(…, verify: [.allModelKeysSet, .shapeMismatch])` ; TTS : shards lus depuis l'index. `TekkenTokenizer.load(modelPath:)
  throws` (`fileNotFound`, `VoxtralError.invalidTokenizer`, vocabulaire vide ou regex invalide) ; `init` déprécié (repli
  démo conservé pour lui seul) ; `demo()` pour les tests ; `fromPretrained` (STT) passe par `load` ; TTS et Realtime
  chargent le tokenizer avant les poids.
- Trouvés par la vérification (sans affaiblir celle-ci) : (1) 3 constantes calculées du TTS étaient comptées comme
  paramètres (`timeEmbedding.invFreq`, `alibiSlopes`, `codebookOffsets`) → préfixe `_` (hors `parameters()`) ;
  (2) **`realtime-4b-fp16` chargeait ses 104 poids d'attention décodeur au hasard** (noms Mistral `attention.wq/wk/wv/wo`
  non traduits en format A) et transcrivait « .. » : traduction ajoutée (choix de Vincent), il transcrit maintenant
  comme le 4 bits ; le test `testFormatADecoderLayers`, qui figeait l'ancien nom, est corrigé.
- Écarts : parité TTS faite par `--voice-embedding …/fr_female.safetensors --seed 42` (le CLI ignore `--seed` pour les
  voix prédéfinies : `-v fr_female` n'est pas reproductible, avant comme après) ; A = worktree `4a51497e`. `tts-4b`
  (poids Mistral) : poids chargés sans clé manquante, mais voix en `.pt` non lues (« 0 voices loaded »), comme avant K-7.
- Catalogue : MLX-018 7 → 3 (restent les chargeurs hérités `MLXLMBridge`, `VoxtralModelLoading.loadWeights`, hors des 3
  vivants) ; MLX-022 4 → 3 (restent `demo()` et le repli de l'`init` déprécié, voulus).
- Porte observée :
  - `GREEN VerifiedLoadingTests + TekkenStrictLoadTests : Executed 7 tests, with 0 failures` ; suite complète
    `Executed 510 tests, with 15 tests skipped and 0 failures (0 unexpected)`
  - `LOAD mini-3b-4bit OK · mini-3b-8bit OK · mini-3b OK · tts-4b-4bit OK · tts-4b-6bit OK · tts-4b-mlx OK · tts-4b OK
    (poids ; voix .pt non lues) · realtime-4b-4bit OK · realtime-4b-fp16 OK (après traduction wq→q_proj)`
  - `MISSING-SHARD → VoxtralError.missingWeights([…])` : 750 clés de couche nommées (ex. `audioTower.layers.0.fc1.bias`) ;
    rouge avant (aucune erreur levée)
  - `PARITY stt identique ; cmp tts : aucune différence` (SHA `66ef9ca0…` A = B, 4/4) ; ids Tekken identiques 20/20
- Complément (vérification du 2026-10-01) : preuve rouge de « dossier privé d'un shard → erreur nommant ≥ 1 clé ».
  `VerifiedLoadingTests` copié dans un worktree du parent `4a51497` (ressource non suivie `VoxtralEncoderFull.mlmodelc`
  recopiée pour la cible app) : RED `Executed 1 test, with 1 failure (0 unexpected)` (`XCTAssertThrowsError failed: did
  not throw an error`, code de sortie 65) ; GREEN sur la branche `Executed 1 test, with 0 failures (0 unexpected)`.
  Packs Small (Voxtral-Small-24B-2507, voxtral-small-8bit, voxtral-small-4bit-mixed) : absents du Mac, LOAD non mesuré.
- Mesure : aucune. Modèles sur `/Volumes/Lexar/models` via liens de dossier depuis `~/Library/Caches/models`.

## K-11 — Exclusion enrôlement / inférence et machine d'états atomique des pipelines — 2026-09-30 — validée
- Fait : `Utils/PipelineGate.swift` (`OSAllocatedUnfairLock`) : état, opération en cours et jeton de génération changés
  sous un seul verrou ; STT, TTS et Realtime lisent `state` depuis la porte (propriété publique en lecture seule, même
  API) ; `begin` refuse une seconde opération (`busy`, nouveau cas additif de `VoxtralPipelineError`,
  `VoxtralTTSError`, `VoxtralRealtimeError`) ; `loadModel` fait un test-et-pose atomique ; `enrollVoice` tient la
  pipeline (synthèse, streaming ou chargement pendant l'enrôlement ⇒ `busy`) ; `unload` pose une nouvelle génération :
  la fin d'une Task périmée n'écrit plus `.ready`. Enrôlement dans `withRandomState(MLXRandom.RandomState())` (graine
  fixe : K-26). Démo : sélecteur de modèle, Load et Play désactivés pendant `isEnrolling`, bouton Annuler branché sur
  `shouldContinue` (« Cancelled »).
- Écart : étape 8 (capture de la démo pendant un enrôlement) non faite ici (clics dans l'app) : **à faire par Vincent**
  à la vérification ; la démo compile.
- Catalogue : `apply.py scan --pattern MLX-024` : 1 → 1 (le détecteur pointe le site du gradient,
  `VoxtralVoiceEnrollment.swift:583` ; la protection est l'exclusion au niveau de la pipeline, qu'il ne voit pas).
- Porte observée :
  - `RED   EnrollInferenceExclusionTests : timeout 120 s / chevauchement détecté (correctif retiré)` — « run 1: no
    completion within 120 s (deadlock) », 0/20 (interblocage compile × vjp reproduit)
  - `GREEN EnrollInferenceExclusionTests : 20/20 OK, refus busy en 0 ms (< 1 s)` (tts-4b-4bit, 50 époques ∥
    `synthesizeStreaming`)
  - `GREEN PipelineStateStressTests (TSan) : 10/10, ThreadSanitizer: 0 warnings` (4 scénarios × 10)
  - `Executed 515 tests, with 16 tests skipped and 0 failures (0 unexpected)`
- Mesure : aucune.

## ASK-26 et ASK-28 répondues — 2026-09-30
- Vincent : ASK-28 = A (mlx-swift-lm sur `main`, `Package.resolved` suivi) ; ASK-26 = B (`VoxtralBenchmark` retiré,
  remplacé par `VoxtralCLI bench`). Inscrites dans `ASK.md` (Réponses). #566 (K-22) et #584 (K-32) rouvertes.
- Chemin vers les profils mesurés : K-22 → K-32 → K-33 → baselines K-34…K-37 → K-76…K-79 (+ K-64) ; restent ouvertes
  pour la suite : ASK-8 et ASK-9 (K-5, prérequis de K-32), ASK-4 (K-76), ASK-5, ASK-13, ASK-18 (K-79).
- ASK-8 = A et ASK-9 = A (Vincent, 2026-09-30, même session) : #579 (K-5) et #564 (K-14) rouvertes. Chemin vers les
  baselines : K-2 (#567) → K-5 (#579) ; K-22 (#566) ; puis K-32 (#584).

## K-22 — Package : dépendances élaguées, résolution reproductible, profiler 1.5.1, `@available` nettoyés — 2026-09-30 — validée
- Fait (ASK-28 = A) : `VoxtralCore` ne dépend plus de `MLXLLM` (import sans symbole retiré), `MLXOptimizers`,
  `ArgumentParser` ni du produit `Transformers` (seulement `Hub`) ; `Scripts/VoxtralGenerate.swift` (interne, mort,
  seul importateur d'`ArgumentParser`) exclu de la cible en attendant sa suppression (K-23) ; `ArgumentParser`
  déclaré par la cible CLI ; swift-mlx-profiler `from: "1.5.1"` ; `Package.resolved` suivi (retiré du `.gitignore`) ;
  28 `@available(macOS 13|14…)` sous le plancher (macOS 15 / iOS 17) retirés. `VoxtralBenchmark` gardé ici : son
  retrait (ASK-26 = B) revient à K-32.
- FluxForge : sources de l'app sans import de `MLXLLM`, `MLXOptimizers`, `Transformers`, `ArgumentParser` ; non
  compilé ici (il suit `main` de ce dépôt par branche et son dépôt porte des changements d'une autre session).
- Porte observée :
  - `** BUILD SUCCEEDED ** × 4 (VoxtralCLI, VoxtralApp, VoxtralBenchmark, VoxtralTTSStreamingDemo)`
  - `Executed 515 tests, with 16 tests skipped and 0 failures (0 unexpected)`
  - `DEPS swift-mlx-profiler 1.5.1 · mlx-swift-lm main@604fae7 · mlx-swift 0.31.6` (seul le profiler a bougé :
    1.4.0@b2a83b3 → 1.5.1@bfe71d8)
  - `GREP @available(macOS 1[34] : 0`
  - `BUILD_TIME clean VoxtralCore avant 122 s après 96 s` (build propre Release de `VoxtralCLI`, dossier neuf,
    `-onlyUsePackageVersionsFromResolvedFile`)
  - `Package.resolved` : révisions identiques après `swift package resolve` sur deux clones neufs (voir rapport)
- Mesure : temps de build seulement (une passe chacun, pas de banc).

## K-2 — STT : cache KV sans fenêtre par défaut + garde-fou explicite — 2026-09-30 — validée
- Fait : les 4 préréglages passent à `maxKVCacheSize: nil` ; les deux boucles `generateStream*` utilisent toujours
  `KVCacheSimple` ; une limite explicite (`contextSize` ou `maxKVCacheSize`) est vérifiée avant tout calcul :
  `invite + maxTokens > limite` ⇒ `VoxtralError.contextTooLong(prompt:maxTokens:limit:)` (nouveau cas) ; l'app ne force
  plus 8 192 (interrupteur « Limit context », éteint par défaut). Test K-1 (b) basculé du déclencheur P-03 (disparu)
  sur P-17 (masque fp32 sur modèle bf16, présent jusqu'à K-3) ; `PerformanceOptimizationTests` (qui figeaient les
  fenêtres des préréglages, et dont un plantait sur `!`) réécrits. CHANGELOG à faire : sémantique des préréglages publics.
- **Diagnostic mémoire (swift-mlx-profiler, `profile run --backend mlx`, C-long, mini-3b-8bit, 4 096 jetons)** : pic
  MLX **actif 8,6 Go**, stable (7,2 → 7,8 Go) ; le pic du processus (70,6 Go ; 74 Go sous `/usr/bin/time -l`) est le
  **cache de buffers MLX** (24,5 Go après le préfill → 56,7 Go en fin de décodage, ≈ 8 Mo par jeton), faute de
  `cacheLimit` (P-09) ; il a fait compresser 45 Go et swapper la machine (96 Go). La porte « pic > 12 Go ⇒ ASK-7 » se lit
  sur la mémoire du modèle : 8,6 Go < 12 Go, **ASK-7 non posée**. Décision de Vincent : la politique mémoire (K-52,
  `cacheLimit`) passe **avant** K-5, K-32 et les baselines (des baselines sous swap ne vaudraient rien).
- Écart (décision de Vincent) : les phrases de la porte sont reportées à K-5 : l'ASR écrit « complete **iCreative**
  studio » (déjà ainsi sur C-moyen) ; la dernière phrase FR exige un budget de jetons selon la durée (en langue auto le
  modèle s'arrête après le 1ᵉʳ segment EN, 4 923 caractères ; en `-l en`, 4 096 jetons sont atteints, 20 733 caractères).
  Au passage : `profile run` en backend `auto`/hybride reste bloqué dans `VoxtralCoreMLEncoder.encode` (`prediction`)
  sur C-long (23 fenêtres, > 20 min) : à instruire (K-25/K-42).
- Catalogue : `apply.py scan --pattern MLX-020` : 2 → 0.
- Porte observée :
  - `RED   LongPromptKVCacheTests.testUltraPreset2600Positions : VoxtralError.mlx (avec K-1)` ([broadcast_shapes]
    (512,2560) vs (1,4,512,2559)) ; `testExplicitLimitThrowsContextTooLong` rouge (mlx au lieu de contextTooLong)
  - `GREEN Executed 3 tests, with 0 failures (0 unexpected)` ; suite complète
    `Executed 515 tests, with 17 tests skipped and 0 failures (0 unexpected)`
  - `LONG_AUDIO ultra(8GB)       : 0 crash · first EN OK · last FR KO (K-5)`
  - `LONG_AUDIO aggressive(16GB) : 0 crash · first EN OK · last FR KO (K-5)`
  - `74063779984  peak memory footprint` (/usr/bin/time -l, C-long, mini-3b-8bit, .mlx) — dont MLX actif 8 557,4 Mo
- Mesure : diagnostic, pas une référence (une passe, machine sous pression mémoire).

## K-52 — Politique mémoire MLX opt-in par pipeline — 2026-09-30 — partielle (avancée par Vincent, hors tracker)
- Clauses non mesurées (vérification du 2026-10-01), reprises par la replanification #590 : temps du profil fast ±5 % ;
  volet lean TTS (pic −20 % pour ≤ +5 % de temps) ; Realtime `step_ms_p50` ±3 % ; max/médiane des pas ≤ 1,3.
- Fait : `Utils/MLXCachePolicy.swift` : `cacheLimitBytes: Int?` opt-in (`MemoryOptimizationConfig`, configurations TTS et
  Realtime ; `nil` partout par défaut = l'hôte n'est pas touché) ; la limite est posée à la fin du chargement, le cache
  vidé en fin de réponse (si la limite est active), la valeur de l'hôte restaurée et le cache vidé à `unload()`.
  `profile run --cache-limit-mb`. App : la fonction morte qui posait `cacheLimit = Int.max` restaure la valeur lue.
- Écarts : exécutée sans K-51 (prérequis) et hors tracker (lot 4, pas encore dispatché) ; mesurée par
  swift-mlx-profiler (`profile run`) et `/usr/bin/time -l`, faute de `$CLI bench` (K-32) ; profils fast/lean non encore
  définis (K-76) : volet « lean » TTS (pic −20 % pour ≤ +5 % de temps) non mesuré ; vidage aveugle Realtime tous les
  256 pas inchangé (step p50 non mesuré).
- Porte observée :
  - `STT 10 min : peak_footprint 10 994 750 480 ≤ active 8 555 Mo + cacheLimit 2 048 Mo + coreml 0 (+5 %)` —
    C-long (11 min 21 s), mini-3b-8bit, `.mlx` ; sans limite : 74 070 104 208 (processus 70,6 Go, swap) ; 6/6 points B
    entre 10 343 et 10 472 Mo de pic processus.
  - Temps (génération, 4 096 jetons) : série 1 A 138,5 / B 166,8 / B 134,9 / A 138,1 s ; série 2 (balise + veille des
    balises étrangères : 0) A 183,3 / B 153,3 / B 133,6 / A 134,3 s ; B3/B4 138,5 / 148,6 s. Dispersion A/A de la machine
    jusqu'à 36 % (GPU 89–99 % occupé ; ReportCrashService à 85 % CPU au machine-check), donc ±5 % non démontrable au sens
    strict ; points voisins B2/A2 : −0,5 %, meilleur B / meilleur A : −0,5 % → aucun coût détecté.
  - `TTS unload : footprint ref + 6 Mo (≤ 200)` ; `RT unload : +43 Mo (≤ 200)` (tts-4b-4bit, realtime-4b-4bit, limite
    512 Mo ; la mémoire revient en ≈ 1–5 s : le pilote GPU reprend les buffers Metal libérés de façon asynchrone ;
    MLX actif et cache à 0 dès `unload()`).
  - `MLXCachePolicyTests` 3/3 ; suite `Executed 520 tests, with 19 tests skipped and 0 failures (0 unexpected)`
- Catalogue : `apply.py scan --pattern MLX-010` : 2 → 0.
- Mesure : diagnostic comparatif A/B, pas une baseline (pas de ligne `BENCHMARKS.md` : l'instrument `bench` est K-32).

## K-5 — Plus de troncature silencieuse : `maxTokens` STT selon la durée, Realtime borné par l'audio — 2026-09-30 — validée
- Fait (ASK-8 = A, ASK-9 = A) : STT `Configuration.maxTokens: Int?`, défaut `nil` ⇒ budget
  `automaticMaxTokens(forDuration:) = max(500, ⌈s × 6⌉ + 64)` (6 = 1,5 × le taux mesuré le plus dense) ; une valeur
  explicite reste un plafond, signalé par `lastResultTruncated` (transcribe et chat). Realtime : boucle extraite
  (`decodeLoop`), un pas par trame audio, `maxTokens` = budget de jetons **texte** (pads et spéciaux gratuits), `>=`,
  `lastGenerationTruncated` / `lastTranscriptionTruncated`. CLI `transcribe`/`chat` et `profile` : `--max-tokens`
  optionnel (auto) ; app : 0 = auto. CHANGELOG à faire (défaut public changé : version mineure, FluxForge prévenu).
- Taux mesurés (C-moyen, 4 096 jetons, mini-3b-8bit, `.mlx`) : EN 520 jetons / 167 s = 3,1 j/s ; FR 693 / 173,8 s = 4,0 j/s.
  L'ancien défaut 500 tronquait déjà C-moyen EN.
- Écarts (décisions de Vincent) : « longueur ≤ 1,5 × la référence » remplacé par un contrôle anti-boucle (jetons ≤ 1,5 ×
  durée × taux) : les références du §5 (`docs/tts_benchmark.md`) sont condensées et ne couvrent pas l'audio (C-moyen EN
  transcrit fidèlement = 2,6 × la référence) : vraies références = K-33. Critères C-long reportés : STT C-long bilingue
  (EN/FR alternés) → K-33 (`-l fr` : fin de séquence après 2 244 jetons ; `-l en` : boucle arrêtée par le budget, 4 153
  jetons, `truncated`) ; Realtime C-long → K-13 : l'encodeur dégénère au-delà de ≈ 30 s (P-62 : sortie correcte puis
  octets NUL, déjà sur C-moyen 167 s, non lié à K-5). Incohérence du plan : K-5 exigeait un résultat que seule K-13
  (qui dépend de K-5) peut donner.
- Porte observée :
  - `RED  RealtimeStepBudgetTests.testStepsEqualFrames : steps=4097 frames=5000 (avant)` (refonte à sémantique d'origine)
  - `GREEN RealtimeStepBudgetTests + MaxTokensForDurationTests : Executed 5 tests, with 0 failures` (dont signal
    `truncated` réel : plafond 5 sur C-court ⇒ vrai ; auto ⇒ faux) ; suite `Executed 525 tests, with 20 tests skipped and 0 failures`
  - `STT  C-moyen EN last OK · C-moyen FR last OK` ; anti-boucle EN 520 ≤ 777, FR 693 ≤ 1 043 jetons ; C-long → K-33
  - `RT   C-long` → K-13 (P-62)
- Mesure : aucune (taux de parole consignés ; balise active pendant les inférences).

## K-32 — Instrument de baseline `VoxtralCLI bench` (STT, TTS, Realtime, enrôlement, chat) — 2026-09-30 — validée
- Fait : `Sources/VoxtralTranscriptionTest/BenchCommand.swift` : `bench stt|tts|realtime|enroll|chat` par les pipelines
  publics ; refus d'un binaire Debug (`REFUSED debug build`) et d'une machine occupée (processus MLX, **balise vivante d'un
  autre runtime** : `REFUSED busy`) ; chargement exclu, `--warmup` exclu, `--cooldown` avant chaque passe ; phases et pas
  par les points d'accroche MLXProfiler (session légère, sans échantillonneur) ; pic MLX exact (remis à zéro par
  l'instrument) + pic `phys_footprint` et pic MLX par phase (échantillonneur 5 ms) ; une ligne `BENCH {json}` par passe
  (stdout + `bench.jsonl`), puis `AA …`. `docs/bench.schema.json`, `docs/eval/chat-questions.json`,
  `VOXTRAL_DTYPE_AUDIT=1` (mel, sortie encodeur, cache KV, logits ; STT et Realtime). Bibliothèque : plus aucun
  `GPU.resetPeakMemory` (le champ public `resetPeakMemory` reste, sans effet, pour la compatibilité). ASK-26 = B :
  `VoxtralBenchmark` retiré (Package.swift, sources, CLAUDE.md, README, docs/Benchmarks.md). `profile` : bloc LLM retiré
  pour le Realtime.
- Écarts : `grep -rn resetPeakMemory Sources/VoxtralCore` = 10 (champ public conservé ; 0 appel) ; `--trace` non livré
  (le diagnostic fin reste `profile run`, avec `--backend/--language/--cache-limit-mb`) ; Realtime : `pad_fraction` et
  `first_text_token_ms` non exposés (pas d'accès aux jetons) ; enrôlement non reproductible avant K-26 (graine).
  Pendant la série : `appstoreagent` ≈ 95 % d'un cœur CPU (noté dans `top_process`), sans effet visible sur la dispersion.
- Porte observée :
  - `AA dispersion total_ms=0.09% step_ms_p50=0.14% out_sha256=identical → PASS (≤ 3 %)` (stt, mini-3b-8bit, C-moyen EN, .mlx)
  - `AA dispersion total_ms=0.16% step_ms_p50=0.00% out_sha256=identical → PASS (≤ 3 %)` (tts, tts-4b-6bit, texte court, graine 42)
  - `AA dispersion total_ms=0.31% step_ms_p50=0.11% out_sha256=identical → PASS (≤ 3 %)` (realtime-4b-4bit, C-moyen EN)
  - `AA q1 dispersion ttft_ms=0.20% tok_s=0.50% out_sha256=identical → PASS (≤ 3 %)` (chat, C-court EN, greedy)
  - 8 lignes valides contre `docs/bench.schema.json` (jsonschema), recopiées dans `BENCHMARKS.md` ; `REFUSED debug build`
    (binaire Debug) ; suite `Executed 525 tests, with 20 tests skipped and 0 failures (0 unexpected)`
- Révisions (sur chaque ligne) : mlx-swift 0.31.6@0bb916c67, mlx-swift-lm main@604fae710, swift-mlx-profiler 1.5.1@bfe71d834.

## ASK-11 et ASK-12 répondues — 2026-10-01
- Vincent : ASK-11 = A (WER +0,2 pt) ; ASK-12 = A **sous condition** : mlx-audio comme référence ponctuelle uniquement,
  sortie figée en fichier, aucune dépendance Python dans le code, les tests ou le build. K-13 (#582) rouverte.
  Vérification des tâches `applied` confiée à la session cloud (Vincent).

## K-13 — Realtime : fenêtres glissantes appliquées (encodeur 750 par tranches, décodeur `RotatingKVCache(8192)`) — 2026-10-01 — validée (porte amendée complétée le même jour)
- Fait (ASK-11 = A, ASK-12 = A sous condition) : `VoxtralRealtimeEncoder.encodeChunked` : tranches de 750 positions,
  `RotatingKVCache(maxSize: 750)` par couche, RoPE à position absolue, masque `makeMask(n:windowSize:)` partagé par les
  couches ; `callAsFunction` garde `encodeFull` (`.causal`) jusqu'à 750 positions. Attention de l'encodeur : paramètre
  additif `maskMode:`. Décodeur : `createCache()` → `RotatingKVCache(maxSize: config.slidingWindow)` (8 192).
- Référence mlx-audio (ASK-12, référence seulement) : mlx-audio 0.5.7 (PyPI) + mlx 0.32.3, venv temporaire hors dépôt,
  modèle local `Voxtral-Mini-4B-Realtime-2602-4bit` ; sorties figées `.local-runs/k13_mlxaudio/{en,fr}.txt`
  (SHA-256 `1a3692f948acb739…`, `7b8c4235046b77ff…`). Aucune dépendance Python dans le code, les tests ou le build.
- Porte observée :
  - `GREEN RealtimeSlidingWindowTests : Executed 3 tests, with 0 failures` — rouge avant correctif :
    `[sliding-window] beyond window: L2 rel = 0.5829416` ; vert : `2.4595144e-07` (fenêtre 8, 40 positions)
  - `EQUIV ≤15 s : L2 rel = 0.0 (< 1e-3)` (`encodeChunked` contre `encodeFull` dans la fenêtre)
  - `WER C-moyen EN/FR : avant 66.47/64.79 · après 164.07/138.50 · mlx-audio 165.87/138.97` contre
    `.local-runs/corpus/long_{en,fr}.txt`. Écart : ces références sont condensées (≈ 1 000 car. pour ≈ 2 800 dits,
    constat K-5/K-33) ; « avant » paraît meilleur parce que la sortie dégénérait en octets NUL après ≈ 30 s (491 / 514
    car. utiles). Swift ≤ mlx-audio + 0,2 pt (ASK-11) sur les deux langues. WER contre la sortie mlx-audio prise pour
    référence : avant 82.67/84.52 · après 5.15/2.44 (écarts restants : espacements « flux 2 »/« flux2 », quelques mots).
    Script (pour K-33) : `norm = NFKD → ASCII, minuscules, ponctuation → espace, NUL retirés ; jiwer.wer(ref, hyp)` (jiwer 3.0.4).
  - Texte identique sur C-moyen avec la fenêtre décodeur : `KVCacheSimple` contre `RotatingKVCache(8192)`, EN et FR
    identiques octet pour octet.
  - `PEAK encode 3/6/12 min : 5272.9/5369.8/5563.4 Mo (±10 %)` → +5,5 % (`peak_mlx_mb_by_phase.encode`, 3 lignes `BENCH`).
  - C-xlong (1 022 s, 12 787 pas, `profile run --per-step-memory`) : MLX actif max pas 8 100–8 300 = 6 392,7 Mo, après
    8 300 = 6 392,7 Mo (0 %) ; ms/pas p50 7 800–8 000 = 33,71, 8 250–8 450 = 34,24 (+1,6 %).
  - Suite `Executed 528 tests, with 20 tests skipped and 0 failures (0 unexpected)`.
- Constat : au-delà de ≈ 9 000 pas, ms/pas monte jusqu'à 55 ms (GPU 59 → 65 %, cache constant). Cause **thermique** :
  un clip de 3 min lancé à chaud juste après C-xlong démarre à 48,9 ms/pas (27,8 à froid) puis redescend en refroidissant.
  Pas lié au code ; à garder en tête pour les baselines longues (K-36).
- Écarts de mesure : premier passage invalidé (Time Machine en copie, `KO` du machine-check), série refaite machine
  propre ; balises : seulement les pids de nos exécutions. Pic encodage : 1 passe + 1 amorçage (mémoire déterministe :
  valeurs identiques entre les deux séries). Catalogue : `apply.py scan --pattern MLX-020` : 0 avant, 0 après (le
  détecteur ne voit pas ce cas).
- Complément (amendement du planificateur du 2026-10-01, vu après le premier rapport) :
  - 0 octet NUL : `TekkenTokenizer.decode(skipSpecialTokens: true)` saute tout id < `default_num_special_tokens`
    (comme mistral-common et mlx-audio) ; seuls BOS/EOS/PAD l'étaient, et le Realtime émet `[STREAMING_PAD]` (32) et
    `[STREAMING_WORD]` (33) entre les mots, décodés en rang 0 = octet 0x00. Test `SpecialTokenDecodeTests` : rouge
    `[decode] "\0\0Hi\0\0 there"` → vert `"Hi there"`.
  - Realtime C-long (681 s) : `truncated=false`, 8 527 pas, 11 707 car., `NUL 0`, dernière phrase FR de la référence
    « Aucune donnee envoyee dans le cloud. » présente ; fin de sortie identique à mlx-audio (« … 16Go recommandé »).
    C-moyen EN/FR : `NUL 0`, texte inchangé (2 755 / 3 086 car.).
  - Référence mlx-audio versionnée : `docs/eval/realtime-reference/` (mlx-audio 0.5.7 = `94c7716`, SHA-256 par fichier).
  - Suite `Executed 529 tests, with 20 tests skipped and 0 failures (0 unexpected)`.
- Hors périmètre, signalé : l'invite Realtime Swift utilise `<pad>` (11) × (1 + délai) ; mlx-audio utilise
  `[STREAMING_PAD]` (32) et 32 jetons de remplissage à gauche (`config.py:93-96`). À traiter dans une fiche dédiée
  (parité de l'invite), mesurée contre la référence figée.
- Révisions : mlx-swift 0.31.6@0bb916c67, mlx-swift-lm main@604fae710, swift-mlx-profiler 1.5.1@bfe71d834.

## Vérification du 2026-10-01 — fait
- Par le planificateur (session cloud), décisions de Vincent du 2026-10-01. Vérifiées et fermées : #558 (K-1), #559
  (K-4), #560 (K-6), #563 (K-11), #565 (K-16), #566 (K-22), #567 (K-2), #573 (K-25), #579 (K-5), #599 (rôles).
- Restent `applied` : #561 (K-7, preuve rouge à fournir) et #584 (K-32, sortie de machine-check à fournir).
- Amendements de porte (décisions de Vincent, blocs « Amendement du planificateur (2026-10-01) » des issues, recopiés
  dans les fiches K-2, K-5, K-13, K-33 et dans `tasks.yaml`) :
  - K-2 : porte C-long réduite à 0 arrêt à `recommended(forRAMGB: 8)` et `(forRAMGB: 16)`, `maxTokens` 4 096, préfixe
    EN stable, pic consigné ; les phrases de référence C-long passent dans K-33. ASK-7 = A (ASK.md).
  - K-4 : clip « Capital » : « One » présent est satisfait par le numéral « 1 » (même mot après normalisation).
  - K-16 : « TSan propre » = 0 alerte dans le code du dépôt ; les 2 alertes de MLX (MetalAllocator, allocator.h:28,
    code tiers) sont exclues de la porte.
  - K-5 : porte réduite (test « pas = trames » rouge, budget STT fonction de la durée, dépassement signalé, dernière
    phrase sur C-moyen EN/FR) ; « longueur ≤ 1,5 × la référence » et « STT C-long : dernière phrase » → K-33 ;
    « Realtime : C-long transcrit en entier » → K-13. Le contrôle anti-boucle n'est pas retenu comme preuve.
  - K-13 : porte complétée (C-long Realtime entier, dernière phrase, `truncated=false`, 0 octet NUL) ; ASK-12 = A sous
    condition (référence mlx-audio figée dans un fichier texte versionné, aucune dépendance Python).
  - K-22 : dérogation (ASK.md) : « FluxForge compile » vérifié à la fusion sur `main`. Libellé du temps de build propre :
    mesuré sur `VoxtralCLI` (122 s → 96 s), pas sur `VoxtralCore` seul.
  - K-33 : porte complétée par les critères C-long de K-2 et K-5 ; juge realtime-4b-4bit : WER publié sur les clips de
    30 s au plus tant que K-13 n'est pas `verified`.
- K-52 : renommée « partielle » (clauses non mesurées listées dans son entrée, reprises par #590).
- Obligations à la fusion de la branche sur `main` : compiler FluxForge Studio contre la branche (K-22) ; publier la
  version mineure décrite dans `CHANGELOG.md` (ASK-9 = A, défauts publics changés).
- Tableau du §3 : colonne État alignée sur le tracker (verified, applied, partielle, en cours).

## K-12 — Streaming TTS réel et annulable (production dans une `Task`, `onTermination`, `checkCancellation`) — 2026-10-01 — validée
- Fait : `VoxtralTTSModel.generateStreaming` rend `AsyncThrowingStream.makeStream()` et produit dans une `Task`
  (`produceStreaming`, `try Task.checkCancellation()` à chaque frame) annulée par `onTermination` ; avant, toute la
  génération s'exécutait dans la closure de construction du flux. `VoxtralTTSPipeline.synthesizeStreaming` : même
  schéma (hunk MLX-003), `checkCancellation()` en tête de la boucle de chunks, état `.ready` rendu à la sortie.
  Compteur interne `streamingFramesProduced` (tests) ; `ttsModel`, `tokenizer`, `voiceEmbeddings` en `private(set)`.
- Porte observée :
  - `GREEN TTSStreamingCancellationTests : Executed 4 tests, with 0 failures` (gardé `VOXTRAL_TTS_STREAM=1`) ; rouge
    sans correctif (3 tests, 6 échecs) : `generateStreaming returned in 676983.9 ms`, annulation : `.ready` non atteint
    en 30 s (`synthesizing`), `frames=1934 > 1607`, puis `busy` sur le test suivant.
  - `STREAM generateStreaming returned in 0.2 ms (< 50) ; first chunk 624.5 ms ≤ 1,5 × ttft batch 527 ms` (Release,
    `bench tts`, machine-check OK, balises : nos pids seulement ; A/A batch 0,03 %).
  - `CANCEL after 5 chunks → .ready in 8 ms (< 1000) ; frames=43 ≤ frames_at_cancel+1 (44)`
  - `PARITY stream concat == batch (seed 42)` au sens décidé par Vincent le 2026-10-01 : codes identiques bit à bit
    (`[1, 2280, 37]`), audio max|Δ| = 1.2423843e-06 ≤ 1e-5 (4 377 600 échantillons ; bruit flottant du re-décodage
    codec par chunk : l'identité bit à bit de l'audio relève de K-43).
  - Suite `Executed 533 tests, with 24 tests skipped and 0 failures (0 unexpected)`. Catalogue : MLX-003 2 → 0.
- Écarts : texte ≈ 350 mots = « Long EN » de `docs/tts_benchmark.md` × 2 (326 mots ; la référence est condensée,
  K-33). Le test Debug borne le premier chunk à 2 × ttft (garde-fou) : la clause 1,5 × est mesurée en Release (un run
  Debug a donné 579 ms contre 571 = 1,5 × 381). Streaming : 494,7 s pour 182,4 s d'audio (re-décodage O(n²), K-43).

## K-26 — Enrôlement reproductible : graine, point de contrôle et reprise, tests de la garde NaN — 2026-10-01 — validée
- Fait : `VoxtralVoiceEnrollment.Config` : `seed`, `checkpointURL`, `checkpointEvery` (additif). Paramètres initiaux tirés
  de la graine ; bruit de Gumbel de chaque époque tiré d'une graine dérivée (SplitMix64(graine, époque)) : la reprise
  n'a pas à restaurer d'état RNG (privé dans mlx-swift). Point de contrôle `.safetensors` (paramètres, moments d'Adam,
  température, époque, meilleur instantané, perte et graine ; empreinte de configuration vérifiée) écrit tous les
  `checkpointEvery` pas et à l'annulation, par remplacement atomique, supprimé en fin de run. Surcharge non levante
  `optimize(reference:progress:)` dépréciée. CLI `enroll` : `--seed`, `--checkpoint`, `--checkpoint-every`,
  `--stop-after`. La démo avait déjà son bouton Annuler (K-11).
- Cause trouvée (hors fichiers listés, nécessaire à la porte) : à graine égale, les codes divergeaient dès l'époque 1.
  Le découpage STFT par `MLX.take` d'indices qui se chevauchent a pour gradient un scatter-add GPU aux additions
  atomiques d'ordre variable. `VoxtralEnrollmentLosses` découpe maintenant en blocs du pas (reshape, tranches,
  concaténation) : mêmes valeurs (test d'égalité sur les 9 tailles), gradient identique d'un appel à l'autre.
- Porte observée :
  - `GREEN EnrollmentReproTests : Executed 5 tests, with 0 failures` (gardé `VOXTRAL_ENROLL_REPRO=1`) ; rouge avant le
    découpage en blocs : `seed 7 ×2 identical=false`, `RESUME … identical=false` ; garde et annulation neutralisées :
    `Executed 3 tests, with 3 failures` (NaN époque 0 sans erreur, NaN époque 5 ≠ meilleur pas, annulation sans erreur).
  - `SEED cmp a.safetensors b.safetensors : identiques ; graine ≠ : différents` (tts-4b-6bit, clone_fr 8 s, 200 époques).
  - `RESUME 2500/5000 : identique bit à bit ; surcoût -1,3 % (≤ 5 %)` (5 000 époques : sans point de contrôle 460,6 s,
    avec 454,4 s, sorties identiques ; arrêt à 2 500 puis reprise = run continu, `cmp`).
  - `VoxtralEnrollmentLossesTests` 6/6 (valeurs PyTorch inchangées) ; suite `Executed 540 tests, with 29 tests skipped
    and 0 failures (0 unexpected)`.
- Écarts : `clone_fr.wav` dure 8,6 s : `--duration 8` (le défaut 16 s refuse la référence) ; machine-check OK, balises :
  nos pids seulement.

## K-32b — Instrument bench : phases sans double comptage, pad_fraction, --trace, blocage Core ML — 2026-10-01 — validée
- Fait (commit `1a38ff86`) : `phases_ms` en temps exclusif (une phase qui en contient d'autres ne compte que son temps
  propre : la phase pipeline STT « Generation » entourait le préfill et le décodage LLM, comptés deux fois) ;
  `pad_fraction` dans `bench realtime` (part des pas de décodage dont le jeton ne porte pas de texte : jetons de
  contrôle `[STREAMING_PAD]`, `[STREAMING_WORD]`… ; nouvelle propriété additive `lastPadFraction`), schéma mis à jour ;
  `--trace` réel : une passe de diagnostic supplémentaire (pas de ligne BENCH) qui écrit une trace Chrome dans `--out`.
- Blocage `.auto` : pas propre à C-long. Core ML en `.cpuAndGPU` (MPSGraph) se bloque dès la 1re fenêtre (aussi sur
  C-moyen) dans `-[AGXG15XFamilyCommandQueue commandBuffer]` → `semaphore_wait_trap`, 0 % CPU, dans le même processus
  que MLX (`sample`, `VoxtralCoreMLEncoder.swift:375`). Un `autoreleasepool` par prédiction n'y change rien. En
  `.cpuAndNeuralEngine` tout passe : décision de Vincent (2026-10-01) : préréglages `default`/`mini`/`small` sur l'ANE
  (`gpuOnly` reste GPU, documenté), CHANGELOG. `.auto` est le défaut de `VoxtralPipeline` : le blocage touchait tout
  consommateur ayant l'encodeur Core ML (FluxForge à prévenir, #557).
- Porte observée (machine-check sans KO recopié dans le rapport ; balises : nos pids seulement ; toutes les lignes
  `dirty:false`, commit `1a38ff86a`) :
  - stt (A/A K-32) : somme des phases 15 897,0 ≤ total 15 898,2 ms ; `AA dispersion total_ms=0.21% step_ms_p50=0.37%
    out_sha256=identical → PASS` ; chat : 1 336,3 ≤ 1 337,3 ms, `AA q1 dispersion ttft_ms=0.64% tok_s=0.17% … PASS`.
  - realtime : `pad_fraction` 0.7226 (C-moyen EN), `AA dispersion total_ms=0.15% step_ms_p50=0.00% … PASS`.
  - `TRACE …/.local-runs/bench.noindex/k32b/trace-realtime-2026-10-01T111033Z.json`.
  - `bench stt --backend auto` C-long : 98 352,8 ms contre `.mlx` 52 414,5 ms = 1,88 × (< 2 ×). Réserve : encode
    identique au diagnostic (14,2 s) mais préfill et décodage doublés pendant cette passe (`top_process` :
    managedappdistri 46 %) ; le diagnostic sur le même binaire avant commit donnait 59,1 s (1,14 ×). Textes C-long
    `.auto` ≠ `.mlx` (5 549 / 5 160 car., C-long bilingue, K-33) ; C-moyen : `.auto` (ANE) = `.mlx` (même sha).
  - Suite `Executed 540 tests, with 29 tests skipped and 0 failures (0 unexpected)`.

## K-3 — Masques d'attention construits par le cache (décodeur STT vivant et décodeur hérité) — 2026-10-01 — validée
- Fait (commit `988c256b`) : décodeur vivant (`LlamaStandardModel`) : masque booléen de `cache.makeMask(n:windowSize:
  returnArray: true)` (forme des clés que le cache présente : offset, fenêtre tournante), sinon
  `MLXLMCommon.createCausalMask(n:offset:)`, `nil` pour un jeton ; décodeur hérité (`LlamaModel`) : plus de masque
  `[T, T]` fait main, `nil` ⇒ `.causal` dans `LlamaAttention` à tout offset. Amendement : le test (b) de
  `MLXErrorBoundaryTests` a un déclencheur durable (poids de forme fausse injecté par `update(parameters:)` sans
  vérification). Point d'accroche de test interne `prefillLogitsObserver`.
- Porte observée :
  - `GREEN AttentionMaskTests : Executed 4 tests, with 0 failures (0 unexpected)` ; rouge avec les anciens masques :
    préfill bf16 → `mlx("[scaled_dot_product_attention] …")`, décodeur hérité 600 positions → `Fatal error: Index out
    of range` (runner arrêté) ; le test du `RotatingKVCache` enroulé (`[[6, 6, 6], [6, 12, 12]]`, masque `.bool`)
    passait déjà avec l'ancien masque (même largeur offset + T).
  - `PARITY greedy 3/3 identiques ; logits L2 rel max=0.0` (C-court EN/FR, C-moyen EN, mini-3b-8bit `.mlx`).
  - `LEGACY prompt 600 positions … : OK (pas d'arrêt)`. Écart : via un décodeur hérité réduit construit comme le fait
    `loadVoxtralModel(modelPath:dtype:lazy:)` (`VoxtralForConditionalGeneration(config:)`) ; ce chargeur ne charge pas
    le seul dossier bf16 présent (`mistralai/Voxtral-Mini-3B-2507` : `keyNotFound audio_tower.conv2.weight` après son
    `sanitize`), défaut séparé du chargeur hérité (K-30).
  - `TIME C-moyen EN A=15.937 B=15.949 B=15.948 A=15.941 s → écart +0,07 % (≤ 5 %)` (`bench stt`, sorties identiques,
    `dirty:false` ; machine-check sans KO, balises : nos pids seulement).
  - Suite `Executed 545 tests, with 30 tests skipped and 0 failures (0 unexpected)` ; MLX-019 : 2 → 1 (reste
    `createCausalMask(N:…)` public, inchangé, K-30).
