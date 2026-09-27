# Audit « Stabilité & nettoyage » — mlx-voxtral-swift

> **Vérification croisée : 29 constats relus, 12 gardés, 0 écartés, 17 amendés** (relecture adverse du code à
> `9392ed1`, de l'amont et des listings HF, 2026-09-27). Amendés : S-02, S-04, S-06, S-07, S-08, S-09, S-14, S-15,
> S-16, S-19, S-20, S-21, S-22, S-25, S-27, S-28, S-29. Sous-constats retirés : annexe finale.

> Skill `mlx-swift-audit`, phase 2 (constats `S-xx`). Révision auditée : `9392ed1` (= tag `v2.2.2`,
> branche `claude/action-plan-skills-beta-wifgmu`). Date : 2026-09-27.
> Périmètre : tout le dépôt — `VoxtralCore`, `VoxtralApp`, CLI (`VoxtralTranscriptionTest`), `VoxtralBenchmark`,
> `VoxtralTTSStreamingDemo`, `Tests/`, `Examples/`, `Scripts/`, racine, docs.
> Entrées de phase 1 : `scan.md` (scan.py), `patterns-scan.md` (apply.py scan).

## 0. Cadre et méthode

- **Environnement** : session cloud Linux, pas de Mac, pas de toolchain Swift, pas de GPU. Aucun build, aucun
  test, aucune mesure. Chaque constat est **vérifié en lisant le code** à `9392ed1` (fichier:ligne cités). Un effet
  qui ne se prouve qu'à l'exécution (plantage, qualité, mémoire) est marqué **À MESURER** ; les fiches qui demandent
  build, tests ou mesure ciblent `macos-gpu`.
- **Dépôt sans `CLAUDE.md` ni `AGENTS.md`** : aucune contrainte locale de build ou de test à respecter ; pas de
  script de tests ni de CI (`.github/` absent).
- **Amont lu pour prouver la redondance** : mlx-swift-lm `main@ee673d6` (2026-09-22), mlx-swift `@9019419` (0.31.6).
- **Consommateurs de l'API publique** : recherche de code GitHub (lecture seule) `user:VincentGourbin NOT
  repo:VincentGourbin/mlx-voxtral-swift`. `VoxtralCore` est importé par `fluxforge-studio-swift` (FluxForge Studio,
  App Store, dont la chaîne LipDub) et `SongAnalysisDb`. **0 résultat** pour `VoxtralGenerator`,
  `loadVoxtralModel`, `VoxtralForConditionalGeneration`, `TekkenTokenizer`, `ChatTemplateProcessor`,
  `synthesizeStreaming`, `VoxtralDebug`. Symboles consommés : `VoxtralPipeline(.mini3b4bit)`,
  `VoxtralCore.ModelRegistry`, `ModelDownloader.customModelsDirectory`, `RuntimeBeacon.isEnabled`,
  `VoxtralTTSPipeline` (dont `recommendedWarmUpVocalise`). Limite : la recherche GitHub n'indexe que les branches
  par défaut ; toute suppression publique reste « cassant » et passe par ASK.
- **Données HF** (lecture seule, 2026-09-27) : listings de dépôts et `config.json`/`tekken.json` de
  `mistralai/Voxtral-Mini-3B-2507`, pour prouver S-01, S-02 et S-07.

## 1. Synthèse

**29 constats** : **6 hauts**, 16 moyens, 7 bas (après vérification croisée : S-07 et S-08 passent en moyenne,
S-19 et S-25 en basse ; la répartition initiale annoncée, « 8 / 14 / 7 », ne correspondait déjà pas au tableau,
qui donnait 8 / 16 / 5). Les hauts sont des **défauts silencieux ou fatals** : sortie tronquée ou fausse sans
erreur, arrêt du processus hôte sur audio long, ou téléchargement doublé. Aucun ne se voit sur un test court et propre.

| Id | Sév. | Constat (une ligne) | Statut |
|---|---|---|---|
| S-01 | haute | Le jeton d'arrêt codé en dur `32000` correspond au mot « ␣Capital » en Tekken : la transcription s'arrête sur ce mot | VÉRIFIÉ |
| S-02 | haute | Le `RotatingKVCache` par défaut (2 048 à 8 192 jetons selon la RAM ; 8 192 imposé par l'app) sur un LM sans fenêtre glissante : invite > fenêtre → **arrêt du processus** au préfill (masque ≠ clés, cf. P-03) ; invite qui tient mais invite + sortie qui déborde → début de l'audio évincé sans erreur | VÉRIFIÉ en lecture (arrêt et perte À MESURER) |
| S-03 | haute | Complétude du téléchargement déduite d'un seul fichier (`config.json`/`params.json`) ; `verifyShardedModel` dit « complet » sans index (famille MLX-012) | VÉRIFIÉ |
| S-04 | haute | Les 3 chargeurs vivants (STT, TTS, Realtime) appliquent les poids sans vérification (`verify: .none`) : un poids manquant reste aléatoire, sans erreur | VÉRIFIÉ |
| S-05 | haute | `TekkenTokenizer` bascule en silence sur un tokenizer « démo » octet par octet si `tekken.json` manque ou ne se décode pas | VÉRIFIÉ |
| S-06 | haute | `small-24b-8bit` : l'enum pointe vers `mzbac/…`, le registre, le README et l'app vers `VincentGOURBIN/…` : double téléchargement (28,06 + 26,50 Go) et chargement impossible hors ligne | VÉRIFIÉ |
| S-07 | moyenne | Le glob `*.safetensors` télécharge aussi `consolidated.safetensors` des dépôts Mistral : ×2 (mini-3b 18,7 Go au lieu de 9,36 ; small-24b 97,0 Go au lieu de 48,5) | VÉRIFIÉ |
| S-08 | moyenne | Le « streaming » TTS ne streame pas : la closure de construction de `AsyncThrowingStream` génère tout de façon synchrone ; en plus, MLX-003 (pas d'`onTermination`) | VÉRIFIÉ |
| S-09 | moyenne | Aucune annulation coopérative (STT, TTS batch, Realtime) ; calcul synchrone dans des `async` sur le pool coopératif | VÉRIFIÉ |
| S-10 | moyenne | États des pipelines `@unchecked Sendable` sans verrou (course, état incohérent après `unload` pendant un stream, enrôlement non exclusif) | VÉRIFIÉ (course À MESURER) |
| S-11 | moyenne | État global mutable non synchronisé (`_melFiltersCache`, `VoxtralMemoryManager.config` écrasé par chaque pipeline, `_hubApi`, `writeDebugToDump`…) | VÉRIFIÉ |
| S-12 | moyenne | Des `MLXArray` paresseux traversent l'isolation dans des types `@unchecked Sendable` (MLX-004, non détecté par apply.py) | VÉRIFIÉ (plantage À MESURER) |
| S-13 | moyenne | Code mort vérifié (0 appel) : ≈ 2 400 lignes, dont ≈ 900 sans aucun risque d'API | VÉRIFIÉ |
| S-14 | moyenne | Famille de chargement STT « legacy » publique : écrit dans `/tmp` à chaque message, souche `downloadModel` qui ne télécharge rien, deux `loadVoxtralModel` homonymes | VÉRIFIÉ |
| S-15 | moyenne | Dépendances inutiles dans `VoxtralCore` : `MLXLLM`, `MLXOptimizers`, `ArgumentParser` ; `Transformers` déclaré mais seul `Hub` est importé | VÉRIFIÉ (build À MESURER) |
| S-16 | moyenne | Triplication interne (3 Llama, 3 configurations, chargeur et quantification maison ; 8 `fatalError` de dispatch dus aux champs typés `Module`) redondante avec l'amont `MLXLMCommon.loadWeights` + `PerLayerQuantization` | VÉRIFIÉ |
| S-17 | moyenne | Conformance `LanguageModel` inutilisée, fausse (`prepare` passe des embeddings comme ids) et cause de la casse #50 | VÉRIFIÉ |
| S-18 | moyenne | `mlx-swift-lm` sur `branch: "main"`, sans `Package.resolved` suivi : builds non reproductibles (casse #50 avérée) | VÉRIFIÉ |
| S-19 | basse | Plateformes et toolchain incohérentes : tools 6.2 / macOS 15 contre README « Swift 6.0 / Xcode 15 / macOS 14 » | VÉRIFIÉ |
| S-20 | moyenne | Docs désynchronisées : llms.txt en 1.0.10 (tag v2.2.2), sans TTS/Realtime, affirmations fausses ; 3 numéros de version différents | VÉRIFIÉ |
| S-21 | moyenne | Surface publique énorme (≈ 1 100 déclarations, 45 fonctions libres dont 39 noms distincts) aux noms génériques, déjà en collision chez FluxForge | VÉRIFIÉ |
| S-22 | moyenne | API publiques trompeuses : paramètre ignoré, souches, métriques toujours à 0 | VÉRIFIÉ |
| S-23 | basse | 155 `print()` dans la bibliothèque (dont chemins vivants) ; `VoxtralDebug` existe mais n'est pas utilisé partout | VÉRIFIÉ |
| S-24 | basse | Chemins développeur `/Users/vincent/…` codés en dur ; recherche de modèles dans le CWD | VÉRIFIÉ |
| S-25 | basse | 24 fichiers suivis malgré `.gitignore` : 22 WAV (70,2 Mio, dont 8 non référencés = 18,3 Mo) + cache `.serena/*.pkl` | VÉRIFIÉ |
| S-26 | basse | Racine et cibles annexes : `create_app_bundle.sh` orphelin et incomplet, ressource `.mlmodelc` ignorée par git mais déclarée, bench mal nommé | VÉRIFIÉ (build À MESURER) |
| S-27 | moyenne | Tests manquants, tautologiques ou vides de sens hors machine du mainteneur ; pas de CI | VÉRIFIÉ |
| S-28 | basse | Erreurs secondaires avalées (`try?`), `as!`, `precondition` atteignables par l'API publique | VÉRIFIÉ |
| S-29 | basse | TODO obsolètes et souches « For now / would integrate » | VÉRIFIÉ |

**Ordre de traitement recommandé** (skill, ordre imposé) :
1. Stabilité bloquante : S-01, S-02 (fiche commune avec P-03 de l'audit perf STT), S-03 + S-04 + S-05 (un lot
   « sortie silencieusement fausse »), S-06.
2. Hygiène sans risque : S-07, S-25, S-13 (privés et internes), S-24, S-23, S-29, S-20 et S-19 (docs).
3. Concurrence et annulation (S-08, S-09, S-10, S-11, S-12), puis l'API (S-14, S-17, S-21, S-22 → ASK) et la
   dette structurelle (S-15, S-16, S-18).

## 2. Faits vérifiés (ne pas re-dériver)

| Fait | Preuve |
|---|---|
| `9392ed1` = tag `v2.2.2` ; 0 issue ouverte, 0 PR ouverte | GitHub (list_tags, list_issues, list_pull_requests), 2026-09-27 |
| Tekken : id = rank + 1 000 (spéciaux) ; **id 32000 = rank 31000 = « ␣Capital »** (`token_bytes: IENhcGl0YWw=`) | `VoxtralComponents.swift:155-176` ; `hf://models/mistralai/Voxtral-Mini-3B-2507/tekken.json` (≈ octet 2 925 780) |
| LM de Voxtral Mini 3B : `sliding_window: null`, `max_position_embeddings: 131072`, `torch_dtype: bfloat16` | `config.json` HF du dépôt `mistralai/Voxtral-Mini-3B-2507` |
| 375 jetons audio par fenêtre de 30 s (12,5 jetons/s) | `VoxtralProcessor.swift:389-390`, `VoxtralModeling.swift:720-739` |
| `Module.update(parameters:)` (non-throwing) = `try! update(parameters:, verify: .none)` | mlx-swift `Source/MLXNN/Module.swift:401-408` |
| Options de vérification disponibles : `.noUnusedKeys`, `.allModelKeysSet`, `.shapeMismatch`, `.all` | mlx-swift `Module.swift:383-399` |
| L'amont charge avec `verify: [.all]`, quantifie si `weights["\(path).scales"] != nil`, charge les shards en concurrence et expose une variante async hors pool coopératif | mlx-swift-lm `MLXLMCommon/Load.swift:368-405`, `:409-438`, `:9-18` |
| `GPU.resetPeakMemory()` n'est **pas** déprécié en 0.31.6 | mlx-swift `GPU+Metal.swift:217-222` (seuls l. 23-190 sont `deprecated`) |
| Les caches KV utilisés sont ceux de l'amont (`KVCacheSimple`, `RotatingKVCache`) ; pas de cache KV maison | `VoxtralModeling.swift:1139,1145` ; `MLXLMBridge.swift:28-37` (implémentation maison archivée) |
| L'encodeur Core ML respecte `customModelsDirectory` depuis `1570294` (#49) | `VoxtralCoreMLEncoder.swift:453-481` |
| 24 fichiers suivis bien qu'ignorés | `git ls-files -ci --exclude-standard` ; `git check-ignore --no-index -v` → `.gitignore:54` (`*.wav`), `:119` (`.serena/`) |
| Tailles HF : mini-3b consolidated 9 348 806 528 o + shards 9 356 474 312 o ; small-24b consolidated 48 519 877 672 o + shards 48 527 546 144 o ; `mzbac/Voxtral-Small-24B-2507-8bit` 28 056 927 031 o ; `VincentGOURBIN/voxtral-small-8bit` 26 499 134 369 o | listings `hf_fs ls` du 2026-09-27 |
| `VoxtralCore` : 20 489 lignes ; ≈ 1 100 lignes de déclaration `public` (1 098 à 1 104 selon le motif `grep` ; 1 065 annoncé à l'origine, non reproduit) ; 487 fonctions de test (XCTest) | `scan.md` §1 ; `grep` (vérification croisée) |

## 3. Constats

Format : sévérité · `fichier:ligne` · constat · preuve · correction · risque API · effort · statut · fiche proposée.

---

### S-01 — Le jeton d'arrêt `32000` coupe la transcription sur « Capital » · **haute**

- **Où** : `Sources/VoxtralCore/VoxtralModeling.swift:1124` (`let stopTokens = [2, 4, 32000]`), `:1262` (break) ;
  même liste au chemin hybride `:1313`, `:1438`.
- **Constat** : `32000` est l'EOS de l'ancien tokenizer Llama/Mistral v1. Avec Tekken (vocabulaire de 131 072
  entrées, 1 000 ids spéciaux), l'id 32000 est un jeton de texte ordinaire : « ␣Capital ». Dès que le modèle
  l'émet, la génération s'arrête après ce mot (le jeton est ajouté à la sortie l. 1225, puis `break` l. 1262). Toute
  transcription ou réponse de chat qui contient « Capital » (précédé d'un espace) est **tronquée sans erreur**, sur
  les deux backends (MLX et hybride).
- **Preuve** : décalage id = rank + `numSpecialTokens` (`VoxtralComponents.swift:155-176`) ; entrée `rank 31000`
  de `tekken.json` = « ␣Capital » ; le Realtime, lui, lit `config.eosTokenId` (`VoxtralRealtimeModel.swift:120`).
- **Correction** : dériver les jetons d'arrêt du tokenizer et de `generation_config.json`
  (`tokenizer.eosToken`, déjà chargé l. 349-367), soit `[2]`, `[/INST]` (4) si voulu. Supprimer 32000. Ajouter
  un test unitaire : « aucun jeton d'arrêt ≥ numSpecialTokens ».
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ (mécanisme) ; effet sur un vrai audio À MESURER.
- **Fiche K-S01** — *Jetons d'arrêt depuis le tokenizer*. **Porte** : test unitaire vert (stop ⊂ ids spéciaux) ;
  la transcription d'un audio qui dit « Capital Gains and Capital One » dépasse le premier « Capital » (texte
  complet, 1/1) ; parité greedy sur 3 audios de référence sans ce mot (identique). Cible `macos-gpu`.

### S-02 — Le `RotatingKVCache` par défaut : arrêt au préfill au-delà de la fenêtre, perte du début en deçà · **haute**

- **Où** : `VoxtralModeling.swift:1119-1147` (cache rotatif `maxSize: maxContext, keep: 4` dès que
  `maxKVCacheSize` est défini ; même logique au chemin hybride `:1308-1311`) ; préfill tranché par 512
  `:1163-1188` (et `:1349-1368`), `attentionMask: nil` ; masque maison `[T, offset + T]`
  (`Utils/VoxtralStandardLoader.swift:450-481`, `offset = cache.first.offset`) passé tel quel au SDPA (`:692-698`) ;
  `Configuration/MemoryOptimizationConfig.swift:41-69` (8 192 / 6 144 / 4 096 / 2 048) et `:77-90`
  (`recommended()`, RAM en Gio entiers : 0-15 → `ultra` 2 048 ; 16-31 → 4 096 ; 32-63 → 6 144 ; ≥ 64 → 8 192) ;
  `Pipeline/VoxtralPipeline.swift:107-115`, `:118-122` (défaut `.recommended()`), `:360`, `:373` ; l'app impose
  8 192 quel que soit la RAM (`Sources/VoxtralApp/TranscriptionManager.swift:70`, `:212`).
- **Constat** : le LM de Voxtral n'a **pas** de fenêtre glissante (`sliding_window: null`, 131 072 positions). Deux
  régimes (vérification croisée, qui corrige « perdu sans erreur ») :
  - **invite > fenêtre** : dès qu'une tranche de T ≥ 2 démarre à `offset ≥ maxSize`, `RotatingKVCache.updateConcat`
    rogne l'avant (`trimSize = idx − maxSize + 1`) et renvoie `maxSize − 1 + T` clés, alors que le masque maison
    en suppose `offset + T` : `broadcast_to` du masque échoue (mlx `fast.cpp:906-909` à `1f8e74e`), l'erreur part
    au gestionnaire par défaut de mlx-swift (`fatalError`), aucun `withError` dans Voxtral → **arrêt du processus
    hôte** (FluxForge, app). Seuils (invite = 375 × N fenêtres de 30 s + quelques jetons,
    `VoxtralProcessor.swift:388-390`) : audio de plus de 2 min 30 s pour 2 048 (Mac 8 Go), de plus de 5 min pour
    4 096 (**Mac 16 Go**), de plus de 8 min pour 6 144, de plus de 10 min 30 s pour 8 192 (≥ 64 Go et app). Même constat que **P-03** de l'audit perf STT,
    qui le simule exactement.
  - **invite ≤ fenêtre < invite + sortie** : au décodage (T = 1, pas de masque) la rotation évince le début de
    l'audio (hors 4 jetons `keep`) **sans erreur**.
  Le seuil « ≈ 2 min sur 16 Go » de la version initiale était faux : un Mac 16 Go tombe dans la tranche 16-31
  (4 096). Voxtral accepte jusqu'à ≈ 30 min. FluxForge utilise le défaut (`VoxtralPipeline(.mini3b4bit)`). À noter :
  `maxTokens = 500` par défaut tronque déjà la sortie au-delà de ≈ 3 min de parole (P-11).
- **Preuve** : lecture ci-dessus ; mlx-swift-lm `KVCache.swift:691-714` (`updateConcat`), `:716-760`
  (`updateInPlace`) ; simulation de la logique amont (scratchpad, `sim_rotating.py`, et annexe A de l'audit perf STT).
- **Correction** : `KVCacheSimple` par défaut (`maxKVCacheSize = nil` dans tous les préréglages STT et dans l'app) ;
  plafonner la mémoire par `Memory.cacheLimit`, bf16 et le préfill tranché (audit perf, P-01, P-09, P-10), pas par une
  fenêtre ; masque amont (P-02) pour les caches à fenêtre restants ; si un plafond est voulu, **lever une erreur
  Swift** quand `invite + maxTokens > maxKVCacheSize`. Une seule fiche avec P-03.
- **Risque API** : aucun (valeurs par défaut) ; pic mémoire à re-mesurer. **Effort** : S (défaut) / M (garde-fou).
  **Statut** : VÉRIFIÉ en lecture ; arrêt et perte de qualité À MESURER.
- **Fiche K-S02 (= fiche P-03)** — *Cache KV sans fenêtre par défaut en STT*. **Porte** : test de non-régression
  (invite synthétique de 2 600 jetons, préréglage `.ultra`) : arrêt avant, passe après ; audio de 12 min en profil
  16 Go avec `maxTokens` ≥ 4 096 : **0 arrêt**, première et dernière minutes présentes dans la transcription ;
  pic `phys_footprint` mesuré et consigné (décision ASK si > 12 Go). Cible `macos-gpu`.

### S-03 — « Téléchargé » déduit d'un seul fichier ; index absent = « complet » (famille MLX-012) · **haute**

- **Où** : `Utils/ModelDownloader.swift:206-210` (`isModelDownloaded` = présence de `config.json`) ;
  `:312-321` (`verifyShardedModel` renvoie `(true, [])` sans `model.safetensors.index.json` **ou** si l'index est
  illisible) ; `:537-581` (TTS : `params.json` seul, jamais `verifyShardedModel`) ; `:644-692` (Realtime :
  `config.json`/`params.json` seul) ; retour anticipé sans reprise `:589-592`, `:699-702` ; `downloadRepoDirect`
  sans marqueur de fin `:79-158`.
- **Constat** : l'arbre HF est listé puis téléchargé fichier par fichier. Pour `mzbac/voxtral-mini-3b-8bit`, le
  listing donne `config.json`, `generation_config.json`, `model-00001…`, `model-00002…`,
  `model.safetensors.index.json`, `params.json`, `preprocessor_config.json`, `tekken.json`. Si le téléchargement est
  coupé pendant le 2ᵉ shard, `config.json` est là et l'index pas encore : `findModelPath` renvoie le dossier comme
  complet, la reprise ne se fait jamais et le chargement part avec un seul shard (voir S-04) et sans `tekken.json`
  (voir S-05). Pour le TTS, une coupure après `params.json` (`tekken.json`, `voice_embedding/*`) passe aussi pour
  complète. `apply.py` n'a rien détecté (sa regex ne cherche que `contains { $0.hasSuffix(".safetensors") }`).
- **Correction** : marqueur `.complete` écrit en fin de `downloadRepoDirect` (après le dernier fichier, avec la
  liste attendue et les tailles) ; `find*ModelPath` exigent le marqueur ou tous les fichiers de l'arbre attendu
  (index compris) ; index illisible = incomplet ; même logique pour TTS et Realtime.
- **Risque API** : aucun (comportement). **Effort** : M. **Statut** : VÉRIFIÉ en lecture ; l'ordre exact de
  l'API `tree` HF (utilisée l. 91) est À MESURER, mais le défaut ne dépend pas de l'ordre pour l'index illisible
  ni pour le TTS.
- **Fiche K-S03** — *Complétude prouvée par marqueur ou index*. **Porte** : 3 tests unitaires (1 shard sur 2 →
  non téléchargé, reprise effective ; index corrompu → non téléchargé ; TTS sans `tekken.json` → non téléchargé) ;
  coupure réseau simulée au milieu d'un shard puis relance → modèle complet, 1/1. Cible `macos-gpu`.

### S-04 — Les trois chargeurs vivants appliquent les poids sans vérification · **haute**

- **Où** : STT `Utils/VoxtralStandardLoader.swift:1323`, `:1338` (`update(parameters:)` = `try! … verify: .none`) ;
  TTS `TTS/VoxtralTTSModelLoading.swift:71` (`verify: .none`) et `:120-131` (le shard 2 est facultatif) ; Realtime
  `Realtime/VoxtralRealtimeModelLoading.swift:55` (`verify: .none`). Paramètre `dtype` ignoré :
  `VoxtralStandardLoader.swift:1285-1343`.
- **Constat** : une clé manquante (shard absent, renommage de clés, modèle d'une autre révision) laisse les poids
  d'initialisation aléatoire : transcription ou voix inintelligible **sans erreur**. Une **forme** incompatible n'est
  pas vérifiée non plus (`.shapeMismatch` absent de `.none`, mlx-swift `Module.swift:475-480`) : le tableau est
  remplacé tel quel et l'erreur surgit plus loin, dans un matmul ; seule une **structure** incompatible
  (`UpdateError.incompatibleItems`) fait planter par le `try!` caché de la surcharge non-throwing (vérification
  croisée : la version initiale attribuait aussi la forme au `try!`). Ironie : le chargeur legacy
  (`VoxtralModelLoading.swift:282`) vérifiait `[.all]`.
- **Preuve** : mlx-swift `Module.swift:401-408`. `.noUnusedKeys` est probablement évité parce que `sanitize` ajoute
  la clé dupliquée `embedTokens.weight` (`VoxtralModelLoading.swift:493-497`).
- **Correction** : `try model.update(parameters:, verify: [.allModelKeysSet, .shapeMismatch])` dans les 3 chargeurs
  (sans `.noUnusedKeys`, ou après avoir retiré les clés en trop) ; erreur typée qui liste les clés manquantes ; TTS :
  lire l'index plutôt que deux noms de shard codés en dur ; supprimer le paramètre `dtype` ou l'appliquer.
- **Risque API** : aucun (erreur au lieu d'une sortie fausse). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S04** — *Chargement vérifié*. **Porte** : les 6 modèles STT, 3 TTS et 1 Realtime se chargent sans
  erreur (0 clé manquante) ; un dossier privé d'un shard lève une erreur qui nomme ≥ 1 clé ; parité greedy et
  audio identiques (graine fixée) avant/après. Cible `macos-gpu`.

### S-05 — Repli silencieux de `TekkenTokenizer` sur un tokenizer « démo » · **haute**

- **Où** : `VoxtralComponents.swift:110-121` (`init` non throwing), `:141-144` (fichier absent → démo),
  `:201-204` (JSON invalide → démo), `:369-394` (vocabulaire de 256 octets). Log seulement via
  `VoxtralDebug.log` (désactivé par défaut, `Utils/VoxtralDebug.swift:11`). Utilisé par le STT
  (`VoxtralProcessor.swift:445`, `fromPretrained` déclaré `throws` mais qui ne lève jamais), le TTS
  (`TTS/Pipeline/VoxtralTTSPipeline.swift:160`) et le Realtime (`Realtime/Pipeline/VoxtralRealtimePipeline.swift:104`).
- **Constat** : `tekken.json` absent (voir S-03) ou corrompu → encodage octet par octet, ids faux → TTS inintelligible
  et STT au décodage absurde, pipeline dans l'état `.ready`. Même famille : l'échec de compilation de la regex
  bascule sans bruit sur un découpage par espaces (`:153`, `:423-426`).
- **Correction** : `static func load(modelPath:) throws -> TekkenTokenizer` qui lève `fileNotFound` ou
  `invalidTokenizer` ; les pipelines l'utilisent ; tokenizer démo seulement par une fabrique explicite de test ;
  regex invalide = erreur.
- **Risque API** : additif (nouvelle fabrique ; l'`init` public peut rester, déprécié). Consommateurs : 0 hit.
  **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S05** — *Tokenizer : erreur au lieu du repli*. **Porte** : `loadModel` lève une erreur typée sur un
  dossier sans `tekken.json` (3 pipelines, 3 tests) ; tokens identiques sur 20 phrases FR/EN avant/après avec le vrai
  fichier. Cible `macos-gpu`.

### S-06 — `small-24b-8bit` : deux dépôts pour le même modèle · **haute**

- **Où** : `Pipeline/VoxtralPipeline.swift:47-48` (`mzbac/Voxtral-Small-24B-2507-8bit`) contre
  `Utils/ModelRegistry.swift:90-91` (`VincentGOURBIN/voxtral-small-8bit`), `README.md:216` (`llms.txt:118` ne nomme
  aucun dépôt, seulement « ~25 GB », cohérent avec le registre).
  Résolution : `VoxtralPipeline.swift:229` passe `model.repoId` à `ModelDownloader.resolveModel`
  (`ModelDownloader.swift:401-429`), qui ne le trouve ni par id ni par repoId et part dans `downloadByRepoId`.
  L'app télécharge l'entrée du registre (`Sources/VoxtralApp/TranscriptionManager.swift:131-141`) puis charge
  l'enum (`:216-231`) ; `VoxtralTranscriptionManager.isModelDownloaded` lit le registre (`:171-176`).
- **Constat** : deux dépôts réels et différents (6 shards 28,06 Go contre 5 shards 26,50 Go). L'app télécharge
  ≈ 26,5 Go puis la pipeline en télécharge ≈ 28 Go de plus. Chaque chargement relance une requête `tree` HTTP
  (l. 91-99) : **pas de chargement hors ligne**. `recommendedModel` renvoie justement `.small24b8bit` à partir de
  64 Go (`VoxtralPipeline.swift:565-566`).
- **Correction** : une seule source de vérité — `Model.repoId` lu dans `ModelRegistry.model(withId: rawValue)` ;
  `loadModel` résout par **id** ; test unitaire : chaque `VoxtralPipeline.Model` existe dans le registre avec le même
  repoId. Choisir le dépôt retenu (ASK : garder `VincentGOURBIN/voxtral-small-8bit`, cohérent avec le README).
- **Risque API** : aucun (la valeur de `repoId` change pour un cas). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S06** — *Source unique des repoId*. **Porte** : test « enum ⊂ registre » vert ; téléchargement dans
  l'app puis `loadModel` en mode avion → chargé (0 octet réseau) ; un seul dossier `small-8bit` sur disque.
  Cible `macos-gpu`.

### S-07 — `consolidated.safetensors` téléchargé en plus des shards (×2) · moyenne

> Vérification croisée : sévérité ramenée de haute à **moyenne**. Faits exacts (listings HF recalculés à l'octet),
> mais seuls les deux dépôts Mistral non quantifiés (non recommandés) sont touchés, sans sortie fausse ni plantage :
> coût disque et réseau (risque de disque plein sur `small-24b`).

- **Où** : `Utils/ModelDownloader.swift:361-365` et `:388-392` (`matching: ["*.json", "*.safetensors"]`) ; le
  chargeur l'ignore ensuite (`VoxtralStandardLoader.swift:997-1000`, `VoxtralModelLoading.swift:105-108`).
  L'intention d'exclure figurait dans la souche morte `VoxtralModelLoading.swift:27-32`
  (`ignore_patterns=["consolidated.safetensors", …]`).
- **Constat** : pour `mini-3b`, 18,71 Go téléchargés au lieu de 9,36 Go ; pour `small-24b`, 97,05 Go au lieu de
  48,53 Go. Le README annonce « ~6 GB » et « ~48 GB » (`README.md:207`, `:215`).
- **Correction** : paramètre `excluding:` dans `downloadRepoDirect` (défaut `["consolidated*.safetensors"]` pour le
  STT) ; **pas** pour le TTS officiel `mistralai/Voxtral-4B-TTS-2603`, qui n'a que `consolidated.safetensors`.
  Mettre à jour les tailles du README.
- **Risque API** : additif. **Effort** : S. **Statut** : VÉRIFIÉ (listings HF + glob).
- **Fiche K-S07** — *Exclure consolidated en STT*. **Porte** : `mini-3b` téléchargé = 9,37 Go ± 1 % (contre 18,7) ;
  test `matchesGlob`/exclusion vert ; TTS `tts-4b` inchangé (consolidated toujours présent). Cible `macos-gpu`.

### S-08 — Le « streaming » TTS ne streame pas et ne s'annule pas · moyenne

> Vérification croisée : sévérité ramenée de haute à **moyenne** (défaut réel et vérifié, mais seul consommateur
> = la démo interne, FluxForge n'appelle pas `synthesizeStreaming` ; ni sortie fausse ni plantage : latence et
> pipeline occupée). Nuance sur l'annulation et porte précisée ci-dessous.

- **Où** : `TTS/VoxtralTTSModeling.swift:569-685` (`AsyncThrowingStream { continuation in … }` : toute la boucle
  `for i in 0..<maxTokens` s'exécute **dans la closure de construction**, qui est synchrone et non échappante) ;
  `TTS/Pipeline/VoxtralTTSPipeline.swift:556-671` (`Task { … }` sans `continuation.onTermination` : MLX-003, seule
  occurrence détectée par apply.py), `:569` (l'appel bloque jusqu'à la fin de la génération), `:581` (la
  consommation commence ensuite), `:669` (`state = .ready` en fin de Task) ; `:506-508` (un appel pendant
  `.synthesizing` lève « Model not loaded », message trompeur).
- **Constat** : `AsyncThrowingStream.init(_:bufferingPolicy:_:)` appelle `build` immédiatement. Tout le texte est
  donc généré avant le premier `yield` consommable : le TTFT du streaming = génération complète + un décodage.
  Le `ttft` du chemin batch (`VoxtralTTSModeling.swift:485-496`, affiché par le CLI `VoxtralCLI.swift:556`) mesure
  le premier frame interne, pas l'audio disponible. En plus : `stop()` de la démo (`StreamingDemoViewModel.swift:552-558`)
  n'annule que le consommateur ; la `Task` productrice n'est jamais annulée (le `Task.isCancelled` de
  `VoxtralTTSModeling.swift:622` lit donc toujours `false`) : si l'arrêt survient pendant la génération, elle va
  jusqu'à EOA ou, au pire, `maxFrames` = 2 500 frames (200 s d'audio) ; s'il survient après le premier chunk, la
  génération est déjà finie et ce sont les décodages complets de chaque chunk restant qui continuent. Dans les deux
  cas la pipeline reste `.synthesizing` et refuse tout nouvel appel. Enfin, chaque chunk décode **toute** la séquence accumulée
  (`VoxtralTTSPipeline.swift:583`), un coût quadratique renvoyé à l'audit perf.
- **Correction** : produire dans une `Task` (ou `AsyncThrowingStream.makeStream()` + Task),
  `continuation.onTermination = { _ in task.cancel() }`, `try Task.checkCancellation()` dans la boucle, état
  remis à `.ready` sur annulation ; décodage incrémental (perf).
- **Risque API** : aucun (même signature). Consommateurs : FluxForge n'utilise pas `synthesizeStreaming` (0 hit).
  **Effort** : M. **Statut** : VÉRIFIÉ (sémantique de la stdlib + lecture).
- **Fiche K-S08** — *Streaming TTS réel et annulable*. **Porte** : texte long (≈ 350 mots, 4 bits), **sans
  warm-up** (`warmUpText: nil` ; avec warm-up le premier chunk attend 3 s d'audio accumulé,
  `VoxtralTTSPipeline.swift:595-596`, et la porte ne serait pas atteignable) : premier chunk
  ≤ 1,5 × `ttft` batch (auparavant ≈ durée de génération) ; annulation après 5 chunks → pipeline `.ready` en < 1 s ;
  audio concaténé identique au batch (graine fixée). Cible `macos-gpu`.

### S-09 — Pas d'annulation coopérative ; calcul bloquant dans des fonctions `async` · moyenne

- **Où** : `grep isCancelled|checkCancellation` → seulement `VoxtralTTSModeling.swift:622` (streaming) et le
  sondage de l'enrôlement (`VoxtralVoiceEnrollment.swift:480-490`). `VoxtralPipeline.transcribe`/`chat`
  (`Pipeline/VoxtralPipeline.swift:316-473`), `VoxtralTTSPipeline.synthesize` (`:190-400`),
  `VoxtralRealtimePipeline.transcribe` (`:118-154`) et `loadModel` (`VoxtralPipeline.swift:237-241`) sont `async`
  mais entièrement synchrones. `VoxtralTTSPipeline.enrollVoice` (long ; durée non sourcée dans le dépôt, À MESURER)
  est synchrone (`:426-452`), mais annulable par son sondage `shouldContinue`.
- **Constat** : une `Task` annulée continue jusqu'à `maxTokens` (Small 24B : plusieurs minutes). Le calcul et le
  chargement bloquent un thread du pool coopératif ; l'amont l'évite explicitement (« cooperative threads must
  never block », `MLXLMCommon/Load.swift:409-438`).
- **Correction** : `try Task.checkCancellation()` à chaque pas des boucles de génération (`VoxtralModeling.swift:1159`,
  `:1345`, TTS batch `VoxtralTTSModeling.swift:488`) ; exécuter chargement et génération hors pool coopératif (file dédiée +
  continuation, modèle de l'amont) ou dans un acteur dédié.
- **Risque API** : aucun. **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche K-S09** — *Annulation et hors pool*. **Prérequis** : K-S02 (sans lui, une transcription de 10 min
  arrête le processus sous 64 Go, voir S-02). **Porte** : annulation d'une transcription de 10 min → retour en
  < 2 s avec `CancellationError`, pipeline `.ready` ; UI de l'app fluide pendant le chargement (0 blocage > 250 ms
  du main thread). Cible `macos-gpu`.

### S-10 — États des pipelines non protégés (`@unchecked Sendable` sans verrou) · moyenne

- **Où** : `VoxtralPipeline.swift:23`, `:173`, `:211-216` (`guard` puis `state = .loading` sans atomicité),
  `:325-333` ; `VoxtralTTSPipeline.swift:20`, `:75`, `:93` (`prefixCacheEntry`), `:126-131`, `:203`, `:313`, `:516`,
  `:669` (écriture depuis une Task non isolée), `:676-683` (`unload`) ; `VoxtralRealtimePipeline.swift:19`, `:51`,
  `:123` ; `VoxtralTTSSynthesisManager.swift:20` ; `VoxtralRealtimeManager.swift:21` ;
  `VoxtralTranscriptionManager.swift:51-52` (`@MainActor` + `@unchecked Sendable`, redondant) ;
  `VoxtralStandardModel` (`VoxtralStandardLoader.swift:241-242`, « caller ensures single-threaded access »).
- **Constat** : deux `loadModel` concurrents passent tous deux le `guard`. `unload()` pendant un stream : la Task
  remet `state = .ready` alors que `ttsModel == nil`, donc `isReady == true` mais `synthesize` échoue.
  `enrollVoice` ne marque pas la pipeline occupée : une synthèse concurrente sur le même modèle est possible ;
  `MLXRandom.seed` est global (`VoxtralTTSModeling.swift:582`) et la reproductibilité est alors perdue.
- **Correction** : machine d'états protégée (`OSAllocatedUnfairLock`, comme `RuntimeBeacon.swift:146`) avec
  transitions atomiques ; état `.enrolling` ; jeton de génération pour ignorer les écritures d'une Task périmée ;
  à terme, acteurs (cassant → ASK).
- **Risque API** : aucun (verrou) / cassant (acteur). **Effort** : M. **Statut** : VÉRIFIÉ en lecture ; course À
  MESURER (Thread Sanitizer).
- **Fiche K-S10** — *Machine d'états atomique*. **Porte** : test de stress (2 × `loadModel` + `unload` pendant un
  stream + synthèse pendant un enrôlement) : 0 alerte TSan, états finaux cohérents (10/10 exécutions). Cible
  `macos-gpu`.

### S-11 — État global mutable non synchronisé · moyenne

- **Où** : `VoxtralFeatureExtractor.swift:249-271` (`nonisolated(unsafe) var _melFiltersCache` ; le commentaire
  « worst case is computing twice » est faux : une mutation concurrente de `Dictionary` est un comportement
  indéfini) ; `Utils/VoxtralMemoryManager.swift:13` (« Thread-safe »), `:23` (`public var config` sans verrou),
  `:49` (`evalCounter` hors verrou), `:149`, `:156` ; `VoxtralPipeline.swift:204` (chaque `init` écrase la
  configuration globale, relue par `generateStream` via `VoxtralModeling.swift:1119`) ;
  `Utils/ModelDownloader.swift:28` (`customModelsDirectory`, écrit par FluxForge), `:32-39` (singleton paresseux
  `_hubApi` non protégé), `:44` (`setenv` process-wide) ; `VoxtralModeling.swift:19` (`writeDebugToDump`),
  `:902`, `:908` (`_mergeCallCount` incrémenté, jamais lu) ; `Utils/VoxtralDebug.swift:11`, `:14` ;
  `CoreML/VoxtralCoreMLEncoder.swift:184`.
- **Correction** : `Mutex`/`OSAllocatedUnfairLock` pour le cache mel et la configuration ; configuration mémoire
  portée par la pipeline (pas de global) ; supprimer `_mergeCallCount` et `writeDebugToDump` (voir S-14) ;
  initialisation `static let` pour `_hubApi`.
- **Risque API** : additif (la propriété `config` peut rester, protégée). **Effort** : S/M. **Statut** : VÉRIFIÉ.
- **Fiche K-S11** — *Globaux protégés*. **Porte** : 8 déclarations `nonisolated(unsafe)` → ≤ 2 documentées ;
  test : deux pipelines de configurations différentes gardent chacune la leur ; TSan propre sur extraction de
  features en parallèle (4 tâches). Cible `macos-gpu`.

### S-12 — `MLXArray` paresseux qui traverse une frontière d'isolation (MLX-004) · moyenne

- **Où** : types `@unchecked Sendable` porteurs de `MLXArray` : `TTS/VoxtralTTSProcessor.swift:13-14`
  (`TTSSynthesisResult.waveform`), `:323-325` (`TTSStreamingChunk.waveform`), `VoxtralTTSModeling.swift:555-557`
  (`GenerationChunk`). Tranches non évaluées rendues : `VoxtralTTSProcessor.swift:134`, `:178`, `:269` ; streaming
  `VoxtralTTSPipeline.swift:641-643`, `:629`. Consommées sur le MainActor (`StreamingDemoViewModel.swift:580-586`)
  et dans FluxForge.
- **Constat** : le pattern MLX-004 (« eval avant tout transfert ») s'applique, mais le détecteur d'apply.py n'a rien
  trouvé : il ne cherche que `nonisolated(unsafe) let` et `UncheckedTransfer(`.
- **Correction** : `MLX.eval(waveform)` avant `return`/`yield` (sans coût si déjà évalué) ; documenter
  « waveform évalué » sur les types publics ; la doc « float32 PCM » (`:324`) est inexacte (dtype du décodeur).
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ en lecture ; plantage À MESURER.
- **Fiche K-S12** — *eval avant transfert*. **Porte** : test « consommer le résultat sur le MainActor après une
  synthèse faite hors MainActor » 20/20 sans plantage ; audio identique. Cible `macos-gpu`.

### S-13 — Code mort vérifié (0 appel dans le dépôt, 0 chez les consommateurs) · moyenne

Vérifié par `grep` des références hors déclaration et commentaires, puis par recherche GitHub chez les
consommateurs.

| Élément | Où | Lignes | Visibilité |
|---|---|---|---|
| `VoxtralGenerate` (`ParsableCommand` jamais invoqué, seule raison d'`import ArgumentParser` dans la bibliothèque) | `Scripts/VoxtralGenerate.swift` | 292 | interne |
| Fichiers de commentaires « équivalent Python `__init__` » | `Scripts/Scripts.swift`, `Utils/Utils.swift`, `Models/Models.swift` | 90 | — |
| `CustomLoadWeights.customLoadWeights` → `replaceAllQuantizedLinearWithWeights` (chaîne jamais appelée ; contient 3 `fatalError`) | `Utils/CustomLoadWeights.swift`, `Utils/QuantizedLinearWeightLoader.swift` | 333 | publique |
| `loadVoxtralModelWithMLXLM` | `Utils/VoxtralMLXLMLoader.swift` | 60 | publique |
| `LlamaModelWrapper` (jamais construit ; seulement des `as?` : `VoxtralModeling.swift:509`, `:638`, `:921`, `:1469`, `:1614`) | `Utils/LlamaModelWrapper.swift` | 36 | publique |
| `AudioEncoder`, `ChatTemplateProcessor` (qui fait un `Data(contentsOf:)` synchrone sur une URL http, l. 770-781), `encodeTranscription` | `VoxtralComponents.swift:564-568`, `:645-781` | ≈ 140 | publique |
| `mlxLMGetModelPath`, `mlxLMInitializeRope`, **les deux** `mlxLMScaledDotProductAttention` | `MLXLMBridge.swift:95`, `:138`, `:178` ; `Models/VoxtralLlama.swift:470` | ≈ 120 | publique |
| `quantizeModel`, `saveModel`, `saveConfig`, `saveQuantizedModel`, `getQuantizationStats`, `loadQuantizedVoxtral` (`VoxtralQuantization.swift:355`), `treeReduce`/`treeFlatten`, `computeBitsPerWeight` ×2, `voxtralMixedQuantizationPredicate` ×2 (la version `MLXLMBridge.swift:681` de `loadQuantizedVoxtral` est, elle, atteinte par le chargeur legacy, voir S-14) | `VoxtralQuantization.swift` (fichier entier, 625 l.) ; `MLXLMBridge.swift:566-679` | ≈ 740 | publique |
| `MLXCoreMLBridge.toMLMultiArrayNoCopy` (copie malgré son nom) | `CoreML/MLXCoreMLBridge.swift:216-222` | 7 | publique |
| Classe `VoxtralCLI` dans la bibliothèque (appelle `exit(1)`) | `VoxtralGenerator.swift:328-369` | 42 | publique |
| Privés morts : `debugModelWeights`, `dumpSwiftAudioFeatures`, `loadPythonAudioFeatures` (chemins `/Users/vincent/…`) ; `debugSwiftWeightLoadingChain` ; `VoxtralGenerator.loadModel/loadProcessor/processAudio/generateStreaming/generateBatch` | `VoxtralModeling.swift:650-711`, `:768-892` ; `MLXLMBridge.swift:390-540` ; `VoxtralGenerator.swift:127-296` | ≈ 500 | privée |
| Bloc commenté `loadVoxtralWithOfficialLlama` | `VoxtralStandardLoader.swift:1254-1281` | 28 | — |

- **Correction** : lot 1 (aucun risque) : privés, internes, commentés et fichiers-commentaires, `_mergeCallCount` ;
  lot 2 : publics → `@available(*, deprecated, message:)` dans une version mineure, suppression dans une majeure (ASK).
- **Risque API** : lot 1 aucun ; lot 2 cassant (aucun consommateur connu). **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche K-S13** — *Retrait du code mort*. **Porte** : lot 1 : −≈ 900 lignes, `BUILD SUCCEEDED`, tests verts
  (487 fonctions) ; lot 2 : 0 avertissement de dépréciation dans FluxForge (build) avant la suppression. Cible
  `macos-gpu`.

### S-14 — Chemin de chargement STT « legacy » public, divergent et bruyant · moyenne

- **Où** : `VoxtralGenerator` (`VoxtralGenerator.swift:69-122`) + `VoxtralGeneratorBridge.swift:18-51` →
  `VoxtralForConditionalGeneration.init(path:)` (`MLXLMBridge.swift:206-285`) / `fromPretrained`
  (`VoxtralModeling.swift:1033-1047`) → `loadVoxtralModel(modelPath:dtype:lazy:)` (`Utils/VoxtralModelLoading.swift:139-360`).
  `writeDebugToDump` (`VoxtralModeling.swift:19-33`) : **216 appels** (MLXLMBridge 102, VoxtralModelLoading 61,
  VoxtralQuantization 26, QuantizedLinearWeightLoader 12, CustomLoadWeights 11, VoxtralMLXLMLoader 4 ; la
  « VoxtralModeling 1 » initiale était la déclaration), tous en code legacy ou mort (0 dans `sanitize`, `:419-501`,
  ni dans le chemin des pipelines), non conditionnés par `VoxtralDebug`, qui ouvrent, écrivent et ferment
  `/tmp/swift_debug_generation.txt` à chaque message (croissance non bornée). `downloadModel(modelId:revision:)` public
  (`VoxtralModelLoading.swift:15-64`) est une **souche** : il crée `~/Documents/models/<id>` vide, affiche
  « Model downloaded » et ne télécharge rien (`downloadFromHuggingFaceHub` n'est qu'un `print`). Deux
  `loadVoxtralModel(modelPath:dtype:)` publics aux types de retour différents (`VoxtralModelLoading.swift:139`,
  `Utils/VoxtralPythonCompatLoader.swift:16`) : l'appel sans étiquette de type de `MLXLMBridge.swift:209` dépend du
  classement des surcharges. Vérification croisée : la règle de Swift qui préfère la surcharge sans argument par
  défaut désigne **probablement** `VoxtralPythonCompatLoader.swift:16`, qui délègue au chargeur vivant
  `loadVoxtralStandardModel` ; `init(path:)` ne passerait alors **pas** par `VoxtralModelLoading.swift:139-360`,
  qui serait du code mort plutôt que legacy atteignable (résolution À MESURER au build, par exemple en rendant une
  surcharge `@available(*, unavailable)`). La correction ne change pas. `llms.txt:227-235` présente `loadVoxtralModel()` comme « Internal function » alors qu'il
  est public.
- **Attention** : `VoxtralModelLoading.swift` contient aussi du code **vivant** : `Module.sanitize` (`:419-501`,
  appelé par `VoxtralStandardLoader.swift:1319`, `:1334`) et `enum VoxtralError` (`:748-764`). À déplacer avant
  toute suppression.
- **Correction** : déprécier toute la famille (ASK) ; en attendant, `writeDebugToDump` devient un no-op sauf si
  `VoxtralDebug.enabled` ; `downloadModel` lève « non supporté » ou délègue à `ModelDownloader`.
- **Risque API** : cassant pour la suppression (0 consommateur) ; aucun pour le no-op. **Effort** : M.
  **Statut** : VÉRIFIÉ.
- **Fiche K-S14** — *Isoler et déprécier la famille legacy*. **Porte** : 0 écriture dans `/tmp` pendant une
  génération **par `VoxtralGenerator` (chemin legacy)** (vérifié par `fs_usage` ou un test ; le chemin des
  pipelines n'écrit déjà rien, une porte sur lui serait verte d'avance) ; `sanitize` et `VoxtralError` déplacés,
  build et tests verts. Cible `macos-gpu`.

### S-15 — Dépendances inutiles dans `VoxtralCore` · moyenne

- **Où** : `Package.swift:63` (`MLXOptimizers` : 0 `import`) ; `:68` (`MLXLLM` : un seul `import MLXLLM`,
  `VoxtralModeling.swift:13`, et aucun symbole MLXLLM référencé : `LlamaModel` y désigne la classe **locale**
  `Models/VoxtralLlama.swift:275`, qui masque celle de l'amont ; le commentaire « Constructor for official
  MLXLLM.LlamaModel » `:585-587` est faux) ; `:65` (`ArgumentParser` seulement pour le mort `VoxtralGenerate`) ;
  `:66` (`Transformers` déclaré, mais seul `Hub` est importé : `ModelDownloader.swift:9`, `VoxtralCoreMLEncoder.swift`).
- **Constat** : chaque consommateur (FluxForge) compile `MLXLLM` (≈ 60 modèles) et `ArgumentParser` pour rien.
- **Correction** : retirer `MLXLLM`, `MLXOptimizers`, `ArgumentParser` de la cible `VoxtralCore` (après S-13) ;
  dépendre du produit `Hub` au lieu de `Transformers`. **Et** déclarer `ArgumentParser` dans la cible
  `VoxtralTranscriptionTest` (`Package.swift:84-90`), qui l'importe (`VoxtralCLI.swift:16`, `ProfileCommand.swift:4`)
  sans le déclarer et ne compile aujourd'hui que par l'import transitif de `VoxtralCore` (vérification croisée :
  la correction initiale cassait le CLI). Les tests importent aussi `MLXLMCommon`, `MLXFFT`, `MLXRandom` par
  transitivité (non touchés ici).
- **Risque API** : aucun (sauf un consommateur qui profiterait de l'import transitif, à vérifier au build de
  FluxForge). **Effort** : S. **Statut** : VÉRIFIÉ en lecture ; compilation À MESURER.
- **Fiche K-S15** — *Élaguer les dépendances*. **Porte** : `BUILD SUCCEEDED` pour VoxtralCore, les 4 exécutables,
  les tests et FluxForge ; temps de build propre de VoxtralCore mesuré avant/après (gain attendu, non chiffré).
  Cible `macos-gpu`.

### S-16 — Triplication interne et redondance avec l'amont mlx-swift-lm · moyenne

- **Où** : 3 LLM : `Models/VoxtralLlama.swift` (`LlamaModel`, 542 l., atteint seulement par `init(config:)`
  legacy, `VoxtralModeling.swift:468-480`), `Utils/VoxtralStandardLoader.swift:311-785` (`LlamaStandardModel`,
  chemin réel) et `LlamaModelWrapper` (mort). Le champ effacé `@ModuleInfo public var language_model: Module`
  (`VoxtralModeling.swift:448`), avec `lm_head` lui aussi typé `Module`, impose un dispatch par `as?` à chaque appel,
  d'où **8 des 15 `fatalError`** (`VoxtralModeling.swift:514`, `:643`, `:930`, `:1478`, `:1624` pour
  `language_model` ; `:1018`, `:1639`, `:1663` pour `lm_head`). Les 2 `fatalError` du TTS
  (`TTS/VoxtralTTSModeling.swift:54`, `:73`) sont un dispatch analogue sur le type d'embedding
  (`Embedding`/`QuantizedEmbedding`), sans lien avec `language_model` : 10 dispatchs au total, comme au §5
  (vérification croisée : « 12 » était faux). Le chemin réel instancie en plus un `VoxtralEncoder` et un projecteur
  vides (`VoxtralModeling.swift:559-573`), et `VoxtralHybridEncoder` un troisième encodeur (`:132`), jamais
  chargés. 3 types de configuration pour le même `config.json` : `PythonVoxtralConfig`
  (`VoxtralConfiguration.swift:322`), `VoxtralConfig`, `VoxtralStandardConfiguration`
  (`VoxtralStandardLoader.swift:23`). Chargeur et quantification maison : `detectQuantizedModules` +
  `convertSnakeCaseToCamelCase` (`VoxtralStandardLoader.swift:1034-1251`, 2ᵉ copie `VoxtralModelLoading.swift:389`).
- **Amont** : `MLXLMCommon.loadWeights(modelDirectory:model:quantization:perLayerQuantization:)`
  (`Load.swift:368-405` : quantification par `.scales`, `verify: [.all]`, shards concurrents), variante async
  (`:409-438`), `BaseConfiguration.PerLayerQuantization` (`BaseConfiguration.swift:71-99`, même format de config
  mixte). Backbone avec injection d'embeddings : `Mistral3TextModel.callAsFunction(_:cache:inputEmbeddings:)`
  (`MLXLLM/Models/Mistral3Text.swift:275-302`) ; le `LlamaModel` amont n'accepte pas d'embeddings
  (`Llama.swift:139`, `:174`). **Non-constats** : caches KV (amont déjà utilisé) ; tokenizer Tekken (pas
  d'équivalent Tekken en Swift dans l'amont).
- **Correction** : (a) typer `language_model` en `LlamaStandardModel` et `lm_head` en `Linear` (dont
  `QuantizedLinear` hérite) et supprimer les branches mortes ; (b)
  migrer le chargement STT vers `MLXLMCommon.loadWeights` + `PerLayerQuantization` (adapter `sanitize`) : l'amont
  exige `model: BaseLanguageModel` (`Load.swift:368-369`, protocole `LanguageModel.swift:8`, qui ne demande que
  `sanitize(weights:)`), compatible avec le retrait de `LanguageModel` proposé en S-17 ; son `verify: [.all]` inclut
  `.noUnusedKeys`, donc retirer d'abord la clé dupliquée `embedTokens.weight` (S-04) ; (c)
  évaluer `Mistral3TextModel` comme backbone (parité greedy obligatoire).
- **Risque API** : cassant pour `LlamaModel`, `VoxtralConfig`, `PythonVoxtralConfig` publics (ASK). **Effort** :
  L. **Statut** : VÉRIFIÉ (lecture de l'amont).
- **Fiche K-S16** — *Un seul LLM, chargeur amont*. **Porte** : parité greedy 32/32 jetons sur 6 audios × 3
  quantisations ; temps de chargement ≤ baseline (chargement concurrent amont attendu plus rapide, À MESURER) ;
  0 `fatalError` de dispatch. Cible `macos-gpu`.

### S-17 — Conformance `LanguageModel` inutilisée, fausse, cause de la casse #50 · moyenne

- **Où** : `VoxtralStandardLoader.swift:242-257` (`prepare` « Simple implementation for testing ») ;
  `VoxtralModeling.swift:434`, `:1596-1647` (`prepare` passe `mergedEmbeddings` comme **ids de jetons** à
  `callLanguageModel(inputs:)` l. 1629 ; l'audio passe par `input.image?.pixels` l. 1602), `:1652-1666`,
  `:1679-1690`.
- **Constat** : aucun appel à `TokenIterator`, `ModelContainer` ni au `generate` amont (grep : 0). Le seul effet
  de cette conformance est le couplage à l'API mouvante de `main` : le correctif #50 (`9392ed1`) répare exactement
  cela. L'amont `main@ee673d6` a encore fait évoluer le protocole (`ChatConventionsProviding`,
  `newCache(parameters:) throws`, `cacheStatus` : `LanguageModel.swift:308-366`, avec implémentations par défaut).
- **Correction** : retirer les conformances `LanguageModel`/`KVCacheDimensionProvider` des deux classes (ou les
  corriger et les tester avec `TokenIterator`).
- **Risque API** : cassant en théorie (0 consommateur) → ASK. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S17** — *Découpler du protocole LanguageModel*. **Porte** : build vert contre `mlx-swift-lm` `main`
  courant et contre le tag 3.31.4 ; tests verts. Cible `macos-gpu`.

### S-18 — `mlx-swift-lm` suit `main`, sans `Package.resolved` suivi · moyenne

- **Où** : `Package.swift:46-52` ; `.gitignore:27` (`Package.resolved` ignoré).
- **Constat** : chaque résolution prend la tête de `main`. La casse de build est avérée (#50, mlx-swift-lm a changé
  la signature de `prepare`) ; la révision résolue n'est notée nulle part (piège 21 du catalogue). La contrainte est
  réelle : FluxForge dépend aussi de `main`, et SwiftPM refuse de résoudre un intervalle de versions contre une
  branche. `main@ee673d6` a 10 jours de plus que `9392ed1` : compatibilité À MESURER.
- **Correction** : suivre `Package.resolved` dans le dépôt (sans effet sur les consommateurs, reproductibilité des
  builds, bancs et CI) ; noter la révision résolue dans chaque ligne de banc ; dès qu'un tag > 3.31.4 existe,
  passer FluxForge et Voxtral en `from:` ensemble (ASK) ; S-17 réduit la surface de couplage.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S18** — *Résolution reproductible*. **Porte** : `Package.resolved` suivi ; `swift package resolve` puis
  build → révision identique sur deux machines ; build vert contre `main@ee673d6` consigné. Cible `macos-gpu`.

### S-19 — Plateformes et toolchain incohérentes · basse

> Vérification croisée : sévérité ramenée de moyenne à **basse** (écart documentaire ; un intégrateur trop ancien
> reçoit une erreur SwiftPM explicite, rien de silencieux) ; fiche scindée par cible.

- **Où** : `Package.swift:1` (`swift-tools-version: 6.2`, soit Swift 6.2 et Xcode 26), `:10-13`
  (`.macOS(.v15)`, `.iOS(.v17)`) ; `README.md:40-43` (« macOS 14.0 / Xcode 15.0 / Swift 6.0 ») ;
  `llms.txt:188-192` (idem) ; 28 annotations `@available(macOS 13/14…)` sous le plancher (par exemple
  `VoxtralPipeline.swift:22`, `:534`, `VoxtralTTSPipeline.swift:19`, `VoxtralCoreMLEncoder.swift:176`) ;
  `Info.plist` : `LSMinimumSystemVersion 15.0` (cohérent).
- **Constat** : un intégrateur en Swift 6.0 ou 6.1 est refusé par SwiftPM ; une cible macOS 14 ne peut pas
  dépendre de `VoxtralCore`.
- **Correction** : README et llms.txt → « macOS 15+ / iOS 17+, Xcode 26+ (Swift 6.2) » ; retirer les
  `@available` obsolètes (code, build).
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S19a** — *Exigences exactes (docs)*. **Porte** : README et llms.txt disent « macOS 15+ / iOS 17+,
  Xcode 26+ (Swift 6.2) », 0 mention de « Swift 6.0 », « Xcode 15 » ou « macOS 14 » (`grep`). Cible `cloud`.
- **Fiche K-S19b** — *`@available` sous le plancher (code)*. **Porte** : `grep -rc '@available(macOS 1[34]' Sources`
  = 0 (28 aujourd'hui) et `BUILD SUCCEEDED` toutes cibles. Cible `macos-gpu` (exige un build).

### S-20 — Documentation désynchronisée du code · moyenne

- **Où** : `llms.txt:5` (« Version: 1.0.10 ») et `:198` (`from: "1.0.8"`) alors que `9392ed1` = v2.2.2 ; llms.txt
  ne mentionne ni TTS, ni clonage, ni Realtime ; `:223` « ModelDownloader — Thread-safe » est faux (S-11) ;
  `:47-50` présente `tokenCount`/`tokensPerSecond` comme des métriques, or `tokenCount` vaut toujours 0
  (`VoxtralTranscriptionManager.swift:126`) ; `context7.json` enregistre le dépôt auprès de Context7, qui indexe sa
  documentation pour les assistants de code (que llms.txt en fasse partie est probable mais non vérifié ici).
  Trois versions différentes : `VoxtralCoreVersion = "0.1.0"` (`VoxtralCore.swift:45`), CLI `version: "2.0.0"`
  (`VoxtralTranscriptionTest/VoxtralCLI.swift:25`), tag v2.2.2. README : dépôt `small-24b-8bit` (`:216`) ≠ code
  (S-06) ; tailles `~6 GB`/`~48 GB` (`:207`, `:215`) ≠ téléchargement réel (S-07) ; `voxtral tts` (`:282`) alors que
  l'exécutable s'appelle `VoxtralCLI` (le nom `voxtral` n'est que le `commandName`, `VoxtralCLI.swift:23`) ;
  l'arbre d'architecture (`:291-309`) omet `Realtime/`, `CoreML/`, `Pipeline/`, `VoiceCloning/`,
  `VoxtralBenchmark` ; le Realtime est absent des Features ; le commentaire « Auto mode (recommended) » (`:266-267`)
  illustre `--backend hybrid` ; « Hybrid … ~660 MB less » (`:263`) n'est pas étayé, puisque les poids MLX de
  l'encodeur restent résidents en hybride (`VoxtralModeling.swift:445`, `:576`, `VoxtralHybridEncoder.swift:499-503`)
  → À MESURER ; `Examples/ReferenceImplementation.swift:7` « Tested with: v1.0.8 » et le fichier n'est compilé par
  aucune cible (l'API qu'il utilise est pourtant à jour).
- **Correction** : une seule source de version (générée depuis le tag) pour `VoxtralCoreVersion`, le CLI et llms.txt ;
  réécrire llms.txt (API des 3 pipelines, exigences, mise en garde de concurrence) ; corriger le README ; compiler
  `Examples/` comme cible d'exemple ou le déplacer dans les docs.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ (sauf la revendication « 660 MB », À MESURER).
- **Fiche K-S20** — *Docs alignées*. **Porte** : 0 écart dans la checklist S-20 (12 points) relue contre
  `9392ed1` ; llms.txt cite v2.2.x. Cible `cloud` pour les docs, `Examples/` déplacé dans `docs/` et les littéraux
  de version remplacés à la main ; une version **générée** depuis le tag ou une cible d'exemple compilée relèvent
  d'une fiche `macos-gpu` distincte (build requis).

### S-21 — Surface publique énorme, noms génériques en collision chez les consommateurs · moyenne

- **Où** : ≈ 1 100 déclarations `public` dans `VoxtralCore` (1 098 à 1 104 lignes selon le motif `grep` ; le
  « 1 065 » initial n'a pas été reproduit) ; 45 fonctions libres publiques au niveau fichier (39 noms distincts ;
  « 41 » initial non reproduit ; hors `loadVoxtralWithOfficialLlama`, commentée) (`loadWeights`,
  `loadConfig`, `loadAudio`, `downloadModel`, `quantizeModel`, `saveModel`, `saveConfig`, `treeFlatten`,
  `treeReduce`, `createCausalMask`, `initializeRope`…) ; constantes globales `N_FFT`, `HOP_LENGTH`, `N_MELS`
  (`VoxtralFeatureExtractor.swift:14-19`) ; types génériques `ModelRegistry`, `ModelDownloader`, `RuntimeBeacon`,
  `LlamaModel`, `AudioEncoder`, `VoxtralModel` (typealias, `MLXLMBridge.swift:195`). API « à la Python » :
  `applyChatTemplate` renvoie `Any` et oblige à faire des `as!` (`VoxtralProcessor.swift:521`, `:171`, `:384`,
  `:631`, `:722` ; `VoxtralPipeline.swift:424-428`) ; faute de frappe publique `applyTranscritionRequest`
  (`VoxtralProcessor.swift:352`, signalée en commentaire `VoxtralPipeline.swift:335`) ; l'app détecte le backend
  hybride en analysant une chaîne (`VoxtralApp/TranscriptionManager.swift:242`).
- **Preuve consommateur** : FluxForge `ViewModels/ModelManager.swift` : « Typealias to avoid name collision with
  VoxtralCore.ModelRegistry » ; `Services/LTXBeaconBridge.swift` : fichier créé uniquement parce que `RuntimeBeacon`
  existe dans VoxtralCore et LTXVideo (recherche GitHub, 2026-09-27).
- **Correction** : revue d'API — façades publiques (3 pipelines, 3 managers, registres, `RuntimeBeacon`,
  enrôlement, `WAVWriter`), le reste en `internal` ; préfixer les noms génériques (`VoxtralModelRegistry`…) via
  typealias déprécié ; résultats typés (`ProcessedInputs`) ; propriété `isCoreMLActive` typée.
- **Risque API** : cassant → ASK (plan de dépréciation sur une version mineure). **Effort** : L. **Statut** : VÉRIFIÉ.
- **Fiche K-S21** — *Revue d'API*. **Porte** : liste des symboles publics conservés validée (ASK) ; FluxForge
  compile sans typealias de contournement ; déclarations publiques ≤ 300. Cible `macos-gpu`.

### S-22 — API publiques trompeuses (souches, paramètres ignorés, métriques à zéro) · moyenne

- **Où** : `VoxtralTranscriptionManager.chat(systemPrompt:userMessage:)` lève toujours (`:153-157`) ;
  `TranscriptionResult.tokenCount` = 0 donc `tokensPerSecond` = 0 (`:124-128`) ; `loadVoxtralStandardModel(dtype:)`
  ignore `dtype` (`VoxtralStandardLoader.swift:1285-1343`) alors que `VoxtralPipeline` passe `.float16`
  (`:237-240`) ; `saveQuantizedModel` n'enregistre pas les poids, affiche « requires external implementation » et
  n'écrit que `config.json` (`VoxtralQuantization.swift:503-525`) ; `downloadModel` (S-14) ;
  `toMLMultiArrayNoCopy` copie (`MLXCoreMLBridge.swift:216-222`) ; `VoxtralForConditionalGeneration.init(officialLlama:)` ne prend pas le Llama « officiel »
  (`VoxtralModeling.swift:585-632`, voir S-15).
- **Correction** : implémenter (compter les jetons générés), retirer, ou déprécier avec un message explicite.
- **Risque API** : additif (implémenter) / cassant (retirer). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S22** — *API honnête*. **Porte** : `tokenCount` > 0 sur une transcription ; 0 souche publique restante
  (checklist S-22). Cible `macos-gpu`.

### S-23 — `print()` dans la bibliothèque · basse

- **Où** : 155 `print` dans `VoxtralCore` (377 au total selon scan.py, CLI et Examples compris). Sur chemins
  vivants : `ModelDownloader.swift:349-375`, `:386-395`, `:595-609` ; `VoxtralCoreMLEncoder.swift:257`, `:269`,
  `:318-322` (à chaque recherche de modèle) ; `VoxtralTTSModeling.swift:515`, `:633` (« [GEN] EOA at frame » à
  chaque synthèse) ; `VoxtralFlowMatching.swift:253-259` (sous `debug:`). `VoxtralDebug.always` n'est lui-même qu'un
  `print` (`Utils/VoxtralDebug.swift:31-33`). Le reste est en code legacy (S-14).
- **Correction** : `os.Logger` (sous-système `…voxtral`, catégories load/stt/tts) derrière `VoxtralDebug` ; le CLI
  garde ses `print`.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S23** — *Journalisation*. **Porte** : `grep -c 'print(' Sources/VoxtralCore` = 0 hors `VoxtralDebug` ;
  une synthèse dans FluxForge n'écrit rien sur stdout. Cible `macos-gpu`.

### S-24 — Chemins développeur codés en dur ; recherche de modèles dans le CWD · basse

- **Où** : `/Users/vincent/Developpements/convertvoxtral/…` dans `VoxtralModeling.swift:769`, `:811`, `:843` et
  `MLXLMBridge.swift:394` (tous en code mort), `VoxtralCoreMLEncoder.swift:305-311` (`#if DEBUG`),
  `Tests/…/TekkenTokenizerTests.swift:8`. Recherche Core ML dans `CWD/Resources` et dans des chemins relatifs à
  l'exécutable (`VoxtralCoreMLEncoder.swift:287-333`) ; recherche de `voxtral_models/` dans le CWD
  (`ModelDownloader.swift:297-306`) : un modèle arbitraire du dossier courant peut être chargé.
- **Correction** : supprimer ; ne garder que la variable `VOXTRAL_RESOURCES_PATH` explicite.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S24** — *Chemins explicites*. **Porte** : `grep -rn '/Users/' Sources Tests` = 0 ; build vert. Cible
  `macos-gpu`.

### S-25 — Hygiène git : fichiers suivis malgré `.gitignore` · basse

> Vérification croisée : chiffres exacts (24 fichiers, 73 583 048 o, 8 WAV non référencés = 18,27 Mo, `.serena`
> 1 790 543 o), sévérité ramenée de moyenne à **basse** : SwiftPM clone l'**historique**, donc retirer les fichiers
> de l'arbre ne réduit pas ce que rapatrie FluxForge ; seul le gain d'hygiène reste sans réécriture d'historique (ASK).

- **Où** : `git ls-files -ci --exclude-standard` → **24** fichiers. 22 WAV (`.gitignore:54` `*.wav`) =
  73 583 048 o (70,2 Mio) : **14 référencés** (`docs/tts_benchmark.md:85-110` : `fluxforge_*` ×12 ;
  `docs/voice_cloning.md` : `clone_en.wav`, `clone_fr.wav`) et **8 non référencés par leur nom**
  (`docs/examples/tts_bench_*.wav` ×6 = 10,7 Mo ; `samples/{en,fr}_voxtral_tts_demo.wav` = 7,5 Mo ; 18,3 Mo en
  tout). `.serena/cache/swift/{document_symbols,raw_document_symbols}.pkl` (1,79 Mo ; cache d'outil IDE ;
  `.gitignore:119` ; ajouté par erreur dans le commit `90d6ab4`, « revert: Restore v1.0.5 weight loading »).
- **Constat** : SwiftPM clone le dépôt avec son historique : chaque machine qui compile FluxForge rapatrie
  ≈ 72 Mo d'audio et de cache inutiles au code.
- **Correction** : `git rm --cached .serena/` ; supprimer ou déplacer vers les assets d'une Release GitHub les
  8 WAV non référencés ; pour les 14 référencés, assets de Release ou Git LFS (liens de la doc à mettre à jour), ou
  exception explicite `!docs/examples/*.wav` si on les garde ; l'historique reste lourd (réécriture = ASK).
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S25** — *Dépôt propre*. **Porte** : `git ls-files -ci --exclude-standard | wc -l` = 0 (ou uniquement
  les exceptions déclarées) ; poids de l'arbre suivi −≥ 20 Mo (samples, tts_bench, .serena) ; liens de la doc
  valides (0 lien mort). Cible `cloud`.

### S-26 — Racine et cibles annexes · basse

- **Où** : `create_app_bundle.sh` (référencé nulle part) : binaire Debug seulement (`:10-17`), n'embarque pas le
  bundle de ressources SwiftPM (`:29`) alors que `Bundle.module` est lu au démarrage
  (`VoxtralApp/VoxtralAppMain.swift:19`) ; son Info.plist fait doublon (`:32-63` et `Sources/VoxtralApp/Resources/Info.plist`).
  `Package.swift:79-81` déclare `.copy("Resources/VoxtralEncoderFull.mlmodelc")` alors que `*.mlmodelc/` est ignoré
  (`.gitignore:41`, `:45`) : sur un clone propre, la ressource manque (avertissement SwiftPM) et `Bundle.module`
  n'est peut-être pas généré, auquel cas VoxtralApp ne compile pas → À MESURER. L'encodeur Core ML est de toute façon
  téléchargé depuis HF (`VoxtralCoreMLEncoder.swift:434-518`). `VoxtralBenchmark` annoncé « Performance benchmark
  tool » (`Package.swift:30`) ne mesure que des conversions Float16 (`BenchmarkCLI.swift:17-20`) : piège 33, renvoi
  à l'audit perf. Le CLI s'appelle `VoxtralTranscriptionTest` (`Package.swift:27-28`, `:85`). `context7.json` : clé
  publique Context7 (normal), mais elle expose un llms.txt périmé (S-20).
- **Correction** : supprimer `create_app_bundle.sh` (ou le documenter en Release + copie du `.bundle`) ; retirer la
  ressource `.mlmodelc` et `resourceBundle` ; renommer ou réécrire le bench (audit perf).
- **Risque API** : aucun (sauf `VoxtralCoreMLEncoder.resourceBundle`, public, à déprécier). **Effort** : S.
  **Statut** : VÉRIFIÉ en lecture ; compilation sur clone propre À MESURER.
- **Fiche K-S26** — *Racine et cibles annexes*. **Porte** : `git clone` + `swift build` (toutes cibles) =
  `BUILD SUCCEEDED` sans avertissement « Invalid Resource ». Cible `macos-gpu`.

### S-27 — Tests manquants, tautologiques ou trompeurs ; pas de CI · moyenne

- **Où / constat** :
  - Pas de CI (`.github/` absent) ni de script de tests ; 487 fonctions XCTest.
  - Sans test : `VoxtralPipeline` (hors la campagne `VOXTRAL_TTS_CAMPAIGN`, `TTSQuantizationCampaignTests.swift:112`),
    `VoxtralTranscriptionManager`, la complétude du téléchargement (`verifyShardedModel`, `findModelPath`,
    `matchesGlob`, `downloadRepoDirect`), `VoxtralStandardLoader` (`detectQuantizedModules`,
    `convertSnakeCaseToCamelCase`, `loadVoxtralStandardModel`), `generateStream` et `mergeInputEmbeddings` (jetons
    d'arrêt, cache rotatif), `VoxtralHybridEncoder`, `VoxtralCoreMLEncoder`, `MLXCoreMLBridge` (conversions fp16),
    `VoxtralMemoryManager`, `VoxtralRealtimeManager` ; 0 test d'annulation ou de concurrence.
  - Tests tautologiques : `Performance/PerformanceOptimizationTests.swift:173-255` recopient l'algorithme (préfill
    tranché, EOA) au lieu d'appeler le code : ils passent même si le code change.
  - `Tokenization/TekkenTokenizerTests.swift:8-19` : hors de la machine du mainteneur, les tests tournent sur le
    tokenizer démo (S-05) et passent sans rien prouver.
  - `Loading/ModelLoadingSymlinkedDirectoryTests.swift:33` teste `loadWeights(modelPath:)` **legacy** et pas
    `loadWeights(from:)` du chemin réel (`VoxtralStandardLoader.swift:986-1024`), corrigé en double dans #49.
  - **Non-constat** : les sondes TTS lourdes sont bien gardées par variables d'environnement (`XCTSkipUnless`,
    par exemple `TTSBlindValidationTests.swift:21-27`).
- **Correction** : tests unitaires sans GPU pour S-01, S-03, S-06, S-07 et la quantification ; tests d'intégration
  gardés pour S-02, S-04, S-08, S-09 ; remplacer les tests tautologiques par des appels au code ; tokenizer de test
  = fixture réduite versionnée ; CI macOS (GitHub Actions `macos-15`, build + tests sans modèle).
- **Risque API** : aucun. **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche K-S27** — *Filet de tests et CI*. **Porte** : CI verte sur un runner macOS arm64 doté d'**Xcode ≥ 26**
  (`swift-tools-version: 6.2` l'exige ; image `macos-15` avec Xcode 26 sélectionné, ou `macos-26`) (build + tests
  unitaires) ; chaque
  fiche K-S01 à K-S12 apporte un test qui **échoue sans le correctif** (piège 38). Cible `macos-gpu`.

### S-28 — Erreurs secondaires avalées, `as!`, `precondition` atteignables · basse

- **Où** : quantification TTS illisible → modèle chargé non quantifié, puis `verify: .none`
  (`VoxtralTTSModelLoading.swift:91-102`, `:71`) ; jetons spéciaux illisibles → valeurs par défaut
  (`VoxtralComponents.swift:349-367`) ; `mergeInputEmbeddings` rogne sans erreur si le nombre de jetons audio ne
  correspond pas (`VoxtralModeling.swift:957-964`) ; `fatalError` (`:916`, dans le privé `mergeInputEmbeddings`)
  atteignable via l'API publique `callAsFunction(inputIds: nil, inputsEmbeds: nil)` (`:987-1000`) ; `precondition` atteignables par
  `optimize(reference:)` public et `EnrollmentLossComputer.init` (`VoxtralVoiceEnrollment.swift:501-502`,
  `VoxtralEnrollmentLosses.swift:86-87` ; `prepareReference` lève correctement, `:129-133`) ; 9 `as!` non comptés
  par scan.py (`VoxtralStandardLoader.swift:1316`, `MLXLMBridge.swift:727-728`, `VoxtralProcessor.swift:171`,
  `:384`, `:631`, `:722`, `VoxtralGenerator.swift:184`, `VoxtralGenerate.swift:98`) ; l'app avale l'erreur de
  suppression (`VoxtralApp/ContentView.swift:798`) ; le bench mesure du vide si la conversion échoue
  (`BenchmarkCLI.swift:167`).
- **Correction** : erreurs typées ; `throw` à la place des `precondition` sur l'API publique ; retirer les `as!`
  avec les types `Any` (S-21).
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-S28** — *Erreurs explicites*. **Porte** : 0 `as!`, 0 `precondition` atteignable depuis l'API publique ;
  un test par cas. Cible `macos-gpu`.

### S-29 — TODO obsolètes et souches · basse

- **Où** : 2 TODO, tous deux obsolètes : `VoxtralComponents.swift:28` (`//import Transformers // TODO: Integrate
  later` alors que swift-transformers est déjà une dépendance) et `:566`. Souches « For now / would integrate » :
  `VoxtralModelLoading.swift:50`, `:60`, `VoxtralLlama.swift:436-442`, `MLXLMBridge.swift:79`, `:182`,
  `VoxtralModeling.swift:1038`, `MLXCoreMLBridge.swift:221`, `VoxtralGenerator.swift:335`,
  `VoxtralQuantization.swift:32`, `:462`, `:512`, `VoxtralMLXLMLoader.swift:40`. En-tête trompeur
  « COMPOSANTS VOXTRAL VALIDÉS - Version Production » (`VoxtralComponents.swift:1-22`) : 2 des 3 composants annoncés
  sont morts (`AudioEncoder`, `ChatTemplateProcessor`) ; le 3ᵉ, `TekkenTokenizer`, est vivant (précision de la
  vérification croisée : l'en-tête ne couvre pas que du code mort).
- **Correction** : traités par S-13 et S-14 ; corriger l'en-tête.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : incluse dans K-S13.

## 4. Carte d'atteignabilité (qui appelle quoi)

| Zone | Atteinte par | Statut |
|---|---|---|
| `VoxtralPipeline` → `loadVoxtralStandardModel` → `VoxtralForConditionalGeneration(standardModel:)` → `generateStream(WithAudioEmbeds)` ; `VoxtralProcessor`/`TekkenTokenizer` ; `VoxtralHybridEncoder`/Core ML ; `ModelDownloader`/`ModelRegistry` | app, CLI, FluxForge, SongAnalysisDb | **vivant** (STT) |
| `VoxtralTTSPipeline` → `loadVoxtralTTSModel`, codec, flow matching, enrôlement, ZeroVoice, presets | CLI, démo, FluxForge (LipDub) | **vivant** (TTS) |
| `VoxtralRealtimePipeline` → `loadVoxtralRealtimeModel` | CLI `realtime` | **vivant** |
| `Module.sanitize`, `VoxtralError` (dans `VoxtralModelLoading.swift`) | chargeur STT vivant, partout | **vivant, dans un fichier legacy** |
| `VoxtralGenerator`(+Bridge), `init(path:)`, `fromPretrained`, `loadVoxtralModel` ×2, `VoxtralLlama` (`LlamaModel`…), `mlxLMCreateAttentionMask`, `writeDebugToDump` | API publique seulement ; 0 appel interne depuis une façade, 0 consommateur | **legacy public** (S-14) |
| Tableau S-13 | personne | **mort** |

Le code legacy et mort pèse ≈ 4 300 lignes, soit ≈ 21 % de `VoxtralCore` (20 489 lignes) : fichiers entiers legacy ou morts (≈ 4 240 l. mesurées par `wc -l`) plus les privés morts, moins le code vivant de `VoxtralModelLoading.swift`.

## 5. Indices du scan examinés et écartés (non-constats vérifiés)

- `GPU.resetPeakMemory()` (7 occurrences) : **pas** déprécié en mlx-swift 0.31.6 ; MLX-001 avait raison de ne rien
  signaler.
- Caches KV : ceux de l'amont sont déjà utilisés ; pas de redondance.
- `Task.detached` (`StreamingDemoViewModel.swift:411`) : voulu (enrôlement long, pipeline « boxée », retour sur le
  MainActor) ; pas de `MLXArray` transféré.
- `value-and-grad` (`VoxtralVoiceEnrollment.swift:558`) : aucun `compile` dans le dépôt, donc le deadlock ABBA
  compile × vjp (piège 20) ne s'applique pas ; reste le risque d'une synthèse concurrente pendant l'enrôlement (S-10).
- Les 15 `fatalError` : 10 sont des dispatchs de type inatteignables avec les modules actuels (symptôme de S-16),
  3 sont en code mort (`CustomLoadWeights`), 1 dans le chargeur legacy (`VoxtralModelLoading.swift:206`) ; seul
  `VoxtralModeling.swift:916` est atteignable par l'API publique (S-28).
- `try!` (1) : `VoxtralTTSModeling.swift:171`, regex littérale constante, sûre (pourrait devenir un `static let`).
- `RuntimeBeacon` : correctement verrouillé (`OSAllocatedUnfairLock`), écriture atomique.
- `MLX-015` (3 occurrences, dans les tests) : ce sont les tests de non-régression des liens symboliques.
- `MLX-002` (28 constantes fp32) et `cacheLimit` absent de la bibliothèque (seule l'app le pose, à 0 puis
  `Int.max`, `TranscriptionManager.swift:293-295`) : relèvent de l'audit perf.
- FluxForge, demande n° 7 de `docs/FRAMEWORK_ASKS_STORAGE.md` (« Core ML encoder ignores customModelsDirectory ») :
  **résolue** dans `1570294` (v2.2.1). La doc de FluxForge est à mettre à jour.

## 6. Décisions à prendre (ASK)

1. **Dépôt retenu pour `small-24b-8bit`** (S-06) : A) `VincentGOURBIN/voxtral-small-8bit` (registre, README) ;
   B) `mzbac/Voxtral-Small-24B-2507-8bit`.
2. **Cache KV par défaut** (S-02) : A) `KVCacheSimple` partout (contexte complet ; pic mémoire à mesurer) ;
   B) garder une limite, mais lever une erreur au dépassement.
3. **API publique legacy** (S-13 lot 2, S-14, S-17, S-21) : A) dépréciation en 2.3, suppression en 3.0 ;
   B) suppression directe en 3.0.
4. **Épinglage de `mlx-swift-lm`** (S-18) : A) suivre `Package.resolved` et rester sur `main` tant que FluxForge
   l'impose ; B) passer ensemble (FluxForge et Voxtral) au prochain tag > 3.31.4.
5. **WAV de la doc** (S-25) : A) assets de Release GitHub ; B) Git LFS ; C) garder les 14 référencés et
   supprimer les 8 autres. Réécrire l'historique : oui ou non.

## Annexe — Constats écartés à la vérification croisée

Relecture adverse (2026-09-27) : chaque constat relu contre le code à `9392ed1`, l'amont (mlx-swift-lm `ee673d6`,
mlx-swift `9019419` et son sous-module mlx `1f8e74e`) et les listings HF. **Aucun constat entier n'est écarté** :
les 29 défauts existent. Les **sous-constats** ci-dessous, réfutés ou non reproduits, ont été retirés du corps ou
corrigés.

| Id | Sous-constat retiré ou corrigé | Raison |
|---|---|---|
| S-02 | « le début des audios longs est perdu **sans erreur** » (régime invite > fenêtre) | Faux : après le premier rognage, `RotatingKVCache.updateConcat` rend `maxSize − 1 + T` clés contre un masque maison `[T, offset + T]` ; `broadcast_to` échoue (mlx `fast.cpp:906-909`) → `fatalError` du gestionnaire par défaut. La perte silencieuse n'existe qu'au décodage (T = 1). Rejoint P-03 de l'audit perf STT. |
| S-02 | « > ≈ 2 min sur 16 Go » | Faux : `recommended()` compte la RAM en Gio entiers ; un Mac 16 Go donne 16 → tranche 16-31 → 4 096 jetons (arrêt au-delà de 5 min d'audio). 2 048 ne vaut que sous 16 Go. |
| S-02 | Porte « WER ≤ WER(KVCacheSimple) + 0,5 pt » | Tautologique : la correction **est** `KVCacheSimple` ; remplacée par « 0 arrêt, première et dernière minutes présentes ». |
| S-04 | « une **forme** incompatible fait planter par le `try!` caché » | Faux : `.shapeMismatch` n'est pas dans `.none` ; la forme n'est pas vérifiée et l'erreur surgit plus loin. Seule une structure incompatible déclenche le `try!`. |
| S-06 | `llms.txt:118` cité comme pointant vers `VincentGOURBIN/…` | La ligne ne nomme aucun dépôt. |
| S-09 | Enrôlement « ≈ 30 min » | Durée non sourcée dans le dépôt ; remplacée par « À MESURER ». |
| S-14 | « 217 appels » dont « VoxtralModeling 1 » | 216 appels : la « VoxtralModeling 1 » était la déclaration. |
| S-14 | `init(path:)` → `VoxtralModelLoading.swift:139` (chemin legacy atteignable) | Non établi : la surcharge sans argument par défaut (`VoxtralPythonCompatLoader.swift:16`) est probablement choisie ; résolution À MESURER au build. |
| S-15 | Correction « retirer `ArgumentParser` de `VoxtralCore` » seule | Incomplète : `VoxtralTranscriptionTest` importe `ArgumentParser` sans le déclarer ; le CLI ne compilerait plus. Correction complétée. |
| S-16 | « 12 des 15 `fatalError` » dus à `language_model` | 10 dispatchs en tout (8 `VoxtralModeling`, dont 3 sur `lm_head` ; 2 TTS sur le type d'embedding, autre cause), cohérent avec le §5. |
| S-19 | Cible `cloud` pour une porte qui exige un build | Fiche scindée : K-S19a docs (`cloud`), K-S19b `@available` (`macos-gpu`). |
| S-21 | « 1 065 déclarations », « 41 fonctions libres » | Non reproduits : ≈ 1 100 lignes publiques (1 098 à 1 104), 45 fonctions libres (39 noms distincts). |
| S-22 | « `VoxtralCoreMLEncoder.isReady` vaut toujours `true` » (API trompeuse) | Faux positif : l'`init` lève si `MLModel(contentsOf:)` échoue et `model` est un `let` non optionnel (`VoxtralCoreMLEncoder.swift:189`, `:209-224`) ; une instance existante est toujours prête. |
| S-28 | « `try? await Task.sleep` avale l'annulation, donc les réessais continuent » (`ModelDownloader.swift:149`) | Faux : après le `sleep` avalé, le `URLSession.download` suivant lève aussitôt `URLError(.cancelled)`, non transitoire (`:166-175`), qui remonte ; au plus le type d'erreur change. |
| S-29 | En-tête « validés production » posé « sur des composants morts » | Imprécis : `TekkenTokenizer`, dans le même fichier, est vivant. |
| S-07, S-08, S-19, S-25 | Sévérités | S-07 et S-08 de haute à moyenne, S-19 et S-25 de moyenne à basse (raisons en tête de section). |
