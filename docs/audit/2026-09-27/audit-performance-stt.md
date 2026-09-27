# Audit « Performance » — chemin STT / chat de mlx-voxtral-swift

> **Vérification croisée : 29 constats relus, 28 gardés (dont 15 amendés), 1 écarté, 15 amendés.**
> Relecture adverse du 2026-09-27 : chaque `fichier:ligne` relu à `9392ed1` ; règles MLX relues dans les sources
> **re-téléchargées à `ce45c52`** (les copies du scratchpad de l'auditeur ne correspondaient pas toutes à ce
> commit ; les numéros de ligne cités sont néanmoins exacts à `ce45c52`) ; mlx-swift `0.31.6` (`ErrorHandler.swift`,
> `Linear.swift`, `MLXFast.swift`) ; mlx-swift-lm `ee673d6` ; issues #12, #13, #14, #16, #17, #18, #22 et commits `70c390b`, `c1942ee`,
> `1eb2cc9`, `d6acbf1`, `41ce59d`, `1528887` relus (les autres faits F-xx sont repris sans relecture).
> Amendés : P-01, P-03, P-04, P-06, P-07, P-08, P-10, P-13, P-14, P-15, P-20, P-21, P-24, P-26, P-29 (passages
> marqués *amendé* / *amendée* / *vérification croisée*) ; écarté : P-25 (annexe B). Corrections hors constats (F-08, §1, §3 bis) :
> annexe C.

> Skill `mlx-swift-audit`, phase 2, constats `P-01…P-29`. Révision auditée : `9392ed1` (tag `v2.2.2`,
> branche `claude/action-plan-skills-beta-wifgmu`). Date : 2026-09-27.
> Périmètre : Voxtral Mini 3B et Small 24B (4 bits mixte, 8 bits, bf16), du fichier audio au texte :
> `VoxtralFeatureExtractor`, encodeur audio MLX et Core ML (hybride), projecteur, fusion des embeddings,
> décodeur Llama (`LlamaStandardModel`, chemin hérité `VoxtralLlama`), boucles de génération
> (`VoxtralModeling.generateStream*`), quantification, chargement, mémoire (`VoxtralMemoryManager`,
> `MemoryOptimizationConfig`), pipelines (`VoxtralPipeline`, `VoxtralTranscriptionManager`), CLI `profile`.
> Hors périmètre : TTS, Realtime, enrôlement, serveur (voir `audit-annexes-serveur.md`).

## 0. Cadre, méthode, sources

- **Environnement** : session cloud Linux, **sans Mac, sans toolchain Swift, sans GPU**. Aucun build, aucun
  test, aucune mesure. Tout gain est **attendu**, jamais obtenu ; tout chiffre qui n'est pas déjà mesuré dans le
  dépôt (issues, commits, docs) est **À MESURER**. `machine-check.sh` n'est pas exécutable ici.
- **Chaque constat est vérifié en lisant le code** à `9392ed1` (fichier:ligne). Quand l'effet dépend du
  comportement de MLX, la règle est lue **dans la source C++ réellement résolue** : Voxtral dépend de mlx-swift
  `from: "0.31.6"` (`Package.swift:42`) ; le tag `0.31.6` = `0bb916c` épingle MLX C++ `ce45c52`
  (`git ls-tree 0.31.6 Source/Cmlx/mlx`, récupéré en lecture dans le scratchpad) ; les fichiers
  `mlx/fast.cpp`, `mlx/ops.cpp`, `mlx/dtype.cpp`, `mlx/backend/metal/allocator.cpp`,
  `mlx/backend/metal/scaled_dot_product_attention.cpp` ont été lus à `ce45c52`. mlx-swift-lm : `main@ee673d6`
  (Voxtral suit `branch: "main"`, `Package.swift:52`, sans `Package.resolved` — S-18).
- **Faits** : issues fermées #12 à #22 (corps **et** commentaires, via MCP GitHub en lecture), messages des
  commits de perf du 2026-04-11, `README.md`, `Scripts/CoreMLConversion/README.md`, Hub HF (listings et
  `config.json`, 2026-09-27). Ils sont repris tels quels en §2 avec leur source ; les conditions de mesure
  d'origine (Release/Debug, refroidissement, A/B/B/A) ne sont pas documentées : ce sont des mesures « en
  session » au sens de `measurement.md`, pas des références.
- **Articulation avec `audit-stabilite.md`** : S-01 (jeton d'arrêt `32000` = « ␣Capital »), S-06
  (`small-24b-8bit` sur deux dépôts), S-07 (`consolidated.safetensors` téléchargé en double), S-14 (famille de
  chargement héritée), S-17 (conformance `LanguageModel`) ne sont **pas** redupliqués ici ; ils faussent aussi
  les mesures (S-01 coupe les sorties, S-07 double le disque) et sont prérequis de la baseline (P-19).
  **P-03 complète et corrige S-02** : au-delà de la fenêtre, le défaut réel au préfill est un **arrêt du
  processus**, pas une troncature (§4, P-03, simulation en annexe A).
- **Consommateurs** (repris de l'audit stabilité, recherche GitHub en lecture) : FluxForge Studio (App Store,
  chaîne LipDub) consomme `VoxtralPipeline(.mini3b4bit)`, `ModelRegistry`, `ModelDownloader`… avec la
  configuration par défaut : **backend `.auto` (Core ML) et préréglage mémoire `.recommended()`** — c'est le
  chemin prioritaire de cet audit.

## 1. Carte du chemin STT (tel qu'exécuté par défaut)

```
fichier audio
 └─ loadAudio : AVAudioFile lu en entier au format natif, AVAudioConverter → Float32 16 kHz mono
    (VoxtralFeatureExtractor.swift:24-86)                                             [P-29]
 └─ processAudioForVoxtral : complétion à un multiple de 30 s, mel log 128×3000 **fp32**,
    deux passes STFT (max global puis normalisation) + .item() par tranche (:341-417) [P-21]
 └─ invite : bos, [INST], begin_audio, 375 × N jetons audio, [/INST], (lang), transcribe
    (VoxtralProcessor.swift:352-432)
 ├─ backend .mlx : VoxtralStandardEncoder (32 couches) sur **toutes** les tranches en un lot,
 │    projecteur 5120→3072 (→5120 Small), eval (VoxtralModeling.swift:723-763)      [P-01, P-13]
 └─ backend .auto/.hybrid (défaut bibliothèque) : Core ML `.cpuAndGPU`, une prédiction par tranche,
      sortie **Float32** [1,375,3072], ponts CPU (VoxtralHybridEncoder.swift:223-256)  [P-01, P-14]
 └─ fusion : cumsum/take/where → **fp32** (promotion) (VoxtralModeling.swift:903-982, 1461-1502)
 └─ préfill tranché 512 : LlamaStandardModel, masque additif **fp32** [T, offset+T] construit sur CPU,
    lm_head sur toutes les positions, eval(logits) par tranche (VoxtralModeling.swift:1163-1197,
    VoxtralStandardLoader.swift:383-481)                                  [P-02, P-03, P-06, P-10]
 └─ cache KV amont : RotatingKVCache(maxSize 2048…8192, keep 4) par défaut, KVCacheSimple sinon
    (VoxtralModeling.swift:1131-1147) — tampons **fp32**                               [P-03, P-10]
 └─ décodage : 1 jeton, pénalité de répétition 1,2 (boucle), argmax, `.item()` par pas, sans asyncEval ;
    tous les 2/4/8/16 pas : eval + (≤ 31 Go) clearCache + resetPeakMemory (:1198-1273)
                                                                     [P-07, P-08, P-12]
 └─ arrêt : [2, 4, 32000] (S-01), maxTokens 500 (P-11) ; fin : clearCache (:1277-1279)
```

Réglages effectifs par point d'entrée :

| Point d'entrée | Backend | maxTokens | Préréglage mémoire (KV) | Remarque |
|---|---|---|---|---|
| `VoxtralPipeline()` / FluxForge | `.auto` → Core ML si téléchargé (`VoxtralPipeline.swift:196`, `:283-296`) | 500 (`:109`) | `.recommended()` : < 16 Go `ultra` (KV 2 048), 16-31 `aggressive` (4 096), 32-63 `moderate` (6 144), ≥ 64 `light` (8 192) (`MemoryOptimizationConfig.swift:77-90`) | température 0, pénalité 1,2 (`:110-112`) |
| `VoxtralTranscriptionManager` | `.auto` (`VoxtralTranscriptionManager.swift:91-95`) | 500 | `.recommended()` | façade FluxForge |
| App `VoxtralApp` | `.auto` (`TranscriptionManager.swift:216-219`) | 500 (`:68`) | KV 8 192 imposé (`:70`, `:212`) | |
| CLI `transcribe` / `chat` | `.mlx` par défaut (`VoxtralCLI.swift:186`) | 500 (`:181`, `:292`) | `.recommended()` | |
| CLI `profile` (source des issues #12-#22) | `.auto` implicite (`ProfileCommand.swift:177`) | 500 (`:61`) | `.recommended()` (96 Go → `light`) | un passage à froid |

## 2. Faits déjà mesurés (repris tels quels, avec source)

| # | Fait | Source |
|---|---|---|
| F-01 | Mini 3B 8 bits, STT, trace de profil (backend Core ML) : préfill 3,59 s à 49 % GPU ; génération 30,6 tok/s (500 jetons, **maxTokens atteint**), 76 % GPU, pas moyen 38 ms, min 28,4 ms, max 3,6 s (= pas 1, préfill inclus) | issues #13, #15, #18 |
| F-02 | Chat Mini : 17,6 tok/s (108 jetons), pas moyen 85,8 ms ; après `41ce59d` (top-p par `top(k: 1000)`) : 33,5 tok/s ; Small chat 8,8 → 11,5 tok/s | #15, #20, commit `41ce59d` |
| F-03 | Retrait des synchronisations de debug : 30,6 → 33,5 tok/s (+9,5 %) Mini STT ; « le préfill reste à 49 % GPU » | commit `c1942ee`, commentaire #13 |
| F-04 | Mémoire MLX active pendant le préfill : 480 → 6 116 Mo (Mini), 473 → 15 575 Mo (Small) ; pic process 11,1 Go (Mini STT), 9,5 Go (chat), 22,0 / 20,0 Go (Small) ; après nettoyage 4 078 Mo (Mini), 13 276 Mo (Small) | #17, #19, #21 |
| F-05 | Préfill tranché 512 : pic MLX Mini 6 116 → 4 878 Mo (−20 %), Small 15 575 → 14 364 Mo (−8 %) ; 32,8 / 11,2 tok/s sans régression ; transcription identique | commit `1eb2cc9` |
| F-06 | Limites KV dans tous les préréglages : « pas de régression sur obama.mp3 (33,2 / 11,1 tok/s) ; **l'essentiel du pic est constitué des poids du modèle chargés pendant le préfill**, pas du cache KV » | commit `d6acbf1` |
| F-07 | Small 4 bits : préfill 19,13 s à 49 % GPU (STT), 18,29 s (chat) ; décodage STT 11,1 tok/s à 92 % GPU, pas moyen 126 ms | #19, #20 |
| F-08 | Extraction des features : 3,79 s (STT) / 3,47 s (chat) à 0 % GPU **avant** `70c390b` ; après : **254 ms** pour le même fichier (MP3 de 203 s, 44,1 kHz stéréo), message de `70c390b` (« Before: 3.79s … After: 254ms », « Fixes #12 ») — *amendé (vérification croisée)* : l'ordre est connu, 3,79 / 3,47 s sont obsolètes | #12, commit `70c390b` |
| F-09 | Encodage Core ML : 4,28 s (STT) / 2,37 s (chat) à 48 % GPU ; pont MLX↔Core ML ≈ 2 % | #14 et commentaire |
| F-10 | Mise en place de l'encodeur : à froid 1 min 09,6 s (Mini) / 2 min 25,2 s (Small), à chaud 1,41 / 2,97 s ; chargement du modèle 175 ms ; tokenizer 268 ms ; parallélisation tokenizer/Core ML ≈ −270 ms | #16, #22, commit `1528887` |
| F-11 | README (M3 Max 96 Go, conditions non documentées) : bf16 (« fp16 ») 90,1 s, 5,6 tok/s, pic GPU 15,26 Go ; 8 bits 34,6 s, 14,5 tok/s, 10,05 Go ; 4 bits mixte 28,2 s, 17,7 tok/s, 8,31 Go ; Small « 0,5 / 0,7 / 1,0 tok/s » (contredit F-07 : 11,1 tok/s) | `README.md` §STT |
| F-12 | Encodeur : « MLX ~500 ms, Core ML (ANE) ~150 ms » par fenêtre ; « GPU ~280 ms avec VoxtralEncoderFull » | `Scripts/CoreMLConversion/README.md` §Benefits ; `VoxtralCoreMLEncoder.swift:118-119` |
| F-13 | Poids (Hub, 2026-09-27) : `torch_dtype: bfloat16` ; Mini 8 bits 5,40 Go ; Mini 4 bits mixte 3,20 Go (tour audio 6 b gs64, attention LM 4 b, MLP des 2 premières/2 dernières couches 6 b, `embed_tokens` 4 b, `lm_head` 6 b gs128) ; Small 4 bits mixte 14,86 Go ; Mini bf16 (shards) 9,36 Go ; Core ML Mini 1,32 Go, Small 1,38 Go, entrée et **sortie Float32**, stockage fp16 | `hf_fs ls/cat` (`config.json`, `metadata.json`) |
| F-14 | Dernier ASR Mistral publié : `Voxtral-Mini-4B-Realtime-2602` (1,77 M téléchargements) ; aucun STT hors ligne plus récent que la génération 2507 | `hf_fs search hf://models/mistralai` |

Géométrie utile (config.json) : Mini LM 30 couches, 8 têtes KV × 128 ; Small 40 couches, 8 × 128 ; vocabulaire
131 072 ; encodeur 32 couches, 1 280, 20 têtes × 64 ; **375 jetons audio par fenêtre de 30 s (12,5 jetons/s)**.
Cache KV par jeton : Mini 61 440 éléments (fp32 240 Kio, bf16 120 Kio), Small 81 920 (fp32 320 Kio, bf16 160 Kio).

## 3. Catalogue T1…T23 appliqué au chemin STT

| T | Technique | État | Où / preuve | Constat |
|---|---|---|---|---|
| T1 | `Memory.cacheLimit` par étape | **absente** | 0 pose dans `VoxtralCore` ; seule pose : app, fonction morte, `0` puis `Int.max` (`VoxtralApp/TranscriptionManager.swift:285-296`) | P-09 |
| T2 | Limites adaptatives | **absente** | les « préréglages » par RAM règlent eval/clearCache/KV, jamais `cacheLimit`/`memoryLimit` (`MemoryOptimizationConfig.swift:32-90`) | P-08, P-09 |
| T3 | `clearCache()` après réponse | **appliquée** (et mal placée en plus) | `VoxtralModeling.swift:1279`, `:1453` ; app `TranscriptionManager.swift:396`, `:467` ; mais aussi **dans** la boucle (`:1252-1254`) | P-08 |
| T4 | Résidence par étape | **partielle, implicite** | poids paresseux : en hybride la tour audio MLX n'est jamais matérialisée ; en MLX pur elle reste résidente pendant tout le décodage | P-23, P-24 |
| T5 | Variante sans tour | **non applicable** (audio obligatoire en STT) ; l'équivalent « sans tour MLX » est l'hybride | `VoxtralPipeline.swift:346-362` | P-14 |
| T6 | Réutilisation de conversation | **n/a** transcription ; **absente** en chat | `VoxtralPipeline.swift:393-473` recalcule tout à chaque question | P-15 |
| T7 | Médias nouveaux seulement | **absente** (chat) | idem | P-15 |
| T8 | Budget de jetons média | **non applicable** : 375 jetons/30 s fixés par le modèle (`VoxtralProcessor.swift:389-396`) ; la dernière fenêtre est complétée par du silence (`VoxtralFeatureExtractor.swift:366`) — ne pas couper (analogue piège 23) | — | — |
| T9 | `prefillStepSize` | **partielle** : 512 codé en dur deux fois, non balayé, pas de « dernier logit seulement » | `VoxtralModeling.swift:1169`, `:1352` ; `:1177-1188` | P-06, P-22 |
| T10 | KV 8 bits (lean) | **absente** (`quantized-kv` = 0 au scan) | — | P-16 |
| T11 | KV préalloué, écriture en place | **appliquée par l'amont** (`KVCacheSimple`, pas 256, `slice_update` : mlx-swift-lm `KVCache.swift:408-464`) ; **mais** réallocation + copie complète à chaque tranche de préfill | `VoxtralModeling.swift:1145`, amont `:424-453` | P-10 |
| T12 | `lm_head` quantifié via `quantizedMatmul` | **appliquée** (packs : `lm_head` 8 b ou 6 b gs128 → `QuantizedLinear`) ; restriction n/a (sortie libre) ; en bf16 le head passe par une conversion fp32 | `VoxtralStandardLoader.swift:1223-1251` ; F-13 | P-05 |
| T13 | Quantification mixte par voie | **appliquée** via les packs « 4bit-mixed » (tour audio 6 b, head 6 b, MLP extrêmes 6 b) ; 8 bits uniforme | F-13 ; prédicat `MLXLMBridge.swift:604-675` | — |
| T14 | Dé-quantifier une étape bornée par le calcul | **absente** (encodeur audio 6/8 b, préfill) | — | P-27 |
| T15 | Pipelining `asyncEval` | **absente** (0 occurrence) | `.item()` par pas `VoxtralModeling.swift:1224`, `:1404` ; `eval` par tranche `:1187`, `:1368` | P-07 |
| T16 | `eval` par couche | **non nécessaire** au décodage (graphe d'un pas) ; encodeur évalué d'un bloc sans borne de lot (`:754`) ; chargement sans aucun `eval` | `VoxtralStandardLoader.swift:832-834` | P-13, P-04 |
| T17 | Fuites de dtype fp32 | **défaut majeur** : tout le chemin calcule en fp32 | voir P-01 | P-01, P-02, P-05 |
| T18 | `compile(shapeless:)` d'activation | **appliquée par l'amont** : `gelu` et `silu` de MLXNN sont compilés (`Activations.swift:212-213`, `244-245`, tag 0.31.6) ; rien d'autre à compiler (R1-R3) | — | — |
| T19 | Cache KV entre étapes | **non applicable** (une seule étape autorégressive) | — | — |
| T20 | Politique de calcul par étape | **absente** : mêmes poids/dtype pour encodeur, préfill, décodage | — | P-27 |
| T21 | Réduction du nombre de pas | **non applicable** (pas de solveur itératif) ; le levier voisin est fonctionnel (`maxTokens`) | — | P-11 |
| T22 | Lecture `pread`/`F_NOCACHE` | **absente** ; chargement paresseux (mmap MLX) : la lecture a lieu au premier préfill | `VoxtralStandardLoader.swift:986-1024`, `:1285-1344` | P-04 (suite) |
| T23 | Reprise / porte GPU iOS | **absente** ; `Package.swift` déclare iOS 17 ; cible iOS du STT non confirmée | `Package.swift:9-12` | ASK |

Rejets du catalogue à ne pas re-proposer : R1-R3 (compiler le pas de décodage), R12
(`MLX_MAX_OPS_PER_BUFFER`), R16 (`iogpu.wired_limit_mb`). R14 (« > 2× MLX ⇒ pas d'asset ») sert de règle de
décision pour l'hybride Core ML (P-14).

## 3 bis. Réponses aux questions du cadrage

1. **Masque causal fp32 `VoxtralLlama.swift:527-536` qui promeut les scores ?** Ce masque n'est utilisé que par le
   décodeur **hérité** (`VoxtralLlama.LlamaModel`, P-17). Le chemin réel utilise
   `VoxtralStandardLoader.swift:450-481`, lui aussi fp32. Le SDPA **ne promeut pas** selon le masque : il exige que
   le type du masque se promeuve vers le type de sortie, sinon il lève (`fast.cpp:796-800` à `ce45c52`). Le masque
   fp32 **verrouille** donc le calcul en fp32 (P-02) ; la promotion elle-même vient des features fp32 (P-01).
2. **Cache KV maison vs `KVCacheSimple` amont ?** Aucun cache maison sur ce chemin : `KVCacheSimple` /
   `RotatingKVCache` de mlx-swift-lm (`VoxtralModeling.swift:1139`, `:1145`). Les défauts sont d'usage : fenêtre
   par défaut incompatible avec le masque maison (P-03), croissance par tranche (P-10).
3. **Logits de toutes les positions au préfill ?** Oui, et forcés par `eval(chunkOutput.logits)` (P-06).
4. **`.item()` dans la boucle de décodage ?** Oui, un par pas (`:1224`, `:1404`), nécessaire au test d'arrêt mais
   sans décalage `asyncEval` (P-07). *Corrigé (vérification croisée)* : 18 des 49 `.item()` du scan sont dans les
   aides de debug de `MLXLMBridge.swift:326-535` et 12 dans celles de `VoxtralModeling.swift:650-845`
   (`debugModelWeights`, bloc `VoxtralDebug.enabled`, `dumpSwiftAudioFeatures`), soit 30 hors chemin ; 2 sur le
   chemin STT (`:1224`, `:1404`) et 2 dans l'extraction (`VoxtralFeatureExtractor.swift:322`, `:360`) ; le reste
   est hors périmètre (TTS, Realtime).
5. **`asyncEval` absent ?** Oui, partout (P-07).
6. **`cacheLimit` jamais posé ?** Jamais dans `VoxtralCore` ; l'app pose `0` puis `Int.max` dans une fonction jamais
   appelée (P-09).
7. **`eval` par couche dans l'encodeur ?** Non ; le graphe de 32 couches est évalué d'un bloc
   (`VoxtralModeling.swift:754`). Ce n'est pas un défaut en soi ; le défaut est le lot non borné (P-13).
8. **Audio long découpé en fenêtres de 30 s et encodé séquentiellement ?** Découpé en fenêtres de 30 s ; encodé en
   **un seul lot** `[N, 128, 3000]` en MLX (pic ∝ durée, P-13), **séquentiellement** fenêtre par fenêtre en Core ML
   (P-14). Surtout, l'invite complète (375 × N jetons) passe dans **un** préfill, ce qui déclenche P-03 au-delà de
   2 min 30 à 10 min 30 selon la RAM.

## 4. Constats

Format : id · sévérité · `fichier:ligne` — titre ; constat ; preuve ; correction ; gain attendu (source
catalogue) ; protocole et porte (voir le protocole commun §5) ; risque API ; effort ; statut ; fiche proposée
(objet + porte chiffrée + cible).

---

### P-01 · haute · `VoxtralModeling.swift:723-763`, `:979`, `:1497` ; `VoxtralFeatureExtractor.swift:298-335` — Tout le chemin STT calcule en fp32 (encodeur, projecteur, préfill, cache KV, décodage)

- **Constat** : les features mel sont en fp32 (audio `Float32` `VoxtralFeatureExtractor.swift:35-40`, STFT et
  filtres fp32 `:298-333`) et ne sont **jamais** castées vers le dtype du modèle (bf16, F-13) :
  `getAudioEmbeds` les passe telles quelles à l'encodeur (`VoxtralModeling.swift:730`), comme
  `encodeMLX` (`VoxtralHybridEncoder.swift:265`, branche pratiquement inatteignable depuis `VoxtralPipeline`, qui
  ne passe par l'hybride que si Core ML est disponible). Le chemin Core ML rend du **Float32** (F-13 ;
  branche `.float32` de `MLXCoreMLBridge.toMLXArray`, `MLXCoreMLBridge.swift:177-180` — *amendé : l'audit citait
  `:170-173`, qui est la branche `int8`*). La fusion `which(mask, audio, texte)` (`VoxtralModeling.swift:979`, `:1497`)
  promeut alors les embeddings texte bf16 en fp32 ; tout le décodeur suit, et le cache KV est **alloué en fp32**
  (dtype des premières clés écrites). Au décodage, le jeton bf16 est réécrit dans ce tampon fp32 et l'attention
  repasse en fp32 dès la couche 0 : les 30 (Mini) / 40 (Small) couches décodent en fp32.
- **Preuve** (MLX `ce45c52`) : conv `ops.cpp:4131` (`out_type = promote_types(in, wt)`) ; `quantized_matmul`
  `ops.cpp:4343` (`promote_types(x, scales)`) ; `matmul` `ops.cpp:3069-3082` ; `where` `ops.cpp:1728` ;
  `slice_update` `ops.cpp:843` (mise à jour castée au dtype du tampon) ; SDPA `fast.cpp:704`
  (`final_type = result_type(q, k, v)`) ; table `dtype.cpp:45`, `:48` : **float16 × bfloat16 = float32** (la
  sortie Core ML fp16 éventuelle ne suffirait donc pas). Tampon KV : mlx-swift-lm `KVCache.swift:439-440`
  (`zeros(..., dtype: keys.dtype)`). **Preuve indépendante** : le préfill passe un masque tableau fp32 (P-02) ;
  `fast.cpp:796-800` lève « Mask type must promote to output type » si q/k/v étaient bf16 — le chemin qui
  fonctionne en production est donc nécessairement fp32.
- **Correction** : (1) d'abord P-02 (sinon l'étape 2 fait lever le SDPA) ; (2) caster les features vers un dtype
  de calcul explicite, lu sur un poids jamais quantifié (ex. `languageModel.model.norm.weight.dtype`, pas
  `weight.dtype` d'un `Linear` — MLX-009) à l'entrée de `getAudioEmbeds` et `encodeMLX` ; (3) caster les
  embeddings audio vers `embeddings.dtype` avant `which` dans les deux fusions (couvre la sortie Core ML
  Float32) ; (4) test de dtype : après un préfill, `cache[0]` doit être bf16 (échoue avant correction).
- **Gain attendu** : cache KV divisé par 2 (Mini 240 → 120 Kio/jeton ; Small 320 → 160 ; à 10 min d'audio,
  7 500 jetons : Mini 1,72 → 0,86 Gio, Small 2,29 → 1,14 Gio) ; GEMM de l'encodeur et du préfill (bornés calcul)
  en bf16 ; lecture KV au décodage ÷ 2. Source : catalogue T17 (Qwen38 : fuite fp32 corrigée, greedy ×2,74 sur
  une autre architecture — **ordre de grandeur non transposable**) et piège 26 / MLX-002. Pour les poids bf16,
  voir P-05 (effet bien plus fort). *Nuance (vérification croisée)* : le gain de **calcul** dépend de la puce et
  reste À MESURER (sur les GPU Apple sans accélérateur matriciel, le débit ALU bf16 n'est pas nécessairement
  supérieur au fp32) ; le gain **mémoire** (KV, activations) est mécanique. Sur une puce NAX (génération M5), le
  fp32 exclut en plus l'attention du chemin NAX : `metal::is_nax_available() && … (env::enable_tf32() ||
  q.dtype() != float32)` (`scaled_dot_product_attention.cpp:177-178` à `ce45c52`).
- **Protocole / porte** : §5, Mini 8 bits et 4 bits mixte, backends `.mlx` et `.auto`, clips C-moyen et C-long.
- **Risque API** : aucun. **Effort** : S (4 lignes) + test. **Statut** : VÉRIFIÉ (lecture + règles MLX) ; gain
  À MESURER.
- **Fiche proposée** — *Calcul bf16 de bout en bout*. **Porte** : dtype du cache KV = bf16 après préfill (test
  rouge avant, vert après) ; encodage + préfill ≥ 5 % plus rapides **ou** pic MLX −≥ 10 % (A/B/B/A) ; parité :
  transcription greedy identique sur ≥ 90 % des clips et WER ≤ WER_ref + 0,3 pt (FR et EN) ; sinon politique
  mixte (encodeur fp32, décodeur bf16) documentée et mesurée. Prérequis P-02, P-19. **Cible : macos-gpu.**

### P-02 · haute · `Utils/VoxtralStandardLoader.swift:450-481`, `:692-698` — Masque additif fp32 reconstruit sur CPU à chaque tranche : il verrouille le fp32 et ne suit pas les caches amont

- **Constat** : `LlamaStandardModel.createCausalAttentionMask` construit à chaque appel (T > 1) deux grilles
  d'indices par des tableaux Swift (`MLXArray((offset..<(offset+T)).map { Float($0) })`, `:468-469`), puis un
  masque **additif fp32** `where(futur, -inf, 0)` (`:476-478`), passé en tableau à
  `scaledDotProductAttention(..., mask:)` (`:692-698`). Conséquences : (a) le SDPA refuse un masque fp32 avec des
  q/k/v bf16 (`fast.cpp:796-800`) : impossible de corriger P-01 sans corriger ce masque ; (b) la forme
  `[T, offset+T]` suppose un cache qui renvoie exactement `offset+T` clés, faux pour `RotatingKVCache` (P-03) ;
  (c) coût CPU O(offset) + un tableau `[512, L]` fp32 par tranche (≈ 16 Mo à L = 8 192).
- **Preuve** : lecture ; `fast.cpp:796-807` à `ce45c52` ; le noyau fusionné accepte un masque tableau pour
  head_dim 128 (`scaled_dot_product_attention.cpp:625-629`), donc pas de repli non fusionné : le coût est le
  verrou de dtype et la forme, pas le noyau. Le scan MLX-002 ne signale que `:477` et **pas** `:476`
  (`MLXArray(-Float.infinity)`) ni les grilles `:468-469`.
- **Correction** : utiliser l'aide amont `createAttentionMask(h:cache:)` (mlx-swift-lm `KVCache.swift:376-397` :
  `.causal` en mode noyau, ou masque **booléen** via `cache.makeMask` pour les caches à fenêtre) et l'appel SDPA à
  `ScaledDotProductAttentionMaskMode` ; `.none` au décodage (T = 1). Le mode `.causal` est aligné en bas à droite
  (`offset = kL − qL`, `fast.cpp` repli causal), correct pour le préfill tranché.
- **Gain attendu** : débloque P-01 ; supprime la construction CPU par tranche ; corrige P-03. Source : amont
  mlx-swift-lm (même règle pour tous ses modèles), piège 26.
- **Risque API** : aucun (`createCausalAttentionMask` est `private`). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Masque d'attention amont*. **Porte** : logits du dernier jeton identiques (écart relatif L2
  ≤ 1e-3) sur 3 clips, greedy identique 100 % ; préfill ±5 % (neutre attendu) ; test : un préfill à entrée bf16 ne
  lève plus. **Cible : macos-gpu.**

### P-03 · haute · `VoxtralModeling.swift:1135-1147`, `:1169-1188` ; `MemoryOptimizationConfig.swift:41-90` ; `VoxtralStandardLoader.swift:460-469` — Au-delà de `maxKVCacheSize`, le préfill **arrête le processus** (complète et corrige S-02)

- **Constat** : tous les préréglages posent `maxKVCacheSize` (commit `d6acbf1`), donc `RotatingKVCache(maxSize,
  keep: 4)` par défaut ; l'app impose 8 192. Au préfill tranché, `RotatingKVCache.updateConcat` rend
  `min(idx, maxSize − 1) + T` clés (mlx-swift-lm `KVCache.swift:691-714`), alors que le masque maison est
  `[T, offset + T]` avec `offset` = total déjà traité. Dès qu'une tranche de T ≥ 2 démarre à `offset ≥ maxSize`,
  masque et clés diffèrent d'**une** colonne : `broadcast_to` échoue dans `fast.cpp:805-807`, l'erreur MLX part au
  gestionnaire de mlx-swift qui, sans gestionnaire de tâche ni global, appelle **`fatalError(message)`**
  (`ErrorHandler.swift:345`, tag 0.31.6 ; *amendé : l'audit citait le commentaire `:4` « print … then exit », qui
  décrit l'ancien gestionnaire C `mlx_error_handler_default_` remplacé par mlx-swift, `:283`*), et
  Voxtral n'utilise jamais `withError` ni `withErrorHandler` (grep : 0). Invite = 375 × N + ≈ 5-8 : arrêt à partir de 6 fenêtres
  (> 2 min 30) sous 16 Go, 11 (> 5 min) de 16 à 31 Go, 17 (> 8 min) de 32 à 63 Go, 22 (> 10 min 30) à partir de
  64 Go et dans l'app. Même backend Core ML (même décodeur). Quand l'invite tient mais que invite + sortie
  dépasse la fenêtre, la rotation fait sortir le début de l'audio pendant la génération (effet décrit par S-02).
- **Preuve** : lecture + simulation exacte de la logique amont (annexe A) : `2048 → 2258 jetons : tranche @2048,
  masque [210, 2258] vs clés 2257` ; `8192 → 8258 : masque [66, 8258] vs clés 8257`. *Vérification croisée* :
  simulation réécrite indépendamment (scratchpad `verif_sim.py`) : mêmes seuils, première invite en échec à 6 / 11 /
  17 / 22 fenêtres pour 2 048 / 4 096 / 6 144 / 8 192. *Amendé* : depuis la vérification croisée de l'audit
  stabilité, **S-02 décrit déjà cet arrêt** ; P-03 et S-02 sont un seul constat, une seule fiche (K-S02 = P-03) —
  « corrige S-02 » ne vaut plus. Le clip du dépôt
  `docs/examples/fluxforge_long_en_6bit.wav` (167,0 s → 6 fenêtres, 2 250 jetons audio) suffit à déclencher
  l'arrêt en préréglage `ultra`.
- **Correction** : STT sans fenêtre : `KVCacheSimple` pré-dimensionné (P-10) ; mémoire tenue par bf16 (P-01), KV
  8 bits (P-16) et `cacheLimit` (P-09), pas par une fenêtre ; P-02 rend le masque correct pour les caches à
  fenêtre restants (chat texte) ; garde-fou : erreur Swift claire si invite + maxTokens > limite voulue.
  Test de non-régression : invite synthétique de 2 600 jetons + `.ultra` (échoue avant, passe après).
- **Gain attendu** : suppression d'un arrêt fatal des hôtes (FluxForge) sur audio long ; KV bf16 à 30 min : Mini
  2,58 Gio, Small 3,43 Gio (au lieu de fp32 5,15 / 6,87 Gio). Source : catalogue piège 29/36 (caches glissants
  dimensionnés à la longueur du run), T11.
- **Risque API** : aucun en signature ; changement de comportement des préréglages publics (documenté).
  **Effort** : M. **Statut** : VÉRIFIÉ (lecture + simulation) ; reproduction sur Mac À MESURER.
- **Fiche proposée** — *Cache KV sans fenêtre en STT + garde-fou* (à fusionner avec K-S02). **Porte** : clip C-long
  (≈ 12 min) en préréglages 8 et 16 Go simulés : plus d'arrêt ; test 2 600 jetons rouge → vert ; WER ≤
  WER(KVCacheSimple) + 0,5 pt ; pic `phys_footprint` consigné (décision ASK s'il dépasse le budget du profil).
  **Cible : macos-gpu.**

### P-04 · moyenne (*amendé : haute → moyenne*) · `Utils/VoxtralStandardLoader.swift:1285-1344` ; `Pipeline/VoxtralPipeline.swift:237-241` — Poids jamais matérialisés au chargement : le premier préfill paie la lecture disque ; les mesures de préfill (#13, #17, #19, #21) sont contaminées

- *Vérification croisée* : constat confirmé (aucun `eval` dans `VoxtralStandardLoader.swift`, ni dans
  `loadWeights` `:986-1024` ni après `update(parameters:)` `:1323`/`:1338` ; issue #17 : « Prefill begin 480 MB »,
  ce qui correspond à la seule table `embed_tokens` 8 bits ≈ 0,43 Go matérialisée par l'`eval` de la fusion
  `VoxtralModeling.swift:1499`). **Sévérité rétrogradée** : le défaut ne coûte ni débit ni temps total (il déplace la
  lecture du chargement vers la première requête) ; sa valeur est méthodologique (baseline P-19) et d'UX (premier
  TTFT), pas une perte de performance.

- **Constat** : `loadVoxtralStandardModel` charge les safetensors (MLX : paresseux), applique `update(parameters:)`
  et rend le modèle **sans aucun `eval`** ; `VoxtralPipeline.loadModel` non plus. La lecture des 3,2 à 14,9 Go de
  poids se produit donc au premier préfill. F-10 (« chargement 175 ms ») et F-06 (« l'essentiel du pic = poids
  chargés pendant le préfill ») le confirment ; F-04 note « Prefill begin 480 MB (weights loaded) » alors que le
  pack Mini 8 bits pèse 5,40 Go. Le diagnostic « 49 % GPU systémique, allocations Metal » (commentaires #13,
  #14, #19) est donc **non établi** : la phase « préfill » mesurée mélange E/S disque, matérialisation et calcul.
- **Preuve** : lecture ; F-04, F-06, F-10 ; catalogue piège 18 (`activeMemory` ne bouge pas après un chargement
  paresseux).
- **Correction** : après `update(parameters:)`, matérialiser **par module** (couche par couche, piège 4) les seuls
  poids qui serviront : `languageModel.*` toujours ; `audioTower.*`/`multiModalProjector.*` seulement en backend
  MLX ; **jamais** les modules factices (P-24). Mesurer ensuite la baseline avec une requête d'amorçage exclue.
  Suite possible (T22) : lecture concurrente des shards si le chargement devient visible (Small).
- **Gain attendu** : premier TTFT honnête (la lecture passe dans « chargement », où la barre de progression
  l'annonce) ; mesures de préfill exploitables. Pas de gain de débit à chaud. Source : piège 18, `measurement.md`
  (« mesures à chaud »).
- **Risque API** : aucun (temps de `loadModel` plus long, attendu). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Matérialiser les poids résidents au chargement, filtrés par voie*. **Porte** : premier
  préfill = préfill à chaud ± 5 % (A/A sur 2 requêtes) ; `activeMemory` après chargement = poids LM (+ tour audio
  en `.mlx`) ± 5 % ; en `.auto`, tour audio MLX non matérialisée ; aucun tenseur aléatoire matérialisé.
  **Cible : macos-gpu.**

### P-05 · haute · `ops.cpp:3069-3082` (MLX `ce45c52`) ; `VoxtralStandardLoader.swift:209-217`, `:1285-1288` — Modèles bf16 (profil 16 bits) : chaque `Linear` convertit son poids en fp32 à chaque appel

- **Constat** : conséquence de P-01 pour les checkpoints non quantifiés (`mini-3b`, `small-24b`) : `Linear` fait
  `matmul(x, weight.T)` (MLXNN `Linear.swift:127-129`, tag 0.31.6) et `matmul` caste l'opérande bf16 vers fp32
  (`ops.cpp:3069-3082`) : **copie fp32 transitoire de chaque matrice, à chaque pas**. Au décodage : lecture bf16
  (2 o/param) + écriture fp32 (4 o) + GEMV fp32 (4 o) ≈ **5×** le trafic d'un GEMV bf16 ; copie transitoire du
  `lm_head` : 131 072 × 3 072 × 4 o = 1,61 Go (Mini), 2,68 Go (Small). Cohérent avec F-11 : bf16 5,6 tok/s contre
  14,5 en 8 bits (×2,6, alors que le rapport d'octets de poids est ≈ ×1,9) et pic 15,26 Go. De plus
  `loadVoxtralStandardModel(dtype:)` **ignore** `dtype` (jamais lu, `:1285-1344`) ; le pipeline passe `.float16`
  (`VoxtralPipeline.swift:239`) et le README appelle « fp16 » un modèle bf16.
- **Preuve** : lecture MLX + MLXNN ; F-11 ; F-13 (`torch_dtype: bfloat16`).
- **Correction** : celle de P-01 (supprime la promotion) ; renommer « fp16 » en « bf16 » (README, registre) ;
  retirer ou implémenter `dtype:` (S-22).
- **Gain attendu** : décodage bf16 nettement plus rapide (borne bande passante ÷ ~5 sur les poids) et pic −1,6 Go
  (Mini) ; source : T17 + mécanisme `matmul`. **À MESURER.**
- **Risque API** : aucun. **Effort** : S (inclus dans P-01) + doc. **Statut** : VÉRIFIÉ (mécanisme) ; gain À MESURER.
- **Fiche proposée** — *Profil 16 bits sans conversion fp32*. **Porte** : `mini-3b` bf16 : décodage ≥ +5 % et pic
  process −≥ 1 Go (A/B/B/A), parité greedy/WER comme P-01. **Cible : macos-gpu.**

### P-06 · moyenne · `VoxtralModeling.swift:1011-1019`, `:1177-1196`, `:1360-1377` — Le préfill calcule les logits de **toutes** les positions et les force par `eval(logits)`

- **Constat** : `callAsFunction` applique `lm_head` à tous les états cachés ; chaque tranche fait
  `eval(chunkOutput.logits)` ; le préfill court (≤ 512) calcule aussi tout, puis ne garde que la dernière ligne
  (`:1210-1212`). Mini : `lm_head` = 2 × 3 072 × 131 072 = 0,81 GFLOP/jeton contre 6,42 GFLOP/jeton pour les
  30 couches (≈ 11 % du calcul de préfill, hors attention) ; Small ≈ 3 %. Tampon de logits par tranche de 512 :
  512 × 131 072 × 4 o = 256 Mio en fp32 (128 Mio en bf16).
- **Preuve** : lecture ; amont mlx-swift-lm `LLMModel.swift:41-58` : les tranches n'évaluent que le cache
  (`asyncEval(cache)`), la dernière position passe au `TokenIterator`.
- **Correction** : par tranche, `eval`/`asyncEval` des états du cache seulement ; dernière tranche : découper
  l'état caché à la dernière position **avant** `lm_head`.
- **Gain attendu** : préfill −≈ 10 % de calcul (Mini), −≈ 3 % (Small) ; pic −256 Mio (fp32) par tranche. Source :
  T9 (`keepLastOnly`, YuE2 `TokenGenerator.swift:15-36`), piège 27, MLX-006.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ ; gain À MESURER.
- **Fiche proposée** — *Dernier logit seulement au préfill*. **Porte** (*amendée*) : préfill Mini −≥ 5 % **ou** pic
  −≥ 128 Mio à 5 min d'audio (A/B/B/A) ; logits du dernier jeton : écart relatif L2 ≤ 1e-5 (fp32) / ≤ 1e-3 (bf16) ;
  greedy identique 100 %. **Cible : macos-gpu.** *Raison de l'amendement* : « bit-exact » n'est pas tenable — le
  `lm_head` passe d'une matrice de T lignes (noyau `qmm`/GEMM) à une seule ligne (noyau `qmv`/GEMV), ordre
  d'accumulation différent ; la porte bit-exacte rejetterait une correction saine.

### P-07 · moyenne · `VoxtralModeling.swift:1187`, `:1224`, `:1368`, `:1404` — Aucune mise en pipeline : préfill et décodage synchrones (pas d'`asyncEval`)

- **Constat** : chaque tranche fait un `eval` bloquant ; chaque pas construit le graphe, échantillonne puis appelle
  `.item()` : le GPU attend pendant que le CPU prépare le pas suivant. F-01 : 76 % GPU en génération Mini,
  pas médian ≈ 30-38 ms. Indicateur de bande passante tirée (calcul d'ordre de grandeur) : Mini 8 bits ≈ 4 Go lus
  par pas × 33,5 tok/s ≈ 135 Go/s ; Small 4 bits mixte ≈ 13,9 Go × 11,1 ≈ 155 Go/s, soit ≈ 45-50 % des 300 Go/s d'un
  M3 Max 96 Go (variante 30 cœurs GPU — à confirmer par `machine-check`), contre ≈ 71 % atteints dans Qwen38.
- **Preuve** : lecture ; scan : `async-eval` = 0 ; amont `Evaluate.swift:882`, `:894`, `:950`.
- **Correction** : motif décalé d'un pas : construire y(n+1) depuis y(n) paresseux, `asyncEval(y(n+1))`, puis
  `y(n).item()` et test d'arrêt sur y(n) ; `asyncEval(cache)` par tranche de préfill. Vérifier que la branche
  asynchrone est réellement prise (piège 6). Ou migration P-18. *Amendé (vérification croisée)* : la réécriture de
  la boucle réinjecte directement `nextToken` au lieu de reconcaténer `generated` pour en relire le dernier jeton
  (`:1239`/`:1200`, `:1417`/`:1381`) — détail repris de l'ex-P-25, écarté (annexe B).
- **Gain attendu** : −13 à −22 % ms/pas dans Qwen38 (T15) — ordre de grandeur non transposable. **À MESURER.**
- **Risque API** : aucun. **Effort** : M. **Statut** : VÉRIFIÉ (absence) ; gain À MESURER.
- **Fiche proposée** — *Pipeline asyncEval*. **Porte** : ms/pas médian −≥ 5 % (Mini 8 bits, C-moyen, A/B/B/A) ;
  sortie greedy identique 100 % ; preuve que le chemin asynchrone est exécuté. **Cible : macos-gpu.**

### P-08 · moyenne · `VoxtralModeling.swift:1247-1259`, `:1424-1436` ; `MemoryOptimizationConfig.swift:56-70` ; `VoxtralPipeline.swift:204` — Les préréglages vident le cache MLX tous les 2 ou 4 jetons sur les Mac ≤ 31 Go, et la configuration est globale

- **Constat** : `aggressive` (16-31 Go) et `ultra` (< 16 Go) activent `clearCacheOnEval` avec `evalFrequency` 4 et 2 :
  toutes les 2-4 étapes de décodage, `eval(generated)` (redondant : `.item()` a déjà synchronisé) puis
  `Memory.clearCache()` rend tous les tampons à Metal, réalloués au pas suivant. Tous les préréglages sauf
  `disabled` remettent aussi le pic à zéro dans la boucle (`:1256-1258`), ce qui fausse toute lecture de
  `peakMemory`. Enfin `VoxtralPipeline.init` écrit la configuration dans le singleton
  `VoxtralMemoryManager.shared` (`:204`) et `generateStream` lit ce singleton (`VoxtralModeling.swift:1119` ; le
  pipeline ne passe jamais `memoryOptimization`, `:353-374`) : le dernier pipeline créé impose sa politique
  d'`eval`/`clearCache`/`resetPeakMemory` aux autres (S-11). *Amendé (vérification croisée)* : la taille du cache
  KV, elle, reste propre à chaque pipeline (passée par `contextSize:`, `VoxtralPipeline.swift:360`, `:373`, `:451`,
  `:463`) ; seul le rythme de nettoyage est partagé. `optimizeIfNeeded(tokenIndex: 0)` après chaque requête (`:332`, `:408`) n'a pas d'effet utile.
- **Preuve** : lecture ; catalogue R13 (limites mobiles fixes : +32 % en 4 bits, +73 % en bf16) et T2.
- **Correction** : aucun `clearCache`/`eval`/`resetPeakMemory` dans la boucle ; `clearCache` après la réponse (déjà
  fait) et après l'encodage ; limites MLX par profil (P-09) ; configuration passée par appel, pas par singleton.
- **Gain attendu** : décodage plus rapide sur les Mac 8-31 Go (les plus courants) ; source R13/T2. **À MESURER.**
- **Risque API** : aucun en signature ; sémantique des préréglages publics modifiée (documentée). **Effort** : S.
  **Statut** : VÉRIFIÉ ; gain À MESURER.
- **Fiche proposée** — *Sortir le nettoyage mémoire de la boucle*. **Porte** : `recommended(forRAMGB: 16)` sur le Mac
  de mesure : décodage ≥ +5 % (A/B/B/A), pic `phys_footprint` ≤ +10 %, greedy identique. **Cible : macos-gpu.**

### P-09 · moyenne · `VoxtralApp/TranscriptionManager.swift:285-296` ; `allocator.cpp:52-54` (MLX `ce45c52`) — `Memory.cacheLimit` jamais posé par la bibliothèque ; la seule pose le rend illimité

- **Constat** : aucune pose dans `VoxtralCore`. La limite par défaut de MLX vaut `block_limit =
  min(1,5 × max_recommended_working_set, 0,95 × RAM)` (`allocator.cpp:52-54`), soit quasiment toute la mémoire.
  L'app pose `cacheLimit = 0` puis `Int.max` « Restore default (unlimited) » dans `aggressiveMemoryCleanup()`,
  fonction jamais appelée ; `Int.max` n'est d'ailleurs **pas** le défaut. F-04 : pic process 11,1 Go pour 6,1 Go
  actifs (Mini) ; 22,0 Go pour 15,5 Go (Small). Le préfill tranché crée des tampons de tailles toutes différentes
  (P-10), exactement le cas qui a donné 74 Go dans Qwen38 (T1).
- **Preuve** : lecture ; grep ; F-04 ; catalogue T1, piège 7, pattern MLX-010 (resté muet ici, voir retours skill).
- **Correction** : `Memory.cacheLimit` posé **après** chargement par profil (fast : quelques Go, à mesurer ; lean :
  `min(1 Go, max(256 Mo, dispo/6))`) et `memoryLimit` adaptatif en lean (T2) ; supprimer la fonction de l'app.
- **Gain attendu** : pic `phys_footprint` ramené vers `actif + cacheLimit` ; temps inchangé en fast. **À MESURER.**
- **Risque API** : additif (réglage de profil). **Effort** : S. **Statut** : VÉRIFIÉ ; gain À MESURER.
- **Fiche proposée** — *cacheLimit par profil*. **Porte** : à 10 min d'audio, pic `phys_footprint` ≤ pic actif MLX +
  cacheLimit + empreinte Core ML (+5 %) ; temps fast ±5 % ; lean ≤ budget du profil. **Cible : macos-gpu.**

### P-10 · moyenne · `VoxtralModeling.swift:1145` ; mlx-swift-lm `KVCache.swift:411`, `:424-453`, `:691-714` — Le cache KV est réalloué et recopié en entier à chaque tranche de préfill

- **Constat** : `KVCacheSimple` grandit par pas de 256 : une tranche de 512 dépasse toujours la capacité restante,
  d'où allocation d'un nouveau tampon et `concatenated([ancien, nouveau])` : **copie de tout le cache à chaque
  tranche** (O(n²/512)), et une nouvelle taille de tampon par tranche et par couche (60 tampons K/V par tranche).
  Même chose pour `RotatingKVCache.updateConcat` jusqu'à la fenêtre. La longueur finale (invite + maxTokens) est
  pourtant **connue avant le préfill**.
- **Preuve** : lecture amont (`step` public, `:411`). Ordre de grandeur : 30 min d'audio, Mini fp32 : 44 tranches,
  ≈ 119 Go de copies cumulées ; surtout 44 tailles distinctes retenues par le cache MLX (alimente P-09).
- **Correction** : `KVCacheSimple` avec `step` = arrondi(invite + maxTokens, 256) avant le préfill, puis `step = 256`
  (une seule allocation, écriture en place).
- **Gain attendu** : préfill et pic mémoire plus bas sur audio long ; sortie bit-exacte. Source : T11 (YuE2
  −16 % en phase AR), T1. **À MESURER.**
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ ; gain À MESURER.
- **Fiche proposée** — *Cache KV pré-dimensionné*. **Porte** (*amendée*) : **30 min** d'audio : préfill −≥ 5 % **ou**
  pic `phys_footprint` −≥ 10 % (A/B/B/A) ; 10 min consigné ; sortie bit-exacte. Prérequis P-03. **Cible :
  macos-gpu.** *Raison* : par le calcul même de l'audit, à 10 min (15 tranches) les copies cumulées ne font que
  ≈ 13 Go (512 × 105 × 240 Kio) ; même à 30 min (44 tranches, ≈ 119 Go, recalcul confirmé) leur temps
  (≈ 0,5 s à 250 Go/s) reste probablement sous 5 % d'un préfill de plusieurs dizaines de secondes. Le gain
  plausible est le **pic** : pendant chaque `concatenated`, l'ancien et le nouveau tampon coexistent (jusqu'à ≈ 2×
  le cache en fin de préfill, soit ≈ +5 Gio transitoires à 30 min en fp32 Mini, calcul, À MESURER). Le critère de
  temps est donc attendu en échec ; c'est le critère de pic qui doit trancher, à 30 min.

### P-11 · moyenne · `VoxtralPipeline.swift:109`, `:118` ; `VoxtralCLI.swift:181`, `:292` ; `VoxtralApp/TranscriptionManager.swift:68` ; `ProfileCommand.swift:61` — `maxTokens = 500` par défaut : transcriptions tronquées, et mesures faites sur des sorties tronquées

- **Constat** : 500 jetons ≈ 3 min de parole. Tout audio plus long est coupé sans signal. F-01 : la trace STT de
  référence « atteint maxTokens » : le débit publié est celui d'une transcription tronquée (le débit par pas reste
  valable, pas le temps total ni le WER).
- **Preuve** : lecture ; F-01, F-07 (« 500 tokens » en STT).
- **Correction** : `maxTokens` = f(durée audio) (taux jetons/seconde de parole mesuré sur le corpus + marge,
  garde-fou anti-boucle déjà présent `:1266-1272`) ; exposer la valeur dans le profil.
- **Gain attendu** : fonctionnel (sortie complète) ; mesures comparables. **Risque API** : défaut modifié
  (comportement). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *maxTokens proportionnel à la durée*. **Porte** : clips C-moyen (167 et 174 s) et C-long :
  dernière phrase de la référence présente ; longueur ≤ 1,5 × la référence. **Cible : macos-gpu** (doc en cloud).

### P-12 · moyenne · `VoxtralPipeline.swift:112`, `:121` ; `VoxtralModeling.swift:1216-1219`, `:1553-1588` — Pénalité de répétition 1,2 appliquée au décodage greedy de transcription, en boucle scalaire

- **Constat** : par défaut, à chaque pas, jusqu'à 20 jetons distincts reçoivent une mise à jour scalaire des logits
  (tranche, comparaison, multiplication, division, `where`, `slice_update` : ≈ 120 nœuds de graphe par pas). En
  transcription greedy, pénaliser les 20 derniers jetons (mots-outils, ponctuation) peut changer l'argmax et donc
  le texte.
- **Preuve** : lecture. **Correction** : (1) A/B WER 1,0 contre 1,2 par langue ; (2) si la pénalité reste,
  implémentation vectorisée (un `take` + `where` + une mise à jour) ou processeur amont
  (`GenerateParameters.repetitionPenalty`, `Evaluate.swift:171`).
- **Gain attendu** : qualité (À MESURER) ; coût CPU par pas réduit (probablement < 5 %). **Risque API** : défaut
  modifié. **Effort** : S. **Statut** : implémentation VÉRIFIÉE ; effet WER À MESURER.
- **Fiche proposée** — *Pénalité de répétition tranchée par le WER*. **Porte** : retenir la valeur au WER le plus bas
  si l'écart ≥ 0,3 pt (sinon statu quo) ; version vectorisée : logits bit-exacts et ms/pas −≥ 5 %, sinon non
  retenue. **Cible : macos-gpu.**

### P-13 · moyenne · `VoxtralModeling.swift:723-755` ; `CoreML/VoxtralHybridEncoder.swift:259-286` — Encodeur MLX : toutes les fenêtres de 30 s en un seul lot, sans borne

- **Constat** : `[N, 128, 3000]` passe d'un bloc dans les 32 couches ; les activations transitoires croissent avec
  la durée. 30 min (N = 60), fp32 : sortie `fc1` 60 × 1 500 × 5 120 × 4 o = 1,84 Go, Q/K/V ≈ 1,38 Go (moitié en
  bf16). Le SDPA de l'encodeur est fusionné (head_dim 64, sans masque) : pas de matrice de scores.
- **Correction** : lots de K fenêtres (K à balayer : 2/4/8) avec `eval` entre lots, concaténation des embeddings
  (375 × 3 072 par fenêtre).
- **Gain attendu** : pic d'encodage borné, temps neutre. Source : esprit T16 (graphe borné). **À MESURER.**
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ ; gain À MESURER.
- **Fiche proposée** — *Encodeur par lots bornés*. **Porte** (*amendée*) : 30 min d'audio : pic MLX de l'encodage
  −≥ 30 % ; temps d'encodage ±5 % ; embeddings bit-exacts attendus, à défaut écart relatif L2 ≤ 1e-5 et greedy
  identique 100 %. **Cible : macos-gpu.** *Raison* : changer la dimension M des GEMM (N × 1 500 → K × 1 500 lignes)
  peut changer la variante de noyau choisie ; l'égalité stricte n'est pas garantie par construction.
  *Note (vérification croisée)* : ce chemin n'est pris qu'en backend `.mlx` (défaut de la CLI, `VoxtralCLI.swift:187`),
  pas en `.auto` (Core ML) ; `encodeMLX` (`VoxtralHybridEncoder.swift:259-286`) est pratiquement inatteignable depuis
  le pipeline.

### P-14 · moyenne · `CoreML/VoxtralCoreMLEncoder.swift:119-160`, `:211-224` ; `VoxtralHybridEncoder.swift:33-35`, `:223-256` ; `MLXCoreMLBridge.swift:90-189` ; `VoxtralPipeline.swift:196`, `:283-296` — Hybride Core ML : défaut de la bibliothèque, jamais comparé à un encodeur MLX bf16

- **Constat** : (a) tous les préréglages Core ML utilisent `.cpuAndGPU` (le libellé dit « Neural Engine ») ;
  `withHuggingFaceDownload` crée l'encodeur avec `VoxtralCoreMLConfig.default` (variante `mini`, même pour Small,
  sans effet sur ce chemin : la variante ne sert qu'à la découverte automatique, `VoxtralCoreMLEncoder.swift:239-240`) ; (b) une prédiction par fenêtre, pas de lot ; (c) *amendé* : pont
  d'entrée élément par élément (`MLXCoreMLBridge.swift:120-126`, après une copie `asArray`) ; pont de sortie
  Float32 = deux copies en bloc (`Array(UnsafeBufferPointer)` puis `MLXArray`, `:177-180`) — la boucle élément par
  élément `:182-189` est la branche fp16, non prise (sortie Float32, F-13) ; mesuré à ≈ 2 % de l'encodage (F-09) ;
  (d) sortie Float32 → P-01 ; (e) coût fixe : 1,32/1,38 Go de téléchargement ; mise en place à froid 1 min 09 /
  2 min 25 — *amendé* : la phase « 4. Encoder Setup » englobe `createHybridEncoderWithDownload` (téléchargement
  éventuel **et** compilation, non séparés ; CPU 5-8 %, GPU 1 %, #16, #22) : l'attribution à la seule compilation
  n'est pas établie ; à chaud 1,41 / 2,97 s (F-10) ; 48 % GPU en encodage (F-09). La comparaison « MLX ~500 ms /
  Core ML ~150 ms » (F-12, conditions non documentées) a probablement été faite avec l'encodeur MLX en **fp32**
  (P-01) : elle est à refaire.
- **Correction** : matrice A/B/B/A par fenêtre et de bout en bout : MLX bf16 (après P-01) / Core ML `.cpuAndGPU` /
  `.cpuAndNeuralEngine` / `.all`, WER compris ; règle de décision (R14 inversée) : Core ML reste le défaut `.auto`
  seulement s'il est ≥ 1,2× plus rapide **et** WER ≤ +0,3 pt ; sinon `.mlx` par défaut (plus de compilation à
  froid ni de téléchargement). `predictions(fromBatch:)` si Core ML est gardé. *Amendé* : la réécriture des ponts
  est retirée de la fiche — F-09 les mesure à ≈ 2 % de l'encodage, sous le seuil de 5 % de `measurement.md`.
- **Gain attendu** : jusqu'à 1 min 09 - 2 min 25 au premier lancement et 1,4-3 s à chaque chargement si `.mlx`
  gagne ; sinon ANE = GPU libre pour le décodage. **À MESURER.**
- **Risque API** : défaut de backend modifié (comportement, FluxForge) → ASK. **Effort** : M.
  **Statut** : configuration VÉRIFIÉE ; gains À MESURER.
- **Fiche proposée** — *Matrice encodeur MLX bf16 / Core ML GPU / ANE*. **Porte** (*amendée*) : décision par la
  règle ci-dessus (≥ 1,2× **et** WER ≤ +0,3 pt), chiffres consignés par fenêtre et de bout en bout, mise en place à
  froid chronométrée en séparant téléchargement et compilation. **Cible : macos-gpu.**

### P-15 · moyenne · `Pipeline/VoxtralPipeline.swift:393-473` — Chat : chaque question sur le même audio refait extraction, encodage et préfill

- **Constat** : pas d'état de session : extraction mel, encodage audio, fusion et préfill de l'audio sont recalculés
  pour chaque question. F-02/F-08/F-09 : pour une question de chat Mini, extraction 3,47 s + encodage Core ML
  2,37 s + préfill 3,33 s contre génération 6,13 s. *Amendé (vérification croisée)* : les 3,47 s d'extraction
  précèdent `70c390b` (≈ 0,25 s après, F-08) et les 3,33 s de préfill incluent la lecture des poids du premier
  préfill (P-04) ; le coût réellement répété par question est donc ≈ 0,25 s + 2,37 s + un préfill audio à chaud
  (< 3,33 s, À MESURER).
- **Correction** : session audio : garder les embeddings audio (clé = empreinte du fichier) et un instantané du
  cache KV à la fin du bloc audio, juste avant la question (motif Qwen38 : instantané avant l'invite de
  génération, `trim` du KV au préfixe) ; ne préremplir que la nouvelle question.
- **Gain attendu** (*amendé*) : de l'ordre de 3 à 6 s par question suivante sur ≈ 9 à 12 s (et non « ≈ 9 s sur
  ≈ 15 s », calculé avec des chiffres obsolètes), avant les autres correctifs. Source : T6/T7 (Qwen38 : tour 3 en
  0,63 s, réponses identiques). **À MESURER.**
- **Risque API** : additif (API de session). **Effort** : M-L. **Statut** : VÉRIFIÉ (absence).
- **Fiche proposée** — *Session audio réutilisable en chat*. **Porte** : 2ᵉ question : latence au premier jeton
  −≥ 50 % ; réponses greedy identiques à un démarrage à froid sur 4 questions ; mémoire consignée.
  **Cible : macos-gpu.**

### P-16 · moyenne · `VoxtralModeling.swift:1131-1147` ; `VoxtralStandardLoader.swift:675-681` — Pas de cache KV quantifié (profil lean, Small 24B, audio long)

- **Constat** : aucune option KV 8 bits. À 30 min d'audio, Small : KV bf16 3,43 Gio (fp32 aujourd'hui 6,87), KV
  8 bits ≈ 1,8 Gio. L'attention appelle `cache.update` puis le SDPA directement (`:675-698`) : un
  `QuantizedKVCache` exige le chemin amont `attentionWithCacheUpdate` (`AttentionUtils.swift:46`).
- **Correction** : en lean, `QuantizedKVCache(groupSize: 64, bits: 8)` via l'aide amont ; parité WER.
- **Gain attendu** : KV −≈ 47 % en lean (T10 ; Qwen38 lean 12,2 contre 16,1 Go à 32 k, effet combiné).
  **À MESURER.** **Risque API** : additif. **Effort** : M. **Statut** : VÉRIFIÉ (absence).
- **Fiche proposée** — *KV 8 bits en lean*. **Porte** : Small 4 bits, 20 min : pic −≥ 1 Go ; WER ≤ +0,3 pt ;
  décodage ≥ −5 % au pire (sinon réservé au lean). Prérequis P-02, P-03. **Cible : macos-gpu.**

### P-17 · moyenne · `Models/VoxtralLlama.swift:315-337`, `:515-541` ; `MLXLMBridge.swift:48-87` ; `Utils/VoxtralModelLoading.swift:139-180` — Décodeur hérité : masque `[T, T]` qui ignore l'offset → arrêt du processus dès la 2ᵉ tranche

- **Constat** : `loadVoxtralModel(modelPath:dtype:lazy:)` (public) construit `VoxtralForConditionalGeneration(config:)`
  dont le décodeur est `VoxtralLlama.LlamaModel` (`VoxtralModeling.swift:480`). Son masque
  (`mlxLMCreateAttentionMask` → `createCausalMask(N: T, offset:)`) construit des lignes et colonnes `0..<T` : forme
  `[T, T]` quel que soit l'offset. Avec le préfill tranché de `generateStream`, la 2ᵉ tranche (offset 512) a 1 024
  clés : `broadcast_to([512, 512] → [1, 32, 512, 1024])` échoue (`fast.cpp:805-807`) → arrêt. Invites > 512 jetons =
  audio > 30 s. Les constantes fp32 relevées par MLX-002 (`:527-536`) sont celles de ce masque.
- **Preuve** : lecture. Atteignabilité : API publique ; 0 consommateur trouvé (audit stabilité, S-14).
- **Correction** : faire pointer ce chemin sur le décodeur standard (ou le masque amont), ou le déprécier (S-14,
  ASK).
- **Risque API** : cassant si suppression → ASK ; aucun si correction du masque. **Effort** : S.
  **Statut** : VÉRIFIÉ (lecture) ; arrêt À MESURER.
- **Fiche proposée** — *Chemin hérité : masque amont ou dépréciation* (avec K-S14). **Porte** : test d'une invite de
  600 jetons via `loadVoxtralModel(modelPath:dtype:lazy:)` sans arrêt, ou dépréciation actée. **Cible : macos-gpu.**

### P-18 · moyenne · `VoxtralModeling.swift:1100-1456`, `:1596-1647` — Boucle de génération maison dupliquée au lieu du `TokenIterator` amont

- **Constat** : deux boucles de ≈ 170 lignes quasi identiques (`generateStream`, `generateStreamWithAudioEmbeds`)
  réimplémentent échantillonnage, pénalité, préfill, choix du cache et arrêt. L'amont (`main@ee673d6`) offre
  `asyncEval` (T15), préfill tranché pipeliné avec `PrefillParameters` (`LLMModel.swift:25-62`,
  `PrefillParameters.swift:15-36`), processeurs de logits, `kvBits`/`maxKVSize`, masques corrects. La conformance
  `LanguageModel` existante est inutilisable (S-17) : `prepare` ne tranche pas, calcule les logits de toutes les
  positions et passe les embeddings comme ids (`:1629`). Le détecteur MLX-006 ne l'a pas vu (signature changée,
  voir retours skill).
- **Correction** : `prepare` conforme (embeddings fusionnés une fois, tranches d'embeddings, `asyncEval(cache)`,
  dernier jeton laissé à l'itérateur) + `TokenIterator` ; `VoxtralPipeline` inchangé en surface.
- **Gain attendu** : union de P-06, P-07, P-12, P-16 et des évolutions amont futures. **Risque API** : aucun en
  surface (interne) ; décision S-17 (ASK). **Effort** : L. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Génération par TokenIterator*. **Porte** : greedy identique 100 % au code maison corrigé ;
  tok/s ≥ boucle maison (≥ +5 % attendu) ; build vert contre `main` et le prochain tag. **Cible : macos-gpu.**

### P-19 · moyenne · `VoxtralTranscriptionTest/ProfileCommand.swift:172-190` ; `README.md` §STT — Aucune baseline exploitable : instrument à un passage froid, corpus externe, chiffres contradictoires

- **Constat** : `profile` charge, transcrit **une fois** (à froid, P-04) et décharge ; backend `.auto` implicite ;
  pas d'A/B/B/A, pas de refroidissement, pas de ligne `BENCHMARKS.md` (absent), révision de mlx-swift-lm non
  notée alors qu'elle suit `main` ; `resetPeakMemory` dans la boucle (P-08) ; fichier `obama.mp3` hors dépôt ;
  sorties tronquées (P-11) ou coupées (S-01). README et issues se contredisent (Small 0,5-1,0 tok/s contre
  11,1). Point positif : l'instrument passe par `VoxtralPipeline`, donc par le chemin des consommateurs (piège 33
  évité).
- **Correction** : corpus du dépôt avec texte de référence : `docs/examples/fluxforge_long_{en,fr}_6bit.wav`
  (167,0 / 173,8 s ; textes dans `docs/tts_benchmark.md` « Full test texts » ; parole synthétique : biais à
  noter), clips courts (≈ 5 s) pour le TTFT, un clip long ≈ 12 min (concaténation) et un enregistrement réel long
  (ASK) ; `profile --passes 2 --cooldown 120 --backend`, requête d'amorçage exclue, une ligne JSON par mesure.
- **Risque API** : aucun. **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Baseline STT mesurée*. **Porte** : A/A ≤ 3 % sur 2 passes ; une ligne `BENCHMARKS.md` par
  (modèle × backend × clip) avec révision résolue ; WER de référence par clip. **Cible : macos-gpu.**

### P-20 · moyenne · (absence) `scan.md` §6 alerte rouge ; `VoxtralPipeline.swift:87-130` — Aucun profil de référence STT : les réglages qui comptent sont dispersés

- **Constat** : backend (`.auto`/`.mlx`), préréglage mémoire par RAM, fenêtre KV, `maxTokens`, pénalité,
  tranche 512 codée en dur, pack de poids (dont S-06) : aucun n'est figé ni mesuré ensemble. Standard absent
  (`profiles-standard.md`).
- **Correction** : type `VoxtralReferenceProfile` `<bits>bit-fast|lean` (4/8/16 × fast/lean, par modèle), chaque
  champ = un bouton existant ou créé par P-01…P-16 ; proposition en §6.
- **Risque API** : additif. **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Profils STT de référence*. **Porte** : 6 profils × 2 modèles mesurés (temps, pic, WER) ;
  `lean` Small 4 bits sous le budget d'une machine 32 Go à 10 min d'audio (valeur ASK) ; CLI `references`.
  **Cible : macos-gpu** (*amendée* : la porte exige des mesures et un build). Sous-fiche documentaire possible en
  **cloud** : squelette `References.md` (12 lignes modèle × profil, chaque valeur sourcée ou « À MESURER ») ;
  porte = document présent et 0 valeur non sourcée ; le type Swift et la CLI restent sur macos-gpu (build).

### P-21 · basse · `VoxtralFeatureExtractor.swift:372-394`, `:322` — Mel calculé deux fois par fenêtre, avec une synchronisation par fenêtre

- **Constat** : première passe (STFT + mel + `max().item()`) pour le maximum global, seconde passe qui recalcule
  tout. `hanning` recalculé à chaque appel (`:298`). Le cache des filtres (`:250`, `:260-271`) est un dictionnaire
  global muté sans verrou (S-11).
- **Correction** : une passe : log-mel de toutes les fenêtres (paresseux), un seul maximum global, une seule
  synchronisation. Découper d'abord la phase « Audio Feature Extraction » en décodage audio / mel. *Amendé
  (vérification croisée)* : F-08 (3,79 s à 0 % GPU) date d'avant `70c390b` ; la phase entière vaut ≈ 254 ms
  depuis pour un fichier de 203 s : le gain absolu de P-21 est de l'ordre de la centaine de millisecondes au plus.
- **Gain attendu** : partie GPU du mel ÷ 2, N−1 synchronisations en moins ; faible au total. **À MESURER.**
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Mel en une passe*. **Porte** : phase mel −≥ 5 % ; features bit-exactes. **Cible : macos-gpu.**

### P-22 · basse · `VoxtralModeling.swift:1169`, `:1352` — Tranche de préfill figée à 512, non balayée, non exposée

- **Constat** : valeur justifiée par `1eb2cc9` (512 contre monolithique seulement, F-05), jamais comparée à 256,
  1 024, 2 048 ; dupliquée dans deux boucles.
- **Correction** : `prefillStepSize` champ de profil, une seule boucle. **Gain attendu** : Qwen38 : 512 plus rapide
  **et** plus léger que 2 048/4 096 ; 256 plus léger en lean (T9). **À MESURER.** **Risque API** : additif.
  **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Balayage de la tranche de préfill*. **Porte** : A/B/B/A 256/512/1 024/2 048 à 5 et 30 min ;
  retenir la plus rapide dont le pic ≤ pic(512) + 5 % ; sortie bit-exacte après factorisation. **Cible : macos-gpu.**

### P-23 · basse · `VoxtralStandardLoader.swift:242-297` ; `VoxtralPipeline.swift:363-376` — En backend MLX, la tour audio reste résidente pendant tout le décodage

- **Constat** : l'encodeur (≈ 637 M paramètres + projecteur ≈ 25 M) ne sert qu'au début de la requête ; résident :
  ≈ 0,52 Go (6 bits), 0,68 Go (8 bits), 1,27 Go (bf16) (calcul, À MESURER). En `.auto` il n'est jamais matérialisé
  (paresseux), ce qui explique « ~660 Mo de moins » du README pour l'hybride.
- **Correction** : en lean MLX, libérer les poids de la tour après l'encodage (remplacement par des zéros non
  évalués, T4) et la recharger paresseusement à la requête suivante.
- **Gain attendu** : pic de décodage −0,5 à −1,3 Go ; coût : relecture par requête. Source T4 (YuE2 : pic 10,3 → 6,4
  Go sans coût temps). **À MESURER.** **Risque API** : aucun. **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Tour audio libérée après encodage (lean)*. **Porte** : pic de décodage −≥ 0,5 Go (Mini
  8 bits) ; temps par requête ≤ +5 % ; sortie identique. **Cible : macos-gpu.**

### P-24 · basse · `VoxtralModeling.swift:560-573` ; `CoreML/VoxtralHybridEncoder.swift:132` — Modules factices aléatoires : coût nul aujourd'hui, piège pour toute évaluation globale

- **Constat** : `init(standardModel:)` crée un `VoxtralEncoder` (32 couches) et un projecteur jamais chargés ;
  l'encodeur hybride en crée un autre. ≈ 662 M paramètres chacun, initialisés en fp32 : ≈ 2,6 Go chacun **s'ils
  sont évalués** (un `eval(model.parameters())` pour corriger P-04, un export, une statistique). Aujourd'hui
  paresseux, donc gratuits.
- **Correction** : ne pas les instancier (ou 0 couche) ; en attendant, filtrer toute passe sur `parameters()`
  (piège 1 du catalogue, transposé).
- **Risque API** (*amendé*) : propriétés publiques `audioTower`/`multiModalProjector` → garder le type et vider le
  contenu ne change aucune signature mais **change le comportement** d'un appel direct (module vide) : aucun
  consommateur connu ne les appelle (le pipeline passe par `standardModel`), à confirmer (ASK) ; suppression =
  cassant → ASK. **Effort** : S. **Statut** : VÉRIFIÉ (création) ; impact latent.
- **Fiche proposée** — *Neutraliser les modules factices*. **Porte** : aucun tenseur aléatoire dans
  `parameters()` du wrapper (test) ; mémoire inchangée ; build vert. **Cible : macos-gpu** (*amendée* : la porte
  exige un test, un build et une mesure mémoire ; le patch peut être rédigé en cloud avec `syntax_guard`, mais la
  fiche ne se ferme que sur macos-gpu).

### P-26 · basse · `VoxtralModeling.swift:1520-1544` — Top-p approché (chat)

- **Constat** : depuis `41ce59d` (F-02), le seuil vient du plus petit des 1 000 meilleurs ; si leur masse < `topP`,
  le seuil tombe à 1e-9 (quasi tout le vocabulaire) : ce n'est plus un nucleus. Softmax fp32 sur 131 072 à chaque
  pas. *Amendé (vérification croisée)* : ce n'est **jamais** un nucleus — quand la masse des 1 000 meilleurs
  atteint `topP`, le seuil est la 1 000ᵉ probabilité (`:1536-1537`), donc un filtre **top-k = 1 000**, pas le plus
  petit ensemble de masse `topP`. Atteignabilité : seulement si `temperature > 0` (`:1512-1514`) — CLI `chat`
  (0,7 par défaut, `VoxtralCLI.swift:295`) et `profile` en chat ; le pipeline et l'app sont à 0 par défaut
  (argmax), donc la transcription n'est pas concernée.
- **Correction** : top-p exact sur les k meilleurs (tri des k, cumul) ou échantillonneur amont, mesuré contre
  l'actuel. **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ ; effet qualité À MESURER.
- **Fiche proposée** — *Top-p exact mesuré*. **Porte** : chat tok/s ≥ actuel − 5 % ; test de distribution sur
  logits synthétiques. **Cible : macos-gpu.**

### P-27 · basse · (absence) — Politique de calcul par étape (T14/T20) : encodeur audio dé-quantifié en profil fast

- **Constat** : l'encodeur (1 500 lignes par fenêtre, borné calcul) tourne en `quantizedMatmul` 6/8 bits ; le
  décodage (borné bande passante) gagne au bas-bits, l'encodeur probablement pas.
- **Correction** : `dequantizeWeights` de la tour audio une fois vers bf16 en profil fast (≈ +0,75 Go pour Mini
  mixte). **Gain attendu** : YuE2 NAR 58-61 → 53-55 s (T14) ; Qwen38 « le préfill est borné par la
  déquantification ». **À MESURER** après P-01. **Risque API** : aucun. **Effort** : S.
- **Fiche proposée** — *Encodeur bf16 en fast*. **Porte** : encodage −≥ 5 %, pic ≤ +0,8 Go, WER ≤ +0,1 pt.
  **Cible : macos-gpu.**

### P-28 · basse · `VoxtralPipeline.swift:336-376` ; `VoxtralModeling.swift:1126`, `:1163-1197` — Étapes strictement séquentielles : l'encodage de toutes les fenêtres précède le premier jeton de préfill

- **Constat** : les fenêtres de 30 s s'encodent indépendamment, et les jetons audio de la fenêtre k ne dépendent
  que d'elle ; pourtant extraction, encodage de tout l'audio puis préfill s'enchaînent. F-01/F-09 : 4,28 s
  d'encodage Core ML puis 3,59 s de préfill (Mini).
- **Correction** : préfill en flux par fenêtre (encoder k+1 pendant le préfill de k ; Core ML sur ANE en parallèle
  du GPU si P-14 le retient).
- **Gain attendu** : TTFT réduit sur audio long. **À MESURER.** **Risque API** : aucun. **Effort** : L.
  **Statut** : VÉRIFIÉ (séquencement).
- **Fiche proposée** — *Préfill en flux par fenêtre*. **Porte** : TTFT −≥ 10 % à 10 min ; greedy identique.
  **Cible : macos-gpu.**

### P-29 · basse · `VoxtralFeatureExtractor.swift:57-85` — Le fichier audio est décodé en entier au format natif avant rééchantillonnage

- **Constat** : un `AVAudioPCMBuffer` de toute la durée au format de traitement (Float32 non entrelacé, fréquence et
  canaux d'origine), puis sortie 16 kHz, copie `Array`, copie `MLXArray`. 30 min stéréo 48 kHz : ≈ 691 Mo pour le
  seul tampon source.
- **Correction** : conversion par blocs (ex. 10 s) avec `AVAudioConverter` en flux, écriture dans un tampon de
  sortie préalloué. **Gain attendu** : pic de l'extraction −≥ 50 % sur fichier long ; temps neutre. **À MESURER.**
  **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche proposée** — *Décodage audio par blocs*. **Porte** (*amendée*) : pic mémoire de l'extraction −≥ 50 %
  (30 min, stéréo 48 kHz) ; échantillons : écart max ≤ 1e-6 hors des 20 dernières millisecondes, features mel
  écart relatif ≤ 1e-5, greedy identique 100 % ; temps ±5 %. **Cible : macos-gpu.** *Raison* : un rééchantillonneur
  alimenté par blocs n'est pas garanti bit-exact aux bords ; de plus le bloc d'entrée actuel rend le **même**
  `sourceBuffer` avec `.haveData` à chaque appel (`:67-70`), si bien que la fin de la sortie actuelle n'est pas une
  référence fiable (indice, non vérifié à l'exécution).

## 5. Protocole commun de mesure (toutes les fiches perf)

- **Binaire Release** (`xcodebuild -scheme VoxtralCLI -configuration Release`), `machine-check.sh --cooldown 120`,
  aucun autre process MLX, sorties dans un dossier `.noindex`. Révision de mlx-swift-lm **résolue** notée sur
  chaque ligne (dépendance sur `main`, S-18).
- **A/B/B/A**, deux passes par variante, une requête d'amorçage exclue (P-04) ; une différence n'est lue que si
  elle dépasse l'écart A/A ; **seuil 5 %** : en dessous, le code est retiré et la fiche passe « retirée » avec la
  mesure.
- **Mesures** : par phase (extraction, encodage, préfill, décodage), TTFT, ms/pas médian et p90, tok/s, pic
  `phys_footprint` et pic MLX actif, `weights_bw_gbps` = taille des poids lus × tok/s.
- **Parité sur checkpoint réel** : greedy (texte identique) ; WER normalisé (casse, ponctuation, accents : la
  référence FR de `tts_benchmark.md` est sans accents) ; tolérance WER par fiche. Pour les fiches bit-exactes
  (P-10, P-21, P-22) : égalité stricte des sorties ; *amendé (vérification croisée)* : P-06, P-13 et P-29 changent
  de noyau ou de découpage et ont une tolérance chiffrée dans leur porte (une porte bit-exacte y rejetterait une
  correction saine).
- **Corpus** : C-court = `docs/examples/fluxforge_short_{en,fr}_6bit.wav` (5,0 / 4,8 s) ; C-moyen =
  `fluxforge_long_{en,fr}_6bit.wav` (167,0 / 173,8 s ; les versions 4 bits et bf16 FR ont été coupées par le TTS,
  `tts_benchmark.md:56`) ; C-long = concaténation ≈ 12 min (déclenche P-03 à 8 192) ; un enregistrement réel long
  (ASK). Modèles : Mini 4 bits mixte, Mini 8 bits, Mini bf16, Small 4 bits mixte ; backends `.mlx` et `.auto`.
- **Ordre** (skill) : stabilité bloquante d'abord (P-03 avec S-01/S-02, P-17 si le chemin hérité est gardé) ;
  **baseline P-19** (rien ne peut conclure sans elle) ; P-04 avant toute mesure de préfill ; puis P-02 → P-01
  (→ P-05), P-06, P-07, P-08, P-09, P-10, P-13, P-14 ; puis P-15, P-16, P-22, P-23, P-27, P-28 ; P-18 en dernier
  (refonte) ; P-20 et la doc à la fin.

## 6. Conséquences pour les profils STT (proposition, toutes les valeurs À MESURER)

| Réglage | `fast` | `lean` |
|---|---|---|
| Poids 4 bits | Mini `mzbac/voxtral-mini-3b-4bit-mixed` (3,20 Go) ; Small `VincentGOURBIN/voxtral-small-4bit-mixed` (14,86 Go) | idem |
| Poids 8 bits | Mini `mzbac/voxtral-mini-3b-8bit` (5,40 Go) ; Small : ASK S-06 (26,50 ou 28,06 Go) | idem |
| Poids 16 bits | Mini `mistralai/Voxtral-Mini-3B-2507` shards (9,36 Go, sans `consolidated`, S-07) ; Small (48,5 Go) | idem |
| Dtype de calcul | bf16 (P-01) | bf16 |
| Encodeur | vainqueur de la matrice P-14 ; dé-quantifié en bf16 si P-27 passe | Core ML ANE ou MLX, tour libérée après usage (P-23) |
| Lot encodeur | K à balayer (P-13) | K petit |
| Cache KV | `KVCacheSimple` pré-dimensionné (P-03, P-10) | KV 8 bits (P-16) |
| Tranche de préfill | 512 (à balayer, P-22) | 256 (à balayer) |
| `cacheLimit` / `memoryLimit` | quelques Go / nil (P-09) | `min(1 Go, max(256 Mo, dispo/6))` / `dispo − 2 Go` (T2) |
| `clearCache` | après réponse | après encodage et après réponse |
| `maxTokens` | f(durée) (P-11) | f(durée) |
| Pénalité de répétition | décision WER (P-12) | idem |

Modèles à septembre 2026 (F-14) : pas de STT hors ligne Mistral plus récent que 2507 ; le plus récent est
`Voxtral-Mini-4B-Realtime-2602` (déjà porté dans le chemin Realtime). Une comparaison WER/temps Mini 3B 2507
contre Realtime 4B sur le même corpus est la question de choix de modèle à trancher (ASK, mesure).

## 7. Décisions à prendre (ASK)

1. **Backend par défaut** (P-14) : A) garder `.auto` (Core ML) ; B) `.mlx` par défaut si la matrice le justifie
   (impact FluxForge : plus de compilation à froid ni de téléchargement Core ML).
2. **Cache KV en STT** (P-03, avec S-02) : A) `KVCacheSimple` pré-dimensionné, limite mémoire par profil ; B) garder
   une limite mais lever une erreur Swift au dépassement.
3. **Chemin hérité** (P-17, S-14) : corriger le masque ou déprécier.
4. **Modules factices publics** (P-24) : vider (additif) ou supprimer (cassant).
5. **Cible iOS du STT** (T23) : oui/non (porte GPU en arrière-plan, limites `lean` iOS).
6. **Corpus réel long** pour le WER (P-19) et modèle de référence (Mini 3B 2507 vs Realtime 4B 2602).

## Annexe A — Simulation du préfill avec `RotatingKVCache` (P-03)

Réplique exacte de `RotatingKVCache.updateConcat` + `temporalOrder` (mlx-swift-lm `KVCache.swift:620-637`,
`:691-714`) et du masque `[T, offset+T]` (`VoxtralStandardLoader.swift:450-481`), tranches de 512 :

```python
def run(prompt, max_size, keep=4, chunk=512):
    keys_len = None; offset = 0
    for start in range(0, prompt, chunk):
        T = min(chunk, prompt - start)
        mask_kl = offset + T                       # masque maison, offset = cache.offset avant update
        if keys_len is None: keys_len = T
        else:
            idx = keys_len; trim = idx - max_size + 1
            keys_len = (keep + (idx - trim - keep) + T) if trim > 0 else idx + T
        offset += T
        if T > 1 and mask_kl != keys_len:          # T == 1 : pas de masque
            return f"échec @{start}: masque [{T}, {mask_kl}] vs clés {keys_len}"
    return "ok"
# 2048 : 2048 → ok ; 2258 → échec @2048 [210, 2258] vs 2257 ; 8192 : 8258 → échec @8192 [66, 8258] vs 8257
```

## Annexe B — Constats écartés à la vérification croisée

| id | Raison |
|---|---|
| P-25 | Pas un constat de performance tenable. (1) Le « masque concaténé » (`VoxtralModeling.swift:1242-1245`, `:1419-1422`) n'est **jamais atteint** depuis les points d'entrée : `VoxtralPipeline.transcribe`/`chat` n'appellent `generateStream*` sans `attentionMask` (`VoxtralPipeline.swift:353-374`, `:444-464`), donc `currentAttentionMask == nil`. (2) `eval(generated)` redondant est déjà compté dans P-08. (3) La reconcaténation de `generated` porte sur un tableau `int32` `[1, L]` (≈ 10 Kio à 2 600 jetons) : négligeable devant un pas de ≈ 30 ms, gain attendu < 5 % selon l'auditeur lui-même, sans porte propre. Le seul détail utile (réinjecter `nextToken`) est repris dans la correction de P-07. |

## Annexe C — Corrections hors constats (vérification croisée)

- **F-08** : 3,79 s / 3,47 s d'extraction sont **antérieurs** à `70c390b` (254 ms après, même fichier, « Fixes #12 ») ;
  répercuté dans P-15 et P-21.
- **§1** : l'app pose `maxKVCacheSize` à `TranscriptionManager.swift:212` (et non `:211`).
- **§3 bis, question 4** : « 37 des 49 `.item()` dans `MLXLMBridge.swift:326-535` » était faux (18) ; décompte corrigé
  en place.
- **Sources MLX** : les fichiers du scratchpad de l'auditeur (`mlx/fast.cpp`, `ops.cpp`, `allocator.cpp`,
  `sdpa_metal.cpp`) **diffèrent** de `ce45c52` (seul `dtype.cpp` est identique) ; tous les numéros de ligne cités ont
  été revérifiés sur les fichiers re-téléchargés à `ce45c52` et sont exacts (`fast.cpp:704`, `:796-800`, `:805-807` ;
  `ops.cpp:843`, `:1728`, `:3069-3082`, `:4131`, `:4343` ; `dtype.cpp:45`, `:48` ; `allocator.cpp:52-54` ;
  `scaled_dot_product_attention.cpp:625-629`).
- **Constats relus sans changement** (preuve et lignes confirmées) : P-02, P-05, P-09, P-11, P-12, P-16, P-17, P-18,
  P-19, P-22, P-23, P-27, P-28.
