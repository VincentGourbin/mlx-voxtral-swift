# Audit « Performance » — chemin TTS Voxtral 4B (bf16 / 6 bits / 4 bits)

> **Vérification croisée : 20 constats relus, 5 gardés, 0 écartés, 15 amendés** (relecture adverse du code à
> `9392ed1`, de MLX C++ `1f8e74e`, de mlx-swift `9019419`, de mlx-swift-lm `ee673d6`, de la référence
> `acoustic_head.py`, de `params.json` (HF), des issues #26 à #29 et #45, des commits `a00024f`, `0be05af` et
> `f4fd21c`, et d'une recherche de code GitHub en lecture seule chez le consommateur FluxForge ; 2026-09-27).
> - Gardés : P-32, P-37, P-42, P-44, P-49.
> - Amendés : P-30, P-31, P-33, P-34, P-35, P-36, P-38, P-39, P-40, P-41, P-43, P-45, P-46, P-47, P-48.
> - Sévérités : P-33 passe de haute à **moyenne** ; P-34, P-43 et P-48 passent de moyenne à **basse**.
> - Cibles : une fiche dont la porte exige un build, un test ou une mesure cible `macos-gpu`. K-P34, K-P35,
>   K-P40, K-P41, K-P46 et K-P48 (« cloud (code) puis macos-gpu ») passent donc en `macos-gpu`. Le code peut être
>   rédigé dans le cloud, mais la fiche ne se ferme que sur Mac.
> - Détail des amendements : annexe finale.
>
> **Faits transverses établis à la vérification** :
> - **F-1 décrit un code antérieur à `a00024f` et à `0be05af`.** L'issue #26 a été ouverte le 2026-04-11 à 09:29 UTC,
>   avant `a00024f` (11:01 UTC le même jour). Le code de l'époque faisait :
>   - un `MLX.eval(hidden)` à chaque frame (extrait cité dans #26) ;
>   - deux passes FM séquentielles (CFG non batché), soit 14 passes par frame ;
>   - un `MLX.eval(xt)` à chaque pas d'Euler (diff de `0be05af` sur `VoxtralFlowMatching.swift`).
>
>   Ses ms par frame et sa répartition CPU/GPU ne décrivent donc pas le code audité. Le 4 bits actuel tourne à
>   ≈ 31,5 fr/s (F-6) ; le bf16 actuel n'a pas été mesuré. Tout calcul dérivé de F-1 (§0.2, P-30, P-31, P-41,
>   P-47) n'est qu'un ordre de grandeur sur un code antérieur.
> - **FluxForge ne livre que le 6 bits.** Sources (branche par défaut) :
>   - `Models/VoxtralTTSVariant.swift` : `case q6`, « The only variant we ship » ;
>   - chargement explicite par `VoxtralTTSService.loadModel(variant:)` ;
>   - voix enrôlées synthétisées avec warm-up via `synthesize(…, warmUpText:)` ;
>   - un `MLX.Memory.clearCache()` déjà appelé côté app avant chaque prévisualisation (« the warm model's buffer
>     cache grows unbounded (~+4 GB) across repeated previews », `GenerationQueueManager+VoiceTraining.swift`).
>
>   Le chemin du consommateur est donc **6 bits + voix clonée + batch** : le défaut bf16 (P-34) ne le concerne
>   pas, et pour P-30 seule la partie 4 et 6 bits le touche.

> Skill `mlx-swift-audit`, phase 2, constats **P-30 à P-49**. Révision auditée : `9392ed1` (2026-09-12, branche
> `claude/action-plan-skills-beta-wifgmu`). Date : 2026-09-27.
> Périmètre : `TTS/VoxtralTTSModeling.swift` (décodeur AR), `TTS/VoxtralFlowMatching.swift` (transformeur
> acoustique, Euler, CFG), `TTS/VoxtralCodecDecoder.swift` (+ `VoxtralCodecEncoder.swift`),
> `TTS/VoxtralTTSProcessor.swift`, `TTS/Pipeline/VoxtralTTSPipeline.swift` (dont le streaming l. 556),
> `TTS/Pipeline/VoxtralTTSSynthesisManager.swift`, `TTS/VoxtralTTSModelLoading.swift`, `TTS/VoxtralTTSRegistry.swift`,
> voix prédéfinies, `VoxtralZeroVoice`, `VoxtralVoiceSLERP`, et l'instrument `profile --pipeline tts`.
> Hors périmètre (renvois) : enrôlement (audit annexes, A-01, A-06), stabilité du streaming (S-08, S-09, S-12).

## 0. Cadre, sources, méthode

- **Environnement** : session cloud Linux, sans Mac, sans toolchain Swift, sans GPU. Aucun build, aucun test,
  aucune mesure. Tout gain est **attendu** ; tout chiffre qui n'est pas déjà mesuré dans le dépôt, ses issues ou
  ses commits est **À MESURER**. Toutes les fiches qui demandent build, tests, mesure ou écoute ciblent
  `macos-gpu`. `machine-check.sh` n'a pas pu tourner ici.
- **Statuts** : `VÉRIFIÉ` = mécanisme lu dans le code à `9392ed1` (et, pour MLX, dans le C++ amont) ;
  `À MESURER` = l'effet chiffré reste à mesurer. Les deux se cumulent souvent : « VÉRIFIÉ (mécanisme), À MESURER (gain) ».
- **Amont lu** :
  - MLX C++ `ops.cpp` @ `1f8e74e` (sous-module de mlx-swift `9019419`) :
    - `matmul` promeut les deux opérandes au type commun et **caste l'opérande qui diffère** (`ops.cpp:3492-3507`) ;
    - `quantized_matmul` fait `dtype = promote_types(x.dtype, scales.dtype)` puis `astype(scales)` et `astype(biases)`
      (`ops.cpp:4803-4818`) ;
    - `conv_general` caste l'entrée et le poids (`ops.cpp:4591-4593`) ;
    - `rms_norm` rend `result_type(x, weight)` (`fast.cpp:82`).
    - Limite : le clone de mlx-swift est superficiel, sans tags, et le sous-module du tag `0.31.6` n'a pas pu être
      comparé. Ces règles de promotion sont anciennes.
  - mlx-swift `9019419` : `Linear` = `matmul(x, weight.T)` (`MLXNN/Linear.swift:124-131`), `QuantizedLinear` =
    `quantizedMM` (`Quantized.swift:369-379`), `silu` déjà compilé (`Activations.swift:212-213`, `:1049`),
    `QuantizationMode` = `affine | mxfp4 | mxfp8 | nvfp4` (`MLX/Ops.swift:1100-1127`).
  - mlx-swift-lm `ee673d6` : `KVCacheSimple` (`MLXLMCommon/KVCache.swift:408-487`).
  - Référence Python mlx-audio `main` (fetch du 2026-09-27) : `voxtral_tts/acoustic_head.py`, `audio_tokenizer.py`.
- **Données HF** (lecture seule, MCP) :
  - `params.json` de `mistralai/Voxtral-4B-TTS-2603` ;
  - `config.json` et `model.safetensors.index.json` du pack mlx-community 4 bits ;
  - listings des packs 4 bits, 6 bits et bf16.
- **Mesures existantes exploitées** : `docs/tts_benchmark.md`, `docs/zerovoice_benchmark.md`,
  `docs/voice_cloning.md`, issues fermées #26-#29 et #45 (corps et commentaires), commits `a00024f`, `0be05af`,
  `f4fd21c` et PR #37.

### 0.1 Faits sourcés (mesures existantes, recopiées telles quelles)

| # | Fait | Source |
|---|---|---|
| F-1 | Pas AR moyen : 4 bits **42,6 ms** (écart-type 23,7), 28 % GPU / 84 % CPU ; bf16 **240,7 ms** (écart-type 14), 46 % GPU / 38 % CPU ; génération sémantique = 87,5 % (4 bits) / 97 % (bf16) du temps total ; 190-191 frames. **Périmé** (vérification croisée) : mesuré avant `a00024f` et `0be05af` (eval à chaque frame, 14 passes FM et 7 `eval` par frame) | issue #26 (profil, M3 Max, 2026-04-11) |
| F-2 | Pic MLX 2,4 Go (4 bits) contre 7,7 Go (bf16) ; RT 1,12× contre 0,28× ; TTFT 406 ms contre 1 624 ms | #27 et son commentaire |
| F-3 | Préfill 789 ms à 40 % GPU, +1 895 Mo (4 bits) ; 1 210 ms à 19 % GPU, +6 595 Mo (bf16). Le propriétaire conclut « inherent to model size » | #28 et son commentaire |
| F-4 | Le « codec 4 bits 3,4× plus lent » est **réfuté** : « codec weights are BF16 in both … 448ms was a cache-cold outlier, subsequent runs show 66ms » (bf16 : 131 ms) | #29, commentaire de clôture |
| F-5 | EOA vérifié tous les 4 frames : RT 1,12× → 1,41×, 16,2 → 17,9 fr/s, audio identique | commit `a00024f` |
| F-6 | CFG en un batch 2 + un seul `eval` par intégration d'Euler + préfill fusionné : TTFT 414 → ≈ 280 ms, 27,7 → 31,5 fps (tts-4b-4bit) | commit `0be05af`, PR #37 |
| F-7 | Cache KV du préfixe de voix : préfill « premier audio » 344-381 ms à froid → 146-150 ms à chaud | commit `f4fd21c`, PR #37 |
| F-8 | Banc 2026-04-02, M3 Max 96 Go. Court EN : 4 bits RTF 1,17 ; 6 bits 1,88 ; bf16 6,86. Long EN (2 266 frames) : 4 bits 120,61 s ; bf16 902,26 s. Long FR : 4 bits et bf16 atteignent `maxFrames` = 2 500 ; 6 bits non | `docs/tts_benchmark.md:38-56` |
| F-9 | Voix clonée, une voix FR : 6 bits couverture ASR 99,4 %, RTF 1,47, 3,5 Go ; bf16 96,5 %, RTF 3,44, 8 Go ; « Defaults stay bf16 » | `docs/voice_cloning.md:122-126` ; #45, commentaires finaux |
| F-10 | Emballement : 197,8 s de babillage (pas d'EOA avant `maxFrames` = 2 500) pour un texte de 2 phrases | #45, point 1 |
| F-11 | Mélanges ZeroVoice : t = 0,10 → 149 frames (contre 95), t = 0,15 → 260 frames, transcription vide ; AR t = 0,05 : « 200s generation! » | `docs/zerovoice_benchmark.md:22-26`, `:98` |
| F-12 | Pack 4 bits : 802 clés, `total_size` 2 509 772 000 o. `.scales` présentes : LLM 183, FM 25 (21 dans les couches, plus `llm_projection`, `time_projection` et les deux têtes). Absentes : `input_projection` (36 entrées), `audio_codebook_embeddings` et les **116 tenseurs du codec** | `model.safetensors.index.json` (HF) |
| F-13 | Architecture : LLM 26 couches, dim 3 072, 32 têtes, 8 têtes KV, `hidden_dim` 9 216, vocabulaire 131 072, embeddings liés. FM : 3 couches de mêmes dimensions. Codec : dim 1 024, 8 têtes, `hidden_dim` 4 096, 4 × 2 couches, fenêtre 16, 12,5 frames/s | `params.json` (HF) |

### 0.2 Budget par frame (dérivé du code et de F-13 ; estimations, pas des mesures)

Comptage fait à la lecture de `decodeOneFrame` (`VoxtralFlowMatching.swift:284-334`) et de `llmForward`
(`VoxtralTTSModeling.swift:300-309`).

| Grandeur | LLM (1 pas) | FM (1 frame = 7 pas d'Euler) |
|---|---|---|
| Paramètres lus | 3,026 G | 7 × 368,3 M + tête sémantique 25,6 M = 2,604 G |
| Appels `Linear` | 182 | 176 |
| Octets lus, 4 bits (0,5625 o/param) | 1,70 Go | 1,46 Go |
| Octets lus, bf16 sans promotion | 6,05 Go | 5,21 Go |
| Octets lus, bf16 **avec** la promotion fp32 actuelle (≈ 10 o/param : lecture bf16 + écriture fp32 + relecture fp32) | 6,05 Go | **26,0 Go** |

Le FM fait donc **la moitié des appels** d'une frame. En bf16, la promotion actuelle (P-30) lui fait déplacer
**≈ 81 % des octets** de la frame.

Indice de cohérence (pas une preuve) : à 240,7 ms par frame (F-1), 11,3 Go par frame sans promotion
représenteraient 47 Go/s, loin d'un décodage borné par la bande passante. Les 32,1 Go par frame avec promotion
représentent 133 Go/s.

> Vérification croisée : ce rapprochement mêle un temps mesuré sur l'ancien code (F-1 : 14 passes FM par frame,
> donc ≈ 2 × 26 Go de trafic FM) et les octets du code actuel (7 passes batch 2). Il ne tient donc que comme ordre
> de grandeur. Le ms par frame du bf16 actuel est **À MESURER** (K-P45).

## 1. Synthèse

**20 constats** (P-30 à P-49) : **3 hauts**, 8 moyens, 9 bas après la vérification croisée (4, 10 et 6 avant).
Aucun n'est un gain « obtenu ».

| Id | Sév. | Constat (une ligne) | Statut |
|---|---|---|---|
| P-30 | haute | Le transformeur FM et la tête sémantique tournent en fp32. En bf16, MLX recopie chaque poids en fp32 à chaque appel (≈ 26 Go par frame). En 4 et 6 bits, ≈ 340 `astype` de scales et biases par frame | VÉRIFIÉ (mécanisme), À MESURER (gain) |
| P-31 | haute | Boucle AR synchrone : `MLX.eval(xt)` dans `decodeOneFrame` à chaque frame ; streaming : un `.item()` et un `eval` par frame ; aucun `asyncEval` (T15) | VÉRIFIÉ, gain À MESURER |
| P-32 | haute | L'attention du codec, à fenêtre ≤ 16, est calculée en T×T fp32 avec masques recalculés par couche : ≈ 10,5 Go par tenseur de scores à 2 266 frames | VÉRIFIÉ (structure), pic À MESURER |
| P-33 | moyenne (était haute) | Le streaming re-décode toute la séquence accumulée à chaque chunk (O(N²), ≈ 33× le travail du codec sur le texte long). Le codec causal permet un décodage incrémental exact. Seul appelant : la démo (FluxForge n'utilise pas le streaming) | VÉRIFIÉ, À MESURER |
| P-34 | basse (était moyenne) | Le modèle par défaut est le bf16 : registre, CLI, `profile`, et `VoxtralTTSSynthesisManager` qui ne permet pas de choisir. Le commentaire de #27 affirme l'inverse. Défaut délibéré et documenté ; FluxForge choisit explicitement le 6 bits | VÉRIFIÉ |
| P-35 | moyenne | T21 non exposé : pas d'Euler et CFG codés en dur ; `flowSteps`, `cfgAlpha` et `temperature` (config et CLI) ne sont jamais lus | VÉRIFIÉ |
| P-36 | moyenne | Invariants du FM recalculés à chaque pas : `llmProjection`, `timeProjection`, masque sémantique construit sur CPU à chaque frame | VÉRIFIÉ, gain À MESURER |
| P-37 | moyenne | Attention du FM non fusionnée (répétition GQA, matmul, softmax, scale fp32) au lieu de `MLXFast.scaledDotProductAttention` | VÉRIFIÉ, gain À MESURER |
| P-38 | basse | Codec entièrement en fp32 : voulu pour le codebook (comme la référence), mais propagé à tout le décodeur, avec cast des poids à chaque appel | VÉRIFIÉ, À MESURER |
| P-39 | basse | Recalculs à chaque décodage : weight norm des 5 convolutions, centroïdes sémantiques, ALiBi et masques par couche, concaténation de padding | VÉRIFIÉ |
| P-40 | moyenne | Pas de cache KV de préfixe pour les voix clonées, ZeroVoice et mélanges (le chemin de LipDub) ; le cache n'a qu'une entrée | VÉRIFIÉ |
| P-41 | moyenne | `maxFrames` fixe à 2 500 quel que soit le texte : un EOA manqué coûte jusqu'à 200 s d'audio (≈ 10 min de calcul en bf16) | VÉRIFIÉ (+ F-8, F-10, F-11) |
| P-42 | moyenne | Aucune politique mémoire sur le TTS (T1, T2, T3) ; `unload()` n'appelle pas `clearCache` | VÉRIFIÉ, effet À MESURER |
| P-43 | basse (était moyenne) | Poids paresseux : la première synthèse paie la lecture disque (le pic de préfill de #28, antérieur au cache de préfixe, correspond aux poids du LLM). Coût déplacé, pas supprimé | VÉRIFIÉ (paresse), À MESURER |
| P-44 | basse | Deux passes LLM sur le chemin du TTFT (suffixe texte, puis jeton AUDIO) | VÉRIFIÉ |
| P-45 | moyenne | Instrument TTS incomplet : `profile` couvre le seul chemin prédéfini en batch, sans graine ; le TTFT exclut le préfixe ; le « froid » confond deux effets ; pas de TTFA en streaming (complète A-15) | VÉRIFIÉ |
| P-46 | basse | Post-traitement : un `.item()` par frame dans les coupes ; en streaming, le balayage du porteur est relancé à chaque chunk | VÉRIFIÉ |
| P-47 | basse | Warm-up (A6b) : les frames du vocalise sont générées puis jetées ; le streaming attend 3 s d'audio avant d'émettre (ASK) | VÉRIFIÉ, coût À MESURER |
| P-48 | basse (était moyenne) | Packs uniformes (LLM + FM), pas de quantification mixte (T13). Le mode de quantification est ignoré (`.affine` dans les deux branches) : un pack mxfp4, mxfp8 ou nvfp4 serait mal chargé, sans erreur. Défaut latent : aucun pack de ce type n'existe | VÉRIFIÉ |
| P-49 | basse | Compile (T18) : les seuls candidats sont dans le codec, à traiter après P-32/P-33, et interdits sous gradient (A-01) | VÉRIFIÉ, À MESURER |

**Priorités de mesure** (après la baseline K-P45) :
1. P-30 : premier levier sur le chemin **par défaut** (bf16, CLI et référence qualité). Pour le consommateur
   (FluxForge, 6 bits), c'est la partie 4 et 6 bits qui compte (−338 `astype` par frame).
2. P-32 et P-33, mémoire des textes longs et streaming réel.
3. P-31, levier sur 4 et 6 bits (chemin borné CPU, F-1).
4. P-35 (T21).
5. P-40 et P-41.

## 2. Techniques du catalogue sur ce chemin (T1…T23)

| T | Statut | Où / preuve | Commentaire |
|---|---|---|---|
| T1 `cacheLimit` par étape | **absente** | 0 `Memory.cacheLimit` dans `Sources/VoxtralCore/TTS/`. Les 2 poses du dépôt sont dans `VoxtralApp/TranscriptionManager.swift:293-295` (scan.md §4). `VoxtralMemoryManager` n'est jamais appelé par le TTS | P-42 |
| T2 limites adaptatives | **absente** | idem ; VoxtralCore cible aussi iOS 17+ (commit `7ddcc4b`) | P-42 (profil `lean`) |
| T3 `clearCache` entre étapes | **absente** | 0 appel dans `TTS/` ; `unload()` à `VoxtralTTSPipeline.swift:676-684` | P-42 |
| T4 résidence par étape | **non applicable** en synthèse répétée ; partielle en fin de chaîne | Les 3 étapes (LLM + FM à chaque frame, codec en fin) servent à chaque synthèse. Pack 4 bits : codec ≈ 0,30 Go sur 2,51 Go (F-12, calcul 149 M × 2 o). Le levier mémoire est le **transitoire** du codec (P-32), pas les poids | En chaîne multi-modèles (FluxForge TTS → LTX) : `unload()` + `clearCache` (P-42). Enrôlement : A-06 |
| T5 variante sans tour | **appliquée de fait** | Pas de `lm_head` (`VoxtralTTSModeling.swift:8`, `:297-309`). `VoxtralCodecEncoder` jamais instancié (0 référence hors de son fichier ; poids absents du checkpoint, `VoxtralCodecEncoder.swift:13-15`) | — |
| T6 réutilisation du préfixe KV | **partielle** | Voix prédéfinies (`VoxtralTTSPipeline.swift:93-103`, `:210-223`) et streaming avec `voiceKey` (`:512-514`). Absente pour `synthesize(text:voiceEmbedding:)` (`:333-340`), ZeroVoice (`:278-281`) et la démo clonée (`StreamingDemoViewModel.swift:507-508`) | P-40 ; mesure F-7 à requalifier (P-45) |
| T7 médias nouveaux seulement | non applicable | Pas d'image ; la voix, seul « média », relève de T6 | — |
| T8 budget de jetons média | non applicable | La longueur de voix (N = 67 à 218 frames, `VoxtralVoiceSLERP.swift:4`) est fixée par le preset ou l'enrôlement : c'est un levier de qualité | — |
| T9 tranche de préfill | non applicable | Invite = voix (≤ ≈ 220 positions, en cache) + texte ; pas de `lm_head`, donc aucun tenseur de logits de vocabulaire. Le pic de F-3 = matérialisation des poids (P-43), pas des activations | — |
| T10 KV quantifié | absente, **non prioritaire** | KV bf16 = 104 Ko par position (26 × 2 × 8 × 128 × 2 o) ; ≤ 0,3 Go à 2 800 positions | — |
| T11 KV préalloué, écriture en place | **appliquée (amont)** | `KVCacheSimple` (`VoxtralTTSModeling.swift:312-314`) : préallocation par pas de 256 et `slice_update` (`mlx-swift-lm KVCache.swift:421-464` @ `ee673d6`). `cloneKVCaches` (`:697-704`) coûte une réallocation par synthèse (≈ 23 Mo) | Dépend de `branch: main` non épinglé (S-18) |
| T12 tête restreinte / quantifiée | quantifiée : **appliquée** (4 et 6 bits, F-12) ; restriction : non applicable | Restreindre la tête sémantique aux 8 194 lignes légales sur 8 320 retirerait 1,5 % de la tête | En bf16, cette tête est promue fp32 (P-30) |
| T13 quantification mixte par voie | **absente** | LLM + FM uniformes, codec bf16 (F-12) | P-48 |
| T14 dé-quantifier pour une étape bornée par le calcul | **non applicable** | Codec **non quantifié** (F-12, F-4). FM à M = 6 lignes (2 × 3 positions) et LLM à 1 position : bornés bande passante et dispatch, le packé gagne. Préfill texte court | #29 **réfuté** : ce n'est pas un levier |
| T15 pipelining `asyncEval` | **absente** | 0 `asyncEval` ; `MLX.eval(xt)` à chaque frame (`VoxtralFlowMatching.swift:321`) ; streaming : `.item()` et `eval` à chaque frame (`VoxtralTTSModeling.swift:631`, `:670`) | P-31 |
| T16 `eval` par couche | **partielle** | Un `eval` par frame pour 26 couches LLM + 21 passes FM : graphe borné, correct. Le codec est un seul graphe sur toute la séquence (`VoxtralTTSPipeline.swift:234-235`, `:350-351`) | P-32, P-33 |
| T17 fuites fp32 | **partielle** | Le LLM reste bf16 : recast de la voix (`VoxtralTTSModeling.swift:369`), RoPE et SDPA MLXFast (`Models/VoxtralLlama.swift:122-154`). En revanche FM et tête sémantique en fp32 (P-30), codec en fp32 (P-38) | Voir §3 |
| T18 compile d'activation | **absente** | 0 `compile` dans le dépôt, mais `silu` est déjà compilé par MLXNN (faux négatif de scan.py, cf. A-01) | P-49 ; risque ABBA (A-01) |
| T19 KV entre étapes | non applicable | Le FM ne lit que l'état caché de la dernière position (`VoxtralTTSModeling.swift:490`), pas le KV | — |
| T20 politique par étape | **partielle** | De fait : LLM et FM quantifiés (bornés bande passante), codec bf16 (calcul). Le gain bf16 est annulé par la promotion fp32 du FM et du codec | P-30, P-38 |
| T21 nombre de pas | **absente (non exposée)** | `nDenoisingSteps = 8` (7 pas d'Euler) et `cfgAlpha = 1.2` en dur (`VoxtralFlowMatching.swift:195-196`) | P-35 |
| T22 `pread` / `F_NOCACHE` | absente, faible applicabilité | `MLX.loadArrays` (`VoxtralTTSModelLoading.swift:118-145`) ; poids résidents | Temps de chargement jamais mesuré |
| T23 reprise / porte GPU iOS | absente | VoxtralCore iOS 17+, aucune porte GPU côté synthèse (piège 22) | Seulement si un hôte iOS synthétise ; aucun consommateur iOS connu |

**Rejets du catalogue confrontés au dépôt** :
- **R15 (CFG batché)** : rejeté chez Y, où le CFG était inactif. **Retenu ici**, et mesuré (F-6) → à capitaliser.
  Vérification croisée : F-6 mesure trois changements ensemble (CFG batch 2, un `eval` par intégration, préfill
  fusionné). La part du CFG batché seul n'est pas isolée ; à capitaliser comme tel.
- **R1 et R2 (compiler le pas)** : ne pas compiler le pas AR ou FM par défaut (P-49).

## 3. Constantes fp32 : voulu ou fuite ?

Les 22 occurrences MLX-002 (sur 28) du chemin TTS et de l'enrôlement (apply.py, `--max 100`) ont été relues une à une.
Aucune ne promeut à elle seule un tenseur bf16. **La promotion réelle vient d'ailleurs.**

| Ligne | Nature | Verdict |
|---|---|---|
| `VoxtralFlowMatching.swift:135` (`invFreq` fp32) | table sinusoïdale | **Voulu**, identique à la référence (`acoustic_head.py:119-121`). Mais la sortie de `timeEmbedding` est fp32 (`:142`, `:146-148`) et nourrit `timeProjection` : elle participe à la promotion de P-30 |
| `:324`, `:326` (clip et round FSQ sur `xt`) | état d'Euler fp32, sortie `int32` | **Voulu**, sans fuite (tenseur (1, 36)) |
| `:341-342`, `:348` (`quantizeToFSQ`, `dequantizeFSQ`) | utilitaires fp32 | **Voulu** ; hors du chemin chaud de synthèse |
| `VoxtralCodecDecoder.swift:219`, `:224` (masques −1e9 et 0) | ajoutés à des scores déjà fp32 | **Voulu**, même softmax fp32 que la référence (`audio_tokenizer.py:289-300`). Le coût vient de la taille T×T (P-32), pas du dtype |
| `:339` (centroïdes `embedding_sum / cluster_usage` en fp32) | codebook | **Voulu**, identique à la référence (`audio_tokenizer.py:390-392`). Conséquence : **tout le codec** tourne en fp32 (P-38) |
| `:360` (décodage FSQ en fp32) | idem | **Voulu** (`audio_tokenizer.py:411`) ; même conséquence |
| `VoxtralVoiceSLERP.swift:53-92` | SLERP en fp32 explicite (`:49-50`, `:89`) | **Voulu** ; recasté au dtype des embeddings (`VoxtralTTSModeling.swift:369`) ; N ≤ 218 lignes |
| `VoxtralTTSProcessor.swift:362` | écriture WAV | Voulu |
| `VoxtralEnrollmentLosses.swift:124-125`, `VoxtralCodecEncoder.swift:189` | accumulateur de perte ; code mort | Voulu ; hors du chemin |

Les **sources réelles de promotion**, non détectées par MLX-002 :
- `llmOutput.asType(.float32)` (`VoxtralFlowMatching.swift:289`) ;
- l'état `xt` et l'embedding temporel en fp32, projetés par des `Linear` (`:295`, `:232-234`) ;
- `* MLXArray(scale)` avec `scale: Float` (`VoxtralFlowMatching.swift:64`, `VoxtralCodecDecoder.swift:210`) : scalaire
  fp32 « fort », hors de portée de la regex, qui exige `Float(` ou `dtype: .float32` ;
- l'entrée fp32 du codec (`:442`).

## 4. Constats

### P-30 · haute · `TTS/VoxtralFlowMatching.swift:289` — FM et tête sémantique en fp32 : cast de poids à chaque appel

- **Constat** : `decodeOneFrame` caste l'état caché en fp32 (`:288-289`, « critical for flow matching stability ») et
  fait tourner tout le FM avec des activations fp32. S'y ajoutent `xt` (`:295`), l'embedding temporel (`:142-148`)
  et `* MLXArray(scale)` (`:64`). Chaque `Linear` du FM reçoit donc une entrée fp32 : 7 pas × (3 couches × 7 +
  4 projections), plus la tête sémantique (`:248`).
- **Preuve** :
  - **Pack bf16** : `Linear.callAsFunction` fait `matmul(x, weight.T)` (mlx-swift `Linear.swift:129`) et `matmul`
    caste l'opérande de type différent (`ops.cpp:3492-3507`). Chaque appel matérialise donc une copie fp32 du poids.
    Cela fait 176 copies par frame, ≈ 26,0 Go déplacés par frame contre 5,2 Go en bf16 (§0.2).
  - **Packs 4 et 6 bits** : `quantized_matmul` promeut au dtype de `x` et caste `scales` et `biases` (`ops.cpp:4803-4818`).
    Cela ajoute ≈ 340 noyaux `astype` par frame (169 appels quantifiés × 2) et ≈ 0,65 Go par frame de trafic
    supplémentaire, sur un chemin déjà borné par le CPU (F-1 : 84 % CPU).
  - La référence Python promeut aussi, **implicitement**, via `t.astype(mx.float32)` et le bruit fp32
    (`acoustic_head.py:128`, `:171-175`, `:209`), mais **ne caste pas** l'état caché (`:205`, `:224`).
  - Le fp32 n'est donc pas une exigence du modèle mais un artefact de promotion MLX. Le modèle d'origine (PyTorch,
    `nn.Linear` sans promotion implicite) tourne vraisemblablement dans le dtype des poids : **À VÉRIFIER**.
- **Correction** :
  - garder en fp32 ce qui est petit et sensible : `xt`, `v`, combinaison CFG, pas d'Euler, logits sémantiques
    (`.asType(.float32)` **après** la tête, comme `acoustic_head.py:185-187`) ;
  - faire tourner le transformeur dans le **dtype de calcul** des poids : caster `x` après le `stacked` (`:237`), et
    l'état caché avant `llmProjection` ;
  - remplacer `MLXArray(scale)` par un scalaire Swift (faible) ou par `dtype: x.dtype` ;
  - dtype de calcul lu sur `scales` si le module est quantifié, **jamais** sur `weight` (MLX-009 : `uint32` packé).
- **Gain attendu** : pas bf16 de 240,7 ms (F-1) divisé par 2 à 2,8 (octets par frame 32,1 → 11,3 Go, estimation
  §0.2). En 4 et 6 bits : −340 dispatches par frame, attendu 5 à 15 % (estimation ; catalogue T17 : ×2,74 sur Q
  pour une fuite comparable).
- **Risque** : numérique. Les 21 niveaux FSQ (pas de 0,1 sur [−1, 1]) absorbent une petite erreur bf16, mais
  l'erreur se propage par la rétroaction AR. **API** : aucun.
- **Effort** : M. **Statut** : VÉRIFIÉ (mécanisme, 3 dépôts lus), À MESURER (gain, qualité).
- **Fiche K-P30** — *FM dans le dtype des poids*.
  - **Porte** : pas bf16 **−40 %** au moins, 4 bits −5 % au moins (A/B/B/A, graine fixée, Release) ;
  - parité **forcée par l'enseignant** (les mêmes états cachés du LLM passés aux deux variantes du FM) : codes
    acoustiques identiques ≥ 99 %, écart ≤ 1 niveau, codes sémantiques identiques 100 % ;
  - couverture ASR (`TTSQuantizationCampaignTests`, 3 textes × 5 graines) ≥ référence −0,5 point ; 0 `maxFrames` ;
    écoute à l'aveugle (ASK).
  - **Cible** : macos-gpu.
- **Vérification croisée (amendé)** :
  - Mécanisme confirmé dans l'amont :
    - `matmul` : `ops.cpp:3493` puis `:3502-3507` ;
    - `quantized_matmul` : `:4803-4804`, puis `:4818` (`astype(scales)` et `astype(biases)`) ;
    - `conv_general` : `:4591-4593` ;
    - `Linear.swift:129`, `Quantized.swift:369-379`.
  - Une copie de `ops.cpp` attribuée au sous-module du tag 0.31.6 (scratchpad de session, provenance non
    re-vérifiée) contient les mêmes règles.
  - La référence Python fait, elle aussi, tourner le transformeur FM en fp32 : `x_t` et `t` sont fp32, puis
    `mx.stack` (`acoustic_head.py:171-176`). Le passage au bf16 **s'écarte donc de la référence** ; la parité
    forcée par l'enseignant et l'écoute de la porte sont indispensables. Seule la tête sémantique en fp32 est un
    écart du port Swift (la référence ne caste qu'après la tête, `:185-187`).
  - Correction complétée : caster aussi l'embedding temporel **avant** `timeProjection` (ou le précalculer,
    P-36), sinon cette projection de 3 072 × 3 072 reste en fp32 aux 7 pas.
  - Gain : la base « 240,7 ms » est F-1, **périmé**. Le ratio d'octets (32,1 → 11,3 Go par frame) reste le seul
    appui de l'estimation « ÷ 2 à 2,8 », qui est un ordre de grandeur.
  - Consommateur : FluxForge ne livre que le 6 bits. Pour lui, le gain attendu est celui de la partie quantifiée
    (−338 `astype` par frame, 5 à 15 % attendus).
  - Sévérité haute maintenue : trafic inutile sur le chemin de référence, et ≈ 338 noyaux de plus par frame sur
    tous les packs quantifiés.

### P-31 · haute · `TTS/VoxtralFlowMatching.swift:321` — Boucle AR synchrone, aucun `asyncEval` (T15)

- **Constat** :
  - **Batch** : `decodeOneFrame` termine par `MLX.eval(xt)` (`:321`). Chaque frame force donc l'évaluation du
    forward LLM de la frame précédente et des 7 pas du FM. Le commentaire « Feed through LLM — no eval() sync between
    EOA checks » (`VoxtralTTSModeling.swift:528`, et `:480-482`) est faux. En plus : `MLX.eval(hidden)` tous les 4
    frames (`:532-534`) et jusqu'à 4 `.item()` par contrôle d'EOA (`:509-511`).
  - **Streaming** : `codes[0,0].item()` à **chaque** frame (`:631`), plus `MLX.eval(hidden)` à chaque frame (`:670`).
    L'optimisation `a00024f` (F-5) n'existe que sur le chemin batch.
  - Aucun recouvrement entre la construction du graphe de la frame i+1 (≈ 360 appels `Linear` et leurs opérations,
    §0.2) et l'exécution GPU de la frame i. F-1 montre pourtant un chemin 4 bits borné par le CPU (84 % CPU, 28 % GPU).
- **Correction** :
  - `decodeOneFrame` rend des codes paresseux (on retire `:321`) ;
  - la boucle construit la frame i+1 sur `codes_i` paresseux, lance `asyncEval(codes_{i+1})`, **puis** lit l'EOA
    de la frame i ;
  - un seul `.item()`, ou un seul `asArray`, sur les codes sémantiques empilés des k dernières frames ;
  - un seul itérateur de frames partagé par le batch et le streaming.
  - Le débordement après l'EOA reste borné, comme aujourd'hui (≤ 3 frames, `:482`). Les tirages du RNG global
    gardent le même ordre (un par frame), donc la sortie reste bit-exacte à graine fixe.
- **Gain attendu** : −13 à −22 % de ms par pas (catalogue T15, `Q log.md:1301-1345`). Streaming : au moins le gain
  de F-5 (+10 % de fr/s).
- **Risque** : piège 6 (chemin async mort : vérifier que le GPU % monte) ; piège 3 (pas de mutation en place sur un
  tableau non évalué). **API** : aucun ; `onFrame` (`:433`, `:500`) reçoit des codes paresseux (déjà le cas).
- **Effort** : M. **Statut** : VÉRIFIÉ, gain À MESURER.
- **Fiche K-P31** — *Pipelining asyncEval du TTS (batch et streaming)*.
  - **Porte** : fr/s +5 % au moins en 4 et en 6 bits (A/B/B/A) ; codes identiques bit à bit à graine fixe (batch
    et streaming) ; GPU % par frame en hausse (`ioreg` ou trace) ; streaming ≥ batch −3 %.
  - **Cible** : macos-gpu.
- **Vérification croisée (amendé)** :
  - Mécanisme confirmé : `MLX.eval(xt)` (`:321`) évalue aussi le forward LLM de la frame, puisque `xt` dépend
    de `h`. Streaming : `:631` et `:670`. 0 `asyncEval` dans `Sources/`.
  - Prémisse corrigée : le « 84 % CPU, 28 % GPU » de F-1 a été mesuré **avant** `a00024f` et `0be05af`, avec
    ≈ 8 synchronisations par frame au lieu d'une. Il ne prouve plus que le 4 bits actuel est borné par le CPU.
  - Appui actuel : le message de `0be05af` (2026-07-10), « first-token path dominated by GPU sync round-trips,
    not compute », et le catalogue T15.
  - Sévérité haute maintenue : c'est le levier de la boucle AR des packs 4 et 6 bits (ceux du consommateur),
    mais le gain reste entièrement À MESURER. Si K-P45 montre un GPU déjà saturé, la fiche est déclassée.

### P-32 · haute · `TTS/VoxtralCodecDecoder.swift:210-231` — Attention du codec en T×T pour une fenêtre ≤ 16

- **Constat** : `CodecAttention` calcule `scores` (B, H, T, T), `alibi` (H, T, T) et `dist`, `causalMask`,
  `windowMask` (T, T), tous en fp32 (`:210-228`), pour ne garder que 17 clés par requête (fenêtre 2, 4, 8 ou 16,
  `:454`, `:467`). Ces tenseurs sont reconstruits dans **chacune** des 2 couches de chaque étage (`:213-226` dans
  `callAsFunction`, `:310-315`). Au dernier étage, T = 8 × frames. Le décodage batch traite toute la séquence d'un
  coup (`VoxtralTTSPipeline.swift:234`, `:350`).
- **Preuve (calcul)** : un seul tenseur (1, 8, T, T) fp32 pèse 0,07 Go à 190 frames, 1,15 Go à 750 frames (60 s),
  4,6 Go à 1 500 frames, **10,5 Go à 2 266 frames** (le texte long EN de F-8) et 12,8 Go à `maxFrames`. Plusieurs
  coexistent (scores, alibi, somme, softmax). Le banc n'a pas mesuré le pic : il tournait sur 96 Go.
- **Correction** : attention **par bandes** (blocs de 128 à 256 requêtes, clés [début − W, fin), ou gather des W + 1
  voisins), exacte puisque les entrées masquées valent exp(−1e9) = 0 ; `dist`, ALiBi et masque calculés une fois par
  étage. À combiner avec le décodage par fenêtres de P-33 (T16 : un `eval` par bloc).
- **Gain attendu** : transitoire du décodage en O(T·W) au lieu de O(T²). À 2 266 frames, le tenseur de scores passe
  de ≈ 10,5 Go à ≈ 10 Mo ; temps du décodage long réduit (À MESURER).
- **Risque** : parité numérique (ordre des réductions). **API** : aucun.
- **Effort** : M. **Statut** : VÉRIFIÉ (structure et calcul), pic À MESURER.
- **Fiche K-P32** — *Attention du codec par bandes*.
  - **Porte** : forme d'onde |Δ|max ≤ 1e-4 contre l'actuel sur 3 textes (court, 60 s, long quand il tient) ; pic
    `phys_footprint` du décodage long ≤ poids + 1 Go (contre ≥ 10 Go attendus) ; temps de décodage court ±5 %.
  - **Cible** : macos-gpu.
- **Vérification croisée (gardé)** :
  - Formes recalculées avec `params.json` : 8 têtes, `head_dim` 128, T = 8 × frames au dernier étage.
  - `alibi` (8, T, T) fp32 (`:215`) et `scores` doivent coexister pour l'addition de `:228`. Le transitoire
    **minimal** est donc ≈ 2 × 10,5 Go à 2 266 frames. Le constat est plutôt sous-estimé.
  - Chemin atteint par le consommateur : batch `synthesize(text:voiceEmbedding:)` (`:350`).

### P-33 · moyenne (était haute) · `TTS/Pipeline/VoxtralTTSPipeline.swift:583` — Le streaming re-décode tout l'accumulé

- **Constat** :
  - Chaque chunk renvoie **tous** les codes accumulés : `MLX.stacked(allCodes, axis: 1)`,
    `VoxtralTTSModeling.swift:636`, `:654`, `:675`, et API `GenerationChunk.accumulatedCodes`, `:555-564`.
  - La pipeline décode alors toute la séquence (`decodeToWaveform(chunk.accumulatedCodes)`, `:583-584`) pour n'en
    garder que la fin (`:637-644`).
  - Le travail du codec devient O(N²/chunk), et même O(N³) à travers P-32.
  - Texte long (2 266 frames, chunks de 10) : Σ 10k ≈ 258 800 frames décodées au lieu de ≈ 7 700 en incrémental,
    **÷ 33** (calcul).
  - Aujourd'hui ce coût tombe après la génération (S-08) ; une fois S-08 corrigé, il concurrencera la génération en
    temps réel.
- **Preuve de faisabilité** : le codec est causal à champ récepteur borné :
  - convolutions causales à gauche (`:77-88`) ;
  - convolutions transposées tronquées (`:92-106`) ;
  - attention causale à fenêtre 2, 4, 8, 16 sur 2 couches par étage ;
  - ALiBi invariant par translation.
  - Estimation à la lecture : ≈ 24 frames de contexte gauche suffisent (**à confirmer par test**).
- **Correction** :
  - décoder [chunk + L frames de contexte] et jeter les échantillons du contexte ;
  - n'empiler que ces frames ;
  - ajout **additif** d'un champ `newCodes` (ou d'une variante) à `GenerationChunk`, sans retirer
    `accumulatedCodes` (public).
- **Gain attendu** : coût du codec par chunk constant ; travail total ÷ 33 sur le texte long (calcul).
- **Risque** : L trop petit, donc discontinuités → test d'égalité avec le batch. **API** : additif.
- **Effort** : M. **Statut** : VÉRIFIÉ, À MESURER.
- **Fiche K-P33** — *Décodage incrémental exact*.
  - **Porte** : forme d'onde concaténée du streaming = batch (|Δ|max ≤ 1e-4) sur 3 textes et 3 graines ; temps de
    décodage par chunk constant ±20 % du chunk 1 au chunk 200 ; RTF streaming du texte long ≤ RTF batch +5 %.
  - Prérequis : K-S08. **Cible** : macos-gpu.
- **Vérification croisée (amendé : sévérité haute → moyenne)** :
  - Mécanisme et faisabilité confirmés :
    - convolutions transposées tronquées à `T × stride` : la sortie ne dépend que des entrées passées ;
    - fenêtres 2, 4, 8 et 16 sur 2 couches par étage : ≈ 16 frames de contexte, plus les noyaux 3, 4, 4, 4 et 7
      (`params.json`) ; l'estimation de ≈ 24 frames est plausible.
  - Mais la seule cliente est la démo interne :
    - `synthesizeStreaming` n'est appelé que par `StreamingDemoViewModel.swift:507-510` et par les tests ;
    - FluxForge ne l'appelle pas (0 résultat de recherche de code) ;
    - aujourd'hui, le coût tombe après la génération (S-08).
  - Même règle que S-08, lui aussi ramené à moyenne par la vérification croisée de l'audit stabilité.

### P-34 · basse (était moyenne) · `TTS/VoxtralTTSRegistry.swift:29-38` — Le défaut est le bf16, le plus lent

- **Constat** :
  - `tts-4b-mlx` (bf16, ≈ 8 Go) porte `recommended: true` depuis `cc77c86` (2026-03-28), donc `defaultModel`
    (`:68-70`).
  - Même défaut pour la CLI `tts` et `enroll` (`VoxtralTranscriptionTest/VoxtralCLI.swift:399-400`, `:587-588`) et
    pour `profile` (`ProfileCommand.swift:51-52`).
  - `VoxtralTTSSynthesisManager.loadModel(progress:)` (`:52-54`) n'accepte **aucun** modèle : l'API « simple »
    charge toujours le bf16, dont le RTF va de 4,86 à 6,86 (F-8) et vaut 3,44 en voix clonée (F-9).
  - Le commentaire du propriétaire sur #27 (« `tts-4b-4bit` is the default in CLI and registry ») est contredit par
    le code. La démo, elle, part du 4 bits (`StreamingDemoViewModel.swift:14`).
  - #45 conserve **volontairement** le bf16 par défaut (qualité de référence), alors que la campagne donne au 6 bits
    une meilleure couverture à 2,3× la vitesse (F-9).
- **Correction** : additif, `loadModel(modelInfo:progress:)` sur le manager. Changer le défaut est une décision de
  produit, donc **ASK**, à trancher avec les profils de la phase 3.
- **Risque** : API additif ; changer le défaut modifie la qualité perçue par les consommateurs → ASK.
- **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-P34** — *Choix du modèle dans le manager + défaut décidé*.
  - **Porte** : test unitaire, le manager charge le modèle demandé ; décision ASK consignée (`docs/knowledge/decisions`).
  - **Cible** : ~~cloud (code sûr, additif, `syntax_guard`) puis build macos-gpu~~ → **macos-gpu** (la porte est
    un test). Le code additif peut être rédigé dans le cloud.
- **Vérification croisée (amendé : sévérité moyenne → basse, cible)** :
  - Code confirmé :
    - `recommended: true` sur `tts-4b-mlx` (`:37`) ; `defaultModel` (`:68-70`) ;
    - CLI : `VoxtralCLI.swift:400` et `:588` ; `profile` : `ProfileCommand.swift:52` ;
    - `VoxtralTTSSynthesisManager.loadModel` (`:52-54`) appelle `pipeline.loadModel(progress:)` sans `modelInfo` ;
    - la contradiction avec le commentaire de #27 est confirmée.
  - Sévérité ramenée à basse, pour trois raisons :
    1. le défaut est **délibéré et documenté** (`docs/voice_cloning.md:116-117` : « The defaults are bf16 because
       it is the safe reference » ; clôture de #45 : « Defaults stay bf16 ») ;
    2. le consommateur connu choisit explicitement le 6 bits (FluxForge : `VoxtralTTSVariant.q6`, « The only
       variant we ship ») et n'utilise pas le manager (0 résultat de recherche de code) ;
    3. la pipeline accepte déjà `modelInfo`.
  - Le défaut ne pèse donc que sur la CLI, `profile` et les intégrateurs qui gardent les valeurs par défaut.

### P-35 · moyenne · `TTS/VoxtralFlowMatching.swift:195-196` — T21 non exposé ; réglages publics ignorés

- **Constat** :
  - `nDenoisingSteps = 8` et `cfgAlpha = 1.2` sont codés en dur.
  - `VoxtralTTSPipeline.Configuration.flowSteps`, `cfgAlpha` et `temperature` (`:26-28`, `:43-53`) ne sont **lus
    nulle part** (grep), pas plus que les options CLI `--flow-steps`, `--cfg-alpha` et `-t` (`VoxtralCLI.swift:423-430`,
    `:461-464`).
  - `docs/tts_benchmark.md:10` présente « cfgAlpha=1.2, flowSteps=8, temperature=0.0 » comme une configuration.
  - Conséquence : le levier T21 (le FM fait la moitié des appels d'une frame, §0.2) n'est pas testable, et un
    utilisateur qui règle ces options croit en changer l'effet.
- **Correction** :
  - faire passer `flowSteps` et `cfgAlpha` jusqu'à `decodeOneFrame` (paramètres, défauts inchangés) ;
  - `temperature ≠ 0` : erreur explicite ou échantillonnage réel ;
  - balayer 8, 6, 5 et 4 pas de temps.
- **Gain attendu** : passer de 7 à 4 pas d'Euler ramène les appels `Linear` de la frame de 358 à 283 (−21 %), soit
  environ −15 à −20 % de ms par frame en 4 bits (estimation). Catalogue T21 : ODE 32 → 16 = 66-69 → 43,9 s chez Y.
- **Risque** : qualité, donc écoute obligatoire. **API** : comportement des champs existants (correctif), pas de
  nouveau symbole.
- **Effort** : S (câblage) + M (campagne). **Statut** : VÉRIFIÉ.
- **Fiche K-P35** — *Pas de flow matching réglables + balayage*.
  - **Porte (câblage)** : le défaut 8 donne une sortie bit-exacte à graine fixe.
  - **Porte (retenue d'une valeur < 8)** : fr/s +10 % au moins ; couverture ASR ≥ référence −0,5 point (12 textes ×
    3 graines) ; 0 `maxFrames` ; écoute à l'aveugle non inférieure (ASK). Sinon, la valeur est rejetée et documentée.
  - **Cible** : ~~cloud pour le câblage, macos-gpu pour la mesure~~ → **macos-gpu** (voir ci-dessous).
- **Vérification croisée (amendé : ligne, cible)** :
  - Confirmé par grep : aucune lecture de `configuration.flowSteps`, `.cfgAlpha` ni `.temperature` dans le chemin
    TTS. Les seules occurrences sont les déclarations, l'`init` et les tests de valeur
    (`VoxtralTTSPipelineTests.swift:17-44`, `:185`).
  - Ligne corrigée : l'affectation CLI ignorée est `VoxtralCLI.swift:462-464` ; `:461` (`maxFrames`) est lu.
  - Cible : **macos-gpu** pour toute la fiche, puisque même la porte du câblage (sortie bit-exacte à graine fixe)
    exige un build et une exécution.
  - Seule la partie documentaire (`docs/tts_benchmark.md:10`, qui présente ces réglages comme effectifs) est
    faisable dans le cloud.

### P-36 · moyenne · `TTS/VoxtralFlowMatching.swift:232-233` — Invariants du FM recalculés à chaque pas

- **Constat** :
  - `llmProjection(llmBoth)` est recalculé aux 7 pas alors que `llmBoth` ne change pas (`:303`, `:311` → `:233`).
  - `timeProjection(timeEmbedding(t))` (`:232`) ne dépend que des 7 pas de temps fixes (`:297`).
  - `MLX.full` et `concatenated([xt, xt])` sont refaits à chaque pas (`:308-309`).
  - Le masque sémantique est reconstruit sur CPU (tableau Swift de 8 320 flottants), puis téléversé à **chaque**
    frame (`:266-273`).
- **Correction** : table (7, 3 072) des projections temporelles précalculée au chargement (dans le dtype de calcul,
  P-30) ; projection de l'état caché une fois par frame ; masque constant construit une seule fois.
- **Gain attendu** : −13 `Linear` 3 072 × 3 072 par frame (123 M paramètres, 4,7 % des octets du FM) et environ
  −50 opérations par frame. Moins de 5 % seul, d'où le regroupement avec P-37.
- **Risque** : parité à l'arrondi près (autres formes de matmul). **API** : aucun.
- **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-P36** (P-36 + P-37) — *Surcoût hôte du FM*.
  - **Porte** : fr/s +5 % au moins en 4 bits (A/B/B/A) ; codes acoustiques forcés par l'enseignant identiques
    ≥ 99,9 % ; sinon, retrait.
  - **Cible** : macos-gpu.
- **Vérification croisée (amendé)** :
  - `concatenated([xt, xt])` (`:309`) **n'est pas** un invariant : `xt` change à chaque pas. On peut seulement le
    remplacer par une diffusion, sans gain notable ; ce sous-point est retiré.
  - Les invariants réels restent : `timeProjection(timeEmbedding(t))` (7 valeurs fixes) ; `llmProjection(llmBoth)`
    (1 par frame au lieu de 7) ; `MLX.full` (`:308`) ; le masque sémantique (`:266-273`).
  - Le compte de −13 `Linear` par frame est exact (6 `llmProjection` + 7 `timeProjection`).

### P-37 · moyenne · `TTS/VoxtralFlowMatching.swift:48-69` — Attention du FM non fusionnée

- **Constat** : répétition GQA de K et V (`MLX.repeated` ×2, `:57-61`), matmul, `* MLXArray(scale)` (scalaire fp32,
  `:64`), softmax, matmul : environ 6 opérations là où `MLXFast.scaledDotProductAttention` n'en fait qu'une. Le LLM
  l'utilise déjà (`Models/VoxtralLlama.swift:148-154`), et elle gère le GQA sans masque. Cela fait 21 passes par
  frame.
- **Correction** : `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask: .none)` sans répétition.
- **Gain attendu** : environ −105 dispatches par frame sur un chemin borné par le CPU (estimation). Voir K-P36.
- **Risque** : parité à l'arrondi près. **API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Vérification croisée (gardé)** :
  - Le FM a 32 têtes Q et 8 têtes KV (`params.json`) ; `MLXFast.scaledDotProductAttention` gère ce GQA sans
    répétition, et le LLM du même dépôt l'utilise (`Models/VoxtralLlama.swift:148-154`).
  - Le compte de ≈ 105 dispatches par frame (5 × 21) est cohérent.

### P-38 · basse · `TTS/VoxtralCodecDecoder.swift:442` — Codec entièrement en fp32

- **Constat** : `quantizer.decode` produit des embeddings fp32 (`:336-341`, `:358-361`), voulus par la référence.
  Toutes les convolutions (`conv_general` caste, `ops.cpp:4591-4593`) et les 56 `Linear` du codec en héritent, avec
  cast du poids bf16 à chaque appel. `MLXArray(scale)` (`:210`) garde les scores en fp32. La note de
  `docs/tts_benchmark.md:22` (« it always runs in bf16 ») est inexacte : les **poids** sont bf16, le **calcul** est
  fp32.
- **Correction** : caster l'entrée du décodeur dans le dtype des poids après le lookup (`:442`), en gardant le
  softmax en fp32 (`:231`).
- **Gain attendu** : décodage −30 à −50 % (estimation). Absolu faible en batch (66-131 ms par synthèse courte, F-4),
  plus fort en streaming et sur les textes longs.
- **Risque** : **précision audio** (c'est la dernière étape avant la forme d'onde) → porte de qualité stricte.
  **API** : aucun.
- **Effort** : S. **Statut** : VÉRIFIÉ, À MESURER.
- **Fiche K-P38** (P-38 + P-39) — *Codec en bf16 et invariants*.
  - **Porte** : décodage −20 % au moins (A/B/B/A) ; SNR ≥ 40 dB contre fp32 **et** couverture ASR identique **et**
    A/B à l'aveugle non inférieur (ASK) ; test : sortie de chaque étage dans le dtype des poids (ajout de la
    vérification croisée) ; sinon, on ne garde que les invariants de P-39.
  - **Cible** : macos-gpu.
- **Vérification croisée (amendé : correction incomplète)** :
  - Constat confirmé : entrée fp32 (`:337-340`, `:360`), `conv_general` promeut (`ops.cpp:4591-4593`), softmax
    rendu en `x.dtype` = fp32 (`:231`).
  - Mais caster seulement à `:442` **ne suffit pas**. Le padding causal `MLX.zeros([B, K - 1, C])` (`:83`) est en
    fp32 par défaut, et `concatenated` promeut : dès la première convolution (noyau 3), tout le décodeur
    repasse en fp32.
  - Correction complète :
    - `MLX.zeros(…, dtype: x.dtype)` à `:83` (ou padding sans concaténation, P-39) ;
    - scalaire faible à la place de `MLXArray(scale)` (`:210`) ;
    - `.asType(x.dtype)` après le softmax (`:231`), déjà présent.
  - Un test de dtype de la sortie de chaque étage doit faire partie de la porte.
  - La correction de `docs/tts_benchmark.md:22` (« it always runs in bf16 ») est documentaire et faisable dans le
    cloud.

### P-39 · basse · `TTS/VoxtralCodecDecoder.swift:59-64` — Recalculs à chaque décodage

- **Constat** :
  - `getWeight()` recalcule la weight norm g·v/‖v‖ à **chaque** appel des 5 convolutions (`:59-64`, `:69` ;
    ≈ 15 M paramètres), à chaque décodage et donc à chaque chunk en streaming.
  - Les centroïdes `embedding_sum / cluster_usage` (8 192 × 256) sont recalculés à chaque `decode` (`:336-341`).
  - `dist`, ALiBi et les masques sont recalculés par couche (P-32).
  - Le padding causal passe par `concatenated` avec des zéros, une copie de l'activation par convolution (`:83`).
- **Correction** :
  - mettre en cache le poids normalisé et les centroïdes, invalidés sur `update(parameters:)` ;
  - les poids du décodeur sont figés pendant l'enrôlement, où le gradient passe par les embeddings
    (`:445-449`), donc un cache n'y change rien ;
  - pour le padding : `padded` ou convolution à padding asymétrique.
- **Gain attendu** : faible (moins de 5 % du codec, estimation). Voir K-P38.
- **Risque** : API aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Vérification croisée (amendé : précision, risque)** :
  - Les 5 convolutions à weight norm (≈ 15,2 M paramètres d'après `params.json`) sont confirmées.
  - Le padding par `concatenated` (`:83`) ne touche que les **2** convolutions non transposées à noyau > 1
    (`conv0`, noyau 3, et `output_proj`, noyau 7). Les 3 convolutions transposées n'ont pas de padding
    (`:92-106`). Ce n'est donc pas « une copie par convolution ».
  - Risque ajouté : dans mlx-swift, une propriété `MLXArray` stockée sur un `Module` est découverte par
    réflexion (`Module.swift:1720-1735`). Le cache (poids normalisé, centroïdes) doit donc porter un nom préfixé
    par `_`, filtré par `Module.parameterIsValid` (`:1058-1060`), ou vivre hors de la réflexion. Sinon il
    apparaît dans `parameters()`, avec des effets sur `update` et la quantification. À couvrir par un test.

### P-40 · moyenne · `TTS/Pipeline/VoxtralTTSPipeline.swift:333-340` — Pas de cache de préfixe pour les voix clonées

- **Constat** :
  - La surcharge `synthesize(text:voiceEmbedding:seed:warmUpText:…)` n'utilise **jamais** le cache
    (`prefixCache` absent, `:333-340`). ZeroVoice et les mélanges y aboutissent (`:278-281`).
  - En streaming, le cache ne s'applique que si l'appelant passe un `voiceKey` (`:512-514`). La démo ne le fait pas
    pour les voix clonées (`StreamingDemoViewModel.swift:507-508`).
  - C'est pourtant le chemin de la chaîne LipDub de FluxForge : voix enrôlées avec graine et warm-up (#45) ;
    `recommendedWarmUpVocalise` consommé (audit-stabilite §0) ; `synthesizeStreaming` non utilisé.
  - Le cache n'a qu'**une entrée** (`:93`, `:99-102`) : deux voix en alternance le recalculent à chaque appel.
- **Correction** :
  - paramètre **additif** `voiceKey:` sur la surcharge batch ;
  - clé vérifiée par une empreinte (forme + somme de contrôle calculée une fois) pour éviter un préfixe périmé si
    une clé est réutilisée pour un autre embedding ;
  - LRU de 2 à 4 entrées (≈ 23 Mo chacune).
- **Gain attendu** : F-7 (−200 ms par synthèse sur les préréglages, **borne haute**, voir P-45) appliqué aux voix
  clonées : T + 1 frames de préfixe, ≈ 200 pour 16 s de référence.
- **Risque** : clé périmée (empreinte) ; équivalence numérique préfixe + suffixe contre préfill complet (déjà
  acceptée par `f4fd21c` à l'écoute). **API** : additif.
- **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-P40** — *Cache de préfixe pour toute voix*.
  - **Porte** : deuxième synthèse d'une voix clonée, préfill −30 % au moins (graine fixée, A/B/B/A) ; mêmes
    transcriptions ASR sur 3 textes × 3 graines ; une clé réutilisée avec un autre embedding est détectée par un
    test ; le test échoue sans le correctif (piège 38).
  - **Cible** : ~~cloud (code) puis macos-gpu~~ → **macos-gpu** (la porte est une mesure et des tests).
- **Vérification croisée (amendé : cible)** :
  - Confirmé : pas de `prefixCache` à `:333-340` ; ZeroVoice y aboutit (`:281`) ; la démo n'envoie pas de
    `voiceKey` pour les voix clonées (`StreamingDemoViewModel.swift:507-508`) ; le cache n'a qu'une entrée
    (`:93`, `:99-102`).
  - Chemin du consommateur confirmé par recherche de code (FluxForge) : `synthesize(…, warmUpText:)` avec une
    voix enrôlée, via `VoxtralTTSService.synthesize(text:embeddingURL:)`.
  - Le warm-up n'invalide pas le préfixe : la vocalise suit `<next>`, donc le préfixe reste la voix seule.

### P-41 · moyenne · `TTS/Pipeline/VoxtralTTSPipeline.swift:43` — Plafond de frames indépendant du texte

- **Constat** : `maxFrames = 2500` (200 s d'audio) quel que soit le texte (`:43`, `:46`). Un EOA manqué coûte
  2 500 frames : 106 s en 4 bits à 42,6 ms par frame, **≈ 10 min** en bf16 à 240,7 ms par frame (F-1). C'est
  observé : F-10 (197,8 s pour 2 phrases), F-8 (FR long, 4 bits et bf16 à 2 500), F-11 (mélanges, « 200s
  generation »).
- **Correction** : plafond `min(maxFrames, a + b·nJetonsTexte)` calé sur le corpus (distribution frames/jeton) ;
  en option, un détecteur d'emballement (K frames consécutives de code « silence » ou de boucle). Nouveau champ de
  configuration, additif.
- **Gain attendu** : les prises ratées coûtent au plus 3× la durée attendue au lieu de 200 s.
- **Risque** : tronquer un texte légitime → marge calibrée. **API** : additif.
- **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-P41** — *Plafond proportionnel au texte*.
  - **Porte** : 0 troncature sur 12 textes × 3 graines × 3 packs ; le reproducteur #45 (ou une voix dégénérée)
    s'arrête sous 3× la longueur attendue.
  - **Cible** : ~~cloud (code) puis macos-gpu~~ → **macos-gpu** (la porte est une campagne de 108 synthèses).
- **Vérification croisée (amendé : chiffres, cible)** :
  - Coûts : préférer les mesures aux calculs sur F-1 (périmé). F-8 a **mesuré** 2 500 frames en 158,75 s (4 bits)
    et 971,31 s ≈ 16 min (bf16) sur le FR long (`docs/tts_benchmark.md:52`, `:54`). C'était sur le code du
    2026-04-02 ; le coût actuel est À MESURER.
  - F-10 avait pour cause un embedding NaN, depuis rendu impossible (#44) : il illustre le coût du plafond, pas
    un cas courant.
  - Preuve supplémentaire, côté consommateur : `CLAUDE.md` de FluxForge, « without it [le warm-up] the voice runs
    away to the frame cap (babble) ».
  - Limite de la correction : sur le FR long de F-8 (≈ 2 175 frames attendues), un plafond à 3 × la longueur
    attendue n'aurait pas arrêté les prises à 2 500. Le plafond vise les emballements sur textes courts.

### P-42 · moyenne · `TTS/Pipeline/VoxtralTTSPipeline.swift:676-684` — Aucune politique mémoire sur le TTS

- **Constat** :
  - 0 `Memory.cacheLimit`, 0 `memoryLimit`, 0 `clearCache` dans `TTS/` (T1 à T3).
  - `unload()` lâche les références sans `Memory.clearCache()`. Les tampons libérés (8 Go de poids bf16, transitoires
    du codec de P-32) restent donc dans le cache MLX, par défaut jusqu'à la limite mémoire du process (piège 7).
  - FluxForge enchaîne le TTS avec d'autres modèles dans le même process (LipDub → LTX, #45).
  - Pics mesurés : 7,7 Go en bf16, 2,4 Go en 4 bits (F-2).
- **Correction** :
  - `Memory.clearCache()` dans `unload()` et après le décodage (profil `lean`) ;
  - `cacheLimit` par profil (étape AR : petits tampons de forme constante) ;
  - **opt-in** : ce sont des réglages **globaux au process**, et une bibliothèque embarquée ne doit pas les imposer
    à son hôte ; restaurer la valeur précédente (cf. A-17).
- **Gain attendu** : `phys_footprint` après `unload()` ramené à celui du process avant chargement (À MESURER).
- **Risque** : thrash si le plafond est trop bas (T2, piège 17). **API** : additif (profil).
- **Effort** : S. **Statut** : VÉRIFIÉ, effet À MESURER.
- **Fiche K-P42** — *Politique mémoire TTS opt-in*.
  - **Porte** : après `unload()`, footprint ≤ référence du process +200 Mo ; `fast` à temps ±5 % ; `lean` ≤ +5 % de
    temps pour un pic −20 % au moins sur le texte long.
  - **Cible** : macos-gpu.
- **Vérification croisée (gardé, preuve renforcée)** :
  - 0 `clearCache` ni `cacheLimit` dans `Sources/VoxtralCore/TTS/` (grep).
  - Le cache MLX garde les tampons libérés jusqu'à `min(1,5 × max_recommended_working_set, 0,95 × RAM)`
    (`allocator.cpp:52-54`).
  - Côté consommateur, FluxForge appelle déjà `MLX.Memory.clearCache()` avant chaque prévisualisation TTS, parce
    que « the warm model's buffer cache grows unbounded (~+4 GB) across repeated previews »
    (`GenerationQueueManager+VoiceTraining.swift`, recherche de code). Le défaut est donc observé chez l'hôte,
    et contourné par lui.

### P-43 · basse (était moyenne) · `TTS/VoxtralTTSModelLoading.swift:70-71` — Matérialisation paresseuse des poids pendant la 1re synthèse

- **Constat** : `model.update(parameters:verify: .none)` ne déclenche aucun `eval`. Les poids `MLX.loadArrays` sont
  lus depuis le disque au premier forward. Le « pic de préfill » de F-3 correspond à la taille des poids du LLM :
  3,026 G × 2 o = 6,05 Go en bf16 (mesuré +6,6 Go), × 0,5625 = 1,7 Go en 4 bits (mesuré +1,9 Go). Le préfill à
  19-40 % de GPU (F-3) est cohérent avec une lecture disque. Le TTFT de la première synthèse inclut donc la lecture
  des poids, alors que #28 la classe « inherent to model size ».
- **Correction** : option `prewarm` au chargement (`eval` tenseur par tenseur, piège 4, ou mini-forward), additive.
- **Gain attendu** : premier TTFT ramené au TTFT « chaud » ; le temps de chargement augmente d'autant (À MESURER).
- **Risque** : chargement plus long et pic de chargement. **API** : additif.
- **Effort** : S. **Statut** : VÉRIFIÉ (lecture + cohérence des tailles), À MESURER.
- **Fiche K-P43** — *Pré-chauffage optionnel des poids*.
  - **Porte** (amendée) : temps jusqu'au premier frame **préfixe inclus** (instrument K-P45) de la 1re synthèse
    ≤ 1,1 × celui d'une 2e synthèse avec une **autre** voix (préfixe froid dans les deux cas), sur une voix clonée
    et sur une voix prédéfinie ; hausse du temps de chargement rapportée.
  - **Cible** : macos-gpu.
- **Vérification croisée (amendé : sévérité moyenne → basse, constat, porte, statut)** :
  - Paresse confirmée :
    - `MLX.loadArrays` crée des tableaux à primitive `Load` (MLX `io/safetensors.cpp:209`) ;
    - aucun `eval` dans `loadWithConfig` (`:37-75`).
  - Le constat « le TTFT de la 1re synthèse inclut la lecture des poids » n'est vrai que **sans cache de
    préfixe**. F-3 (#28, 2026-04-11) est antérieur à `f4fd21c` (2026-07-10). Depuis, pour une voix prédéfinie, la
    lecture des poids du LLM tombe dans `voicePrefix` → `precomputeVoicePrefixCache`, dont le
    `MLX.eval(cache…)` (`VoxtralTTSModeling.swift:415`) s'exécute **avant** le chronomètre du TTFT (`:441`). Seuls
    les poids FM et codec restent dans le TTFT et le décodage.
  - Pour une voix clonée (chemin du consommateur, sans cache de préfixe), tout reste dans le préfill.
  - Porte reformulée : « temps jusqu'au premier frame **préfixe inclus** » (instrument K-P45), sur une voix
    clonée et sur une voix prédéfinie.
  - Sévérité basse : le pré-chauffage **déplace** le coût vers le chargement sans le réduire (≈ 1,7 à 8 Go lus
    une fois) ; c'est un choix d'expérience utilisateur.
  - Statut : VÉRIFIÉ (paresse), À MESURER (coût).

### P-44 · basse · `TTS/VoxtralTTSModeling.swift:463-476` — Deux passes LLM avant le premier frame

- **Constat** : le préfill du suffixe (`:463-470`) puis le forward du jeton AUDIO (`:473-476`) forment deux passes
  complètes de 26 couches. `0be05af` a retiré l'`eval` intermédiaire mais pas la seconde passe. Même chose en
  streaming (`:596-609`).
- **Correction** : concaténer l'embedding AUDIO au suffixe et faire une seule passe ; lire l'état caché de la
  dernière position.
- **Gain attendu** : environ −10 à −20 ms de TTFT en 4 bits (estimation : une passe LLM à 1 position). Faible.
- **Risque** : arrondi (formes différentes). **API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-P44** — **Porte** : TTFT −5 % au moins (A/B/B/A) ; 20 premiers codes sémantiques identiques à graine
  fixe ; sinon, retrait. **Cible** : macos-gpu.
- **Vérification croisée (gardé)** : deux `llmForward` (`:466`/`:469`, puis `:475`) avant le premier
  `decodeOneFrame`. `0be05af` n'a retiré que l'`eval` intermédiaire (message du commit). La fusion est exacte
  (attention causale). Le gain est faible et la porte à −5 % peut échouer : le retrait est prévu.

### P-45 · moyenne · `VoxtralTranscriptionTest/ProfileCommand.swift:227-277` — Instrument TTS incomplet (complète A-15)

- **Constat** :
  - `profile --pipeline tts` ne mesure que la surcharge **prédéfinie en batch** : pas de voix clonée (le chemin de
    FluxForge) ni de streaming, et sans graine (A-15). Les prises froide et chaude ont donc des longueurs différentes.
  - Le « froid » est le tout premier forward après un chargement paresseux (P-43). Le gain de F-7 (≈ 200 ms) est
    donc une **borne haute** du cache de préfixe.
  - Le TTFT rapporté exclut le calcul du préfixe : `voicePrefix(…)` s'exécute avant `model.generate`
    (`VoxtralTTSPipeline.swift:210`), et `genStart` démarre dans `generate` (`VoxtralTTSModeling.swift:441`).
  - La « Semantic Code Generation » ne sépare pas le LLM du FM. Le chronomètre par pas (`:489-504`) mesure du temps
    hôte entre deux synchronisations.
  - Ni TTFA streaming, ni `phys_footprint` par phase, ni `BENCHMARKS.md` (scan.md §7).
  - `machine-check.sh` ne tourne que sur Mac.
- **Correction** :
  - `profile --pipeline tts --seed --voice-embedding --streaming --runs N`, avec l'ordre des runs contrôlé (préfixe
    froid sur une autre voix) ;
  - phases préfixe, préfill, AR-LLM, FM (mode diagnostic à `eval` bracketé, hors mesure de débit), codec et
    post-traitement ;
  - TTFA en streaming, pic `phys_footprint`, une ligne JSON par mesure avec la révision résolue de mlx-swift-lm.
- **Gain** : prérequis de toutes les fiches perf (ordre du plan, étape 3).
- **Risque** : API aucun (CLI). **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche K-P45** — *Baseline TTS*.
  - **Porte** : A/A ≤ 3 % sur la même commande (piège 33).
  - Baseline enregistrée : 3 packs × {court, 60 s, long} × 3 graines × {prédéfinie, clonée} × {batch, streaming} ;
    fr/s, ms par frame LLM et FM, TTFT, TTFA, décodage, pic.
  - **Cible** : macos-gpu.
- **Vérification croisée (amendé)** :
  - Confirmé à `ProfileCommand.swift:253-262` : deux synthèses prédéfinies, sans graine, la première juste après
    un chargement paresseux. Le TTFT rapporté est celui de la 2e.
  - Nuance : « le gain de F-7 est une borne haute » est une **hypothèse**, pas un constat. Le « froid » de 344 à
    381 ms (« prefill isolated », `f4fd21c`) est bien inférieur aux 789 ms du préfill paresseux de F-3 : on ne
    sait pas quelle part de lecture disque il contient. À MESURER par K-P45.
  - Correction à ajouter : faire mesurer par l'instrument le chemin du consommateur, soit 6 bits + voix clonée +
    warm-up en batch.

### P-46 · basse · `TTS/VoxtralTTSProcessor.swift:66-69` — Un `.item()` par frame dans les coupes

- **Constat** :
  - `rms()` fait un `.item()` par frame.
  - `trimLeadInSilence` et `trimTrailingSilence` balaient jusqu'à 50 frames chacun (`:80-95`), plus un `.item()`
    de seuil (`:56`).
  - `trimLeadingCarrierAdaptive` appelle `frameDB` deux fois par frame (`:255-259`) et avance sans borne jusqu'à
    `totalFrames` (`:263-264`).
  - En streaming, ce balayage est relancé sur toute la forme d'onde à chaque chunk tant que la coupe n'est pas
    trouvée (`VoxtralTTSPipeline.swift:595-621`).
  - Au total, jusqu'à environ 150 synchronisations par synthèse.
- **Correction** : RMS par frame vectorisé (`reshape(n, 1920)`, moyenne des carrés, un seul `asArray`), puis
  balayage en Swift.
- **Gain attendu** : post-traitement ≤ 5 ms (estimation ; aujourd'hui quelques dizaines de ms, À MESURER).
- **Risque** : API aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-P46** — **Porte** : indices de coupe identiques sur `TrimSilenceTests`, `TrimLeadingCarrierTests` et
  `TTSWarmUpCarrierTrimTests` ; phase « Audio Post-processing » ÷ 5 au moins. **Cible** : ~~cloud (code) puis
  macos-gpu (tests)~~ → **macos-gpu** (la porte est faite de tests et d'une mesure).
- **Vérification croisée (amendé : précision, cible)** :
  - `.item()` par frame confirmé (`:66-69`, `:56`).
  - `trimLeadingCarrierAdaptive` appelle `frameDB` **une** fois par frame dans la boucle de `:255` (plus une fois
    à la frame retenue, `:257`), et une fois par frame dans la marche avant non bornée (`:264`) ; pas « deux fois
    par frame ».
  - Compte réaliste : ≈ 100 synchronisations par synthèse (5 de référence, ≤ 35 de balayage, la marche avant, puis
    51 pour `trimTail` s'il est activé). Ce n'est pas borné en théorie à cause de `:264`.
  - En streaming, chaque tentative de coupe relance le balayage sur la fenêtre de 3 s, plus la marche avant. Le
    `asType(.float32)` (`:236`) ne coûte rien : la sortie du codec est déjà en fp32 (P-38).
  - Le vectorisé doit conserver le repli de la dernière frame partielle (`:90`).

### P-47 · basse · `TTS/Pipeline/VoxtralTTSPipeline.swift:595-599` — Coût du warm-up (A6b)

- **Constat** :
  - Le vocalise recommandé (`:72`), ajouté pour les voix clonées (démo `StreamingDemoViewModel.swift:503` ; CLI
    `--warm-up`), est **généré, décodé puis jeté** à chaque synthèse.
  - En streaming, rien n'est émis avant 3 s d'audio accumulé (`scanWindowSamples`, `:595-599`), soit au moins
    38 frames générées avant le premier échantillon utile. Cela représente 1,2 à 1,6 s en 4 bits et ≈ 9 s en bf16
    au rythme de F-1 (calcul) — une fois S-08 corrigé.
  - La réutilisation d'un cache après le porteur est **réfutée** (#45 : « a second utterance after EOA is out of
    distribution », `TTSTwoPassWarmUpProbeTests`).
- **Correction** : mesurer la longueur réelle du porteur (frames) ; évaluer un vocalise plus court, ou une fenêtre
  d'attente adaptée à la durée attendue du porteur. C'est un compromis qualité ↔ latence → **ASK**.
- **Risque** : réintroduire les fuites ou les coupes du porteur (1/8, #45). **API** : aucun. **Effort** : S.
  **Statut** : VÉRIFIÉ, coût À MESURER.
- **Fiche K-P47** — **Porte** (amendée) : prérequis K-S08 ; TTFA streaming des voix clonées −30 % au moins, avec
  des fuites du porteur ≤ la référence (0/10) sur 10 prises ; frames du porteur sur frames totales rapportées en
  batch ; décision ASK. **Cible** : macos-gpu.
- **Vérification croisée (amendé : chiffres, prérequis)** :
  - Premier chunk utile : l'attente de 3 s (`:595-599`) ne concerne que le streaming **avec warm-up**
    (`contentStart` vaut `nil` seulement si `hasWarmUp`, `:563`).
  - Avec `chunkSize` 10 et un premier chunk de 3 frames (`VoxtralTTSModeling.swift:617`, `:652`), les chunks
    tombent à 3, 13, 23, 33 puis 43 frames. Le premier à dépasser 72 000 échantillons est donc celui de
    **43 frames** (82 560 échantillons), pas 38.
  - Les coûts « 1,2 à 1,6 s / ≈ 9 s » reposent sur F-1, périmé ; avec 43 frames au rythme de F-6, on attend
    ≈ 1,4 s en 4 bits (calcul, À MESURER).
  - **Prérequis K-S08** : tant que le streaming ne streame pas, le TTFA égale la génération complète et la porte
    n'est pas mesurable.
  - Pour le consommateur (batch, FluxForge), le coût utile est ailleurs : les frames du porteur générées puis
    jetées à **chaque** synthèse, soit une part importante sur des répliques courtes de doublage. Ajouter à la
    porte : frames du porteur sur frames totales, par synthèse (À MESURER).

### P-48 · basse (était moyenne) · `TTS/VoxtralTTSModelLoading.swift:58` — Quantification uniforme et mode ignoré

- **Constat** :
  - `let mode: QuantizationMode = quantConfig.mode == "affine" ? .affine : .affine` : le mode est **ignoré**. Un
    pack `mxfp4`, `mxfp8` ou `nvfp4`, modes disponibles dans mlx-swift (`Ops.swift:1100-1127`), serait chargé en
    affine.
  - Les packs publiés quantifient uniformément LLM et FM (F-12). T13 (voies à précisions différentes) est absent.
  - Signaux de qualité par largeur : le 4 bits rate l'EOA sur le FR long, pas le 6 bits (F-8) ; le 6 bits couvre
    mieux que le bf16 en voix clonée (F-9).
- **Correction** :
  - correspondance chaîne → `QuantizationMode`, avec erreur si le mode est inconnu ;
  - campagne sur des packs mixtes : LLM 4 bits + FM 6 ou 8 bits, LLM 6 bits + FM 4 bits ;
  - quantification au chargement depuis le bf16 avec un prédicat par voie, ou publication d'un pack
    `VincentGOURBIN/…` avec SHA-256 ;
  - puis évaluer les formats mx et nv sur Apple Silicon.
- **Gain attendu** : profil `4bit-fast` sans échec d'EOA sur le long FR, à ±10 % de la vitesse du 4 bits (À MESURER).
- **Risque** : API additif (mode). **Effort** : S (mode) + L (campagne, packs).
- **Statut** : VÉRIFIÉ (code et index), À MESURER.
- **Fiche K-P48** — *Mode de quantification + packs mixtes*.
  - **Porte (mode)** : test de chargement d'un pack non affine, ou erreur explicite.
  - **Porte (profil)** : couverture ASR ≥ 6 bits −0,5 point ; 0 `maxFrames` sur le long FR (3 graines) ;
    fr/s ≥ 0,9 × le 4 bits.
  - **Cible** : ~~cloud (mode) puis macos-gpu~~ → **macos-gpu** (même la porte du mode est un test de
    chargement).
- **Vérification croisée (amendé : sévérité moyenne → basse, preuve, cible)** :
  - `:58` confirmé : les deux branches valent `.affine`.
  - Défaut **latent** : aucun pack Voxtral publié n'est non affine (`docs/tts_benchmark.md:19-20` : 4 et 6 bits
    « affine, group_size=64 » ; le bf16 n'est pas quantifié). Avec
    `verify: .none` (`:71`), un tel pack serait chargé sans erreur mais faux : pas de `biases` en mxfp4 ou nvfp4.
  - La motivation T13 est plus faible qu'écrit. Sur le FR long de F-8, le **bf16 atteint aussi** `maxFrames`,
    pas seulement le 4 bits. L'échec d'EOA n'est donc pas monotone en largeur de bits, et il repose sur une prise
    unique, sans graine (banc du 2026-04-02).
  - La campagne de packs mixtes reste exploratoire : sévérité basse.

### P-49 · basse · `TTS/VoxtralCodecDecoder.swift:295-296` — Compile (T18) : candidats limités et risqués

- **Constat** : chaînes élémentaires à forme stable dans le codec :
  - `w2(silu(w1(x)) * w3(x))` (`:295-296`) ;
  - LayerScale et résiduel (`:270-278`) ;
  - sur T × 4 096 en fp32 au dernier étage.
  - `silu` est déjà compilé (mlx-swift `Activations.swift:212-213`) ; le produit et les échelles restent des noyaux
    séparés.
  - Le même forward du codec est **différencié** pendant l'enrôlement (`VoxtralVoiceEnrollment.swift:558`) :
    compiler davantage élargit la surface du deadlock ABBA (A-01, piège 20).
  - Rejets du catalogue : ne pas compiler le pas AR ou FM par défaut (R1, R2).
- **Correction** : seulement après P-32 et P-33 (qui retirent le coût dominant), derrière un coupe-circuit et
  désactivé sous gradient ; tests en parallélisme 1.
- **Gain attendu** : ≤ 10 % du décodage codec (Y : −30 % sur un VAE, autre opération ; estimation).
- **Risque** : ABBA (A-01). **API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ, À MESURER.
- **Vérification croisée (gardé)** :
  - 0 `compile(` dans `Sources/`. `compiledSilu` est déjà compilé (mlx-swift `Activations.swift:1049`).
  - Le forward du codec est bien différencié à l'enrôlement (`MLX.valueAndGrad`,
    `VoxtralVoiceEnrollment.swift:558`).
  - La porte bit-exacte et le coupe-circuit suffisent.
- **Fiche K-P49** — **Porte** : décodage −5 % au moins, bit-exact, 0 blocage sur 20 exécutions « enrôlement +
  synthèse » ; sinon, retrait. **Cible** : macos-gpu.

## 5. Optimisations existantes relues (non-constats et confirmations)

- **`0be05af`** (F-6) :
  - CFG en batch 2 : **appliqué** (`VoxtralFlowMatching.swift:299-316`), comme la référence (`acoustic_head.py:219-225`).
  - Un seul `eval` par intégration : appliqué (`:318-321`), mais c'est lui qui bloque le pipelining (P-31).
  - Préfill « fusionné » : un seul graphe, mais deux passes (P-44).
  - Premier chunk de 3 frames (`VoxtralTTSModeling.swift:617`) : sans effet tant que S-08 n'est pas corrigé.
- **`f4fd21c`** (F-7) :
  - appliqué pour les préréglages et le streaming avec clé ;
  - invariant de clonage gardé par `KVCacheCloneTests` ;
  - trous : P-40 (voix clonées) et P-45 (mesure).
- **`a00024f`** (F-5) : batch seulement (P-31).
- **Graine (A6c)** :
  - `MLXRandom.seed` (`VoxtralTTSModeling.swift:440`, `:582`) : coût nul ;
  - **indispensable** à toute mesure A/B/B/A, puisqu'une prise sans graine change de longueur (84 s contre 5 s,
    `docs/streaming_demo.md:32-35`) ;
  - RNG global partagé avec l'enrôlement (A-01).
- **Voix prédéfinies** : 20 fichiers chargés au `loadModel` (`VoxtralTTSPipeline.swift:166-175`), ≤ 54 Mo même en
  fp32 (borne haute 20 × 218 × 3 072 × 4 o) → non-constat.
- **ZeroVoice / SLERP** :
  - SLERP en fp32 par construction, sur ≤ 218 lignes, recasté (`VoxtralTTSModeling.swift:369`) → non-constat ;
  - `zeroVoice` recrée un objet léger à chaque accès (`VoxtralTTSPipeline.swift:403-406`) → négligeable ;
  - leur coût réel : pas de cache de préfixe (P-40) et inflation de frames (F-11, P-41).
- **KV du LLM** : `KVCacheSimple` préalloué (T11), SDPA MLXFast et RoPE MLXFast → non-constat.
- **Encodeur du codec** : jamais instancié, poids absents du checkpoint → hors du chemin (code mort, S-13).

## 6. Entrées pour la phase 3 (profils TTS)

Le standard `<bits>bit-fast|lean` ne prévoit que 4, 8 et 16 bits. **Voxtral publie 4, 6 et 16 bits**, et aucun pack
8 bits n'existe chez mlx-community (listing HF) → à trancher (ASK ; voir le retour sur le skill).

| Profil candidat | Pack | Réglages à figer | Valeurs connues | À mesurer |
|---|---|---|---|---|
| `4bit-fast` | mlx-community 4 bits (2,51 Go) ou pack mixte (P-48) | flowSteps (P-35), asyncEval (P-31), cache de préfixe LRU (P-40), plafond de frames (P-41), codec par bandes (P-32) | 31,5 fps et TTFT ≈ 280 ms (F-6) ; échecs d'EOA sur le FR long (F-8) | tout, après K-P45 |
| `6bit-fast` | mlx-community 6 bits (3,47 Go) | idem | voix clonée 99,4 % / RTF 1,47 (F-9) ; long FR sans `maxFrames` (F-8) ; **seul pack livré par FluxForge** (vérification croisée) | tout |
| `16bit-fast` (référence qualité) | mlx-community bf16 (8,0 Go) | idem + P-30 **obligatoire** | RTF 3,44 à 6,86 (F-8, F-9) | gain de P-30 |
| `*-lean` | idem | `clearCache` après décodage, `cacheLimit` adaptatif opt-in (P-42), décodage par fenêtres (P-33) | — | pic, temps |

## 7. Ordre proposé pour le plan

1. Stabilité bloquante : K-S08 (le streaming ne streame pas), prérequis de P-33 et P-47.
2. Hygiène sans risque. Le code peut être **rédigé** dans le cloud, mais chaque fiche ne se ferme que sur
   `macos-gpu` (portes à tests ou à mesures, vérification croisée) :
   - K-P34 (manager additif) ;
   - K-P35, partie câblage ;
   - K-P40 et K-P41 (code additif) ;
   - K-P48, partie mode ;
   - K-P46, partie code.

   Seules les corrections **documentaires** se ferment dans le cloud (porte : relecture du diff, aucun code
   exécutable modifié) :
   - `docs/tts_benchmark.md:10` (réglages présentés comme effectifs, P-35) et `:22` (« runs in bf16 », P-38) ;
   - commentaires faux `VoxtralTTSModeling.swift:480-482` et `:528` (P-31) ;
   - commentaire du propriétaire sur #27 (P-34), à signaler.
3. **Baseline** K-P45 (sans elle, aucune fiche perf ne conclut).
4. Leviers, par gain attendu décroissant : K-P30 (bf16, voix clonées), K-P32 et K-P33 (mémoire, streaming), K-P31,
   K-P35 (mesure), K-P40, K-P36, K-P42, K-P43, K-P48 (campagne), K-P38, K-P44, K-P46, K-P47, K-P49.
5. Profils TTS (§6) et CLI `--reference` ; lignes `BENCHMARKS.md`.

## Annexe — Constats écartés à la vérification croisée

Aucun constat n'est écarté : les 20 mécanismes sont confirmés dans le code à `9392ed1`. Les sous-points retirés
figurent ci-dessous, avec les amendements.

| Id | Sous-point retiré ou corrigé | Raison |
|---|---|---|
| P-36 | « `concatenated([xt, xt])` refait à chaque pas » compté comme invariant | `xt` change à chaque pas d'Euler (`VoxtralFlowMatching.swift:316`) : ce n'est pas un invariant |
| P-46 | « `frameDB` deux fois par frame », « ≈ 150 synchronisations » | Une fois par frame (`:255`, `:264`) ; ≈ 100 en pratique |
| P-47 | « au moins 38 frames » avant le premier échantillon | Avec les chunks 3 + 10k, le premier chunk utile arrive à 43 frames |
| P-39 | « une copie de l'activation par convolution » (`:83`) | Seulement pour les 2 convolutions non transposées à noyau > 1 |
| P-45 | « le gain de F-7 est donc une borne haute » (présenté comme établi) | C'est une hypothèse, À MESURER |

## Annexe — Amendements de la vérification croisée

| Id | Amendement |
|---|---|
| P-30 | La référence Python fait aussi tourner le FM en fp32 : le bf16 s'en écarte, donc la porte de parité est indispensable. Correction complétée (`timeProjection`). Base de gain F-1 périmée. Pour le consommateur (6 bits), le gain est celui de la partie quantifiée. Sévérité haute maintenue |
| P-31 | Prémisse « borné CPU » (F-1) périmée : remplacée par `0be05af` et T15. Déclassement prévu si K-P45 montre un GPU saturé |
| P-33 | Sévérité haute → moyenne : seul appelant, la démo ; FluxForge n'utilise pas `synthesizeStreaming` ; cohérence avec S-08 |
| P-34 | Sévérité moyenne → basse : défaut bf16 délibéré et documenté ; FluxForge charge explicitement le 6 bits et n'utilise pas le manager. Cible cloud → macos-gpu |
| P-35 | Ligne CLI `:461-464` → `:462-464`. Cible → macos-gpu (même la porte du câblage exige une exécution) ; seule la doc est faisable dans le cloud |
| P-36 | Sous-point `concatenated([xt, xt])` retiré |
| P-38 | Correction incomplète : le padding fp32 `MLX.zeros` (`:83`) et `MLXArray(scale)` (`:210`) re-promeuvent tout le décodeur. Test de dtype par étage ajouté à la porte |
| P-39 | Précision sur le padding (2 convolutions). Risque ajouté : réflexion `Module` de mlx-swift sur un cache `MLXArray` (préfixe `_`) |
| P-40 | Cible cloud → macos-gpu. Chemin consommateur confirmé (FluxForge, batch, voix enrôlée, warm-up) |
| P-41 | Coûts : mesures F-8 (158,75 s et 971 s) au lieu du calcul sur F-1. Cause NaN de F-10 précisée. Limite de la correction sur le FR long. Cible → macos-gpu |
| P-43 | Sévérité moyenne → basse (coût déplacé, pas supprimé). Depuis `f4fd21c`, la lecture des poids LLM d'une voix prédéfinie tombe hors du TTFT (`voicePrefix`). Porte « préfixe inclus » |
| P-45 | « Borne haute » de F-7 requalifiée en hypothèse. Chemin consommateur (6 bits + voix clonée + warm-up) à instrumenter |
| P-46 | Compte d'appels corrigé (≈ 100, non borné via `:264`). Cible → macos-gpu |
| P-47 | 43 frames (et non 38), streaming avec warm-up seulement. Prérequis K-S08 explicite. Métrique batch (frames du porteur) ajoutée |
| P-48 | Sévérité moyenne → basse : mode ignoré latent (aucun pack non affine) ; motivation T13 affaiblie (le bf16 atteint aussi `maxFrames` sur le FR long). Cible → macos-gpu |

Gardés sans changement de fond : P-32 (transitoire minimal ≈ 2 × 10,5 Go, plutôt sous-estimé), P-37, P-42 (preuve
renforcée par le contournement côté FluxForge), P-44, P-49.
