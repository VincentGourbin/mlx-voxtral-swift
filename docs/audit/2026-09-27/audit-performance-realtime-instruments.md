# Audit « Performance » — chemin Realtime et instruments de mesure de mlx-voxtral-swift

**Vérification croisée : 20 constats relus, 10 gardés tels quels, 0 écarté, 10 amendés** (P-60, P-61, P-62, P-63,
P-64, P-66, P-70, P-73, P-77, P-78 ; détail dans l'annexe finale). Relecture adverse du 2026-09-27 à `9392ed1` :
code Voxtral, MLX C++ `ce45c52` (épinglé par mlx-swift 0.31.6), MLXNN 0.31.6, mlx-swift-lm `ee673d6`,
swift-mlx-profiler tags `1.1.1`…`1.5.1`, issues #23-#25 (corps et commentaires), Hub HF, et la **référence déclarée
en tête des fichiers Realtime**, mlx-audio `main` (`stt/models/voxtral_realtime/{encoder,decoder,voxtral_realtime}.py`,
récupérés le 2026-09-27). Les amendements sont signalés en ligne par « *Vérif. croisée* ».

> Skill `mlx-swift-audit`, phase 2, constats `P-60…P-79`. Révision auditée : `9392ed1` (branche
> `claude/action-plan-skills-beta-wifgmu`). Date : 2026-09-27.
> Périmètre (a) : chemin **Realtime** (`Sources/VoxtralCore/Realtime/*` : encodeur causal, décodeur Ada-RMSNorm,
> modèle, pipeline, manager, registre, configuration, chargement). Périmètre (b) : **instruments de mesure** :
> `VoxtralBenchmark/BenchmarkCLI.swift`, `VoxtralTranscriptionTest/ProfileCommand.swift` (+ commande `realtime` de
> `VoxtralCLI.swift`), `Utils/VoxtralProfilerBridge.swift` et la dépendance `swift-mlx-profiler`,
> `Utils/RuntimeBeacon.swift`, harnais de test sous variable d'environnement (`TTSQuantizationCampaignTests`,
> `TTSGenerateStabilityReproTests` et apparentés).
> Hors périmètre : STT (`audit-performance-stt.md`, P-01…P-29), TTS (`audit-performance-tts.md`), stabilité
> (`audit-stabilite.md`), annexes et serveur (`audit-annexes-serveur.md`). Ces rapports sont cités, pas redupliqués.

## 0. Cadre, méthode, sources

- **Environnement** : session cloud Linux, sans Mac, sans toolchain Swift, sans GPU. Aucun build, aucun test,
  aucune mesure. Tout gain est **attendu**, jamais obtenu. Tout chiffre qui n'est pas déjà mesuré (issues, docs)
  est **À MESURER**. Toute fiche qui exige build, test, mesure ou écoute cible `macos-gpu`.
- **Chaque constat est vérifié en lisant le code** à `9392ed1`. Quand l'effet dépend de MLX, la règle est lue
  dans la source C++ **réellement épinglée**. Voxtral dépend de mlx-swift `from: "0.31.6"` (`Package.swift:43` ; *Vérif. croisée : :42 est un
  commentaire*).
  Le tag 0.31.6 épingle MLX C++ `ce45c52`, déjà récupéré en lecture dans le scratchpad par l'audit STT. Les noyaux
  Metal ont été lus sur MLX `1f8e74e` (sous-module de mlx-swift `9019419`) : même logique d'éligibilité.
  MLXNN a été lu sur mlx-swift `9019419`, mlx-swift-lm sur `main@ee673d6`. *Vérif. croisée* : la version résolue
  est 0.31.6 ; les numéros de ligne MLXNN diffèrent de `9019419` (ex. `QuantizedEmbedding.asLinear` =
  `Quantized.swift:213-217` @0.31.6, pas `:240-245`). Les citations MLXNN sont donc données @0.31.6 quand elles
  ont été corrigées.
- **swift-mlx-profiler** a été cloné **en lecture** dans le scratchpad : tags `v1.0.0` … `1.5.1`, HEAD `bfe71d8`.
  Voxtral le requiert `from: "1.4.0"` (`Package.swift:53`), sans `Package.resolved` suivi (S-18). La version
  résolue dépend donc de la machine (P-74).
- **Faits** : issues fermées #23, #24, #25 (corps **et** commentaires, via MCP GitHub en lecture), `README.md`,
  `docs/voice_cloning.md`, `docs/zerovoice_benchmark.md`, Hub HF (listings, `config.json`,
  `model.safetensors.index.json`, cartes de modèles, lus le 2026-09-27).
- **Consommateurs** : recherche de code GitHub `VoxtralRealtime user:VincentGourbin NOT
  repo:VincentGourbin/mlx-voxtral-swift` → **0 résultat**. L'audit stabilité liste les symboles consommés par
  FluxForge (chaîne LipDub) et SongAnalysisDb : aucun symbole Realtime n'y figure. Limite : la recherche n'indexe que
  les branches par défaut. Toute modification de signature publique reste **« cassant » → ASK**. Les corrections
  proposées gardent les signatures.
- **Articulation avec les autres rapports** :
  - P-60 étend au Realtime le mécanisme fp32 de STT P-01/P-05.
  - P-66 est l'homologue Realtime de STT P-04 (poids paresseux).
  - P-64 complète STT P-11 (`maxTokens`).
  - P-75 à P-79 complètent A-15, A-16, A-22, STT P-19 et TTS P-45 : ils fusionnent en **un seul instrument**
    (P-79).
  - S-04 (`verify: .none`) et S-09 (pas d'annulation) ne sont pas redupliqués.

## 1. Carte du chemin Realtime (tel qu'exécuté par défaut)

Modèle par défaut : `realtime-4b-4bit` = `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`
(`VoxtralRealtimeRegistry.swift:28-38`). Sa configuration, lue sur HF :

- **Décodeur** : dim 3 072, 26 couches, 32 têtes de requête et 8 têtes KV, `head_dim` 128, FFN 9 216,
  vocabulaire 131 072, **fenêtre glissante 8 192**, embeddings liés.
- **Encodeur** : dim 1 280, 32 couches, 32 têtes, `head_dim` 64, FFN 5 120, **fenêtre glissante 750**,
  sous-échantillonnage ×4.
- **Cadences** : 100 trames mel/s ; 50 positions/s dans l'encodeur après la conv de pas 2 ; 12,5 pas/s dans le
  décodeur (1 pas = 80 ms d'audio).

| Étape | Code (`9392ed1`) | dtype aujourd'hui | Remarque |
|---|---|---|---|
| Lecture + rééchantillonnage | `VoxtralRealtimePipeline.swift:195` → `VoxtralFeatureExtractor.swift:24-86` | fp32 (CPU) | fichier décodé en entier (STT P-29) |
| Remplissage gauche/droite | `VoxtralRealtimePipeline.swift:197-212` | fp32 | une concaténation **par fichier**, pas par pas |
| Mel log (`globalMax` fixe) | `VoxtralRealtimePipeline.swift:215-223` → `VoxtralFeatureExtractor.swift:277-336` | **fp32** | graphe **paresseux** : calculé dans la phase suivante |
| Conv causales (×2) | `VoxtralRealtimeEncoder.swift:23-48`, `:291-304` | fp32 (promotion) | `concatenated` de zéros une fois par appel (`:44`) |
| 32 couches causales | `VoxtralRealtimeEncoder.swift:329-342` | fp32 | masque `.causal` sur tout l'audio : **fenêtre 750 ignorée** |
| Sous-éch. ×4 + adaptateur | `VoxtralRealtimeEncoder.swift:310-318` | fp32 | adaptateur 16 bits non quantifié → copie fp32 des poids |
| Ada-RMSNorm (retard) | `VoxtralRealtimeModel.swift:69-70`, `VoxtralRealtimeDecoder.swift:24-31`, `:166-174` | **fp32** | `tCond` construit depuis `[Float]` |
| Préfill (1 + 1 + n_delay ≤ 32 jetons) | `VoxtralRealtimeModel.swift:79-110` | fp32 | logits de la **dernière** position seulement (`:108`) |
| Décodage, 1 pas par trame | `VoxtralRealtimeModel.swift:115-141` | fp32 | `eval` puis `.item()` à chaque pas ; `clearCache` tous les 256 pas |
| Tête (embeddings liés) | `VoxtralRealtimeDecoder.swift:187-189` | fp32 | `matmul(h, W.T)` : **W (16 bits) converti en fp32 à chaque pas** |
| Cache KV | `VoxtralRealtimeDecoder.swift:192-194` | fp32 | `KVCacheSimple` sans borne : **fenêtre 8 192 ignorée** |

**Budget de poids du pack 4 bits.** La quantification affine en 4 bits, groupe de 64, coûte 0,5625 o/param.
Chaque entrée a été recalculée depuis la config et recoupée avec l'index HF (`total_parameters` 4 429 679 360,
`total_size` 3 133 615 104 o).

| Bloc | Stockage | Taille |
|---|---|---|
| Couches du décodeur (3,026 G params) | 4 bits | 1 702 Mo (1 623 Mio) |
| `tok_embeddings` (402,7 M params) | 16 bits, non quantifié (`VoxtralRealtimeModelLoading.swift:39`) | 805 Mo (768 Mio) |
| Encodeur (0,965 G params) | 4 bits | 543 Mo (518 Mio) |
| Adaptateur | 16 bits | 50 Mo |
| Conv et Ada-RMSNorm | 16 bits | ≈ 21 Mo |

La copie fp32 de la tête pèse 1 611 Mo (1 536 Mio).

## 2. Faits déjà mesurés (repris tels quels, avec source)

Toutes ces mesures sont « en session » au sens de `measurement.md` : ni machine, ni configuration de build, ni
fichier audio, ni durée ne sont notés dans les issues.

- **F-R1 — #23, #24, #25 (2026-04-11, `realtime-4b-4bit`, trace `/tmp/voxtral_realtime_trace.json`)** : les phases
  mesurées sont les suivantes.

  | Phase | Durée | GPU | CPU | Mémoire MLX |
  |---|---|---|---|---|
  | Mel Spectrogram | 3,28 s | 0 % | 99,8 % | — |
  | Audio Encoding | 5,44 s | 49 % | 7,6 % | 32 → 705 Mo |
  | Prefill | 448 ms | 28 % | 104 % | 639 → 4 619 Mo |
  | Realtime Generation | 23,89 s | 0 % | 49,5 % | — |
  | Token Decoding | 50 ms | 0 % | 100 % | — |

  Pas de décodage : 501 pas, moyenne 33,7 ms, écart-type 2,7 ms, min 0,7 ms, max 67 ms. Pic MLX 4 619 Mo, pic
  process 7 949 Mo.
- **F-R2 — conclusions de fermeture (même jour)** :
  - #23 : « measurement artifact — IOKit sampling … 21 tok/s is reasonable ».
  - #24 : « systemic 49 % GPU cap … root cause is MLX/Metal memory allocation pattern ».
  - #25 : « acceptable — short prompt doesn't saturate GPU ».
- **F-R3 — version du profiler à cette date** : le tag `1.2.0` date du 2026-04-11 et `1.1.1` du 2026-04-10. Tous les
  tags jusqu'à `1.4.0` présentent le même comportement :
  - chaque bord de phase lit IOKit ;
  - `recordStep` lit IOKit (`ProfilingSession.swift:102` @1.4.0) ;
  - le GPU % est agrégé par une variable `currentPhase` unique (`:200-213` @1.4.0).
- **F-R4 — profiler 1.5.x (2026-09-09, `6885d48`)** :
  - Le bord de phase ne coûte plus qu'un signpost. Avant, une paire début/fin coûtait ≈ 4,7 ms
    (`ProfilingSession.swift:14-21`).
  - La lecture IOKit de l'ancien lecteur coûtait 1 511 µs, contre 16,7 µs pour le nouveau
    (`GPUUtilization.swift:31-36`).
  - « Device Utilization % » est une valeur **instantanée**. Lue à un bord, elle donne « 41-49 % where it was
    ~82 % » (`GPUUtilization.swift:11-16`). *Vérif. croisée* : ce cas documenté porte sur une phase de **5 ms** ;
    il montre qu'une lecture de bord n'est pas une mesure, pas que « 49 % » vaille ≈ 82 % en général.
  - *Vérif. croisée* — calcul du GPU % par phase en ≤ 1.4 (`ProfilingSession.swift:200-213` et `:229` @1.4.0,
    identique en `1.1.1`, version la plus récente au moment de la trace : les issues datent de 09:03 UTC, le tag
    `1.2.0` de 09:33 UTC) : moyenne **entière** des seules lectures prises aux bords (plus les pas, s'ils tombent dans
    la phase courante). Une phase sans pas n'a que deux lectures ; (≈ 0 au début, GPU au repos) + (≈ 98 à la fin,
    juste après un `eval`) donne ≈ 49. Hypothèse compatible avec le « 49 % systémique », non établie.
  - Un échantillonneur de fond tourne à 16 ms. Les phases imbriquées sont gérées (`:377-423`).
  - Le type de build est enregistré (`RunEnvironment.swift:24-32`).
- **F-R5 — le Realtime sert de juge ASR** dans `docs/zerovoice_benchmark.md:5,10` (M3 Max 96 Go, 2026-03-31) :
  il transcrit en retour des clips TTS de 7,6 à 20,8 s.
- **F-R6 — HF (2026-09-27)** :
  - `mistralai/Voxtral-Mini-4B-Realtime-2602` reste le Realtime le plus récent de Mistral.
  - Packs MLX : `mlx-community …-4bit` (3,13 Go, format mlx-audio), `…-fp16` (8,87 Go),
    `mlx-community/Voxtral-Mini-4B-Realtime-6bit` (3,61 Go) et `ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx`
    (4,71 Go). Ces deux derniers sont au **format voxmlx** : clés `adapter.w_in.*` et
    `encoder.layers.N.attention.q_proj.*`, config au format `params.json` avec `quantization`.
  - La carte du pack 4 bits indique : « Both components use sliding window attention for unbounded audio length »
    et `model.generate(…, stream=True)` (mlx-audio).

## 3. Catalogue T1…T23 appliqué au chemin Realtime

| # | Technique | Statut Realtime | Où / preuve | Suite |
|---|---|---|---|---|
| T1 | `cacheLimit` par étape | **absente** (substitut : `clearCache` tous les 256 pas) | aucune pose dans `Realtime/` (scan §4 : seules poses dans `VoxtralApp`) ; `VoxtralRealtimeModel.swift:138-140` | P-67 |
| T2 | limites adaptatives | absente | — | P-67 (lean) |
| T3 | `clearCache` après réponse / entre étapes | **partielle** | périodique seulement ; rien en fin de `generate` ni dans `unload()` (`VoxtralRealtimePipeline.swift:181-186`), alors que STT `unload()` → `fullCleanup()` (`VoxtralPipeline.swift:485`) | P-67 |
| T4 | résidence par étape | absente, **masquée par le chargement paresseux** | encodeur (518 Mio en 4 bits) résident pendant tout le décodage ; décodeur non matérialisé tant qu'on n'appelle que `extractAudioEmbeddings` | P-66 |
| T5 | variante « sans tour » → **encodeur seul** | absente | `extractAudioEmbeddings` exige le modèle complet chargé (`VoxtralRealtimePipeline.swift:160-177`) | P-66 |
| T6 | réutilisation de préfixe KV | N/A (un passage par fichier) ; l'équivalent utile est le **streaming incrémental** | `VoxtralRealtimePipeline.swift:118-154` | P-71 |
| T7 | médias nouveaux seulement | N/A | — | — |
| T8 | budget d'entrée exposé → **retard** exposé | appliquée (analogue) | `Configuration.transcriptionDelayMs` (`VoxtralRealtimePipeline.swift:23-36`), CLI `--delay` (`VoxtralCLI.swift:692-693`) | — |
| T9 | tranche de préfill / logits de la dernière position | préfill ≤ 32 jetons : tranche N/A ; **dernière position appliquée** (`VoxtralRealtimeModel.swift:108`) ; en revanche l'encodeur traite tout l'audio en un graphe | `VoxtralRealtimeModel.swift:74-76` | P-62 |
| T10 | KV quantifié | absente | 104 Kio/jeton en bf16 (208 en fp32) ; plafonné à 832 Mio en bf16 une fois la fenêtre 8 192 posée → faible priorité | après P-63 (lean) |
| T11 | KV préalloué, écriture en place | **appliquée (amont)** | `KVCacheSimple` : pas de 256, `slice_update` (mlx-swift-lm `KVCache.swift:408-464`) ; dtype hérité de `keys` (`:439-440`) → fp32 aujourd'hui | P-60 |
| T12 | tête quantifiée via `quantizedMatmul` | **absente, et pire** : tête liée 16 bits par `matmul` brut + conversion fp32 à chaque pas | `VoxtralRealtimeDecoder.swift:187-189` ; `VoxtralRealtimeModelLoading.swift:39` | **P-61** |
| T13 | quantification mixte par voie | partielle (recette du pack) | prédicat `VoxtralRealtimeModelLoading.swift:34-41` ; seuls 4 bits et fp16 sont chargeables | P-70 |
| T14 | dé-quantifier une étape bornée par le calcul | absente | encodeur en `quantizedMatmul` 4 bits sur 50 positions/s d'audio | P-69 |
| T15 | pipelining `asyncEval` | **absente** | `MLX.eval` (`VoxtralRealtimeModel.swift:109`, `:133`) + `.item()` (`:161`) : deux synchronisations par pas | P-65 |
| T16 | `eval` par couche | décodage : `eval` par pas (OK) ; encodeur : **un seul graphe** de 32 couches sur tout l'audio | `VoxtralRealtimeModel.swift:76` | P-62 (tranches) |
| T17 | fuites fp32 | **présente, 3 sources** | mel, tables RoPE, `tCond` | **P-60** |
| T18 | `compile` d'une chaîne élémentaire | absente (0 `compile`) ; la RoPE manuelle se remplace par le noyau fusionné, sans `compile` | `VoxtralRealtimeEncoder.swift:55-83` | P-68 |
| T19 | KV entre étapes | N/A | — | — |
| T20 | politique de calcul par étape | absente | même quantification pour l'encodeur (borné calcul) et le décodeur (borné bande passante) ; tête 16 bits | P-61, P-69 |
| T21 | réduction du nombre de pas | N/A tel quel : 1 pas = 80 ms d'audio, imposé par l'architecture | candidat : pas spéculatifs « remplissage » | P-72 |
| T22 | `pread` / `F_NOCACHE` | N/A (priorité basse) : un fichier de 3,1 Go | d'abord P-66 | — |
| T23 | reprise / porte GPU iOS | absente ; **applicable si iOS est visé**. `Package.swift:10-13` déclare iOS 17 et `@available(macOS 14.0, *)` n'exclut pas iOS. Le pic process de 7 949 Mo (F-R1) est incompatible avec jetsam sur iPhone | — | ASK Q5 |

## 3 bis. Réponses aux questions du cadrage (a)

- **« 0 % GPU pendant 24 s » (#23)** : c'est un artefact, mais pas pour la raison écrite, et la conclusion
  « 21 tok/s is reasonable » est fausse.
  - La phase « Realtime Generation » **contient** « Audio Encoding » et « Prefill » (`VoxtralRealtimePipeline.swift:134`
    ouvre la phase, `VoxtralRealtimeModel.swift:73` et `:106` ouvrent les sous-phases).
  - Le profiler ≤ 1.4 remet `currentPhase` à `nil` à la fin de la sous-phase. Les 501 échantillons de pas ne sont
    donc plus attribués à la phase englobante : son GPU % est la moyenne de ses **deux** lectures de bord, prises
    GPU au repos → 0 %.
  - Les « 71 % du temps » additionnent des phases imbriquées : 23,89 / (3,28 + 5,44 + 0,448 + 23,89 + 0,05)
    = 72 %.
  - Le vrai décodage vaut 501 × 33,7 ms = 16,9 s, soit **29,7 pas/s**. Le chiffre de 21 tok/s divise par une
    durée qui inclut encodage et préfill.
  - Surtout, 33,7 ms/pas n'est pas « raisonnable » : ≈ 5,8 Go sont lus ou écrits par pas, dont 4,0 Go pour la seule
    tête (P-61) (P-73).
- **Encodeur causal à 49 % GPU (#24)** : non établi. La phase n'a que deux lectures instantanées, prises à ses bords,
  moyennées en entier (F-R4) : ce n'est pas une mesure d'occupation. *Vérif. croisée* : la phrase d'origine (« 49 %
  est exactement la valeur … quand l'occupation réelle est d'environ 82 % ») généralisait un cas documenté sur une
  phase de 5 ms ; l'occupation réelle de l'encodage reste **À MESURER**.
  - La phase de 5,44 s mélange trois choses : la lecture disque des poids de l'encodeur (chargement paresseux,
    ≈ 576 Mio de poids = 518 + 48 + 10 sur les +673 Mio observés ; *Vérif. croisée* : le reste, ≈ 97 Mio, n'est pas
    ventilé — sortie `adapterOut` fp32, mel, transitoires), le calcul mel/STFT resté paresseux depuis la phase
    précédente, et un encodeur **fp32** en attention **pleine** sur tout le fichier.
  - La durée de l'audio n'est pas connue, donc aucun ms par seconde d'audio n'est dérivable.
  - La cause « allocation Metal » n'a aucune preuve (P-66, P-73).
- **Préfill à 28 % GPU (#25)** : 8 jetons (BOS + 1 remplissage gauche + 6 de retard à 480 ms), soit environ le coût
  d'un pas.
  - Les 448 ms et les +3 980 Mio sont **compatibles**, à 1,1 % près, avec la matérialisation paresseuse du décodeur
    plus la copie fp32 de la tête : 1 623 (couches 4 bits) + 768 (embeddings) + 1 536 (copie fp32) + ≈ 10 (Ada)
    = 3 937 Mio.
  - *Vérif. croisée* — ce recoupement est un indice, pas une preuve. La valeur « 639 → 4 619 » est
    `Memory.activeMemory` lue par `takeSnapshot()` **immédiatement** après `MLX.eval(logits)` (mémoire lue avant
    IOKit, `ProfilingSession.swift:143-148` @1.1.1). MLX détache le graphe à l'évaluation
    (`transforms.cpp:306-307` @`1f8e74e`, `ce45c52` non disponible hors ligne) ; la copie fp32 n'est donc encore comptée que si le gestionnaire de complétion du
    command buffer ne l'a pas encore rendue au pool. Les instantanés de pas, pris après ≈ 1,5 ms de lecture IOKit
    (`:102-104`), ne la voient pas, ce qui est cohérent avec « Peak MLX Active » = 4 619 Mo = la valeur de fin de
    préfill (maximum des instantanés, pas `Memory.peakMemory`). Le **mécanisme** (copie à chaque pas) est, lui,
    vérifié dans le code MLX (`ops.cpp:3069-3082` @`ce45c52`).
  - Ce n'est ni un préfill lent ni un « prompt trop court » (P-61, P-66).
- **`Memory.clearCache()` l. 139** : toutes les 256 générations, à peu près aligné sur la croissance de
  `KVCacheSimple` (pas de 256).
  - Sans `cacheLimit`, c'est la seule borne du pool, qui garderait sinon les anciens tampons KV de tailles
    croissantes.
  - Revers : le vidage libère aussi les tampons réutilisables d'un pas, dont la copie fp32 de 1,5 Gio aujourd'hui.
    Le pas suivant réalloue, ce qui peut expliquer le max à 67 ms (non établi).
  - Rien n'est vidé en fin de transcription ni au `unload()` (P-67).
- **Concaténation de remplissage par pas ?** Non, vérifié. Les concaténations du chemin Realtime sont toutes
  **par appel** :
  - remplissage audio, une fois par fichier (`VoxtralRealtimePipeline.swift:208-212`) ;
  - zéros des conv causales, deux fois par encodage (`VoxtralRealtimeEncoder.swift:44`) ;
  - préfixe si l'audio est plus court que l'invite (`VoxtralRealtimeModel.swift:99`) ;
  - table de temps, une fois (`VoxtralRealtimeDecoder.swift:30`).

  La boucle de décodage n'en fait aucune : tranche de `adapterOut`, ligne d'embedding, forward. La seule
  concaténation par pas est dans `KVCacheSimple` amont, une fois tous les 256 pas (T11). Les indices `concat` du
  scan sur ces fichiers sont donc des faux positifs de performance.

## 4. Constats (a) — chemin Realtime

### P-60 · haute · `Realtime/Pipeline/VoxtralRealtimePipeline.swift:215-218` ; `Realtime/VoxtralRealtimeEncoder.swift:76-83`, `:138-139` ; `Realtime/VoxtralRealtimeDecoder.swift:24-31`, `:49-56` ; `Realtime/VoxtralRealtimeModel.swift:69-70`, `:93`, `:129` — Tout le chemin Realtime calcule en fp32 (encodeur, décodeur, cache KV)

- **Constat** : trois sources fp32 indépendantes promeuvent le graphe (T17, piège 26).
  1. Le mel sort en fp32 (`logMelSpectrogram` sur l'audio fp32). `conv_general` promeut entrée et poids vers fp32
     (`ops.cpp:4131` @`ce45c52`). L'encodeur entier et l'adaptateur (`Linear` 16 bits → copie fp32 des poids,
     `ops.cpp:3069-3082`) tournent donc en fp32.
  2. Même si le mel était converti, `computeRoPEFreqs` bâtit `cos`/`sin` en fp32 (`MLXArray(theta)`,
     `positions.asType(.float32)`). `interleavedRoPE` multiplie q/k bf16 par ces tables, ce qui donne q/k fp32, puis
     `scaledDotProductAttention` convertit q, k, v vers `result_type` = fp32 (`fast.cpp:704`).
  3. `computeTimeEmbedding` part de `MLXArray([Float])`, donc `tCond`, puis `adaScale` via les `Linear` non
     quantifiés, sont en fp32. `x * (1.0 + adaScale)` passe le flux résiduel du décodeur en fp32 dès la couche 0.
     Il l'était déjà, d'ailleurs : l'entrée `audio_embed (fp32) + tok_embed (16 bits)` est fp32
     (`VoxtralRealtimeModel.swift:93`, `:129`).

  Conséquences :
  - `quantizedMatmul` calcule en fp32 (`ops.cpp:4343`) et le cache KV est alloué en fp32 (dtype de `keys`,
    mlx-swift-lm `KVCache.swift:439-440`).
  - Avec le pack `realtime-4b-fp16` ou l'original bf16, **chaque `Linear` recopie son poids en fp32 à chaque
    appel**, comme STT P-05 : ≈ 5× le trafic d'un GEMV 16 bits au décodage.
  - Cause racine de P-61.
- **Preuve** :
  - Lecture du code Voxtral et des règles de promotion MLX citées. `rms_norm` rend `result_type(x, weight)`
    (`fast.cpp:82`).
  - Les noyaux SDPA fusionnés acceptent `head_dim` 64 et 128 (`scaled_dot_product_attention.cpp:618-634`), donc pas
    de repli matérialisé : le coût fp32 est en bande passante et en calcul, pas en mémoire L×L.
  - Diagnostic sans code : `VoxtralCLI realtime x.wav --embeddings` imprime `dtype` (`VoxtralCLI.swift:739`). On
    attend `float32` aujourd'hui.
- **Correction** :
  - Déterminer `computeDType` depuis un tenseur **jamais quantifiable** (ex. `decoder.norm.weight`, ou
    `ada_rms_norm_t_cond.ada_down.weight` comme la référence mlx-audio, `voxtral_realtime.py:130-133`), ou le figer
    au chargement **avant** toute quantification. *Vérif. croisée* : la version d'origine prenait
    `decoder.tokEmbeddings.weight`, que P-61 (étape 2) propose justement de quantifier : le dtype deviendrait `uint32`
    (pièges 2 et 30). Ne pas utiliser `asType(layer.weight.dtype)` sur un module quantifiable.
  - Convertir le mel à l'entrée de l'encodeur, `cos`/`sin` après calcul en fp32, et `adaScale` une fois dans
    `precomputeAdaScales`.
  - `extractAudioEmbeddings` rend alors ce dtype. C'est additif mais visible : documenter.
- **Gain attendu** : activations et KV divisés par 2 en octets. Suppression des copies fp32 de l'adaptateur et, en
  16 bits, de tous les poids. Prérequis du gain principal de P-61. Source : T17 (Q ×2,74 sur un cas comparable),
  STT P-01/P-05. **À MESURER.**
- **Risque API** : aucun en signature. Le dtype des embeddings extraits change (0 consommateur connu).
- *Vérif. croisée* — **risque de parité** : la référence mlx-audio `main` garde elle aussi le mel en fp32
  (`voxtral_realtime.py:108`, `conv_stem` sans conversion, `encoder.py:169-186`). Après P-60, Voxtral s'écarte donc
  numériquement de la référence comme de son propre état actuel : la porte ne peut être que WER avec tolérance
  (ASK Q1), jamais « identique à mlx-audio ».
- *Vérif. croisée* — **effet sur la tête** : dès que `h` est dans le dtype de `W`, `matmul` n'insère plus d'`astype`
  (`ops.cpp:3078-3083` @`ce45c52`) : P-60 **supprime à lui seul** la copie fp32 de 1,5 Gio par pas décrite en P-61.
  Le gain principal du pack 4 bits appartient donc à la porte de P-60.
- **Effort** : S. **Statut** : **VÉRIFIÉ** (mécanisme) ; effet **À MESURER**.
- **Fiche proposée** — *Realtime en dtype du modèle*.
  - **Porte** : dtype de l'embedding = dtype du modèle, et dtype des logits = dtype du modèle (audit de dtype P-79).
  - Encodage ≥ −10 % sur C-moyen (A/B/B/A) et pic process −≥ 300 Mo.
  - *Vérif. croisée* — pack 4 bits, C-moyen : ms/pas médian ≥ −30 % (A/B/B/A) et pic process −≥ 1 Go (copie de la
    tête supprimée ; reprise de l'ancienne porte de P-61).
  - **Parité** : texte identique sur C-court, **ou** WER ≤ référence + 0,2 pt sur C-moyen fr/en (tolérance
    numérique fp32 → 16 bits). La décision revient à Vincent (ASK Q1).
  - En `realtime-4b-fp16` : ms/pas ≥ −40 %.
  - **Cible : macos-gpu.**

### P-61 · haute · `Realtime/VoxtralRealtimeDecoder.swift:177-189` ; `Realtime/VoxtralRealtimeModelLoading.swift:39` ; `ops.cpp:3069-3082` (MLX `ce45c52`) — Tête liée : la table 131 072 × 3 072 est relue en 16 bits **et recopiée en fp32 à chaque pas**, y compris dans le pack 4 bits recommandé

- **Constat** :
  - `logits(h)` = `MLX.matmul(h, tokEmbeddings.weight.transposed())`. `h` est fp32 (P-60), donc `matmul` exécute
    `astype(W, float32)` : 1 611 Mo écrits puis relus, en plus des 805 Mo lus, **à chaque pas**.
  - La tête n'est jamais quantifiée : le prédicat saute `tok_embeddings` (`VoxtralRealtimeModelLoading.swift:39`).
    Même sans fuite fp32, elle pèse 805 Mo par pas, soit 32 % du trafic de poids du pack « 4 bits ».
  - `embedToken` lit `tokEmbeddings.weight[tokenId]` (`:177-179`). Le jour où l'embedding sera quantifié, ce code
    lira des `uint32` packés (piège 2).
  - Trafic par pas aujourd'hui : 1,70 Go (couches 4 bits) + 4,03 Go (tête : lecture 16 bits, écriture et lecture
    fp32) ≈ **5,8 Go**. À 33,7 ms/pas (F-R1), cela fait ≈ 173 Go/s effectifs.
- **Preuve** :
  - Lecture du code et des règles de promotion ; mlx-swift `Embedding.asLinear` (`Embedding.swift:47-49`) et
    `QuantizedEmbedding.asLinear` → `quantizedMM` (`Quantized.swift:213-217` @0.31.6 ; *Vérif. croisée* : l'original
    citait `:240-245`, numérotation de `9019419`) existent.
  - Index HF du pack 4 bits relu : `decoder.tok_embeddings.weight` sans `scales`/`biases`, adaptateur idem → non
    quantifiés.
  - La référence mlx-audio `main` fait ce que propose la correction : `tok_embeddings.as_linear(h)`
    (`decoder.py:266`) et un prédicat de quantification qui **inclut** `tok_embeddings` et l'adaptateur
    (`voxtral_realtime.py:572`).
  - **Recoupement chiffré** : +3 980 Mio pendant un préfill de 8 jetons (#25) contre un budget de 3 937 Mio
    (couches 1 623 + embeddings 768 + copie fp32 1 536 + Ada ≈ 10), soit un écart de 1,1 %. *Vérif. croisée* :
    compatible mais non probant (instantané `activeMemory` pris juste après `eval`, voir §3 bis) ; le mécanisme reste
    VÉRIFIÉ par le code MLX.
  - Pic process 7 949 Mo (#23) : cohérent avec le pool qui retient la copie de 1,5 Gio, sans `cacheLimit` (P-67).
- **Correction** :
  1. Après P-60, calculer les logits par `tokEmbeddings.asLinear(h)` dans le dtype du modèle : aucune copie.
     *Vérif. croisée* : `Embedding.asLinear` = `matmul(x, weight.T)`, identique au code actuel ; seule la
     disparition de la fuite fp32 (P-60) supprime la copie. Si P-61 devait passer **avant** P-60, l'étape 1 est un
     cast local `h.asType(W.dtype)` dans `logits(_:)`.
  2. T12 : profil `fast` avec tête **8 bits** et profil `lean` avec tête **4 bits**. Soit on quantifie au
     chargement (`QuantizedEmbedding` + `asLinear` → `quantizedMM` ; jamais dé-quantifier en entier), soit on publie
     un pack `-head8`.
  3. `embedToken` passe par `tokEmbeddings(MLXArray(tokenId))`, qui dé-quantifie une seule ligne.
- **Gain attendu** (calcul, si le pas reste borné par la bande passante à efficacité égale) :

  | Configuration | Trafic par pas | ms/pas attendus |
  |---|---|---|
  | Aujourd'hui | 5,8 Go | 33,7 (mesuré) |
  | Tête 16 bits, sans copie | 2,6 Go | ≈ 15 |
  | Tête 8 bits | 2,2 Go | ≈ 12,6 |
  | Tête 4 bits | 2,0 Go | ≈ 11,4 |

  Soit ×2,2 à ×3 en attendu. Pic −1,5 Gio. Budget temps réel de 80 ms/pas : sur une puce à ≈ 100 Go/s, le code
  actuel dépasserait 58 ms/pas contre 20 à 26 ms après correction. Source : T12 (Y ABC ×2,3, pic −283 Mo),
  piège 26. **À MESURER.**
- **Risque API** : aucun. `logits`, `embedToken` et `embedTokens` gardent leurs signatures.
- **Effort** : S (1.) ; M (2., avec parité). **Statut** : **VÉRIFIÉ** (mécanisme) ; gain **À MESURER**.
- **Fiche proposée** — *Tête liée sans copie fp32, puis quantifiée*.
  - *Vérif. croisée* — **porte rebasée** : l'ancienne porte unique (ms/pas ≥ −30 % et pic −≥ 1 Go) mesurait en
    réalité l'effet de P-60, ordonné avant P-61 au §6 ; mesurée après P-60, la tête 8 bits n'offre qu'≈ −15 % de
    trafic (2,5 → 2,1 Go/pas) et −377 Mo : la fiche aurait échoué à tort.
  - **Porte, étape 1** (seulement si P-61 passe avant P-60 : cast local) : ms/pas médian ≥ −30 % (A/B/B/A, pack 4 bits,
    C-moyen) et pic process −≥ 1 Go ; si P-60 est déjà livré, l'étape 1 est sans objet (vérifier par l'audit de dtype
    que les logits ne sont plus fp32).
  - **Porte, étape 2** (tête quantifiée, baseline = après P-60) : ms/pas médian ≥ −8 % pour la tête 8 bits
    (≈ −15 % attendus), `activeMemory` après chargement −≥ 300 Mo.
  - **Parité** : texte identique pour 1. ; WER ≤ référence + 0,2 pt pour la tête 8 bits et + 0,5 pt pour la tête
    4 bits, sinon la tête 4 bits est rejetée et documentée.
  - **Cible : macos-gpu.**

### P-62 · haute · `Realtime/VoxtralRealtimeEncoder.swift:151-159`, `:320-342` — L'encodeur ignore sa fenêtre glissante de 750 : attention pleine sur tout l'audio (divergence au-delà de 15 s, coût quadratique, mémoire non bornée)

- **Constat** :
  - `encodeFull` applique le masque `.causal` à toute la séquence. Le paramètre `cache` de l'attention n'est jamais
    utilisé.
  - Le commentaire de la fonction le reconnaît : « for audio within sliding window » (`:327`).
  - La configuration porte `sliding_window: 750` (= 15 s à 50 positions/s), mais le champ `slidingWindow` n'est lu
    que dans l'`init` (`:105`) et n'est jamais appliqué.
  - La carte du modèle (F-R6) indique que les deux composants utilisent l'attention glissante « for unbounded audio
    length ».
  - Au-delà de 15 s, chaque position voit des distances jamais vues à l'entraînement. La sortie diverge de la
    référence ; la qualité est à mesurer.
- **Coût** (calcul) : une paire requête-clé coûte ≈ 8 192 FLOP par couche ; les linéaires coûtent 60,3 MFLOP par
  position et par couche.

  | Durée audio | Coût plein / coût fenêtré |
  |---|---|
  | 40 s | ×1,05 (*Vérif. croisée* : ×1,03 à l'origine ; 2 000 positions, 1 000 clés en moyenne contre 610) |
  | 5 min | ×1,8 |
  | 30 min | ×6,5 |
  | 1 h | ×12 |

  De plus, le graphe de 32 couches sur tout le fichier (T16) garde des activations proportionnelles à la durée :
  FFN fp32 de 1,8 Go par tenseur à 30 min.
- **Instrument touché** : le Realtime est le juge ASR de `docs/zerovoice_benchmark.md` (F-R5, clips jusqu'à 20,8 s).
  Sa validité au-delà de 15 s n'est pas établie.
- *Vérif. croisée* — **référence** : mlx-audio `main` (dont se réclament les en-têtes des fichiers Realtime) encode
  en `encode_full` seulement si `seq_len <= sliding_window`, sinon par tranches de 750 avec
  `RotatingKVCache(max_size=750)` par couche (`encoder.py:188-219`, `:271-275`). Le port Swift n'a gardé que la
  branche courte : la divergence est donc aussi une divergence **par rapport à la référence déclarée**.
- **Correction** : encodage **par tranches** causales avec un `RotatingKVCache(maxSize: 750)` par couche
  (mlx-swift-lm `KVCache.swift:558`, `makeMask(n:windowSize:returnArray:)` `:253-263`).
  - *Vérif. croisée* — deux points indispensables absents de la version d'origine : (a) la RoPE doit recevoir la
    position **absolue** du début de tranche (`rope_offset = chunk_start` dans la référence) ; `computeRoPEFreqs`
    part aujourd'hui de 0 à chaque appel (`VoxtralRealtimeEncoder.swift:331-334`) ; (b) le masque doit être passé
    explicitement (fenêtre + causal dans la tranche), car `RealtimeEncoderAttention` bascule en `.none` dès qu'un
    cache est fourni sans masque (`:153-158`) : une tranche de 750 positions sans masque verrait son futur.
  - Porter l'état des conv causales d'une tranche à l'autre : 2 trames mel pour conv1, 1 sortie pour conv2. Pour le
    mode fichier, la référence garde la conv sur tout le mel et ne découpe que le transformer, ce qui suffit ; l'état
    des conv ne sert qu'au streaming (P-71).
  - Même code pour le fichier et pour le streaming (P-71).
  - Piège 29 : aucun rollback ici, la fenêtre n'est jamais « reculée ».
- **Risque API** : aucun. **Effort** : M.
- **Statut** : **VÉRIFIÉ** (divergence architecturale) ; effet sur le WER et le temps **À MESURER**.
- **Fiche proposée** — *Encodeur à fenêtre 750 par tranches*.
  - **Porte** : embeddings identiques (L2 relative < 1e-3) à `encodeFull` sur un clip ≤ 15 s. Sur C-moyen, WER ≤ la
    valeur actuelle, **et** comparaison à mlx-audio Python sur le même clip (ASK Q2).
  - Encodage de C-long ≥ −30 % ; pic process borné, indépendant de la durée (±10 % entre 3 et 12 min).
  - **Cible : macos-gpu.**

### P-63 · moyenne · `Realtime/VoxtralRealtimeDecoder.swift:191-194` ; `Models/VoxtralLlama.swift:136-146` — Le décodeur ignore sa fenêtre de 8 192 : cache KV sans borne

- **Constat** :
  - `createCache()` crée des `KVCacheSimple`. `LlamaAttention` utilise `.causal` au préfill et `.none` au décodage :
    l'attention couvre tout l'historique.
  - Au-delà de 8 192 pas (≈ 10 min 55 s), le calcul diverge de la référence et le cache grossit sans fin :
    104 Kio/jeton en bf16, 208 Kio en fp32 aujourd'hui.
  - Pour 1 h, cela représente 45 000 pas, 4,5 Gio de KV en bf16 (8,9 Gio en fp32), **relus à chaque pas**, contre
    832 Mio en bf16 une fois la fenêtre posée. (*Vérif. croisée* : 4,6 / 9,1 à l'origine, confusion Go/Gio :
    45 000 × 104 Kio = 4,79 Go = 4,46 Gio.)
  - *Vérif. croisée* — la référence mlx-audio utilise `RotatingKVCache(max_size=sliding_window)` pour le décodeur
    (`decoder.py:226-229`) : écart confirmé avec la référence déclarée.
  - Aujourd'hui masqué par P-64 : le défaut de 4 096 pas coupe à ≈ 5 min 27 s.
- **Correction** : `RotatingKVCache(maxSize: config.decoder.slidingWindow)`. Au décodage d'un jeton, le masque `.none`
  reste correct sur l'anneau, car les clés sont tournées (RoPE) avant l'écriture.
- **Risque API** : aucun. **Effort** : S. **Statut** : **VÉRIFIÉ** ; parité au-delà de 11 min **À MESURER**.
- **Fiche proposée** — *Cache KV décodeur à fenêtre 8 192*.
  - **Porte** : texte identique sur C-moyen. Sur un clip d'environ 15 min, avec `maxTokens` relevé : pic process
    stable après 8 192 pas (±5 %) et ms/pas au-delà de 8 192 = ms/pas à 8 000 (±5 %).
  - **Cible : macos-gpu.**

### P-64 · moyenne · `Realtime/VoxtralRealtimeModel.swift:115-124`, `:143-149` ; `Realtime/Pipeline/VoxtralRealtimePipeline.swift:29`, `:32` ; `VoxtralTranscriptionTest/ProfileCommand.swift:60-61`, `:287-288` — `maxTokens` compte des **trames audio** : troncature silencieuse à ≈ 5 min 27 s par défaut (≈ 39 s sous `profile`)

- **Constat** :
  - Le modèle émet un jeton par trame de 80 ms, remplissages compris. `generated.count` est donc le nombre de pas,
    et `generated.count > maxTokens` coupe l'**audio**, sans erreur ni indicateur.
  - Défaut de la bibliothèque : 4 096, soit une coupure au-delà d'environ 327 s d'audio.
  - `profile` passe 500 : coupure au-delà d'environ 39 s. Les « 501 pas » de #23 valent exactement 500 + 1, ce qui
    est cohérent avec une trace **tronquée** (*Vérif. croisée* : très probable, non prouvé ; un EOS émis
    exactement au 501ᵉ jeton donnerait le même compte, et la durée de l'audio n'est notée nulle part).
  - Hors par un : `>` rend `maxTokens + 1` jetons.
  - Le « min 0,7 ms » de #23 est **vraisemblablement** le pseudo-pas de sortie (`:120-123`), enregistré sans forward
    (P-75) (*Vérif. croisée* : « est » à l'origine ; aucun autre pas n'évite le forward, mais la trace n'est pas
    disponible).
  - *Vérif. croisée* — la sémantique est **héritée** de la référence mlx-audio (`len(generated) > max_tokens`, défaut
    4 096, `voxtral_realtime.py:274`) : défaut réel, mais commun au port et à la référence ; la parité avec mlx-audio
    sur un clip > 5 min 27 s exige de relever `max_tokens` des deux côtés.
- **Correction** :
  - Borner la boucle par la durée audio (`nAudioTotal - promptLen`) et faire de `maxTokens` un budget de jetons
    **texte** (hors remplissage), ou le retirer.
  - Corriger le hors-par-un.
  - Signaler une troncature (champ de résultat ou erreur).
  - `profile` doit dériver la borne de l'audio.
- **Risque API** : additif si le champ reste avec une sémantique documentée ; « cassant » en sémantique, avec
  0 consommateur connu → ASK Q3.
- **Effort** : S. **Statut** : **VÉRIFIÉ**. Complète STT P-11.
- **Fiche proposée** — *Transcription Realtime non tronquée*.
  - **Porte** : un clip d'environ 12 min est transcrit en entier (dernier mot de la référence présent), et un test
    unitaire vérifie le nombre de pas = trames. Le test doit échouer sans le correctif (piège 38).
  - **Cible : macos-gpu** (test) ; le code est écrivable en cloud.

### P-65 · moyenne · `Realtime/VoxtralRealtimeModel.swift:115-141`, `:159-164` — Décodage synchrone : deux synchronisations par pas, aucun `asyncEval` (T15)

- **Constat** : chaque pas fait `MLX.eval(logits)` (`:133`), puis `argMax(logits).item()` (`:161`), ce qui fait un
  second graphe et une seconde synchronisation. `embedToken(token)` (`:128`) exige l'entier sur le CPU avant de
  construire le pas N+1. Le GPU attend pendant la construction du graphe, et inversement.
- **Correction** :
  - Jeton en `MLXArray` (`argMax` paresseux), embedding par `tokEmbeddings(tokenArr)`.
  - Construire le pas N+1, `asyncEval`, puis `.item()` du pas N.
  - Test d'EOS avec un pas de retard : un pas jeté en fin de fichier.
  - Vérifier que le chemin asynchrone est réellement pris (piège 6).
- **Gain attendu** : T15 (Q −13 à −22 % ms/pas). Ici le pas est d'abord borné par la tête (P-61) : mesurer
  **après** P-61. **À MESURER.**
- **Risque API** : aucun. **Effort** : S.
- **Statut** : **VÉRIFIÉ** (absence) ; gain **À MESURER**.
- **Fiche proposée** — *Pipelining du décodage Realtime*.
  - **Porte** : ms/pas médian ≥ −5 % (A/B/B/A, après P-61) et texte identique sur C-court et C-moyen.
  - **Cible : macos-gpu.**

### P-66 · moyenne · `Realtime/VoxtralRealtimeModelLoading.swift:44-58` ; `Realtime/Pipeline/VoxtralRealtimePipeline.swift:94-100` — Poids jamais matérialisés au chargement : l'encodage et le préfill paient la lecture disque ; pas de chargement « encodeur seul »

- **Constat** :
  - Les safetensors sont paresseux et `update(parameters:)` s'exécute sans `eval`. La lecture disque des
    518 + 48 + 10 Mio de l'encodeur tombe dans « Audio Encoding » (+673 Mio, #24 ; *Vérif. croisée* : 576 Mio de
    poids sur 673, le reste n'est pas ventilé). Celle des 2,4 Gio du décodeur tombe dans « Prefill » (+3 980 Mio en
    448 ms, #25 ; recoupement compatible, non probant, voir §3 bis).
  - Le premier `transcribe` après `loadModel` n'est pas représentatif, et un A/A froid/chaud échoue par
    construction.
  - La paresse a un effet utile, fragile : `extractAudioEmbeddings` seul ne matérialise jamais le décodeur
    (T4/T5 de facto). Une matérialisation globale naïve le casserait.
- **Correction** :
  - Matérialiser **par voie** au chargement, couche par couche (piège 4) : encodeur et adaptateur toujours ;
    décodeur et tête si la transcription est voulue.
  - Option additive `loadModel(…, components: .encoderOnly)` pour l'extraction d'embeddings.
  - Homologue de STT P-04.
- **Gain attendu** : premier TTFT honnête et phases mesurables ; en encodeur seul, −2,4 Gio résidents garantis.
  Source : piège 18, `measurement.md` (mesures à chaud).
- **Risque API** : additif. **Effort** : S. **Statut** : **VÉRIFIÉ**.
- **Fiche proposée** — *Matérialisation Realtime par voie*.
  - **Porte** : première transcription = transcription chaude ±5 % (A/A) ; `activeMemory` après chargement = budget
    du pack ±5 %.
  - En `.encoderOnly` : `activeMemory` après extraction ≤ encodeur + 10 %.
  - **Cible : macos-gpu.**

### P-67 · moyenne · `Realtime/VoxtralRealtimeModel.swift:138-140` ; `Realtime/Pipeline/VoxtralRealtimePipeline.swift:181-186` — Politique mémoire Realtime : pas de `cacheLimit`, vidage périodique aveugle, rien au `unload()`

- **Constat** :
  - Aucune `Memory.cacheLimit` n'est posée (T1) ; le préréglage `MemoryOptimizationConfig` de STT ne s'applique pas
    au Realtime.
  - `clearCache` tous les 256 pas vide aussi les tampons réutilisables (3 bis). La fin de `generate` et `unload()`
    ne vident rien. L'app hôte garde donc le pool (jusqu'à la copie fp32 de 1,5 Gio et les transitoires de
    l'encodeur) après déchargement.
  - Par contraste, STT `unload()` → `fullCleanup()` (`VoxtralPipeline.swift:485`).
  - Pic process à 7 949 Mo contre pic MLX à 4 619 Mo (#23), cohérent avec le pool retenu.
- **Correction** :
  - Limites par étape portées par un profil Realtime : encodage ≈ 1 Go, décodage ≈ 2 Go (valeurs de départ Y/Q,
    À MESURER). Limites adaptatives en `lean` (T2).
  - `clearCache` en fin de `generate` (lean) et dans `unload()`.
  - Garder ou retirer le vidage périodique selon la mesure (porte 5 %).
- **Risque API** : aucun. **Effort** : S.
- **Statut** : **VÉRIFIÉ** (absences) ; effets **À MESURER**.
- **Fiche proposée** — *Politique mémoire Realtime*.
  - **Porte** : après `unload()`, `phys_footprint` = avant chargement ±200 Mo. ms/pas inchangé (±3 %). Pic ≤
    baseline, et max/médiane des pas ≤ 1,3.
  - **Cible : macos-gpu.**

### P-68 · basse · `Realtime/VoxtralRealtimeEncoder.swift:55-83`, `:137-139` — RoPE entrelacée écrite à la main dans l'encodeur (32 couches × q et k), alors que le noyau fusionné fait la même chose

- **Constat** :
  - `interleavedRoPE` enchaîne, pour q et pour k à chaque couche : reshape, deux tranches, six opérations
    élémentaires, `stacked` et reshape, sur des tables fp32 (P-60).
  - La rotation par paires consécutives est exactement `RoPE(traditional: true)`. Le décodeur l'utilise déjà
    (`VoxtralRealtimeDecoder.swift:76-90` → `initializeRope`, `Models/VoxtralLlama.swift:450-465`).
  - Les fréquences sont identiques : `base^(-2i/d)`, contre `1/θ^(2i/headDim)` à `:77-80`.
- **Correction** : appliquer `MLXFast.RoPE(…, dimensions: headDim, traditional: true, base: θ, scale: 1, offset: 0)`
  sur la disposition `[1, H, L, D]` avant la SDPA. Aucune table, dtype conservé. Pas de `compile` (piège 20 :
  l'enrôlement utilise `valueAndGrad` dans le même processus).
- **Gain attendu** : encodeur, moins de noyaux et d'octets. **À MESURER.**
- **Risque API** : aucun pour les fonctions internes. Attention : `interleavedRoPE` et `computeRoPEFreqs` ont une
  visibilité `internal`, pas publique. **Effort** : S. **Statut** : **VÉRIFIÉ** (équivalence) ; gain **À MESURER**.
- **Fiche proposée** — *RoPE fusionnée dans l'encodeur*.
  - **Porte** : embeddings L2 relative < 1e-2 contre la version actuelle convertie au même dtype ; WER identique sur
    C-moyen ; encodage ≥ −5 %.
  - **Cible : macos-gpu.**

### P-69 · basse · (absence) `Realtime/VoxtralRealtimeModelLoading.swift:31-42` — Même calcul pour l'encodeur (borné calcul) et le décodeur (borné bande passante) (T14/T20)

- **Constat** : l'encodeur traite 50 positions par seconde d'audio en `quantizedMatmul` 4 bits : de grandes
  activations, un calcul borné par la dé-quantification (Q : préfill bf16 151 contre 2 bits 113 tok/s). Le décodeur,
  lui, gagne à rester packé.
- **Correction** : en profil `fast`, dé-quantifier l'encodeur vers le dtype du modèle au chargement
  (`dequantizeWeights`, Y `WeightResidency.swift:92-118`). Coût : +1,4 Go (0,965 G params × 2 o − 543 Mo).
- **Risque API** : aucun. **Effort** : S. **Statut** : **À MESURER** (après P-60 et P-62).
- **Fiche proposée** — *Encodeur dé-quantifié (fast)*.
  - **Porte** : encodage ≥ −5 % sur C-moyen (A/B/B/A) ; WER identique ; sinon fiche retirée avec la mesure.
  - **Cible : macos-gpu.**

### P-70 · moyenne · `Realtime/VoxtralRealtimeConfiguration.swift:195-205`, `:245-250` ; `Realtime/VoxtralRealtimeModelLoading.swift:199-214` ; `Realtime/VoxtralRealtimeRegistry.swift:28-57` ; `Realtime/Pipeline/VoxtralRealtimePipeline.swift:78-79` — Seuls le pack 4 bits, le pack fp16 et l'original bf16 sont chargeables ; les packs 6 et 8 bits publiés seraient chargés silencieusement faux ; aucun profil Realtime

*Vérif. croisée* — titre amendé : l'original bf16 `mistralai/…` (format A, entrée `realtime-4b` du registre) est lui
aussi chargeable ; le mode d'échec des packs voxmlx est précisé ci-dessous.

- **Constat** :
  - `mlx-community/Voxtral-Mini-4B-Realtime-6bit` et `ellamind/…-8bit-mlx` sont au format voxmlx (F-R6). Leur
    `config.json` a la forme `params.json` : `load` échoue sur la forme mlx-community et retombe sur
    `loadFromParamsJSON`, qui pose `quantization: nil` (`:249`).
  - Leurs clés (`adapter.w_in`, `encoder.layers.N.attention.q_proj`, adaptateur quantifié) ne sont pas mappées par
    le « Format B ».
  - Avec `verify: .none` (S-04), l'encodeur resterait aléatoire, sans erreur. *Vérif. croisée* : c'est **tout** le
    modèle — les clés du décodeur voxmlx sont `language_model.layers.N.*` (index HF relu), aucune ne commence par
    `decoder.` ; si aucune clé ne correspond (cas de toutes celles observées dans l'index), aucun poids `uint32` n'atteint un `Linear` non quantifié : pas de
    plantage attendu, un modèle entièrement à l'initialisation aléatoire (sortie incohérente, silencieuse).
  - *Vérif. croisée* — second repli silencieux : ces packs ne sont pas au registre, et un `modelId` inconnu retombe
    sur le modèle par défaut (`VoxtralRealtimePipeline.swift:78-79`) ; seule `loadVoxtralRealtimeModel(from:)`
    (publique) atteint le cas ci-dessus.
  - Le standard des profils (`<bits>bit-fast|lean`) n'a donc aujourd'hui qu'une largeur Realtime utilisable,
    4 bits (et fp16 pour 16 bits), et aucun type de profil (scan §6).
- **Correction** :
  1. Lire `quantization` aussi dans la forme `params.json`.
  2. Assainisseur du format voxmlx, ou export d'un pack 8 bits propre (format mlx-audio) avec SHA-256. ASK Q4.
  3. Profils Realtime : proposition au §7.
  4. En 16 bits, préférer l'original bf16 (`mistralai/…`, format A) au pack `fp16`, dtype différent de
     l'entraînement.
- **Risque API** : additif. **Effort** : M.
- **Statut** : **VÉRIFIÉ** (code + index HF) ; sortie effective du chargement voxmlx **À MESURER** (S-04 le rend
  silencieux).
- **Fiche proposée** — *Packs Realtime 8 bits et profils*.
  - **Porte** : le pack 8 bits se charge avec `verify: [.all]` (0 clé manquante ou en trop) ; WER 8 bits ≤ WER
    4 bits sur C-moyen ; une ligne `References.md` par profil.
  - **Cible : macos-gpu** ; lecture de config et assainisseur écrivables en cloud.

### P-71 · moyenne · (absence) `Realtime/Pipeline/VoxtralRealtimePipeline.swift:118-154` ; `Realtime/VoxtralRealtimeModel.swift:57-157` — « Realtime » sans API de streaming : tout le fichier est encodé avant le premier jeton

- **Constat** :
  - L'API prend une URL de fichier. Mel complet, puis encodage complet, puis décodage.
  - La latence au premier texte vaut mel + encodage de **tout** le fichier + préfill + pas jusqu'au premier jeton
    non-remplissage : 3,28 + 5,44 + 0,45 s dans F-R1, avant même le premier pas.
  - Le modèle est pourtant conçu pour le flux : encodeur causal à fenêtre, décodeur synchrone à la trame. mlx-audio
    expose `stream=True` (carte HF) ; voxmlx et Supervoxtral visent la transcription en direct sur macOS (carte
    ellamind).
- **Correction** : API incrémentale additive. Morceaux PCM en entrée, `AsyncThrowingStream` de deltas de texte en
  sortie, avec `onTermination` (MLX-003, S-01 TTS). L'encodeur par tranches de P-62 et le décodeur pas à pas
  partagent l'état entre morceaux.
- **Gain attendu** : latence au premier texte ≈ retard (480 ms) + un pas, au lieu d'une durée proportionnelle au
  fichier. **À MESURER.**
- **Risque API** : additif. **Effort** : L. **Statut** : **VÉRIFIÉ** (absence).
- **Fiche proposée** — *Streaming Realtime*.
  - **Porte** : même texte qu'en mode fichier sur C-moyen, à morceaux de 80 ms à 1 s.
  - Latence d'un mot (fin de parole → delta émis) ≤ retard + 150 ms, médiane sur 20 mots.
  - Temps par pas < 80 ms au p99 sur la machine de référence.
  - **Cible : macos-gpu.** Périmètre produit : ASK Q6.

### P-72 · basse · (proposition) `Realtime/VoxtralRealtimeModel.swift:115-141` — Levier R&D : pas spéculatifs « remplissage » pour réduire le nombre de passes séquentielles (analogue T21)

- **Constat** :
  - Chaque trame exige un forward séquentiel. La parole produit ≈ 3-4 jetons texte/s pour 12,5 trames/s, donc une
    grande part des pas émet probablement un jeton non textuel. La part réelle n'est pas mesurée : c'est le point 1
    de la fiche.
  - Proposer k trames « remplissage » et les vérifier en **un** forward de k positions (masque causal sur le cache)
    : un forward lit les poids une fois pour k pas acceptés.
  - Rollback par `trim` du cache : `KVCacheSimple` le permet. Avec un `RotatingKVCache` (P-63), le dimensionner à
    fenêtre + k (pièges 29 et 36).
- **Gain attendu** : jusqu'à ÷(pas acceptés par vérification) sur les passes de décodage, sans valeur connue.
  **À MESURER** ; aucune source catalogue, technique nouvelle.
- **Risque** : numérique d'un forward k contre k forwards (argmax qui bascule). **Porte bit-exacte exigée** (texte
  identique). **Effort** : M-L. **Statut** : **À MESURER**.
- **Fiche proposée** — *Spéculation « remplissage » (R&D)*.
  - **Porte** : étape 1, la part de jetons de remplissage sur C-moyen est ≥ 50 %, sinon abandon.
  - Étape 2 : passes de décodage ≥ −25 % et texte identique sur C-court, C-moyen et C-long.
  - **Cible : macos-gpu**, après P-61 et P-65.

## 5. Instruments de mesure (b)

### 5.1 Réponses aux questions du cadrage

Légende de la table :
- **« Chemin bib. (piège 33) »** : l'instrument passe-t-il par le chemin de la bibliothèque qu'exécutent les
  consommateurs ?
- **« Ligne JSON / mesure »** : sort-il une ligne JSON par mesure, comme l'exige `measurement.md` ?
- **« phys_footprint + pic MLX par phase »** : mesure-t-il l'empreinte process et le pic MLX, phase par phase ?

| Instrument | Chemin bib. (piège 33) | Validable en A/A | Ligne JSON / mesure | phys_footprint + pic MLX par phase | Release / Debug | Coût du profiler |
|---|---|---|---|---|---|---|
| `VoxtralBenchmark` (`BenchmarkCLI.swift`) | **non** : conversions Float16 sur des données aléatoires, copie « same as MLXCoreMLBridge » (`:187-241`), constante `typicalInferenceMs = 30000` (`:118`) — voir **A-16** | oui en théorie, mais ne mesure rien d'utile | non (texte) | non | non noté | — |
| `VoxtralCLI profile` (`ProfileCommand.swift`) | **oui** : `VoxtralPipeline`, `VoxtralTTSPipeline.synthesize` (voix prédéfinie seulement), `VoxtralRealtimePipeline.transcribe` ; pas d'enrôlement, ni de TTS clonée ou streaming (A-15, P-45) | **non en l'état** : un seul passage froid (P-66), Realtime tronqué (P-64), aucune option de répétition | **non** : rapport texte + trace Chrome (`:131-158`) | profiler ≥ 1.5 : oui par phase, échantillonné à 16 ms ; ≤ 1.4 : bords seulement (P-74) | ≥ 1.5 : type de build dans la trace (`RunEnvironment`) ; ≤ 1.4 : non | ≤ 1.4 : 4,7 ms/paire (8 paires Realtime ≈ 38 ms) + ≈ 1,5 ms/pas via `recordStep` ; ≥ 1.5 : ≈ µs |
| `VoxtralCLI realtime` / `transcribe` / `tts` | oui (pipelines) | non : `Date()` global, un passage | non | non | non noté (README : Release, `README.md:73-74`) | — |
| Phases internes (`session?.beginPhase`, `recordStep`) | oui (dans la bibliothèque) | selon l'instrument appelant | — | via le profiler | — | coût nul si aucune session active (`MLXProfiler.shared.activeSession == nil`) |
| `TTSQuantizationCampaignTests` | **oui** (`VoxtralTTSPipeline.synthesize` + ASR `VoxtralPipeline`) | partiellement : graines et textes répétés, mais pas d'A/A sur une même entrée (P-76) | non (`[campaign] …` en texte) | non | binaire XCTest : Debug par défaut (À VÉRIFIER, `docs/voice_cloning.md:141-145`) | — |
| `TTSGenerateStabilityReproTests` | **non** : `model.generate` direct, qui contourne le cache de préfixe, le porteur et les coupes du pipeline | **oui pour les sorties** : groupe SEED identique (`:153-160` : même nombre de trames et somme de forme d'onde à 1e-2 près — *Vérif. croisée* : « bit-identique » à l'origine, c'est plus faible), c'est un bon gabarit de porte de parité | non | MLX actif et cache seulement ; ni pic ni footprint | idem | — |
| `RuntimeBeacon` | sans objet (signal de présence, pas une mesure) | — | manifeste JSON de présence | — | — | nul si désactivé ; activé : une écriture atomique au début et à la fin, `update` limité (TTS streaming par morceau, enrôlement ≈ 100 fois). Realtime : `begin` sans `model` ni `update` (`VoxtralRealtimePipeline.swift:125`) |

### 5.2 Constats (b)

### P-73 · haute · issues fermées #23, #24, #25 ; `Realtime/Pipeline/VoxtralRealtimePipeline.swift:129-142` ; `Realtime/VoxtralRealtimeModel.swift:73-77`, `:106-110` ; swift-mlx-profiler `ProfilingSession.swift:200-213` (@1.4.0) — Les trois diagnostics Realtime reposent sur des artefacts d'instrument, et la fermeture de #23 a masqué le défaut principal (P-61)

- **Constat** : six biais se superposent dans la trace d'avril.
  1. **Phases imbriquées** : GPU % englobant = moyenne de deux lectures de bord, soit 0 %. Les pourcentages de
     temps sont calculés sur une somme qui compte l'encodage et le préfill deux fois (72 %), et « 21 tok/s » utilise
     le mauvais dénominateur (§3 bis).
  2. **Lecture instantanée** de « Device Utilization % » à des bords ou juste après une synchronisation : la valeur
     « 49 % systémique » (#13, #14, #24) n'est pas une mesure d'occupation (F-R4). *Vérif. croisée* : l'original
     affirmait qu'elle « est celle que le profiler documente pour ≈ 82 % réels » ; ce cas documenté concerne une
     phase de 5 ms et ne se transpose pas. Hypothèse compatible, À MESURER : moyenne entière de deux lectures de bord,
     (≈ 0 + ≈ 98) / 2 ≈ 49 (`ProfilingSession.swift:204-213`, `:229` @1.4.0).
  3. **Mel paresseux** : la phase « Mel Spectrogram » ne mesure que la lecture audio sur le CPU ; le STFT et le mel
     s'exécutent dans « Audio Encoding » (`VoxtralRealtimeModel.swift:76`).
  4. **Poids paresseux** (P-66) : la lecture disque est facturée à l'encodage et au préfill.
  5. **Run tronqué** (P-64) : 501 pas = plafond + 1 (très probable, non prouvé). La durée d'audio n'est notée nulle
     part (`ProfileCommand.swift:284`, `:300`).
  6. **fp32** (P-60/P-61) : jamais soupçonné. Les budgets mémoire (+3 980 Mio contre 3 937 Mio calculés) étaient
     **compatibles** avec lui (*Vérif. croisée* : « le montraient » à l'origine ; instantané `activeMemory` pris
     juste après `eval`, donc non probant, §3 bis).
  7. *Vérif. croisée* — **« Peak MLX Active » n'est pas un pic** : ≤ 1.4, c'est le maximum des instantanés
     `activeMemory` pris aux bords et aux pas (`ProfilingSession.swift:256-261` @1.1.1), pas `Memory.peakMemory` ;
     un transitoire intra-pas (la copie fp32 de la tête) lui échappe.
- **Preuve** : lecture des deux dépôts (profiler par tags `1.1.1` à `1.4.0`, même logique) et arithmétique ci-dessus.
- **Correction** :
  - Ne pas rouvrir d'issue depuis cet audit. Consigner dans `docs/knowledge/` une décision (« les conclusions
    #23-#25 sont caduques ; re-mesure avec l'instrument P-79 ») et un piège (phases imbriquées, lecture
    instantanée).
  - Re-mesurer : baseline P-79, puis Metal System Trace (profiler 1.5 : `MetalSystemTrace.swift`) pour l'occupation
    GPU réelle.
- **Risque API** : aucun. **Effort** : S (doc) + mesure. **Statut** : **VÉRIFIÉ**.
- **Fiche proposée** — *Re-diagnostic Realtime*.
  - **Porte** : trace 1.5.x (`.ioReportResidency`) et Metal System Trace sur C-moyen avec audio complet ;
    occupation GPU du décodage rapportée des deux façons (écart ≤ 10 pts).
  - Décision et piège écrits.
  - **Cible : macos-gpu** (mesure) ; décision et piège rédigeables en **cloud**.

### P-74 · moyenne · `Package.swift:53` ; `VoxtralTranscriptionTest/ProfileCommand.swift:100-108` ; swift-mlx-profiler `ProfilingConfig.swift:71`, `:75`, `:115-124` (@`bfe71d8`) — Sémantique de l'instrument dépendante d'une version non enregistrée ; lecteur GPU instantané par défaut

- **Constat** :
  - `from: "1.4.0"` sans `Package.resolved`. Un clone neuf résout 1.5.1 (2026-09-20) ; une machine déjà résolue
    peut rester en 1.4.x.
  - Entre les deux changent : le coût des bords (4,7 ms → ≈ 0), celui de `recordStep` (lecture IOKit ≈ 1,5 ms →
    appels Mach), la définition du GPU % (bords contre échantillonneur de 16 ms), les pics par phase (bords contre
    échantillons), la gestion des phases imbriquées et l'enregistrement du type de build.
  - Une même commande produit donc des chiffres **non comparables**. `ProfileCommand` n'enregistre ni la version du
    profiler, ni les révisions mlx-swift et mlx-swift-lm (métadonnées `:108`, `:170`, `:284`).
  - `ProfilingConfig(...)` y garde `gpuBackend: .deviceUtilization`, lecture instantanée que le profiler déclare
    « only meaningful averaged over many samples », au lieu de `.ioReportResidency`, moyenne d'intervalle, utilisée
    par `.fineGrained`.
- **Correction** :
  - `from: "1.5.1"` (ou `Package.resolved` suivi, S-18).
  - Configuration de diagnostic `.fineGrained`.
  - Versions résolues dans les métadonnées et dans chaque ligne JSON (P-79).
  - Pas de `snapshotAtPhaseBoundaries` dans un run chronométré.
- **Risque API** : aucun (cibles exécutables ; la bibliothèque ne fait que réexporter,
  `VoxtralProfilerBridge.swift:4`). **Effort** : S. **Statut** : **VÉRIFIÉ**.
- **Fiche proposée** — *Profiler 1.5.1 épinglé et enregistré*.
  - **Porte** : `swift package show-dependencies` affiche `swift-mlx-profiler 1.5.1` ; la trace porte
    `build_configuration` et les révisions ; le coût d'un bord est mesuré ≤ 50 µs (10 000 paires à vide).
  - **Cible : macos-gpu.**

### P-75 · moyenne · `VoxtralTranscriptionTest/ProfileCommand.swift:60-61`, `:99-158`, `:281-303` ; `Realtime/VoxtralRealtimeModel.swift:120-123` — `profile --pipeline realtime` n'est pas un instrument de baseline (complète A-15, STT P-19, TTS P-45)

- **Constat** : l'instrument passe bien par la bibliothèque (piège 33 évité), mais :
  - un seul passage froid, sans amorçage, répétition, refroidissement ni ordre A/B/B/A ;
  - borne de 500 trames (P-64) ;
  - aucune durée d'entrée ni ligne JSON ;
  - bloc « LLM Metrics » imprimé pour le Realtime (`:138-141`) alors que le chemin Realtime n'appelle aucun crochet
    `startPrefill`/`startGeneration` du profiler : durées et jetons à zéro, affichage trompeur ;
  - les statistiques de pas incluent le pseudo-pas de sortie sans forward (`VoxtralRealtimeModel.swift:120-123`,
    « min 0,7 ms »).
- **Correction** : remplacé par P-79. D'ici là : `--max-tokens` dérivé de l'audio, suppression du bloc LLM pour le
  Realtime, pseudo-pas exclu.
- **Risque API** : aucun (CLI). **Effort** : S. **Statut** : **VÉRIFIÉ**.
- **Fiche** : fusionnée dans P-79. **Cible : macos-gpu.**

### P-76 · basse · `Tests/VoxtralCoreTests/TTS/TTSQuantizationCampaignTests.swift:89-109`, `:120` ; `TTS/Pipeline/VoxtralTTSPipeline.swift:322-327`, `:370-372` ; `docs/voice_cloning.md:122-126`, `:141-145` — Les RTF publiés de la campagne (q6 1,47 contre bf16 3,44) ne sont pas des références

- **Constat** :
  - **Binaire XCTest** : `test-without-building` sans `-configuration`, donc Debug par défaut (piège 9, ×1,8 sur le
    coût hôte). La commande `build-for-testing` n'est pas documentée : À VÉRIFIER.
  - **Premier échantillon froid** de chaque modèle (poids paresseux, TTS P-43) inclus dans la moyenne.
  - **Modèles en séquence** A puis B, sans A/B/B/A ni refroidissement : la dérive thermique pèse sur le second.
  - **RTF = `genTime` / `r.duration`**, où `genTime` inclut la génération du porteur « La la la… » alors que
    `duration` est mesurée après sa coupe : RTF surestimé. Le biais est commun aux deux modèles, donc la comparaison
    relative reste possible, pas l'absolue.
  - Aucune empreinte ni pic mémoire (les « 3,5 contre 8 Go » ne sortent pas de ce harnais).
- **À garder** : les répétitions de graines et le juge ASR commun. Le groupe SEED de
  `TTSGenerateStabilityReproTests` est une porte de déterminisme réutilisable. Pour l'enrôlement, qui n'a pas de
  graine (`VoxtralVoiceEnrollment.swift:516-517` utilise `MLXRandom.normal` sans clé), l'instrument doit poser
  `MLXRandom.seed`.
- **Correction** :
  - Le chronométrage passe par P-79 (Release, amorçage, A/B/B/A, JSON).
  - La campagne reste un **harnais qualité**.
  - Documenter la configuration de build des chiffres de `docs/voice_cloning.md`, ou les marquer « en session ».
- **Risque API** : aucun. **Effort** : S. **Statut** : **VÉRIFIÉ** (lecture) ; configuration réelle **À VÉRIFIER**.
- **Fiche** : fusionnée dans P-79. **Cible : cloud** (documentation) + **macos-gpu** (re-mesure).

### P-77 · basse · `Utils/VoxtralMemoryManager.swift:46-49`, `:89-93` ; `Pipeline/VoxtralPipeline.swift:332`, `:408`, `:485` ; `VoxtralModeling.swift:1256-1257`, `:1433-1434` — La bibliothèque remet à zéro le pic MLX : un instrument qui lit `Memory.peakMemory` ne voit pas le pic du run

- **Constat** :
  - `fullCleanup()` (appelé par `VoxtralPipeline.unload()`) et `optimizeIfNeeded` (préréglages avec
    `resetPeakMemory`) appellent `GPU.resetPeakMemory()`.
  - *Vérif. croisée* — site manquant et le plus actif : les boucles de génération STT elles-mêmes
    (`VoxtralModeling.swift:1256-1257`, `:1433-1434`, selon `memConfig.resetPeakMemory`). Or le préréglage par
    défaut `.recommended()` rend `ultra`, `aggressive`, `moderate` ou `light` selon la RAM
    (`Configuration/MemoryOptimizationConfig.swift:77-90`), tous avec `resetPeakMemory: true` (`:44`, `:52`, `:60`,
    `:68`) : le pic est remis à zéro tous les 2 à 16 jetons pendant la génération STT, quelle que soit la machine.
  - Tout pic MLX lu après, ou à travers ces appels, est faux : pic de toute la vie du process, ou pic depuis la
    dernière remise à zéro.
  - Le profiler 1.5 prend le max des échantillons d'`activeMemory`, ce qui le rend robuste à ce reset, mais il
    échantillonne à 16 ms et peut manquer un transitoire plus court (la copie fp32 de P-61 vit moins d'un pas).
- **Correction** :
  - Laisser la remise à zéro du pic à l'**instrument** (début de phase) et la retirer de la bibliothèque. Garder
    `clearCache`.
  - L'instrument P-79 lit `Memory.peakMemory` exact par phase, en plus du footprint échantillonné.
- **Risque API** : **comportemental** (effet de bord retiré de méthodes publiques). *Vérif. croisée* : 0 appel
  direct connu de `VoxtralMemoryManager` chez les consommateurs, mais FluxForge l'atteint indirectement par
  `VoxtralPipeline(.mini3b4bit)` (`unload()` → `fullCleanup()`, génération → préréglage). La recherche de code
  GitHub `peakMemory user:VincentGourbin` ne trouve aucun lecteur dans FluxForge (limite : branches par défaut
  indexées seulement) : pas de cassure attendue, à annoncer au CHANGELOG. **Effort** : S. **Statut** : **VÉRIFIÉ**.
- **Fiche** : fusionnée dans P-79. Porte grep à étendre : aucun `resetPeakMemory` dans `Sources/VoxtralCore`
  (`VoxtralMemoryManager` **et** `VoxtralModeling`). **Cible : macos-gpu** (lecture du pic) ; le retrait et le grep
  sont faisables en cloud.

### P-78 · basse · `docs/zerovoice_benchmark.md:5`, `:10` ; `Realtime/VoxtralRealtimeEncoder.swift:327-342` — Le chemin Realtime sert d'instrument (juge ASR) sans validation

- **Constat** : le benchmark ZeroVoice juge la qualité TTS en retranscrivant avec `realtime-4b-4bit`. Des sorties
  « *(empty)* » y apparaissent (clips de 11,9 et 20,8 s, et ES/DE).
  - Le juge n'a jamais été validé : ni parité avec mlx-audio, ni WER sur un corpus de référence.
  - Il est hors de sa fenêtre au-delà de 15 s (P-62), ce qui concerne le clip de 20,8 s (18,8 s aussi), pas celui de
    11,9 s. *Vérif. croisée* : l'original citait aussi « fp32 (P-60) » ; des activations fp32 coûtent en vitesse et en
    mémoire mais ne dégradent pas l'exactitude : ce n'est pas un motif d'invalidité du juge (retiré).
  - Le code Realtime est inchangé sur le fond depuis la mesure (`1cbf014`, 2026-03-31 ; seuls profiler, beacon et
    `Memory.clearCache` ont été ajoutés depuis) : le juge actuel est celui du benchmark.
  - On ne peut donc pas attribuer ces sorties vides au TTS plutôt qu'au juge.
- **Correction** : valider le juge (WER sur C-court et C-moyen, clip de 20 s) avant tout nouveau benchmark
  ZeroVoice, puis figer sa version (pack + commit) dans le document.
- **Risque API** : aucun. **Effort** : S. **Statut** : **VÉRIFIÉ** (absence de validation) ; cause des sorties vides
  **À MESURER**.
- **Fiche** : fusionnée dans P-79 (le juge est une mesure du corpus). **Cible : macos-gpu.**

### P-79 · haute (prérequis) · (absence) `VoxtralTranscriptionTest/` ; `scan.md` §7 (`BENCHMARKS.md` absent) — Aucun instrument de baseline pour STT, TTS, Realtime et enrôlement : proposition d'instrument minimal

Sans baseline mesurée, aucune fiche perf (P-60…P-72, STT, TTS) ne peut conclure. C'est l'étape 3 imposée par le
plan.

**Fiche proposée — K-P79 *Instrument de baseline minimal* (`VoxtralCLI bench`)**

- **Source** : P-73, P-74, P-75, P-76, P-77, P-78 ; A-15, A-16, A-22 ; STT P-19 ; TTS P-45 ; piège 33 ;
  `measurement.md`.
- **Prérequis** : P-74 (profiler 1.5.1), P-64 (pas de troncature), P-66 (matérialisation), ou une requête
  d'amorçage exclue.
- **Porte** : **A/A** sur la même commande (deux passes après 120 s de refroidissement). Dispersion ≤ 3 % sur le
  total hors chargement et sur le ms/pas médian ; sorties identiques (hash des jetons ou de la forme d'onde, graine
  fixée) ; lignes JSON valides contre le schéma ; refus d'un binaire Debug (`RunEnvironment.isDebugBuild`, ou ligne
  marquée `debug`).
- **Effort** : M. **Risque API** : aucun (cible CLI).
- **Cible** : écriture du squelette et du schéma en **cloud** (`syntax_guard.py`, sans build), porte **macos-gpu**.

**Ce que la commande fait** :

1. **Chemins de la bibliothèque uniquement** (piège 33) :
   - `stt` → `VoxtralPipeline.transcribe`, avec `--backend mlx|auto` ;
   - `tts` → `VoxtralTTSPipeline.synthesize` / `synthesizeStreaming`, voix prédéfinie ou `--voice-embedding` ;
   - `realtime` → `VoxtralRealtimePipeline.transcribe`, et bientôt l'API de streaming (P-71) ;
   - `enroll` → `VoxtralVoiceEnrollment.optimize` avec `MLXRandom.seed`.
2. **Hygiène avant chaque point** :
   - `--cooldown 120` ;
   - processus le plus gourmand noté ;
   - refus si un autre processus MLX tourne (`pgrep`) **ou si un manifeste `RuntimeBeacon` vivant appartient à un
     autre runtime** (`~/Library/Application Support/ai-runtime-beacons/`, schéma partagé avec
     ltx-video-swift-mlx) — piège 8 ;
   - sorties dans `.local-runs/bench.noindex/`.
3. **Déroulé** : `--warmup 1` (exclu) puis `--passes N`. L'ordre A/B/B/A se fait par deux invocations taguées
   (`--tag A|B`). Profiler **désactivé** pendant le chronométrage ; `--trace` produit un run séparé de diagnostic en
   `.fineGrained`.
4. **Une ligne JSON par mesure** (stdout et `bench.jsonl`, recopiée dans `BENCHMARKS.md` sans jamais modifier une
   ligne existante) :
   - **Communs** : `date`, `commit` (+ `dirty`), `build` (Release/Debug), `mlx_swift`, `mlx_swift_lm`,
     `mlx_profiler` (révisions **résolues**), `chip`, `ram_gb`, `macos`, `power`, `top_process`, `pipeline`,
     `model`, `pack_sha256`, `profile`, `input`, `input_s`, `seed`, `pass`, `tag`, `warm`.
   - **Par phase** : `phases_ms` (chargement, matérialisation, audio/mel, encodage, préfill, décodage, codec, post) ;
     `peak_mlx_mb` (exact : `Memory.peakMemory` remis à zéro par l'instrument au début de chaque phase, P-77) ;
     `peak_footprint_mb` (`task_info(TASK_VM_INFO).phys_footprint`, échantillonneur à 5 ms).
   - **Débit** : `steps`, `step_ms_p50`, `step_ms_p90` (pseudo-pas exclus), `ttft_ms`, `rtf`, `weights_bw_gbps` =
     octets de poids lus par pas × pas/s.
   - **Parité** : `out_sha256` (ids de jetons ou forme d'onde quantifiée), `wer` si une référence existe.
   - **Spécifiques** :
     - Realtime : `pad_fraction`, `encode_ms_per_audio_s`, `first_text_token_ms`, `truncated`.
     - TTS : `frames`, `ttfa_ms`.
     - Enrôlement : `epochs`, `epoch_ms_p50`, `final_loss`.
5. **Mode `VOXTRAL_DTYPE_AUDIT=1`** (analogue de `QWEN38_DTYPE_AUDIT`) : imprime le dtype du mel, de la sortie de
   l'encodeur, du cache KV et des logits. C'est la preuve rapide de P-60 et P-01.
6. **Corpus du dépôt** (celui de STT P-19) :
   - C-court : `docs/examples/fluxforge_short_{en,fr}_6bit.wav` (5,0 / 4,8 s) ;
   - C-moyen : `fluxforge_long_{en,fr}_6bit.wav` (167,0 / 173,8 s), références dans `docs/tts_benchmark.md` ;
   - C-long ≈ 12 min par concaténation : dépasse les 5 min 27 s de P-64 et, pour le décodeur, rien de plus ;
     ≈ 15 min pour P-63 ;
   - un enregistrement réel long (ASK Q7).

   Parole synthétique : biais à noter.

**Validation attendue (lignes exactes)** :
```
BUILD SUCCEEDED   (xcodebuild -scheme VoxtralCLI -configuration Release)
BENCH {"pipeline":"realtime","model":"realtime-4b-4bit","build":"Release",...,"pass":1,...}
BENCH {"pipeline":"realtime","model":"realtime-4b-4bit","build":"Release",...,"pass":2,...}
AA dispersion step_ms_p50=…% total_ms=…% out_sha256=identical  → PASS (≤ 3 %)
```

**Pièges à cocher** : 8, 9, 11, 18, 21, 33, 38, 39.

## 6. Protocole de mesure Realtime (pour toutes les fiches P-60…P-72)

- Instrument P-79, binaire Release, `--cooldown 120`, A/B/B/A, seuil de 5 %, un levier par comparaison. Révision de
  mlx-swift-lm résolue sur chaque ligne (dépendance sur `main`).
- **Mesures** :
  - encodage en ms par seconde d'audio ;
  - préfill ;
  - ms/pas p50 et p90 (budget temps réel 80 ms) ;
  - latence jusqu'au premier jeton texte ;
  - pics `phys_footprint` et MLX par phase ;
  - `weights_bw_gbps`.
- **Parité** : le texte greedy identique suffit pour les fiches exactes (P-63 ≤ 8 192 pas, P-65, P-66, P-67). Pour
  les fiches numériques (P-60, P-61, P-62, P-68, P-69), le WER normalisé avec tolérance est décidé par Vincent
  (ASK Q1). La comparaison se fait sur le checkpoint réel (4 bits recommandé), jamais sur des poids aléatoires.
- **Ordre** (skill) :
  1. Stabilité bloquante : P-64, puis P-62 et P-63 (divergences silencieuses).
  2. **Instrument P-79 + P-74 + P-66**, puis baseline.
  3. Leviers par gain attendu : **P-60 → P-61** (le gros morceau ; *Vérif. croisée* : l'essentiel du gain, la copie
     fp32 de la tête, est porté par P-60 ; P-61 se mesure ensuite contre la baseline post-P-60), P-65, P-62 (côté
     perf), P-67, P-68, P-69, puis P-72 (R&D).
  4. P-70 (packs et profils), P-71 (streaming, produit), P-73, P-76, P-78 (documentation, re-diagnostic).

## 7. Conséquences pour les profils Realtime (proposition ; toutes les valeurs À MESURER)

| Profil | Pack | Calcul | Tête | Encodeur | KV | Mémoire | Remarques |
|---|---|---|---|---|---|---|---|
| `4bit-fast` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` (3 133 798 126 o ; SHA-256 à relever) | dtype du modèle (P-60) | 8 bits via `QuantizedEmbedding.asLinear` (P-61) | dé-quantifié si ≥ 5 % (P-69) ; tranches fenêtre 750 (P-62) | `RotatingKVCache(8192)` 16 bits (P-63) | `cacheLimit` encodage 1 Go / décodage 2 Go (P-67) | `asyncEval` (P-65), retard 480 ms |
| `4bit-lean` | idem | dtype du modèle | 4 bits si la parité tient, sinon 8 bits | packé, par tranches | fenêtre 8 192, `kvBits` 8 à qualifier (T10) | limites adaptatives (T2), `clearCache` après transcription, `.encoderOnly` pour l'extraction | cible iOS éventuelle (ASK Q5) |
| `8bit-fast/lean` | aucun pack chargeable aujourd'hui (P-70) : support voxmlx (`ellamind …-8bit-mlx`, 4,71 Go) ou export propre | — | 8 bits | — | — | — | ASK Q4 ; 6 bits (`mlx-community …-6bit`, 3,61 Go) en largeur intermédiaire, comme le TTS q6 |
| `16bit-fast/lean` | original bf16 `mistralai/Voxtral-Mini-4B-Realtime-2602` (format A) plutôt que le pack `fp16` | bf16 | 16 bits (8 bits en option) | bf16 | fenêtre 8 192 | comme en 4 bits | **inutilisable avant P-60** : chaque `Linear` recopie ses poids en fp32 |

## 8. Décisions à prendre (ASK)

1. **Q1 — Tolérance de parité** pour les fiches numériques P-60, P-61, P-62, P-68 et P-69 : texte identique exigé,
   ou WER ≤ référence + x pt (proposé : +0,2 pt en 16 et 8 bits, +0,5 pt pour la tête 4 bits) ?
2. **Q2 — Référence** : accepter une sortie de mlx-audio (Python, Mac) comme référence de parité pour les fenêtres
   (P-62, P-63), puisque la sortie Swift actuelle est justement fausse au-delà de 15 s ? (*Vérif. croisée* :
   mlx-audio `main` implémente déjà les deux fenêtres ; épingler son commit dans la fiche. Il garde en revanche le
   mel fp32 : pour P-60, la référence n'est pas bit-comparable.)
3. **Q3 — Sémantique de `maxTokens`** (P-64) : budget texte, borne supprimée, ou champ déprécié ?
4. **Q4 — Largeur 8 bits** (P-70) : supporter le format voxmlx ou exporter un pack propre au format mlx-audio
   (SHA-256, `Weights.md`) ?
5. **Q5 — iOS** : le Realtime doit-il tourner sur iPhone ? Si oui : T23, profil `lean` obligatoire et pic à
   mesurer contre jetsam.
6. **Q6 — Streaming** (P-71) : objectif produit (FluxForge, démo) ou hors périmètre ?
7. **Q7 — Corpus** : un enregistrement réel long (≥ 12 min, avec référence) peut-il être ajouté au dépôt, ou
   hors dépôt avec un hash ?
8. **Q8 — `VoxtralBenchmark`** (A-16) : le réorienter vers `bench` (P-79) ou retirer le produit (« cassant » pour
   le produit exécutable seulement) ?

## 9. Capitalisation proposée (phase 6, vers le catalogue du skill)

Tout ce qui suit est vérifié en lecture ; les effets sont À MESURER.

- **Piège** — *tête liée par `matmul` brut* : `matmul(h, emb.weight.T)` en activations fp32 recopie toute la table
  du vocabulaire en fp32 à chaque pas. Il contourne aussi `QuantizedEmbedding.asLinear`. Source :
  `Realtime/VoxtralRealtimeDecoder.swift:187-189`, recoupement #25 (+3 980 Mio contre 3 937 Mio de budget,
  compatible mais non probant). La copie disparaît dès que les activations sont dans le dtype de la table (P-60).
- **Piège** — *fenêtre glissante déclarée mais ignorée* (encodeur 750, décodeur 8 192) : divergence silencieuse
  au-delà de la fenêtre, coût quadratique, cache sans borne. Pour le chunking, la RoPE doit porter la position
  absolue et le masque doit être explicite quand un cache est présent.
- **Piège d'instrument** — *phases imbriquées* (swift-mlx-profiler ≤ 1.4) : GPU % englobant à 0 %, temps comptés
  deux fois. *Lecture instantanée* de « Device Utilization % » aux seuls bords d'une phase : moyenne entière de deux
  échantillons, sans valeur de mesure (*Vérif. croisée* : « ≈ 49 % pour ≈ 82 % réels » retiré, cas de 5 ms non
  transposable). « Peak MLX Active » ≤ 1.4 = maximum d'instantanés, pas `Memory.peakMemory`.
- **Piège d'instrument** (*Vérif. croisée*) — *`activeMemory` lue juste après `eval`* : peut encore compter des
  intermédiaires que le gestionnaire de complétion n'a pas rendus ; l'ordre des lectures d'un instantané (mémoire
  avant ou après IOKit) change le résultat.
- **Piège** — *`maxTokens` sur un modèle synchrone à la trame* : il compte des trames audio, d'où une troncature
  silencieuse.
- **Technique de vérification sans Mac** — *réconcilier les deltas mémoire d'une issue avec un budget
  poids + transitoires*. Elle a rendu la copie fp32 plausible à 1,1 % près (*Vérif. croisée* : indice, pas preuve ;
  valider d'abord la sémantique exacte du chiffre relu : métrique, instant de lecture, ordre des lectures).
- **Technique de vérification sans Mac** (*Vérif. croisée*) — *différentiel contre la référence déclarée* : les
  en-têtes des fichiers Realtime citent mlx-audio ; sa version `main` a déjà le chunking à fenêtre 750, le
  `RotatingKVCache(8192)`, l'`async_eval` double tampon, `as_linear` et la tête quantifiée. Le différentiel aurait
  trouvé P-61 (étape 2), P-62, P-63, P-65 et P-68 en une lecture, et donne la référence de parité de l'ASK Q2.
- **Technique candidate** — *RoPE entrelacée manuelle → `RoPE(traditional: true)` fusionnée* (déjà faite par la
  référence mlx-audio) ; *pas spéculatifs « remplissage »* pour les décodeurs ASR synchrones à la trame (R&D).

## Annexe — Constats écartés à la vérification croisée

**Aucun constat écarté.** Les 20 mécanismes décrits se retrouvent dans le code à `9392ed1`, aux lignes citées, sur des
chemins vivants (`VoxtralRealtimePipeline.transcribe` → `VoxtralRealtimeModel.generate`, commande CLI `realtime` et
`profile --pipeline realtime`). La référence déclarée (mlx-audio `main`) corrobore P-61 (étape 2), P-62, P-63, P-65
et P-68.

### Constats amendés

| Id | Amendement | Raison |
|---|---|---|
| P-60 | Source du `computeDType` changée (tenseur jamais quantifiable) ; porte « pack 4 bits : ms/pas ≥ −30 %, pic −≥ 1 Go » ajoutée ; risque de parité (mlx-audio garde le mel fp32) | La correction d'origine prenait `tokEmbeddings.weight`, que P-61 quantifie (→ `uint32`) ; P-60 supprime seul la copie fp32 de la tête, donc son gain principal |
| P-61 | Porte rebasée en deux étapes (étape 1 sans objet après P-60 ; étape 2 : ms/pas ≥ −8 %, active −≥ 300 Mo contre la baseline post-P-60) ; citation MLXNN @0.31.6 (`Quantized.swift:213-217`) ; recoupement #25 « compatible, non probant » | L'ancienne porte (−30 %, −1 Go) mesurait P-60 et aurait fait échouer la tête quantifiée à tort ; lignes lues sur `9019419` au lieu de la version résolue ; la valeur #25 est un instantané `activeMemory` pris juste après `eval` |
| P-62 | Correction complétée (RoPE à position absolue, masque explicite sinon `.none` avec cache, `:153-158`, `:331-334`) ; référence mlx-audio citée ; ×1,03 → ×1,05 à 40 s | Sans ces deux points, l'encodage par tranches serait faux ; coquille de calcul |
| P-63 | 4,6 / 9,1 Gio → 4,5 / 8,9 Gio ; référence mlx-audio citée | Confusion Go/Gio |
| P-64 | « 501 = 500 + 1 » et « min 0,7 ms » passés de certains à très probables ; sémantique héritée de mlx-audio notée | La trace n'est pas disponible et un EOS au 501ᵉ pas donne le même compte |
| P-66 | +673 Mio ventilés : 576 Mio de poids, ≈ 97 Mio non ventilés | L'égalité « +673 ≈ 518 + 48 + 10 » était fausse de 17 % |
| P-70 | Titre : l'original bf16 est chargeable ; tout le modèle voxmlx (décodeur `language_model.*` compris) resterait aléatoire, sans plantage ; repli silencieux d'un `modelId` inconnu (`VoxtralRealtimePipeline.swift:78-79`) | Index HF relu en entier ; registre relu |
| P-73 | Point 2 (« 49 % = ≈ 82 % réels ») remplacé par « pas une mesure » + hypothèse (0 + 98)/2 ; points 5 et 6 nuancés ; point 7 ajouté (« Peak MLX Active » = maximum d'instantanés) | Généralisation d'un cas documenté sur une phase de 5 ms ; recoupement mémoire non probant |
| P-77 | Sites `VoxtralModeling.swift:1256-1257`, `:1433-1434` ajoutés (remise à zéro tous les 2-16 jetons avec tout préréglage `.recommended()`) ; risque requalifié « comportemental », atteint FluxForge par `VoxtralPipeline` | Le site le plus actif manquait ; la porte grep l'aurait laissé passer |
| P-78 | « fp32 » retiré des motifs d'invalidité du juge ; fenêtre > 15 s limitée aux clips concernés ; code du juge inchangé depuis la mesure | Des activations fp32 ne dégradent pas l'exactitude |

### Constats gardés tels quels (motif de la vérification)

- **P-65** : `MLX.eval(logits)` (`VoxtralRealtimeModel.swift:133`) puis `argMax(…).item()` (`:161`) ; 0 `asyncEval`
  dans `Realtime/` ; la référence utilise `async_eval` en double tampon (`voxtral_realtime.py:268`, `:296`).
- **P-67** : 0 `cacheLimit` dans `Realtime/` ; `unload()` (`VoxtralRealtimePipeline.swift:181-186`) ne vide rien, STT
  appelle `fullCleanup()` (`VoxtralPipeline.swift:485`).
- **P-68** : `interleavedRoPE` = paires consécutives, fréquences `θ^(-2i/d)` (`VoxtralRealtimeEncoder.swift:55-83`) ;
  la référence fait `mx.fast.rope(traditional=True)` ; helpers `internal`, utilisés seulement dans ce fichier.
- **P-69** : levier À MESURER, correctement étiqueté ; coût +1,4 Go recalculé.
- **P-71** : aucune API de flux (`VoxtralRealtimeManager` : `transcribe`, `extractEmbeddings` par URL).
- **P-72** : R&D À MESURER, porte en deux étapes avec abandon ; `KVCacheSimple.trim` existe.
- **P-74** : `Package.resolved` ignoré par git ; `ProfileCommand` n'utilise que les 4 paramètres communs à 1.4.0 et
  1.5.1 (les deux compilent, même plateforme macOS 15) ; défaut `gpuBackend: .deviceUtilization`
  (`ProfilingConfig.swift:75`), `.fineGrained` en `.ioReportResidency` (`:115-124`).
- **P-75** : `LLM Metrics` imprimé pour le Realtime (`ProfileCommand.swift:137-141`) ; `startPrefill`/`startGeneration`
  appelés seulement par `VoxtralModeling.swift:1168-1231`, `:1351-1411`.
- **P-76** : `genTime` inclut le porteur (`VoxtralTTSPipeline.swift:322-327`), `duration` est calculée après la coupe
  (`:385`) ; modèles en séquence (`TTSQuantizationCampaignTests.swift:89-109`).
- **P-79** : `BENCHMARKS.md` absent ; `VoxtralTranscriptionTest/` = `ProfileCommand.swift` + `VoxtralCLI.swift` ;
  corpus relu (5,0 / 4,8 / 167,0 / 173,8 s).
