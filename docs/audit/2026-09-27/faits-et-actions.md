# Faits vérifiés, inventaire des actions et capitalisation — mlx-voxtral-swift

> Skill `mlx-swift-audit` : phase 0 (cadrage), faits pour `PLAN.md` §1 et la baseline, inventaire exhaustif des
> actions Voxtral (« ne perdre aucune action »), matière de la phase 6 (capitalisation).
> Révision auditée : `9392ed1` (= tag `v2.2.2`, branche `claude/action-plan-skills-beta-wifgmu`). Date : 2026-09-27.
> Constats de ce rapport : `FA-xx` ; faits : `FV-xx` ; actions : `ACT-xx` ; capitalisation : `V-Tx` (retenues),
> `V-Rx` (rejetées ou réfutées), `V-Px` (pièges). Renvois aux rapports frères : `S-xx` (audit-stabilite.md),
> `A-xx` (audit-annexes-serveur.md), `P-01…29` (audit-performance-stt.md), `P-30…49` (audit-performance-tts.md),
> `P-60…79` (audit-performance-realtime-instruments.md).

## 0. Cadre et méthode

- **Environnement** : session cloud Linux, 4 CPU, sans Mac, sans toolchain Swift, sans GPU. Aucun build, aucun test,
  aucune mesure. `machine-check.sh` rend ici `NA Linux/x86_64 : pas un Mac Apple Silicon` (code 0). Tout chiffre
  ci-dessous est **copié tel quel** d'une source du dépôt ou de GitHub, avec sa machine, sa révision et sa source ;
  rien n'est « obtenu » dans cette session.
- **Sources lues** : `README.md`, `llms.txt`, `docs/*.md`, `Scripts/*/README.md`, `Package.swift`, `Tests/`, le
  `git log` complet (174 commits, messages intégraux des 60 commits de perf, correctifs et docs), les issues #1-#45
  (corps + commentaires, MCP GitHub en lecture), les PR #3-#50 (corps), les 15 plans `project:mlx-voxtral-swift`
  du dépôt `VincentGourbin/action-plans` (corps + commentaires), `knowledge/` d'action-plans, le code à `9392ed1`.
  Amont : `git ls-remote --tags` de `ml-explore/mlx-swift-lm` et `ml-explore/mlx-swift` (le MCP GitHub refuse ces
  dépôts : accès limité à `vincentgourbin/{action-plans,mlx-voxtral-swift,claude-skills}`). Hub HF : MCP
  Hugging Face en lecture.
- **Règle appliquée** : un fait = une source vérifiable (fichier:ligne, commit, issue/PR, commande). Une affirmation
  d'issue ou de commit **contredite par le code** est signalée comme telle (FA-03, FV-34, FV-54).
- **Dépôt sans `CLAUDE.md`, `AGENTS.md`, `BENCHMARKS.md` ni `docs/knowledge/`** (`scan.md` §7) : la « mémoire du
  projet » exigée par la phase 0 n'existe pas ; elle est reconstituée ici à partir des messages de commit, des
  issues et des PR (voir `skill_feedback`).

---

## 1. Cadrage (phase 0)

### 1.1 Documents et leur état à `9392ed1`

| Document | Dernière mise à jour | État |
|---|---|---|
| `README.md` | `27a04fa` (2026-07-16, merge RuntimeBeacon) ; tables STT de `e376f05` (2026-01-30), TTS de `1bb54bf` (2026-04-02) | Tables STT antérieures à toutes les optimisations d'avril (FV-10) ; exigences macOS 14 / Swift 6.0 / Xcode 15 fausses (S-19) ; description du gate de référence périmée (FA-05) ; pas de Realtime dans les Features (S-20) |
| `llms.txt` | `b355644` (2026-02-01, « v1.0.10 ») | Figé en 1.0.10 alors que le tag est v2.2.2 ; ni TTS, ni clonage, ni Realtime (S-20) ; publié aux assistants par `context7.json` |
| `docs/tts_benchmark.md` | `6ad4e56` (2026-04-02) | Mesures antérieures à `a00024f`, `0be05af`, `f4fd21c` (FV-30) |
| `docs/zerovoice_benchmark.md` | `1cbf014` (2026-03-31) | Juge ASR = Realtime 4 bits non validé (P-78) |
| `docs/voice_cloning.md` | `bff9788` (2026-07-27) | À jour du code (PR #46, #48) ; un chiffre à requalifier (FV-54) |
| `docs/streaming_demo.md` | `db34ca0` (2026-07-26) | À jour ; sa métrique « TTFT » ≈ génération complète à cause de S-08 (FA-04) |
| `Scripts/CoreMLConversion/README.md` | `1944576` (2026-01-06) | Non reproductible, affirmations fausses (A-03, A-12) |
| `Scripts/VoiceCloningResearch/README.md` | `ca49c7a` (2026-07-09) | « Next step: Swift/MLX port » déjà fait dans `bd59931` (FA-06) |
| `Package.swift` | `9392ed1` | `mlx-swift-lm` sur `branch: "main"` (l. 46-52) ; tools 6.2, macOS 15 / iOS 17 (l. 1, 10-13) |
| `Tests/VoxtralCoreTests` | — | 487 fonctions `test…` (grep) ; « 483 tests, 0 failures » au commit `a7045f5` ; 11 harnais lourds sous variable d'environnement (`VOXTRAL_TTS_REPRO`, `…_CAMPAIGN`, `…_COMPARE`, `…_PROBE`, `…_DEFICIT`, `…_TWOPASS`, `…_STREAM_SEED`, `VOXTRAL_ENROLL_*`) ; aucune CI (`.github/` absent) |

### 1.2 Points d'entrée (tous, pas seulement l'inférence principale)

| Famille | Point d'entrée | `fichier:ligne` (9392ed1) | Couvert par |
|---|---|---|---|
| STT transcription | `VoxtralPipeline.transcribe(audio:language:)`, `quickTranscribe` | `Pipeline/VoxtralPipeline.swift:316`, `:542` | P-01…P-29, S-01…S-07 |
| STT chat (Q/R sur l'audio) | `VoxtralPipeline.chat(audio:prompt:language:)` | `Pipeline/VoxtralPipeline.swift:393` | P-15, P-26 |
| STT façade | `VoxtralTranscriptionManager` (`@MainActor`) | `Pipeline/VoxtralTranscriptionManager.swift:50-52`, `:90-153` | S-22 |
| STT backends | `.mlx`, `.hybrid` (Core ML + MLX), `.auto` (défaut) | `Pipeline/VoxtralPipeline.swift:73-85`, `CoreML/VoxtralHybridEncoder.swift:63` | A-12, A-13, P-14 |
| STT legacy | `VoxtralGenerator`, deux `loadVoxtralModel` homonymes, `loadVoxtralModelWithMLXLM`, souche `downloadModel` (publics, 0 consommateur) | `VoxtralGenerator.swift:69` ; `Utils/VoxtralModelLoading.swift:15`, `:139` ; `Utils/VoxtralPythonCompatLoader.swift:16` ; `Utils/VoxtralMLXLMLoader.swift:16` | S-13, S-14 |
| Realtime | `VoxtralRealtimePipeline.loadModel/transcribe`, `VoxtralRealtimeManager` | `Realtime/Pipeline/VoxtralRealtimePipeline.swift:68`, `:118` ; `…Manager.swift:21` | P-60…P-73 |
| TTS batch | `VoxtralTTSPipeline.synthesize(text:voice:seed:)`, `(text:voiceEmbedding:seed:warmUpText:warmUpLeadInFrames:)`, `(text:voiceCoordinate:)` (ZeroVoice), `synthesizeToFile`, `blendVoicePresets` | `TTS/Pipeline/VoxtralTTSPipeline.swift:190`, `:302`, `:274`, `:261`, `:455` | P-30…P-49 |
| TTS streaming | `synthesizeStreaming(text:voice:…)`, `(text:voiceEmbedding:…voiceKey:…)` | `TTS/Pipeline/VoxtralTTSPipeline.swift:472`, `:497` | S-08, P-33 |
| TTS façade | `VoxtralTTSSynthesisManager` | `TTS/Pipeline/VoxtralTTSSynthesisManager.swift:20`, `:52-104` | P-34 |
| Enrôlement (clonage, entraînement par vjp) | `VoxtralTTSPipeline.enrollVoice`, `VoxtralVoiceEnrollment.optimize` (deux surcharges) | `TTS/Pipeline/VoxtralTTSPipeline.swift:426` ; `TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:26`, `:464`, `:483` | A-01, A-06…A-11 |
| Téléchargement | `ModelDownloader.downloadRepoDirect`, `resolveModel`, `downloadTTSModel`, `downloadRealtimeModel` ; Core ML `VoxtralCoreMLEncoder.downloadFromHuggingFace` | `Utils/ModelDownloader.swift:79`, `:401`, `:584`, `:695` ; `CoreML/VoxtralCoreMLEncoder.swift:444` | S-03, S-07, A-02, A-04, A-05 |
| Conversion Core ML (Python) | `Scripts/CoreMLConversion/convert.sh`, `convert_to_coreml_ane.py` | — | A-03 |
| Recherche clonage (Python) | `Scripts/VoiceCloningResearch/enroll_voice.py` | — | A-21, FA-06 |
| Observabilité | `RuntimeBeacon` (opt-in), `MLXProfiler` | `Utils/RuntimeBeacon.swift:51` | A-20 |
| CLI `voxtral` (produit `VoxtralCLI`) | `list`, `download`, `transcribe` (défaut), `chat`, `tts`, `enroll`, `realtime`, `profile run` | `VoxtralTranscriptionTest/VoxtralCLI.swift:21-38`, `ProfileCommand.swift:12-36` | A-15, P-45, P-75 |
| Bench | `VoxtralBenchmark` (« Benchmark Float16 conversion ») | `VoxtralBenchmark/BenchmarkCLI.swift:17-20` | A-16 |
| Apps | `VoxtralApp` (STT, macOS), `VoxtralTTSStreamingDemo` (TTS + clonage : fichier, extrait vidéo ffmpeg, micro) | `Sources/VoxtralApp/`, `Sources/VoxtralTTSStreamingDemo/` | A-14, A-18, A-19 |
| Serveur d'inférence | **absent** | — | A-23 (conditionnel) |

### 1.2 bis Couverture de chaque point d'entrée par les phases 2 à 4 (critique de complétude du 2026-09-27)

Règle du skill (phase 0) : chaque point d'entrée est couvert par la stabilité, les annexes et la performance, puis
par les profils et le plan. « N/A » est justifié ; un point d'entrée qui n'est qu'un **enrobage** d'un autre hérite de
ses constats (fonction appelée citée).

| Point d'entrée | Stabilité (S-xx) | Annexes (A-xx) | Performance (P-xx, T1…T23) | Profil / baseline | Fiches |
|---|---|---|---|---|---|
| STT transcription (`transcribe`) | S-01, S-02, S-04, S-05, S-09…S-12 | A-22 (éval WER) | P-01…P-29 ; colonne STT du §2 d'`audit-performance.md` | profils.md §2-§3 ; K-34 | K-1…K-5, K-40, K-45, K-51…K-56, K-62, K-63 |
| `quickTranscribe` | enrobage de `loadModel` + `transcribe` (`VoxtralPipeline.swift:542-547`) : hérite ; course sur `isReady` couverte par S-10 | — (enrobage) | hérite de STT | hérite | K-11 (machine d'états) |
| STT chat (`chat`) | S-01 (réponse tronquée sur « ␣Capital »), S-02 (même `contextSize`), S-09, S-22 (`VoxtralTranscriptionManager.chat` lève toujours) | A-15 (`profile` chat, backend non figé) | P-15, P-26 ; `audit-performance.md` §2.0 bis | **baseline chat ajoutée à K-34** (le 2026-09-27 : elle manquait) | K-49, K-70, K-27 |
| Façade `VoxtralTranscriptionManager` | S-10, S-22 | N/A (inférence principale) | hérite de STT ; blocage du MainActor → K-15 | hérite | K-15, K-27 |
| Backends `.mlx` / `.hybrid` / `.auto` | S-26 (ressource `.mlmodelc`) | A-04, A-12, A-13 | P-14, M-07 ; §2.0 bis (hybride) | champ « backend » (profils.md §2.2) ; K-34 (`.mlx` et `.auto`) | K-25, K-42 |
| STT legacy (`VoxtralGenerator`, `loadVoxtralModel`…) | S-13, S-14, S-16, S-17 | A-05 (souche `downloadModel`) | P-17, P-24 | N/A (0 consommateur, déprécié) | K-3 (masque hérité), K-23, K-30, K-74, K-75 |
| Realtime (`VoxtralRealtimePipeline`, `VoxtralRealtimeManager`) | S-04, S-09, S-10 | A-02 (complétude `config.json`) ; M-01, M-05 | P-60…P-72 | profils.md §4 ; K-36 | K-5, K-9, K-13, K-38, K-46, K-47, K-60, K-65, K-73, K-78 |
| TTS batch (préréglage, embedding, ZeroVoice, mélange, `synthesizeToFile`) | S-04, S-09, S-10 ; ZeroVoice et `synthesizeToFile` sont des enrobages de `synthesize` (`VoxtralTTSPipeline.swift:261-282`) ; `blendVoicePresets` (`:454-458`) ne fait que produire un embedding (SLERP, voir patterns-verdicts.md §1.1) | A-02 (`params.json`) | P-30…P-49 | profils.md §5 ; K-35 | K-14, K-39, K-41, K-44, K-48, K-50, K-57, K-58, K-66…K-68, K-71 |
| TTS streaming | S-08 (MLX-003, MLX-019) | A-19 (démo) | P-33, P-40 ; §2.0 bis | bouton `chunkSize` (profils.md §5) ; K-35 (`streaming`) | K-12, K-43, K-44, K-50 |
| Façade `VoxtralTTSSynthesisManager` | S-10 | N/A (inférence principale) | P-34 (aucun choix de modèle) | K-76 (`loadModel(modelInfo:)`) | K-76, K-79 |
| Enrôlement (`enrollVoice`, `optimize`) | S-10 (enrôlement non exclusif) | A-01, A-06…A-11 | colonne Enrôlement du §2 (A-06, A-10) | `enroll-fast\|lean` (profils.md §6) ; K-37 | K-11, K-26, K-37, K-64 |
| Téléchargement (`ModelDownloader`, encodeur Core ML) | S-03, S-06, S-07 | A-02, A-04, A-05 | octets ×2 (S-07, M-03) ; T22 N/A ; débit non audité (réseau, hors MLX) | `weights:` de chaque profil ; `Weights.md` | K-6, K-10, K-24, K-25, K-81, K-82 |
| Conversion Core ML (Python) | S-26 | A-03 | produit évalué par P-14 / A-12 (le script lui-même : N/A, hors ligne) | — | K-19, K-42 |
| Recherche clonage (Python) | N/A (hors bibliothèque) | A-21, FA-06 | N/A (recherche, non exécutée par les hôtes) | — | K-19 |
| Observabilité (`RuntimeBeacon`, `MLXProfiler`) | aucun constat : `RuntimeBeacon` verrouillé (audit-stabilite.md §5, indices écartés) | A-20 | P-73 (profileur : phases imbriquées) ; surcoût du beacon nul désactivé (audit-performance-realtime-instruments.md §5) | — | K-28, K-32, K-36 |
| CLI `voxtral` | hérite des pipelines ; S-23 (`print` de la bibliothèque sur sa sortie) | A-15 | P-45, P-75 | CLI `references` / `--reference` (K-76) | K-23, K-32, K-76 |
| Bench `VoxtralBenchmark` | S-26 (bench mal nommé) | A-16 | P-79 (piège 33) | — | K-32 (ASK-26) |
| Apps `VoxtralApp`, `VoxtralTTSStreamingDemo` | S-02 (fenêtre 8 192 imposée par l'app), S-26 | A-14, A-17, A-18, A-19 | P-09 (pose `cacheLimit` morte), FA-03 (démo en 4 bits) | défauts à aligner (K-79) | K-15, K-23, K-28, K-79 |
| Éval / qualité | S-27 (tests) | A-22 | P-76, P-78 | K-33 (outil de toutes les portes WER / couverture) | K-29, K-33 |
| Serveur d'inférence | N/A (absent) | A-23 (prérequis si retenu) | N/A (absent) | N/A | ASK-1 ; PLAN.md §6 |
| Export de packs | N/A : absent du dépôt (packs produits hors dépôt par mlx-audio / noScribe, Python) | N/A | T13 (quantification mixte par voie) → PK-1…PK-4 | `weights:` des profils | K-80 |
| Entraînement LoRA, drafter, spéculatif | N/A : absents (l'enrôlement est le seul entraînement) | N/A | drafter externe `jburtoft/…-draft-4layer` hors plan ; pas spéculatifs Realtime (P-72) | champ « spéculatif » : « — » | K-73 ; PLAN.md §6 |

Bilan : aucun point d'entrée sans couverture ; un trou **corrigé** (baseline du chat, absente de K-32/K-34 alors que
K-49 et K-70 mesurent ce chemin).

### 1.3 Modèles et quantisations supportés (registres)

| Registre | Id → dépôt HF | Défaut / recommandé |
|---|---|---|
| STT `ModelRegistry.swift:46-107` | `mini-3b` → `mistralai/Voxtral-Mini-3B-2507` ; `mini-3b-8bit` → `mzbac/voxtral-mini-3b-8bit` ; `mini-3b-4bit` → `mzbac/voxtral-mini-3b-4bit-mixed` ; `small-24b` → `mistralai/Voxtral-Small-24B-2507` ; `small-24b-8bit` → `VincentGOURBIN/voxtral-small-8bit` (l'enum `VoxtralPipeline.Model` pointe vers `mzbac/Voxtral-Small-24B-2507-8bit`, `Pipeline/VoxtralPipeline.swift:47-48` : S-06) ; `small-4bit` → `VincentGOURBIN/voxtral-small-4bit-mixed` | `mini-3b-8bit` (`:76`) |
| Encodeur Core ML `VoxtralCoreMLEncoder.swift:52-54` | variantes `mini` (sortie 3072) et `small` (5120) | `.mini` |
| TTS `VoxtralTTSRegistry.swift:28-66` | `tts-4b-mlx` → `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` ; `tts-4b` → `mistralai/Voxtral-4B-TTS-2603` ; `tts-4b-4bit` → `…-mlx-4bit` ; `tts-4b-6bit` → `…-mlx-6bit` | **`tts-4b-mlx` (bf16)** (`:37`) ; démo : `tts-4b-4bit` (FA-03) |
| Realtime `VoxtralRealtimeRegistry.swift:28-57` | `realtime-4b-4bit` → `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` ; `realtime-4b-fp16` → `…-2602-fp16` ; `realtime-4b` → `mistralai/Voxtral-Mini-4B-Realtime-2602` | `realtime-4b-4bit` (`:37`) |
| Voix | 20 préréglages (`TTS/VoxtralVoicePresets.swift`), ZeroVoice SLERP (`maxBlendWeight = 0.05`, `TTS/VoxtralZeroVoice.swift:70`), voix enrôlées `[T+1, 3072]` | — |

**État de l'art au 2026-09-27 (Hub HF, MCP)** : les dépôts `mistralai` Voxtral sont toujours `Mini-3B-2507`,
`Small-24B-2507`, `Mini-4B-Realtime-2602` et `4B-TTS-2603` ; **aucun checkpoint plus récent**. `mlx-community`
publie TTS 4/6 bits/bf16 (2026-03-27), Realtime `2602-4bit`, `2602-fp16` et `Realtime-6bit` (ce dernier absent du
registre, P-70), `Mini-3B-2507-bf16`. Le dépôt TTS ne contient que `consolidated.safetensors`, `params.json`,
`tekken.json` et 20 `voice_embedding/*.pt` : **aucun encodeur de codec publié** (prémisse du clonage toujours
vraie). Les modèles « optimisés à septembre 2026 » passent donc par les **packs et réglages** (profils, phase 3), pas
par un nouveau checkpoint.

### 1.4 Consommateurs connus de l'API

- **FluxForge Studio** (App Store, `fluxforge-studio-swift`) et **SongAnalysisDb** importent `VoxtralCore` ;
  symboles consommés : `VoxtralPipeline(.mini3b4bit)`, `ModelRegistry`, `ModelDownloader.customModelsDirectory`,
  `RuntimeBeacon.isEnabled`, `VoxtralTTSPipeline` dont `recommendedWarmUpVocalise` ; 0 usage de
  `synthesizeStreaming`, `VoxtralGenerator`, `loadVoxtralModel`, `TekkenTokenizer` (repris de audit-stabilite.md §0,
  recherche de code GitHub ; non re-vérifiable ici, le MCP n'a pas accès à ces dépôts).
- **Chaîne LipDub / LTX** (`ltx-video-swift-mlx`) : consomme les **formes d'onde** des voix enrôlées
  (`AudioPreprocessor.detectSpeechWindow`, commentaire #45 du 2026-07-27) ; sensible au niveau, aux silences
  numériques et au rythme (FV-43, FV-56).
- **SiliconScope** : lit les manifestes `RuntimeBeacon` (schéma partagé avec LTX, PR #38).
- Demandes d'intégration tracées côté FluxForge : `docs/FRAMEWORK_ASKS_VOICE_LIPDUB.md` (A1-A6), `FRAMEWORK_ASKS_STORAGE.md`
  (#5-#8), ask « `LanguageModel.prepare` a changé » (PR #50).
- Toute suppression ou tout renommage public reste **cassant → ASK** ; tout changement de défaut (modèle TTS,
  backend) est un changement de **comportement** → ASK.

### 1.5 Question groupée (ASK de cadrage)

1. **Profils** : quels modèles viser pour la matrice `<bits>bit-fast|lean` — STT Mini 3B seul ou aussi Small 24B ;
   TTS en 4/6 bits/bf16 (le standard ne prévoit pas le 6 bits, voir `skill_feedback`) ; Realtime 4 bits seul ou
   4/6 bits/fp16 ?
2. **iOS** : cible réelle ou seulement « compile » (PR #32 : compilation seule, jamais exécuté sur appareil) ?
3. **Défaut TTS** : garder bf16 (référence de qualité) ou passer au 6 bits (FA-03, P-34) ?
4. **Hybride Core ML** : garder `.auto` par défaut malgré un gain publié de −4,3 % (FV-12, A-12) ?
5. **Serveur** : hors périmètre (A-23) ?

---

## 2. Faits vérifiés (pour `PLAN.md` §1 et la baseline)

### 2.1 Règles de lecture indispensables (sinon les chiffres se contredisent)

| Règle | Preuve |
|---|---|
| **Deux conventions de RTF opposées** coexistent. `TTSSynthesisResult.realTimeFactor` = génération / audio (< 1 = plus vite que le temps réel), utilisé par la CLI, `docs/tts_benchmark.md`, `README.md:100`, la campagne q6. `profile` affiche « RT factor » = audio / génération (> 1 = plus vite), utilisé par les commits `a00024f`, `1eb2cc9` et les issues #26/#27. « RTF 1.12x » (6 bits, FR long, doc) et « RT factor 1.12x » (4 bits, #26) disent l'inverse l'un de l'autre. | `TTS/VoxtralTTSProcessor.swift:30-33` ; `VoxtralTranscriptionTest/ProfileCommand.swift:274` ; `Tests/…/TTSQuantizationCampaignTests.swift:120` |
| **TTFT TTS publié = premier frame de codes interne**, pas le premier audio disponible : `ttft` est pris au premier `eval(codes)` de `generate` ; il exclut le calcul du préfixe de voix (P-45). | `TTS/VoxtralTTSModeling.swift:494-497` ; `VoxtralCLI.swift:555` ; `TTS/Pipeline/VoxtralTTSPipeline.swift:210-225` |
| **Le streaming TTS ne streame pas** : toute la boucle s'exécute dans la closure de construction synchrone d'`AsyncThrowingStream` ; le « TTFT » de la démo (`chunk.elapsed` du premier chunk) ≈ génération complète. L'optimisation « premier chunk de 3 frames » (`0be05af`) est sans effet côté consommateur. | `TTS/VoxtralTTSModeling.swift:580-683`, `:617` ; `StreamingDemoViewModel.swift:517-520` ; S-08 |
| **« tok/s » STT du README = jetons / temps total** (encodage et préfill inclus ; inclusion du chargement non précisée), 500 jetons plafonnés ; les issues d'avril donnent le débit de **décodage** (30,6 tok/s en 8 bits contre 14,5 au README). | `e376f05` (README) : 500 / 34,6 s = 14,5 ; 500 / 90,1 s = 5,55 ; issue #15 |
| **Aucune mesure publiée n'est A/B/B/A avec refroidissement** ; plusieurs sont des premiers passages froids (poids paresseux : P-04, P-43, P-66) ; la configuration de build (Debug/Release) n'est notée nulle part hors README. Toutes les valeurs ci-dessous sont **« en session »** au sens de `references/measurement.md`. | FV-* ; P-19, P-45, P-76, P-79 |
| **Machine** : « M3 Max 96 Go » quand elle est notée (README, `docs/*benchmark.md`, `1944576`) ; les issues #12-#29 et les commits de juillet ne la notent pas. **Révision de dépendance** jamais notée (piège 21). | — |
| **Graine** : le TTS n'est déterministe qu'avec `seed:` (depuis `07e6317`, 2026-07-19, batch ; `d7a414e`, 2026-07-24, streaming). Toute « parité A/B » TTS antérieure (dont `0be05af`, `f4fd21c`) est **à l'écoute**, pas bit-exacte. | `TTS/VoxtralTTSModeling.swift:440`, `:582` |

### 2.2 Contexte et amont

| # | Fait | Preuve |
|---|---|---|
| FV-01 | `9392ed1` = tag `v2.2.2` ; `v2.2.1` = `1570294`, `v2.2.0` = `f5842ec` (merge PR #37), `v2.1.0` = `744048b`, `v2.0.0` = `54976d9`, `v1.0.10` = `1c6d59c`. 174 commits. **0 issue ouverte, 0 PR ouverte**, seule branche distante : `main`. | MCP `list_tags`, `list_issues`, `list_pull_requests` ; `git ls-remote --heads origin` (2026-09-27) |
| FV-02 | Dernier tag `mlx-swift-lm` = **3.31.4** ; `main@ee673d6` (2026-09-22) ; dernier tag `mlx-swift` = 0.31.6. Voxtral : `mlx-swift from: 0.31.6`, `mlx-swift-lm branch: "main"`, `Package.resolved` ignoré. Le commentaire « Revisit once ml-explore cuts a tag beyond 3.31.4 » reste d'actualité. | `git ls-remote --tags` (2026-09-27) ; `Package.swift:43`, `:46-52` ; `.gitignore:27` |
| FV-03 | Casse #50 : mlx-swift-lm `main` a changé `LanguageModel.prepare` en `prepare(_:cache:state:prefill:)` ; l'ancienne signature `windowSize:` est dépréciée côté amont. | `9392ed1` ; mlx-swift-lm `Libraries/MLXLMCommon/LanguageModel.swift:340-343`, `:381-390` |
| FV-04 | Aucun nouveau checkpoint Voxtral depuis `4B-TTS-2603` ; pas d'encodeur de codec publié. | §1.3 (MCP HF) |

### 2.3 STT (Mini 3B, Small 24B)

| # | Mesure (copiée telle quelle) | Machine · révision | Source | Statut |
|---|---|---|---|---|
| FV-10 | Mini 3B transcription, 500 jetons, ~8,5 min d'audio, hybride : fp16 90,1 s · 5,6 tok/s · pic GPU 15,26 Go ; 8 bits 34,6 s · 14,5 tok/s · 10,05 Go ; 4 bits mixte 28,2 s · 17,7 tok/s · 8,31 Go. Chat Mini : 2,2 / 3,6 / 3,4 tok/s ; chat Small : fp16 0,54 · 8 bits 0,74 · 4 bits 1,00 tok/s, pics 55,56 / 30,96 / 20,55 Go | M3 Max 96 Go · `e376f05` (2026-01-30) | `README.md:205-217`, `:250-256` ; diff README de `e376f05` | en session ; antérieur aux optimisations d'avril ; **à requalifier** |
| FV-11 | Python 131 s ; Swift MLX 85,6 s (pic 10,79 Go, final 4,66 Go) ; hybride 81,9 s (pic 10,20 Go, final 4,00 Go) — podcast ~7 min, 8 000 jetons max | M3 Max · `1944576` (2026-01-06) | message de commit | **seule comparaison hybride/MLX publiée : −4,3 %, sous le seuil de 5 %** (A-12) |
| FV-12 | Profil mini-3b-8bit (STT) : extraction audio 3,79 s à 0 % GPU ; préfill 3,59 s à 49 % GPU ; pic MLX 480 → 6 116 Mo ; empreinte 11 Go (STT) / 9,5 Go (chat) ; décodage 30,6 tok/s STT contre 17,6 chat, pas moyen 38 contre 85,8 ms ; « Encoder Setup » froid 1 min 09,6 s, chaud 1,41 s | non notée (96 Go implicite, commentaire #13) · branche `feat/mlx-profiler-integration` ≈ `d1e2d5d` (2026-04-11) | issues #12, #13, #15, #16, #17, #18 | en session, passage froid (P-04) |
| FV-13 | small-4bit : préfill 19,13 s (49 % GPU), pic MLX 473 → 15 575 Mo, empreinte 22,0 Go STT / 20,0 Go chat, après nettoyage 13,3 Go ; STT 11,1 tok/s à 92 % GPU ; chat 8,8 tok/s ; « Encoder Setup » froid 2 min 25,2 s, chaud 2,97 s | idem | issues #19, #20, #21, #22 | en session |
| FV-14 | Chargement audio : 3,79 s → 254 ms (MP3 de 203 s, 44,1 kHz stéréo), ×14,9 | · `70c390b` | message de commit (fixes #12) | en session |
| FV-15 | Retrait des synchronisations de debug : 30,6 → 33,5 tok/s (+9,5 %), mini-3b-8bit STT | · `c1942ee` | idem | en session |
| FV-16 | Top-p par `top(k: 1000)` : chat Mini 18,1 → 33,5 tok/s ; chat Small 8,8 → 11,5 tok/s | · `41ce59d` | idem | en session ; top-p approché (P-26) |
| FV-17 | Limites de cache KV dans tous les préréglages : 33,2 / 11,1 tok/s, pic « marginal » | · `d6acbf1` | idem | introduit `RotatingKVCache` par défaut : S-02, P-03 |
| FV-18 | Préfill par tranches de 512 : pic MLX Mini 6 116 → 4 878 Mo (−20 %), Small 15 575 → 14 364 Mo (−8 %) ; 32,8 / 11,2 tok/s ; « identical output verified » | · `1eb2cc9` | idem (fixes #13, #26) | en session ; le commentaire #13 prévoyait 6,1 → 1-1,5 Go et 15,5 → 2-3 Go (piège 25) |
| FV-19 | Tokenizer et compilation Core ML en parallèle : −≈ 270 ms à froid, aucune différence à chaud | · `1528887` | idem | marginal |
| FV-20 | Pics MLX annoncés pour iOS : mini-3b-4bit ~2,4 Go ; mini-3b-8bit ~6 Go ; small-4bit ~15,5 Go ; tts-4b-4bit ~2,4 Go ; realtime-4b-4bit ~4,6 Go | non notée · PR #32 | corps de PR #32 | **contradictoire** avec le README (« GPU Peak ~8 GB » pour mini-3b-4bit) ; jamais mesuré sur appareil |

### 2.4 Realtime

| # | Mesure | Machine · révision | Source | Statut |
|---|---|---|---|---|
| FV-21 | realtime-4b-4bit : mel 3,28 s (0 % GPU) ; encodage 5,44 s à 49 % GPU (+673 Mo) ; préfill 448 ms à 28 % GPU (639 → 4 619 Mo) ; génération 23,89 s, 501 pas, 33,7 ms/pas (σ 2,7), 21,0 tok/s ; pic MLX 4 619 Mo, processus 7 949 Mo | non notée · 2026-04-11 | issues #23, #24, #25 | le « 0 % GPU » est un artefact d'instrument (fermeture #23) qui a masqué P-61 (P-73) |

### 2.5 TTS

| # | Mesure | Machine · révision | Source | Statut |
|---|---|---|---|---|
| FV-30 | cfgAlpha 1,2 · 8 pas · T = 0. Court EN : 4 bits TTFT 400 ms · 72 frames · 5,68 s · gén 6,63 s · RTF 1,17x ; 6 bits 616 ms · 63 · 5,04 s · 9,46 s · 1,88x ; bf16 1 454 ms · 76 · 5,60 s · 38,43 s · 6,86x. Court FR : 4 bits 224 ms · 53 · 4,16 s · 3,54 s · 0,85x ; 6 bits 338 ms · 65 · 4,80 s · 6,69 s · 1,39x ; bf16 864 ms · 63 · 4,96 s · 25,81 s · 5,20x. Long EN : 4 bits 909 ms · 2 266 · 181,28 s · 120,61 s · 0,67x ; 6 bits 795 ms · 2 101 · 166,96 s · 155,46 s · 0,93x ; bf16 1 523 ms · 2 314 · 185,04 s · 902,26 s · 4,88x. Long FR : 4 bits* 1 412 ms · 2 500* · 200,00 s* · 158,75 s · 0,79x ; 6 bits 1 132 ms · 2 175 · 173,76 s · 194,41 s · 1,12x ; bf16* 1 696 ms · 2 500* · 200,00 s* · 971,31 s · 4,86x (* = plafond `maxFrames` atteint, EOA manqué) | M3 Max 96 Go · `6ad4e56` (2026-04-02) | `docs/tts_benchmark.md:5-56` | en session ; TTFT = premier frame ; RTF = gén/audio ; antérieur à `a00024f`/`0be05af`/`f4fd21c` ; « up to 19 fps » du README = 2 266 / 120,61 s |
| FV-31 | 4 bits « ~10x faster than bf16 (RTF 0.39x) » | M3 Max · `8f095df` | message de commit | en session |
| FV-32 | 4 bits : génération sémantique 11,75 s (28 % GPU, 87,5 % du total), 16,2 fr/s, pas 42,6 ms (σ 23,7), « RT factor 1.12x », TTFT 1 159 ms (corps) / 406 ms (commentaire), pic MLX 2,4 Go ; bf16 : 53,8 s (46 % GPU), 3,5 fr/s, 240,7 ms, RT 0,28x, TTFT 1 624 ms, pic 7,7 Go ; préfill 789 ms / 40 % GPU (+1,9 Go) contre 1 210 ms / 19 % (+6,6 Go) | non notée · 2026-04-11 | issues #26, #27, #28 | RT = audio/gén ; deux TTFT 4 bits contradictoires dans la même issue |
| FV-33 | « Codec 4 bits 3,4× plus lent que bf16 (448 contre 131 ms) » : **faux positif**, codec en bf16 dans les deux packs, 448 ms = passage froid, 66 ms ensuite | · 2026-04-11 | issue #29 (commentaire de clôture) | réfuté ; à ne **pas** cataloguer comme technique |
| FV-34 | Vérification d'EOA tous les 4 frames : « RT factor 1.12x → 1.41x, 16.2 → 17.9 fr/s », audio identique | · `a00024f` | message de commit | **incohérence interne** : +26 % de RT pour +10,5 % de fr/s, non expliquée → à requalifier |
| FV-35 | CFG en lot de 2, une synchro par intégration d'Euler, préfill + 1er jeton AUDIO fusionnés : TTFT 414 → ~280 ms (−32 %), 27,7 → 31,5 fps, tts-4b-4bit, « Audio A/B vs main unchanged » | non notée · `0be05af` (PR #37) | message de commit, corps PR #37 | en session ; parité à l'écoute, sans graine |
| FV-36 | Cache KV du préfixe de voix : préfill « premier audio » 344-381 ms à froid → 146-150 ms à chaud (même voix répétée) | · `f4fd21c` | idem | en session ; préréglages seulement (P-40) |
| FV-37 | Stabilité (6 bits + voix enrôlée, 5 runs) : SEED (`seed: 1234`) EOA = [174 ×5], crête −13,4 dBFS ×5, bit-identique, cache GPU +1 859 Mo ; NONE EOA = [192, 166, 175, 205, 145] (+4 056 Mo) ; CLEAR (`clearCache`) EOA = [158, 156, 174, 198, 156] (+1 822 Mo) | · `07e6317` (PR #41) | message de commit, table PR #41 | déterminisme prouvé |
| FV-38 | Invariant du clone : 3 synthèses consécutives à graine fixe bit-identiques sur les deux chemins ; **0/26** couches du cache de préfixe mutées | 6 bits réel · `6502b7d` (PR #40) | idem | vérifié |
| FV-39 | Streaming : sans graine 2 017 920 contre 119 040 échantillons (~84 s contre ~5 s) pour la même entrée ; graine 42 : 134 400 = 134 400 ; warm-up + graine 42 : 107 520 = 107 520 ; 3 aperçus clonés consécutifs 19,76 / 15,76 / 16,00 s, corrélation croisée ≈ 0 | 4B + voix « vincent » · `d7a414e` (PR #43) | corps PR #43 | vérifié |
| FV-40 | Composants traversés par l'enrôlement (`audio_tokenizer`, `audio_codebook_embeddings`) : 0 tenseur quantifié dans les checkpoints 6 et 4 bits ; `language_model` 183 et `acoustic_transformer` 25 tenseurs quantifiés → enrôler sur bf16 ou 6 bits donne le même embedding | · `eac0f93` (PR #41) | idem | vérifié (lecture des checkpoints) |
| FV-41 | Campagne q6 (1 voix FR masculine ~90 Hz, 3 phrases × 5 graines, juge `mini-3b-8bit`, avec warm-up) : `tts-4b-6bit` couverture 99,4 %, RTF 1,47, fuites 2-3/15 ; `tts-4b-mlx` 96,5 %, RTF 3,44, 0/15 ; sans warm-up 93-96 % ; q6 99,4 % avec contre 93,4 % sans | non notée · `e83778a` (PR #47) | message de commit, PR #47, `docs/voice_cloning.md:122-126` | qualité : comparaison entre bras seulement ; RTF non référence (P-76) |
| FV-42 | Porteur de warm-up : pause terminale −55 / −65 / −126 dB selon la graine (parole −16…−37 dB) ; code sémantique à la frontière 3648 / 3579 / 8032 ; deux passes sur un cache : 2/3 graines continuent à vocaliser ; coupe adaptative 7/8 correct contre 5/12 | · `1c130aa` (PR #44) | message de commit, #45 | vérifié |
| FV-43 | Enrôlement corrigé : 0 échec de coupe sur 10 prises contre 1/8 ; facteur de crête `fr_male` 18,8 dB contre voix enrôlée 21,2 dB (+6 dB de marge → 20,1 dB) ; zéros numériques 3,4 % → 10,5 % d'une génération à l'autre, dont une plage de 786 ms en milieu d'énoncé | · 2026-07-26/27 | commentaires #45 | vérifié |
| FV-44 | Coupe du porteur au seuil relatif au pic : 1re phrase mangée (18,9 → 11,8 s) ; plancher absolu : 9,52 contre 10,08 s ; version adaptative en streaming : 1,36 s de moins que sans warm-up (= le porteur) | · `7cf0b5f`, `85e3be3` | messages de commit | vérifié |
| FV-45 | Lead-in FR ~560 ms à RMS 0,017 → 1er frame RMS 0,073 (codes 855, 10) ; plancher de bruit d'une voix enrôlée −32,5 dBFS ; lead-in de 22 frames (1,76 s) et 32 frames observés → plafond de balayage 20 → 50 | · `b263eaa`, `c2edaea`, `b4714a5` | messages de commit | vérifié |
| FV-46 | Sans ponctuation finale, bf16 génère « 200s+ » d'audio pour une phrase FR de 5 mots (EOA jamais prédit) | · `e8d7f09` | idem | vérifié |
| FV-47 | ZeroVoice : t max recommandé FR 0,15 · EN 0,05 · IT 0,15 · PT 0,10 · AR 0,10 · ES 0,05 · NL 0,05 · DE inutilisable · HI 0,02 ; `maxBlendWeight` 0,20 → 0,05 | M3 Max 96 Go, TTS bf16, juge Realtime 4 bits · `1cbf014` | `docs/zerovoice_benchmark.md:120-132` ; `VoxtralZeroVoice.swift:70` | juge non validé (P-78) |

### 2.6 Enrôlement (clonage de voix)

| # | Mesure | Machine · révision | Source | Statut |
|---|---|---|---|---|
| FV-50 | 1 500 époques en 1 min 46 s contre ≈ 26 min PyTorch-MPS (≈ 15×) ; ECAPA (référence obama, texte inédit) Swift 0,580 · Python 0,541 · Python + perte ECAPA 0,590 ; pertes : L1 exacte, STFT/mel à 0,03 près | Mac M · `bd59931` | message de commit | en session |
| FV-51 | Obama : norme ≈ 4,45, 0/199 trames effondrées, similarité 0,68 | · `b8d98bc` | idem | en session |
| FV-52 | Longueur de référence 4 / 8 / 16 / 24 s → similarité 0,67 / 0,69 / 0,72 / 0,72 (2 000 époques) ; préréglages ≈ 0,84 ; inter-locuteurs ≈ 0,05 ; exemples LibriVox EN 0,67, FR 0,74 (3 000 époques) | · 2026-07-09 | `docs/voice_cloning.md:36-45`, `:225-232` | en session |
| FV-53 | Ancien passe-haut (FIR 64 taps) : −27,0 dB à 100 Hz, −23,9 à 120, −15,5 à 200 ; Butterworth ordre 2 zéro-phase : −1,9 dB à 100 Hz, −1,0 à 120, −29,7 à 30 ; F0/H2 : préréglage +0,9 dB, voix enrôlée −16,3, préréglage passé dans l'ancien filtre −10,5 ; E2E (bf16, 5 000 époques, même référence dégradée) : NaN contre perte 1,55 ; 197,8 s de babillage contre 7,6 s ; RMS −35,7 contre −30,6 dB ; trous ≥ 0,1 s 56 contre 4 | · `f63e2a8`, `1c7b57e` (PR #44) | corps PR #44 | vérifié (tests de réponse en fréquence) |
| FV-54 | Réenrôlement de la **même** référence micro (ancien/nouveau filtre) : F0/H2 −14,5 → −3,4 dB ; F0 132,6 → 89,6 Hz (micro brut 89,9 Hz) ; zéros 21,8 % → 0,3 % ; RMS −38,2 → −33,2 dB | · `4fb44b7` | message de commit | **à requalifier** : `docs/voice_cloning.md:89-92` écrit « 6,4 dB de fondamentale récupérés, 144 → 88 Hz (97 Hz réels) », chiffres différents pour la même expérience annoncée |
| FV-55 | Chaîne F0/H2 (75-105 contre 150-210 Hz, locuteur ~90 Hz) : micro +0,5 dB ; référence préparée −2,1 ; reconstruction depuis les codes +2,8 ; synthèse sur texte nouveau −5,4. Coin 70 Hz : −2,7 dB à 90 Hz ; 50 Hz : −0,8 dB à 90 Hz, réjection à 30 Hz 19 dB contre 30 dB | · `a7045f5` | message de commit, `docs/voice_cloning.md:78-84` | vérifié |
| FV-56 | Référence à −38 dB RMS actif → synthèses à −38 dB (préréglages −23,6 dB) ; prise à −30 dB → −27 dB ; prise à −23 dB → −22 dB ; sortie avec normalisation −21,7 dB (préréglages ≈ −24 dB) | · `1c7b57e`, #45 | messages, `docs/voice_cloning.md:59-63` | vérifié |
| FV-57 | Annexe Python : gradient `torch.stft` MPS, cosinus par seconde contre CPU `[1.0, 1.0, 0.73, 0, 0, 0, 0, 0]` (torch 2.12) ; distance mel de reconstruction 3,10 → 0,69 (pertes sur CPU) ; sans trame END_AUDIO : similarité 0,43 (1re moitié) contre 0,65 (2e) ; plafond Python 0,56-0,59 | · `f9d0ec9` | `Scripts/VoiceCloningResearch/README.md:47-79` | en session |

### 2.7 Téléchargement

| # | Mesure | Source | Statut |
|---|---|---|---|
| FV-60 | `swift-huggingface` bloqué à ~30 Mo sur la redirection CDN des fichiers LFS ; `URLSession` direct ≈ 5,8 Mo/s ; `tts-4b-4bit` (2,4 Go) complet | `fc9013a` | vérifié par l'auteur |

### 2.8 Ce que la baseline (phase 4, point 3) doit re-mesurer

Aucun chiffre de §2.3-§2.6 n'est une référence au sens du skill : révisions anciennes, passage froid, pas
d'A/B/B/A, métriques ambiguës. La baseline (fiche P-79) doit produire, à `9392ed1` + révision résolue de
mlx-swift-lm notée : STT (8 bits et 4 bits, backend `.mlx` **et** `.hybrid`) préfill tok/s, décodage tok/s, TTFT,
pic `phys_footprint` ; TTS (4/6 bits/bf16, préréglage et voix enrôlée, graine fixée) TTFT-frame, **TTFA streaming**,
fps, RTF (génération/audio), pic ; Realtime 4 bits ; enrôlement (s/époque, pic). Premier passage jeté.

---

## 3. Actions — inventaire exhaustif

Dispositions : **fermer avec preuve** · **fiche** (dans le plan, avec porte) · **plan upstream-blocker**
(action-plans) · **hors plan** (motif + déclencheur).

### 3.1 GitHub `mlx-voxtral-swift`

| Id | Source | Action | État vérifié | Disposition |
|---|---|---|---|---|
| ACT-01 | issues / PR / branches | Reprendre tout élément ouvert | **0 issue ouverte, 0 PR ouverte, seule branche distante `main`** (FV-01) | rien à reprendre |

### 3.2 Plans `action-plans` du label `project:mlx-voxtral-swift` (15 plans lus)

| Id | Plan | État vérifié | Disposition |
|---|---|---|---|
| ACT-02 | **#71** `[PR mlx-voxtral-swift#34]` — `status:ready-to-act` depuis le 2026-07-10 (**79 jours**) | PR #34 fusionnée le 2026-07-09T07:37:04Z (merge `374fa7a`, contenu `bd59931`…`b8d98bc` sur `main`) | **fermer `verified`** avec preuve (§3.9) |
| ACT-03 | **#307** `[PR mlx-voxtral-swift#41]` — `ready-to-act` depuis le 2026-07-21 (**68 jours**) | PR #41 fusionnée le 2026-07-20T14:06:41Z (merge `2b421df`) ; son « vrai correctif côté enrôlement (en cours, hors PR) » a été livré par PR #44 (`f63e2a8`, `1c7b57e`) et soldé dans #45 | **fermer `verified`** |
| ACT-04 | **#349** `[issue mlx-voxtral-swift#45]` — `ready-to-act` depuis le 2026-07-28 (**61 jours**) | #45 fermée `completed` le 2026-07-27T08:16:47Z, cinq points tranchés ; résidus ventilés en ACT-10…ACT-17 | **fermer `verified`** avec renvoi à ce rapport |
| ACT-05 | #66 `[PR #33]` — `verified`, fermé le 2026-06-02 | PR #33 fusionnée ; **un point de son plan de test n'a jamais été coché** (ACT-30) | pas de réouverture ; nouvelle fiche (FA-07) |
| ACT-06 | #260, #261, #304, #305, #308, #338, #339, #350, #351, #352, #353 (`stale-branch …`) | tous `verified`, fermés les 2026-07-29 ; branches effacées (seule `main` subsiste) | rien |
| ACT-07 | **absence** de plan pour `Package.swift:51` (« Revisit once ml-explore cuts a tag beyond 3.31.4 ») | 0 plan `kind:upstream-blocker` pour mlx-swift-lm (11 plans upstream-blocker lus, aucun Voxtral) ; dernier tag 3.31.4 (FV-02) | **plan upstream-blocker** (YAML §3.9) — FA-02 |
| ACT-08 | #536 (`project:mlx-swift`, « eval() host cost ~1.5x Python ») | `monitoring`, source `github_release any_new` sur mlx-swift ; les boucles AR synchrones TTS et Realtime y sont exposées (P-31, P-65) | **hors plan Voxtral** (suivi existant) ; citer #536 dans les fiches asyncEval |

### 3.3 Résidus de l'issue #45 (et des éléments nommés par la demande)

| Id | Élément | État vérifié | Disposition |
|---|---|---|---|
| ACT-10 | Item 1 — divergence NaN silencieuse | Corrigé : meilleur état fini, arrêt sur perte non finie, `EnrollmentDivergedError`, refus d'écrire un embedding non fini (`VoxtralVoiceEnrollment.swift:473-491`, `:525-620`) | fermé ; résidu « garde NaN non testée, surcharge non levante » → **fiche A-09** |
| ACT-11 | Item 2 — **coupe alignée sur les jetons** | **Réfutée** : prompt préfillé en une passe, le modèle cadence ses frames, aucun alignement texte → frame ; marqueur de code et deux passes aussi réfutés (FV-42) ; `trimLeadingCarrierAdaptive` livré (`TTS/VoxtralTTSProcessor.swift:222`) ; 0 échec sur 10 avec enrôlement corrigé (FV-43) | **fermer** ; capitalisé V-R1…V-R4 |
| ACT-12 | Item 3 — **q6 contre bf16 par défaut pour les voix enrôlées** | Mesure inverse de l'observation initiale (FV-41) ; défaut **conservé bf16** par décision (`bff9788`) ; surfaces incohérentes (registre/CLI bf16, démo 4 bits) ; une affirmation de #45 sur la démo est fausse (FA-03) | **ASK** + **fiche** campagne multi-voix (FA-03, P-34) |
| ACT-13 | Item 4 — niveau de sortie | −21,7 dB contre ≈ −24 dB pour les préréglages (FV-56) ; documenté | fermé |
| ACT-14 | Item 5 — zéros numériques + **dither optionnel** | Côté codec, 3,4 → 10,5 % selon la génération (FV-43) ; LTX tolère (`detectSpeechWindow`) ; « nothing to act on » (clôture #45) | **hors plan** ; déclencheur : un consommateur dont la détection de silence suppose un plancher naturel → fiche « dither de sortie opt-in » |
| ACT-15 | **`--high-pass-hz 50`** pour voix graves | Documenté (`docs/voice_cloning.md:78-84`) ; défaut 70 Hz conservé (`VoxtralVoiceEnrollment.swift:47`) pour la réjection du grondement et le test 20 dB à 30 Hz (FV-55) | **hors plan** ; à reprendre comme réglage d'un **profil d'enrôlement** (A-11, phase 3) |
| ACT-16 | **« re-enroll from raw mic »** (et contrôle F0/H2 + timing LipDub) | Fait : `4fb44b7` (réenrôlement de la même référence micro brute, F0/H2 −14,5 → −3,4 dB) et `a7045f5` (chaîne complète depuis le micro brut) ; volet LipDub soldé côté LTX (#45, 2026-07-27) | **fermer** ; résidu consommateur : réenrôler les voix FluxForge antérieures à `f63e2a8` (ACT-40) |
| ACT-17 | Bonus — ~5 dB de fondamentale perdus à la génération | Inhérent (codes hors variété faute d'encodeur de codec), non atteignable par une pondération de perte (FV-55) | **hors plan** ; déclencheur : publication d'un encodeur de codec par Mistral (aucun au 2026-09-27, FV-04) |
| ACT-18 | Détecteur de fuite de vocalise de la campagne | PR #47 : « still misses accented "Là, là" » — confirmé : `words()` garde les diacritiques (`TTSQuantizationCampaignTests.swift:41-47`), le test cherche `"lala"` (`:124-126`) | **fiche cloud** : pliage `.diacriticInsensitive` (code de test, sans risque) |

### 3.4 Résidus des issues fermées #11-#29

| Id | Issue(s) | Ce qui reste | Disposition |
|---|---|---|---|
| ACT-20 | #11 (build, `swift build`) | Corrigé (`d94172f`) ; la consigne « utiliser `xcodebuild` » n'existe que dans un commentaire d'issue (pas de `CLAUDE.md`) | fiche hygiène : créer `CLAUDE.md` (commande de build, wrapper de tests, épinglage) — cloud |
| ACT-21 | #12 (extraction audio CPU) | Résolu par `70c390b` (FV-14) ; la demande initiale (mel sur GPU) reste partielle : mel calculé deux fois par fenêtre (P-21) | **fiche P-21** |
| ACT-22 | #13, #17, #19, #21 (préfill 49 % GPU, pic mémoire) | Fermées sur une **hypothèse non vérifiée** (« allocation Metal systémique ») ; tranches de 512 livrées mais −20/−8 % au lieu de −75/−85 % prévus (FV-18) ; « Small utilisable sur 32 Go » jamais testé ; mesures contaminées par les poids paresseux (P-04) | **fiche FA-08** (Metal System Trace + budget 32 Go simulé), macos-gpu |
| ACT-23 | #14 (encodage Core ML 48 % GPU) | Même hypothèse non vérifiée ; hybride jamais comparé proprement (FV-11) | **fiche A-12** |
| ACT-24 | #15, #20 (chat plus lent) | Résolus par `41ce59d` (FV-16) ; top-p approché (P-26) | fiche P-26 (basse) |
| ACT-25 | #16, #22 (compilation Core ML à froid 1 min 09,6 s / 2 min 25,2 s) | « Not fixable » ; les pistes de l'issue (repli MLX au premier lancement, message de progression « compilation ») jamais faites | **fiche A-12** (décider l'hybride ; si `.mlx` par défaut, le problème disparaît) |
| ACT-26 | #18 (pas 1 lent) | Attendu (préfill inclus) | fermé, rien |
| ACT-27 | #23, #24, #25 (Realtime) | Fermées comme artefacts ; la fermeture de #23 a masqué la tête liée recopiée en fp32 (P-61) | **fiches P-61, P-73** |
| ACT-28 | #26 (génération sémantique) | `a00024f` + `0be05af` livrés ; « batch lookup » (5-10 % estimé) et spéculatif (rejeté a priori, non mesuré) jamais essayés ; `asyncEval` absent (P-31) | **fiche P-31** ; spéculatif hors plan (R&D) |
| ACT-29 | #27 (bf16 lent) | Fermée sur une affirmation **fausse** (« 4-bit is the default in CLI and registry ») ; question qualité 4 bits/6 bits/bf16 sur préréglages jamais mesurée | **FA-03 / P-34** + campagne qualité |
| ACT-30 | #28 (préfill TTS en deux passes) | Fusion dans un graphe (`0be05af`) mais toujours deux passes LLM (P-44) | fiche P-44 (basse) |
| ACT-31 | #29 (codec 4 bits lent) | Faux positif (FV-33) | fermé ; capitalisé V-P7 |

### 3.5 Résidus des PR fusionnées #32-#50

| Id | PR | Élément | Disposition |
|---|---|---|---|
| ACT-32 | #33 | « [ ] Spot-check `quickTranscribe` / manager with a non-English sample to confirm auto-detection accuracy » — **jamais coché**, aucun test ne couvre `language: nil` (grep `Tests/` : 0) ; changement de défaut public livré sans ce contrôle | **fiche FA-07**, macos-gpu |
| ACT-33 | #32 | iOS : compilation seule (simulateur), aucun test sur appareil ; table de pics non sourcée (FV-20) | **ASK cadrage** (§1.5 Q2) ; si iOS retenu → fiche appareil |
| ACT-34 | #40 | Suggestion à FluxForge : retirer le contournement « décharger le modèle après chaque aperçu » et relever mémoire active/cache à chaque aperçu | hors dépôt (ACT-40) |
| ACT-35 | #41 | « Le vrai correctif est côté enrôlement (en cours) » | soldé par #44 → rien |
| ACT-36 | #44 | « Remaining, tracked in #45 » | soldé (§3.3) |
| ACT-37 | #47 | Détecteur de fuite incomplet | ACT-18 |
| ACT-38 | #49 | Ask #8 (swift-transformers/HubApi) « hors périmètre du paquet » | hors plan Voxtral (FluxForge) |
| ACT-39 | #50 | « Revisit once ml-explore cuts a tag ≥ 604fae71 » | ACT-07 / FA-02 ; S-17 (retirer la conformance `LanguageModel` inutile) supprime la cause de la casse mais pas la contrainte SwiftPM (FluxForge aussi sur `main`) |

### 3.6 TODO et marqueurs du code

| Id | `fichier:ligne` | Texte | Disposition |
|---|---|---|---|
| ACT-41 | `Package.swift:46-51` | « Pinned to branch … Revisit once ml-explore cuts a tag beyond 3.31.4 » | ACT-07 (plan upstream-blocker) |
| ACT-42 | `VoxtralComponents.swift:28` | `//import Transformers  // TODO: Integrate later…` | fiche S-29 (supprimer : `Hub` est déjà importé ailleurs, S-15) |
| ACT-43 | `VoxtralComponents.swift:566` | `// TODO: implémenter la logique spécifique transcription si nécessaire` (`encodeTranscription`) | fiche S-29 / S-13 (code mort ou souche publique → ASK si public) |
| ACT-44 | `Utils/VoxtralMLXLMLoader.swift:40` | `quantization: nil,  // No quantization for now` | S-14 / S-16 (chargeur legacy) |
| ACT-45 | `MLXLMBridge.swift:694` | « workaround for Swift MLX limitations » (`embed_tokens` jamais quantifié) | S-16 (redondance amont, `QuantizedEmbedding` supporté) — à vérifier en fiche |
| ACT-46 | `VoxtralApp/TranscriptionManager.swift:290-294` | `Memory.cacheLimit = Int.max` « Restore default (unlimited) » | A-17 / P-09 |
| ACT-47 | `VoxtralModeling.swift:838-845` | chemin `/Users/vincent/Developpements/convertvoxtral/…` (« workaround testing ») | S-24 |
| ACT-48 | `TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:51` | doc de `gateReference` : « to true silence » (périmé depuis `1c7b57e`) | FA-05 |

### 3.7 Documentation résiduelle

| Id | Où | Disposition |
|---|---|---|
| ACT-50 | `README.md:185-187` (gate « true silence », normalisation absente) | FA-05 (cloud) |
| ACT-51 | `Scripts/VoiceCloningResearch/README.md:72-79`, `:91-95` (« Next step: Swift/MLX port », plafond 0,56-0,59) | FA-06 (cloud) |
| ACT-52 | `docs/voice_cloning.md:89-92` contre `4fb44b7` (FV-54) | requalifier par re-mesure (fiche macos-gpu) ou citer la source exacte (cloud) |
| ACT-53 | `llms.txt`, README exigences/architecture/Realtime | S-19, S-20 |
| ACT-54 | Métriques TTS (RTF, TTFT, TTFA) | FA-04 |

### 3.8 Actions hors dépôt (consommateurs)

| Id | Action | Disposition |
|---|---|---|
| ACT-40 | FluxForge : (a) réenrôler les voix créées avant `f63e2a8` (2026-07-25) — gain mesuré FV-54 ; (b) retirer le déchargement du modèle après chaque aperçu (PR #40) ; (c) mettre à jour la doc de stockage (résolu en v2.2.1, audit-stabilite l. 684) ; (d) ask #8 HubApi | **plan manuel `project:fluxforge-studio-swift`** (à créer par le planificateur, non vérifiable ici : dépôt inaccessible) |
| ACT-55 | FluxForge et Voxtral passent ensemble de `branch: main` à `from:` quand mlx-swift-lm publie > 3.31.4 | remédiation du plan ACT-07 (ASK : synchronisation des deux paquets) |

### 3.9 Textes prêts pour le tracker (à exécuter par une session autorisée à écrire ; rien n'est écrit ici)

Clôtures (`update-plan.py <n> --close --comment "<preuve>"`, patch `{"status":"verified"}` ; sans `gh` :
`--issue-json … --emit-json`) :

- **#71** — « Vérifié le 2026-09-27 (audit mlx-swift-audit, docs/audit/2026-09-27/faits-et-actions.md ACT-02) :
  PR VincentGourbin/mlx-voxtral-swift#34 fusionnée le 2026-07-09T07:37:04Z (merge 374fa7a) ; l'enrôlement natif
  (`bd59931`) est sur `main` et dans les tags ≥ v2.2.0. Rien à appliquer. »
- **#307** — « Vérifié le 2026-09-27 (ACT-03) : PR #41 fusionnée le 2026-07-20T14:06:41Z (merge 2b421df) ; le
  correctif de fond annoncé côté enrôlement est livré par PR #44 (f63e2a8, 1c7b57e) et tranché dans #45. »
- **#349** — « Vérifié le 2026-09-27 (ACT-04) : issue #45 fermée `completed` le 2026-07-27T08:16:47Z. Résidus
  ventilés dans faits-et-actions.md §3.3 : défaut q6/bf16 → ASK + fiche ; dither, `--high-pass-hz 50`, déficit de
  fondamentale → hors plan documenté ; coupe alignée jetons → réfutée. »

Nouveau plan (ACT-07) — `new-plan.py`, **à vérifier d'abord** que `ml-explore/mlx-swift-lm` publie des *GitHub
Releases* (le plugin `github_release` lit `repos/{repo}/releases/latest` ; s'il n'y a que des tags, il échoue,
voir `skill_feedback`) :

```yaml
subject: "mlx-voxtral-swift épinglé sur mlx-swift-lm branch main faute de tag > 3.31.4 (tag v2.2.2 inconsommable par version)"
kind: upstream-blocker
project: mlx-voxtral-swift
severity: medium
sources:
  - kind: github_release
    repo: ml-explore/mlx-swift-lm
    match_when: semver_gt "3.31.4"
remediation:
  - action: shell
    confirm: required
    cmd: |
      echo "1) Vérifier que le tag contient prepare(_:cache:state:prefill:) (9392ed1).
      2) ASK : passer Voxtral ET FluxForge Studio de branch: main à from: <tag> ensemble.
      3) Package.swift:52 → from: <tag> ; supprimer le commentaire l. 46-51 ; build + tests macos-gpu ; tag v2.2.3."
context:
  project_path: /Users/vincent/Developpements/mlx-voxtral-swift
  branch: main
  related_files: [Package.swift]
  notes: |
    Casse avérée #50 (prepare). Règle SwiftPM : un paquet requis par version ne peut dépendre d'une branche.
    Source : docs/audit/2026-09-27/faits-et-actions.md FA-02, audit-stabilite.md S-18.
```

---

## 4. Constats (format du skill)

Format : sévérité · `fichier:ligne` · constat · preuve · correction · risque (API ?) · effort · statut · fiche
(objet + porte chiffrée + cible).

### FA-01 · moyenne · action-plans #71, #307, #349 — trois plans `ready-to-act` depuis 61 à 79 jours alors que leur source est close

- **Constat** : les plans auto-suivis d'une PR/issue du propriétaire flippent à la fermeture mais personne ne les
  solde ; la remédiation (`gh pr view --web`) n'apporte rien.
- **Preuve** : ACT-02…ACT-04 (dates de fusion et de fermeture, commentaires de transition du watcher
  2026-07-10, 2026-07-21, 2026-07-28 ; `last_check` 2026-09-27).
- **Correction** : clôturer avec les commentaires-preuves de §3.9 ; côté outil, voir `skill_feedback` (source
  `github_issue` sur une URL de PR ne distingue pas fusionnée/fermée ; pas de clôture automatique).
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-FA01** — *Solder les plans Voxtral*. **Porte** : 0 plan `project:mlx-voxtral-swift` en
  `status:ready-to-act` ; #71, #307, #349 fermés `status:verified` avec un commentaire citant PR/commit/date.
  **Cible** : cloud (session autorisée à écrire dans action-plans).

### FA-02 · moyenne · `Package.swift:46-52` + tag `v2.2.2` — dépendance de branche dans une version taguée, sans plan de suivi

- **Constat** : `v2.2.2` (= `9392ed1`) dépend de `mlx-swift-lm` en `branch: "main"`. SwiftPM refuse qu'un paquet
  requis **par version** dépende d'un paquet en branche : un consommateur `from: "2.2.0"` ne peut pas résoudre 2.2.2
  (il échoue ou reste sur ≤ 2.2.1, qui dépend de `3.31.4` et ne compile pas contre `main`). `llms.txt:198` conseille
  encore `from: "1.0.8"`, le README `branch: "main"`. Le « Revisit » n'a aucun plan de suivi (ACT-07).
- **Preuve** : `Package.swift:52` ; tag v2.2.2 → `9392ed1` (MCP) ; dernier tag amont 3.31.4 (`git ls-remote`) ;
  0 plan upstream-blocker Voxtral (MCP `list_issues kind:upstream-blocker`).
- **Correction** : plan upstream-blocker §3.9 ; README/llms.txt : « tags ≥ v2.2.2 : dépendre par `branch`/`revision`
  tant que mlx-swift-lm n'a pas de tag > 3.31.4 » ; S-18 (`Package.resolved` suivi) ; S-17 réduit le couplage.
- **Risque API** : aucun (doc et dépendance). **Effort** : S. **Statut** : VÉRIFIÉ en lecture (règle SwiftPM) ;
  comportement exact de résolution chez un consommateur **À MESURER**.
- **Fiche K-FA02** — *Consommabilité des tags*. **Porte** : plan upstream-blocker créé en `monitoring` ; projet
  témoin `from: "2.2.2"` → sortie de `swift package resolve` consignée ; README et llms.txt disent comment dépendre de
  v2.2.x. **Cible** : cloud (plan, docs) puis macos-gpu (résolution).

### FA-03 · moyenne · `TTS/VoxtralTTSRegistry.swift:37` ; `VoxtralCLI.swift:400`, `:588` ; `StreamingDemoViewModel.swift:14` — défaut TTS incohérent et deux clôtures appuyées sur des affirmations fausses (complète P-34)

- **Constat** : bibliothèque et CLI par défaut en bf16 (le plus lent : RTF 4,86-6,86, FV-30) ; la démo en 4 bits ;
  le README recommande 4/6 bits (`README.md:83-87`) ; `docs/voice_cloning.md:117-120` « defaults are bf16 ». #27 a été
  fermée sur « `tts-4b-4bit` is the default in CLI and registry » (faux depuis `8f095df`) ; #45 (commentaire du
  2026-07-25) affirme que « the streaming demo already defaults enrolled-voice synthesis to bf16 » (faux : 4 bits
  depuis `e8d7f09`, aucune bascule pour les voix clonées, `StreamingDemoViewModel.swift:460-511`). La campagne FV-41
  (n = 15, une voix) donne l'avantage au 6 bits.
- **Preuve** : lignes citées ; `git log -S` sur les deux défauts (`cc77c86`, `8f095df`, `b8d98bc`, `e8d7f09`).
- **Correction** : **ASK** — une politique de défaut unique pour registre, CLI `tts`/`enroll`, démo, README ;
  décision dans `docs/knowledge/decisions/`, liée aux profils (phase 3).
- **Risque API** : comportement (valeur de `VoxtralTTSRegistry.defaultModel` lue par les consommateurs) → ASK.
  **Effort** : S (code) + M (campagne). **Statut** : VÉRIFIÉ.
- **Fiche K-FA03** — *Défaut TTS mesuré*. **Porte** : `TTSQuantizationCampaignTests` étendue à ≥ 3 voix (2
  préréglages + 1 enrôlée) × FR/EN × 5 graines, binaire Release : couverture ASR q6 ≥ bf16 − 1 pt **et** RTF
  (gén/audio) q6 ≤ 0,5 × bf16 → q6 par défaut sur les 4 surfaces ; sinon bf16 documenté partout. **Cible** :
  macos-gpu.

### FA-04 · moyenne · `TTS/VoxtralTTSProcessor.swift:30-33` ; `VoxtralTranscriptionTest/ProfileCommand.swift:274` ; `TTS/VoxtralTTSModeling.swift:494-497`, `:580-683` — métriques TTS publiées ambiguës (RTF inversé, TTFT ≠ premier audio)

- **Constat** : deux RTF de sens opposés ; le TTFT publié mesure le premier frame interne, hors préfixe, et le
  streaming ne streame pas (S-08) : la démo affiche un « TTFT » ≈ génération complète. Les chiffres de FV-30, FV-32,
  FV-34, FV-35 ne sont pas comparables entre eux sans requalification.
- **Preuve** : §2.1.
- **Correction** : un glossaire unique (TTFT-frame, **TTFA** = premier échantillon audio reçu par le consommateur,
  RTF = génération/audio, fps) dans `docs/Benchmarks.md` ; renommer « RT factor » de `profile` en « speed ×
  (audio/gen) » ou l'inverser ; annoter chaque tableau publié (définition, révision, « en session »).
- **Risque API** : aucun (sortie CLI de diagnostic). **Effort** : S. **Statut** : VÉRIFIÉ. Recoupe P-45, P-76, S-08.
- **Fiche K-FA04** — *Glossaire des métriques*. **Porte** : `grep -rn "RT factor" Sources` = 0 ; chaque tableau de
  `README.md` et `docs/*benchmark*.md` porte définition + révision + mention « en session » ; `profile` rapporte TTFA.
  **Cible** : cloud (docs, CLI sûre sous `syntax_guard`) puis macos-gpu (TTFA après K-S08).

### FA-05 · basse · `README.md:185-187` ; `TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:51` — description périmée de la préparation de référence

- **Constat** : le README dit que les fenêtres sous le seuil « become true silence » et omet la normalisation ;
  depuis `1c7b57e`, le gate atténue de 24 dB (`:67`) et la référence est normalisée à −20 dBFS actifs (`:76`). PR #46
  a corrigé `docs/voice_cloning.md` et `docs/streaming_demo.md`, pas le README ni ce commentaire.
- **Correction** : aligner les deux textes sur `docs/voice_cloning.md:64-76`.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-FA05** — *README clonage à jour*. **Porte** : `README.md` décrit passe-haut → normalisation −20 dBFS →
  gate −24 dB ; `grep -n "true silence" README.md` = 0 ; commentaire l. 51 corrigé (la mention l. 276, cas `nil`,
  reste exacte). **Cible** : cloud.

### FA-06 · basse · `Scripts/VoiceCloningResearch/README.md:72-79`, `:91-95` — « Next step » déjà livré

- **Constat** : « Next step: Swift/MLX port of the enrollment loop » a été fait dans `bd59931` (PR #34) ; le
  « quality ceiling (current) 0.56-0.59 » est celui du Python, la voie Swift atteint 0,72 à 16 s (FV-52).
- **Correction** : remplacer par un renvoi à `docs/voice_cloning.md` et à `bd59931`.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche K-FA06**. **Porte** : `grep -n "Next step" Scripts/VoiceCloningResearch/README.md` = 0 ; plafond cité avec
  sa voie (Python / Swift). **Cible** : cloud.

### FA-07 · moyenne · `Pipeline/VoxtralPipeline.swift:316`, `:393` ; `Pipeline/VoxtralTranscriptionManager.swift:115-153` — auto-détection de langue par défaut jamais validée

- **Constat** : PR #33 a fait passer le défaut public de `"en"` à `nil` (auto-détection) ; l'élément de plan de
  test « spot-check with a non-English sample » n'a jamais été coché et aucun test ne couvre `language: nil`.
- **Preuve** : corps de PR #33 (case non cochée) ; `grep -rn "language: nil" Tests` = 0.
- **Correction** : test d'intégration sous variable d'environnement (FR, DE, ES) ; si l'auto-détection dégrade,
  documenter ou détecter explicitement la langue.
- **Risque API** : aucun (test). **Effort** : S. **Statut** : VÉRIFIÉ (absence) ; qualité **À MESURER**.
- **Fiche K-FA07** — *Auto-détection validée*. **Porte** : 3 langues × 3 clips, `mini-3b-8bit` greedy : WER(`nil`) ≤
  WER(langue explicite) + 2 pts, lignes recopiées dans `BENCHMARKS.md`. **Cible** : macos-gpu.

### FA-08 · moyenne · issues fermées #13, #14, #17, #19, #21, #24 — « 49 % GPU systémique » et « Small sur 32 Go » jamais vérifiés

- **Constat** : un même 48-49 % GPU sur quatre opérations différentes (Core ML, préfill MLX, encodeur Realtime) est la
  signature d'un instrument (échantillonnage IOKit), pas d'un goulot ; la cause « allocation Metal » n'a jamais été
  confirmée par Metal System Trace ; la prévision de gain du préfill tranché était ×4-5 trop optimiste (FV-18) ;
  « rendrait le small utilisable sur 32 Go » (#13, #21) n'a jamais été mesuré.
- **Preuve** : commentaires de clôture #13/#14/#24 ; FV-12, FV-13, FV-18 ; P-73 montre la même erreur sur #23.
- **Correction** : la baseline (P-79) inclut une Metal System Trace (union d'intervalles) et un recoupement `ioreg`,
  et un run small-4bit à budget 32 Go simulé.
- **Risque API** : aucun. **Effort** : M. **Statut** : À MESURER.
- **Fiche K-FA08** — *Requalifier le « 49 % GPU »*. **Porte** : occupation GPU réelle du préfill Mini et Small
  consignée (Metal System Trace + `ioreg`, écart entre instruments noté) ; small-4bit transcription 8 min à budget
  32 Go : pic `phys_footprint` ≤ 24 Go et fin sans swap, ou verdict « non supporté » documenté. **Cible** :
  macos-gpu.

### FA-09 · basse · `Package.swift:12` ; PR #32 — support iOS déclaré, jamais exécuté

- **Constat** : `.iOS(.v17)` compile (simulateur) mais n'a jamais tourné sur appareil ; les pics annoncés (FV-20)
  contredisent le README ; piège 22 (simulateur, arrière-plan GPU, thermique).
- **Correction** : ASK de cadrage (§1.5 Q2) ; si iOS est une cible, fiche appareil (profils `lean`).
- **Risque API** : aucun. **Effort** : M. **Statut** : VÉRIFIÉ (absence) ; mémoire **À MESURER**.
- **Fiche K-FA09** — *iOS réel*. **Porte** : iPhone 8 Go, `mini-3b-4bit` transcription 60 s et `tts-4b-4bit` 10 s :
  pic `phys_footprint` consigné ≤ 4 Go, aucun jetsam, au premier plan. **Cible** : macos-gpu (+ appareil).

---

## 5. Capitalisation (phase 6) — ajouts candidats à `references/techniques.md`

Le catalogue ne couvre que YuE2 (Y), Qwen38 (Q) et Gemma : **aucun modèle audio**. Proposition : une section
« **V = github.com/VincentGourbin/mlx-voxtral-swift** (STT, Realtime, TTS à flow-matching, clonage par optimisation de
codes) ». Format : **Problème** · **Mécanisme** · **Gain / mesure (source)** · **Risque** · **Règle**. Toutes les
mesures sont « en session » (§2.1) sauf mention.

### 5.1 Techniques RETENUES

**V-T1. Moins de synchronisations dans une boucle AR à sous-étapes (CFG en lot, une synchro par frame).**
Problème : le TTFT TTS était dominé par des allers-retours GPU → CPU, pas par le calcul. Mécanisme : passes CFG
conditionnelle et inconditionnelle en un lot de 2 ; un seul `eval` pour les 7 pas d'Euler ; préfill et premier
jeton AUDIO dans un seul graphe (`TTS/VoxtralFlowMatching.swift:298-321`, `TTS/VoxtralTTSModeling.swift:455-477`).
Gain : TTFT 414 → ~280 ms (−32 %), 27,7 → 31,5 fps, tts-4b-4bit (`0be05af`). Risque : parité à l'écoute seulement
(pas de graine avant `07e6317`) ; TTFT = premier frame interne. Règle : dans une boucle autorégressive dont chaque
pas contient un graphe court (ODE à quelques pas), une synchro **par frame**, jamais par sous-pas, et les passes
indépendantes en lot. Nuance T16 : l'`eval` par couche gagne sur un graphe profond (Q) ; ici le graphe d'un frame
est court et une synchro unique gagne.

**V-T2. Cache KV du préfixe de conditionnement (voix) cloné, jamais prêté.** Problème : les frames de voix précèdent
le texte, leur préfill O(T²) est refait à chaque synthèse. Mécanisme : précalcul par voix, `cloneKVCaches` puis
préfill du seul suffixe avec l'offset RoPE du préfixe (`TTS/VoxtralTTSModeling.swift:393-420`, `:697` ;
`TTS/Pipeline/VoxtralTTSPipeline.swift:93-103`). Gain : préfill « premier audio » 344-381 → 146-150 ms à chaud
(`f4fd21c`) ; invariant épinglé : 0/26 couches mutées, synthèses graine-fixe bit-identiques (`6502b7d`,
`KVCacheCloneTests`). Risque : cache à une entrée ; absent pour voix clonées/ZeroVoice (P-40) ; clé de voix
périmée. Règle : T6 s'applique au TTS dès que le conditionnement précède le contenu dans le prompt ; cloner
(`state` tronqué à l'offset) et tester que les écritures du clone ne fuient pas dans la source.

**V-T3. Condition d'arrêt lue par lots.** Problème : un `.item()` par frame pour détecter la fin d'audio.
Mécanisme : test EOA tous les 4 frames, frames excédentaires coupées (`TTS/VoxtralTTSModeling.swift:480-535`).
Gain : 16,2 → 17,9 fr/s, « RT 1,12x → 1,41x » (convention audio/gén, incohérence FV-34) (`a00024f`). Risque :
jusqu'à 3 frames générés pour rien. Règle : grouper les lectures CPU d'une condition d'arrêt quand le surcoût du
dépassement est borné ; préalable à T15 (`asyncEval`).

**V-T4. Préfill STT par tranches (T9 confirmé sur un modèle audio).** Mécanisme : tranches de 512 sur les embeddings
audio + texte, cache KV accumulé (`VoxtralModeling.swift:1166-1188`, `:1352-1370`). Gain : pic MLX −20 % (Mini),
−8 % (Small), débit inchangé, sortie identique (`1eb2cc9`). Risque : gain modeste quand les poids dominent ;
interaction avec `RotatingKVCache` (S-02, P-03) ; cache réalloué à chaque tranche (P-10). Règle : T9 vaut aussi pour
les longues séquences audio ; balayer la taille (P-22) et **ne pas croire l'estimation** (piège 25 : ×4-5 prévu).

**V-T5. Top-p sans tri complet du vocabulaire.** Mécanisme : `top(k: 1000)` partiel, seuil de probabilité, masque
(`VoxtralModeling.swift:1520-1544`). Gain : chat Mini 18,1 → 33,5 tok/s (×1,85), Small 8,8 → 11,5 (`41ce59d`).
Risque : top-p approché (queue au-delà de 1 000 exclue, P-26). Règle : jamais d'`argSort` sur 130 k logits par pas.

**V-T6. Aucune statistique de debug dans le chemin chaud.** `.item()` (min/max/mean) et `eval` superflus retirés du
forward : 30,6 → 33,5 tok/s (+9,5 %) (`c1942ee`). Règle : tout `.item()`/`print` de tenseur hors de la boucle
chronométrée, derrière un drapeau de diagnostic.

**V-T7. Décodage/rééchantillonnage audio par `AVAudioConverter`.** 3,79 s → 254 ms pour un MP3 de 203 s (×14,9)
(`70c390b`, `VoxtralFeatureExtractor.swift:44-45`). Risque : conversion en un seul appel = silence au
**suréchantillonnage** (V-P3). Règle : convertisseur système en boucle d'entrée ; tester sous- et
sur-échantillonnage.

**V-T8. Téléchargement HF direct par `URLSession` (liste par l'API tree, fichier par fichier, reprise réseau).**
Problème : `swift-huggingface` reste bloqué à ~30 Mo sur la redirection CDN des fichiers LFS. Mécanisme : `GET
/api/models/{repo}/tree/{rev}?recursive=true`, téléchargement vers un temporaire puis déplacement, saut des
fichiers de taille égale, 5 tentatives avec attente 2/4/8/16 s sur erreurs transitoires
(`Utils/ModelDownloader.swift:79-175`). Gain : ~5,8 Mo/s, 2,4 Go complet (`fc9013a`, `e5c93b9`, `1d6b85c`). Risque :
complétude = taille (pas de SHA-256 alors que l'API tree fournit `lfs.oid`), révision `main` non épinglée,
tentative = redémarrage du fichier à zéro (pas de `Range`), complétude du modèle déduite d'un fichier (S-03, A-02).
Règle : pour les gros LFS, liste + tailles + SHA de l'API tree, écriture atomique, `Range` en reprise ; famille
MLX-012.

**V-T9. Graine sur tous les chemins stochastiques (batch ET streaming).** Problème : le flow-matching tire un bruit
frais réinjecté dans l'état AR : même texte, même voix, sortie différente même à T = 0. Mécanisme : `seed:
UInt64?` sur `generate`, `generateStreaming` et toutes les surcharges (`TTS/VoxtralTTSModeling.swift:440`, `:582`).
Gain : EOA [174 ×5] bit-identique contre [192…145] ; streaming 84 s contre 5 s sans graine (`07e6317`, `d7a414e`).
Risque : une surcharge oubliée = régression de couverture (le streaming l'a été 5 jours). Règle : pour un modèle à
échantillonnage, la graine est un paramètre de **chaque** point d'entrée, et la porte de parité d'une optimisation
TTS est « bit-identique à graine fixée » (pas « écoute A/B »).

**V-T10. Porteur de chauffe (« warm-up ») puis coupe adaptative.** Problème : les voix enrôlées dégradent la 1re
phrase (effondrement à ~−54 dBFS sur ~2,5 s). Mécanisme : préfixer « La la la la la la la la. », couper au premier
frame ≥ 20 dB sous la médiane des frames d'ouverture puis avancer jusqu'à la reprise
(`TTS/VoxtralTTSProcessor.swift:222-270` ; `VoxtralTTSPipeline.recommendedWarmUpVocalise`, `:72`). Gain : 1er mot
correct (« Flux Forge » contre « Sorche »/« Loxforge ») ; coupe 7/8 contre 5/12 ; couverture q6 99,4 % contre 93,4 %
(`eac0f93`, `1c130aa`, `e83778a`). Risque : ~1 prise sur 8 fuit ou rogne → vérification ASR en production ; coût du
porteur (P-47). Règle : seuil relatif au signal **lui-même**, ni au pic global ni absolu (V-R4).

**V-T11. Filtre passe-haut réalisable (Butterworth ordre 2 zéro-phase) sur la référence.** Problème : FIR
complémentaire de 64 taps à 24 kHz, incapable d'un coin à 70 Hz : −27 dB à 100 Hz. Mécanisme : biquad aller-retour
(filtfilt) (`TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:225-260`). Gain : −1,9 dB à 100 Hz, −29,7 dB à 30 Hz ;
synthèse F0/H2 −14,5 → −3,4 dB, F0 132,6 → 89,6 Hz (micro 89,9) (`f63e2a8`, `4fb44b7`). Risque : coin 70 Hz encore
−2,7 dB à 90 Hz (V-R6). Règle : un FIR de N taps à fs ne résout pas un coin < ~fs/N ; tester la réponse aux
fréquences qui comptent (fondamentale), pas à 1 kHz.

**V-T12. Normalisation du niveau actif et gate atténuant (pas de zéros).** Problème : le conditionnement est un
préfixe **continué** : niveau, bruit et silences numériques de la référence se retrouvent dans chaque synthèse.
Mécanisme : RMS des fenêtres actives → −20 dBFS, crête ≤ 0,98 ; fenêtres fermées à −24 dB
(`VoxtralVoiceEnrollment.swift:67`, `:76`, `:123-150`). Gain : RMS −35,7 → −30,6 dB, trous ≥ 0,1 s 56 → 4, zéros
21,8 % → 0,3 %, sortie −21,7 dB (préréglages ≈ −24) (`1c7b57e`, #45). Règle : préparer une référence de clonage comme
un signal que le modèle va **imiter** : niveau cible, plancher naturel, fin sur une pause (FV-57).

**V-T13. Garde de divergence d'une optimisation qui écrit un artefact.** Meilleur état fini gardé, arrêt sur perte
non finie, exception si aucun pas fini, refus d'écrire un embedding non fini (`VoxtralVoiceEnrollment.swift:473-491`,
`:525-620`). Gain : NaN (époques 4 000 → 4 500) et 197,8 s de babillage → perte 1,55 et 7,6 s (`f63e2a8`). Règle :
aucun artefact persistant non fini ; test de la garde (A-09).

**V-T14. Enrôlement natif MLX plutôt que PyTorch-MPS.** 1 500 époques 1 min 46 s contre ≈ 26 min (≈ 15×), ECAPA
0,580 contre 0,541, gradients STFT corrects sur tout le signal (`bd59931`). Règle : vérifier les gradients spectraux
par segment contre le CPU avant d'accuser la perte (V-P9).

**V-T15. Campagne qualité à juge ASR commun.** Chaque modèle chargé une fois, graines × phrases **ordinaires**, ASR
chargé une fois, couverture de mots, drapeaux (fuite, incomplet) (`Tests/…/TTSQuantizationCampaignTests.swift`,
`e83778a`). Gain : a renversé une recommandation fondée sur n = 1 (FV-41). Risque : juge et RTF non références
(P-76, P-78). Règle : une recommandation de quantification TTS exige n ≥ 15 et un juge commun ; n = 1 est
disqualifiant.

**V-T16. Borne de mélange ZeroVoice mesurée par aller-retour ASR.** `maxBlendWeight` 0,20 → 0,05, DE/HI
incompatibles avec la SLERP (`1cbf014`, `docs/zerovoice_benchmark.md`). Règle : toute interpolation d'embeddings de
conditionnement se borne par un aller-retour génération → ASR, par langue.

### 5.2 Techniques ESSAYÉES ET REJETÉES (ou réfutées) — avec mesure

| # | Technique / hypothèse | Mesure | Raison | Source |
|---|---|---|---|---|
| V-R1 | Coupe du porteur **alignée sur les jetons** | non implémentable | prompt préfillé en une passe, le modèle cadence ses frames : aucun alignement texte → frame | `1c130aa`, #45 |
| V-R2 | Marqueur de code sémantique à la frontière | codes 3648 / 3579 / 8032 selon la graine | rien de constant à reconnaître | `1c130aa` |
| V-R3 | **Deux passes sur un même cache KV** (porteur jusqu'à EOA, puis contenu) | 2/3 graines ignorent le texte et continuent à vocaliser | un 2e énoncé après EOA est hors distribution | `TTSTwoPassWarmUpProbeTests`, `1c130aa` |
| V-R4 | Seuil relatif au pic / plancher absolu / minimum global pour la coupe | 1re phrase mangée (18,9 → 11,8 s) ; fuite 2/3 graines ; sur-coupe 2,5 s | pause terminale −55 à −126 dB fixée par la génération | `7cf0b5f`, `ba7647d`, `1c130aa` |
| V-R5 | « Les résolutions STFT ne voient pas 90 Hz » | reconstruction +2,8 dB F0/H2, mieux que sa cible (−2,1) ; synthèse −5,4 | déficit à la génération (codes hors variété), pas dans la perte | `a7045f5` |
| V-R6 | Passe-haut par défaut 50 Hz | 50 Hz : −0,8 dB à 90 Hz mais réjection 19 dB contre 30 dB à 30 Hz | pénalise tout le monde pour les voix graves ; option `--high-pass-hz 50` | `a7045f5` |
| V-R7 | « q6 fait sauter des mots en voix enrôlée » | q6 99,4 % contre bf16 96,5 % (n = 15) | observation n = 1 antérieure au correctif du filtre | `e83778a`, `bff9788` |
| V-R8 | « Buffers MLX périmés du pool de cache » (A6a) ; `clearCache()` comme remède | SEED bit-identique malgré cache +1 859 Mo ; CLEAR EOA 156-198 ≈ NONE 145-205 | la variance vient de l'échantillonnage ; `clearCache` borne la mémoire seulement | `07e6317` |
| V-R9 | « Le cache de préfixe est muté par la génération » (A6) | 0/26 couches mutées | le clone existait depuis `f4fd21c` | `6502b7d` |
| V-R10 | « Codec 4 bits 3,4× plus lent que bf16 » | 448 ms froid, 66 ms ensuite ; codec bf16 dans les deux packs | faux positif de première mesure | issue #29 |
| V-R11 | Perte locuteur ECAPA dans l'enrôlement | Swift sans ECAPA 0,580 ≈ Python avec 0,590 | inutile avec des gradients spectraux corrects | `bd59931` |
| V-R12 | Hybride Core ML + MLX comme défaut | 85,6 → 81,9 s (−4,3 %), pic −590 Mo ; 1 min 09,6 s de compilation au 1er lancement | **sous le seuil de 5 %** : à trancher par A-12 (règle R14) ; aujourd'hui retenu sans preuve | `1944576`, #16 |

Non cataloguables (aucune mesure) : spéculatif TTS (rejeté a priori, #26), conversion Core ML du LLM complet
(`Scripts/CoreMLConversion/README.md:27`), cache binaire du tokenizer « 10-100× » (`90fefc0`), transfert WAV en bloc
(`88e08ad`, « ~1,9 M appels → 1 » sans chronométrage).

### 5.3 Pièges généralisables

1. **V-P1 · Le préfixe de conditionnement est imité.** Tout ce qui est dans une référence de voix (niveau, bruit,
   zéros numériques, filtrage, coupure en plein mot, trame de fin absente) se retrouve dans **chaque** synthèse.
   Mesures : −38 dB → −38 dB (FV-56) ; zéros 21,8 % (FV-54) ; sans END_AUDIO 0,43 contre 0,65 (FV-57).
2. **V-P2 · Filtre irréalisable validé par un test hors bande.** Le test du passe-haut ne sondait que 1 kHz et
   passait malgré −27 dB à 100 Hz (`f63e2a8`). Règle : tester aux fréquences critiques (cousin du piège 38).
3. **V-P3 · Rééchantillonnage « one-shot » silencieux.** `AVAudioConverter` en un appel produisait du **silence**
   au suréchantillonnage (22,05 → 24 kHz) → embedding dégénéré, identique pour toutes les références (`6491973`).
   Règle : test de non-silence en sous- et sur-échantillonnage ; filtre anti-repliement avant sous-échantillonnage
   (`VoxtralVoiceEnrollment.swift:201-222`).
4. **V-P4 · Deux conventions de RTF dans le même dépôt** (FA-04). Règle : RTF = génération / audio, défini une fois,
   rappelé dans chaque tableau.
5. **V-P5 · `AsyncThrowingStream { continuation in … }` exécute sa closure de construction de façon synchrone** : tout
   est généré avant que le consommateur ne reçoive le premier élément ; « premier chunk plus petit » (`0be05af`) sans
   effet ; TTFT de démo ≈ génération complète (S-08). Règle : produire dans une `Task` (ou `makeStream()` + `Task`),
   avec `onTermination` (MLX-003), et mesurer le **TTFA** côté consommateur.
6. **V-P6 · Correctif appliqué à une seule surcharge.** Graine et warm-up ajoutés au batch, pas au streaming utilisé
   par l'app : « régression de couverture » (`d7a414e`). Règle : un paramètre de qualité/déterminisme se propage à
   toutes les surcharges, test par surcharge.
7. **V-P7 · Première mesure froide d'une phase = faux positif** (#29 : 448 contre 66 ms). Extension du piège 11 aux
   phases d'un pipeline : jeter le premier passage (poids paresseux, compilation Metal).
8. **V-P8 · Même pourcentage GPU sur des opérations différentes = signature d'instrument.** 48-49 % sur Core ML,
   préfill et encodeur ; « 0 % GPU » pendant 24 s à 21 tok/s (#23). Règle : recouper par Metal System Trace / `ioreg`
   avant de conclure (FA-08, P-73).
9. **V-P9 · Backward de `torch.stft` faux sur MPS au-delà de ~2,5 s** (cosinus 0 contre CPU, torch 2.12) : une
   référence Python sur MPS n'est pas une référence de gradients (FV-57).
10. **V-P10 · Pas de ponctuation finale ⇒ pas d'EOA ⇒ génération jusqu'au plafond** (200 s+ pour 5 mots, `e8d7f09`) ;
    même symptôme avec un embedding NaN (197,8 s). Règle : assainir le texte et borner `maxFrames` par la longueur du
    texte (P-41).
11. **V-P11 · Nom inventé dans un corpus d'évaluation ASR** : toutes les voix l'écorchent (préréglages compris), le
    signal disparaît (`e83778a`). Règle : phrases ordinaires ; détecteurs textuels normalisés (lettres, sans
    diacritiques : ACT-18).
12. **V-P12 · Deux post-traitements qui s'annulent.** `applyTrims` retirait le lead-in voulu par
    `warmUpLeadInFrames` : les variantes d'un test à l'aveugle sonnaient identiques (`9fc331e`). Règle : un test qui
    distingue les variantes (durée) avant tout test d'écoute.
13. **V-P13 · Test qui ne voit qu'une direction.** Le test de coupe ne vérifiait que « porteur gardé » ; « contenu
    mangé » (1re phrase perdue) passait (`7cf0b5f`). Règle : garder les deux directions d'erreur.
14. **V-P14 · Affirmation de clôture non vérifiée dans le code.** #27 (« 4 bits par défaut ») et #45 (« démo en bf16
    pour les voix enrôlées ») sont faux (FA-03). Règle : un commentaire de clôture cite `fichier:ligne`.
15. **V-P15 · Tag de version qui dépend d'une branche** : inconsommable par exigence de version (FA-02).
16. **V-P16 · Estimation de gain d'un commentaire d'issue** : ×4-5 prévu, −20 %/−8 % mesuré (FV-18) — piège 25
    confirmé hors YuE2.

---

## 6. Synthèse

- **Actions** : 0 issue et 0 PR ouvertes ; 3 plans action-plans à solder (ACT-02…04, textes prêts) ; 1 plan
  upstream-blocker manquant (ACT-07) ; résidus de #45 tous tranchés (1 ASK, 1 fiche de test, 4 hors plan motivés,
  2 fermés) ; 12 résidus d'issues fermées dont 4 fermetures sur hypothèse ou affirmation fausse (FA-03, FA-08) ;
  1 contrôle de PR jamais fait (FA-07) ; 8 marqueurs de code ventilés ; 1 plan consommateur FluxForge à créer.
- **Faits** : ~50 mesures publiées, **aucune de référence** (révisions anciennes, passages froids, deux RTF,
  TTFT ≠ TTFA) → la baseline P-79 est un prérequis de toute fiche perf.
- **Capitalisation** : 16 techniques retenues, 12 rejetées/réfutées mesurées, 16 pièges — premier apport « audio »
  (TTS à flow-matching, clonage, STT) au catalogue.
