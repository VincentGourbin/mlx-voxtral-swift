# Audit « Annexes & serveur » — mlx-voxtral-swift

**Vérification croisée : 23 constats relus, 23 gardés, 0 écartés, 13 amendés** (A-01, A-04, A-06, A-07,
A-09, A-10, A-12, A-13, A-14, A-15, A-19, A-21, A-22). Relecture adverse le 2026-09-27, code relu à
`9392ed1`, amont relu aux tags résolus (mlx-swift `0.31.6` = `0bb916c`, swift-transformers `1.3.4`), Hub HF et
PyPI interrogés. Aucun constat n'est faux dans son ensemble. En revanche, dix sous-affirmations (lignes, chiffres,
raisonnements) ont été retirées ou corrigées : voir l'annexe finale.

> Révision auditée : `9392ed1` (branche `claude/action-plan-skills-beta-wifgmu`) · date : 2026-09-27 ·
> skill `mlx-swift-audit` phase 2, constats `A-01…A-23`.
> Environnement : session cloud Linux, **sans Mac, sans toolchain Swift, sans GPU**. Aucun build,
> aucun test, aucune mesure n'a été faite : tout gain est « attendu », tout chiffre non déjà mesuré
> dans le dépôt (ou ses PR/issues) est « À MESURER ». Seules exécutions faites ici : lecture de code,
> `git`, contrôle `argparse` des scripts Python (sans torch), inspection de paquets PyPI, lecture du
> Hub HF et des sources amont.
>
> Amont lu : mlx-swift `main` @ `9019419` (local) **et** le tag `0.31.6` = `0bb916c` (fichiers lus sur
> `raw.githubusercontent.com`), celui que Voxtral résout réellement (`Package.swift:43`
> `from: "0.31.6"` + mlx-swift-lm `Package.swift:64` `.upToNextMinor(from: "0.31.6")`, aucun tag
> > 0.31.6 d'après `git ls-remote`) ; swift-transformers `1.3.4` (dernier tag compatible avec
> `from: "1.3.3"`), `Sources/Hub/HubApi.swift`.
>
> Pas de `CLAUDE.md` dans le dépôt. Consommateurs identifiés : FluxForge Studio (App Store, macOS) —
> « Simplified facade for Fluxforge Studio » (`Pipeline/VoxtralTranscriptionManager.swift:4`), demandes
> de stockage citées par #49 (`FRAMEWORK_ASKS_STORAGE.md`), texte d'annonce « Vous pouvez maintenant
> entrainer vos propres voix » (`Tests/.../TTSBlindValidationTests.swift:35-36`, **indice** que FluxForge
> embarque l'enrôlement) ; chaîne LipDub / ltx-video-swift-mlx (issue #45). Toute suppression d'API
> publique ci-dessous est donc marquée « cassant » et passe par ASK.

## 1. Carte des points d'entrée annexes

| # | Point d'entrée | Fichiers | Consommateurs | Tests | État résumé |
|---|---|---|---|---|---|
| 1 | Enrôlement de voix (descente de gradient) | `TTS/VoiceCloning/VoxtralVoiceEnrollment.swift`, `VoxtralEnrollmentLosses.swift`, `TTSPipeline.enrollVoice` (l. 425-452), CLI `enroll`, démo | CLI, démo, FluxForge (indice) | pertes, préparation de référence, gradients élémentaires ; **pas** la garde NaN ni l'annulation | garde NaN OK ; **concurrence non gardée (ABBA)**, pas de graine, pas de reprise, modèle entier chargé (résidence du LLM à mesurer : chargement paresseux) |
| 2 | Conversion Core ML + encodeur hybride | `Scripts/CoreMLConversion/*`, `CoreML/*.swift`, ressource `VoxtralEncoderFull.mlmodelc` | `VoxtralPipeline` (défaut `.auto`), FluxForge via `VoxtralTranscriptionManager` | aucun test Core ML | **script de conversion cassé**, chemin de cache incohérent, ANE non mesuré |
| 3 | Recherche clonage (Python) | `Scripts/VoiceCloningResearch/*` | aucun (annexe) | — | amont non épinglé, divergence Swift non documentée |
| 4 | Téléchargement / registres | `Utils/ModelDownloader.swift`, `ModelRegistry.swift`, `TTS/VoxtralTTSRegistry.swift`, `Realtime/VoxtralRealtimeRegistry.swift` | tous | tailles + liens symboliques (#49) | **complétude TTS/Realtime = un seul `.json`**, pas de SHA-256, révision non épinglée, API publique factice |
| 5 | CLI, bench, apps | `VoxtralTranscriptionTest/*`, `VoxtralBenchmark`, `VoxtralApp`, `VoxtralTTSStreamingDemo` | utilisateurs | aucun | profil sans backend/graine, bench hors chemin, code mort mémoire, `FFmpeg` sans lecture asynchrone |
| 6 | RuntimeBeacon | `Utils/RuntimeBeacon.swift` | CLI (`--beacon`), hôtes | `RuntimeBeaconTests` | bon ; course théorique `update`/`end` |
| 7 | Serveur | — | homelab (stt-mcp, selon cadrage) ? | — | **absent** : décision → ASK (§ 4) |
| 8 | Éval / qualité | `Tests/.../TTS*Campaign*`, `TTSBlind*`, sondes | mainteneur | tout est `XCTSkip` sans variable d'env. | **aucune éval WER / parité automatisée** ; pas de CI |

## 2. Constats

Format : id · sévérité · `fichier:ligne` · constat · preuve · correction · risque API · effort · statut ·
fiche proposée (objet + porte chiffrée + cible).

### A-01 · haute · Enrôlement (vjp) et inférence peuvent tourner en même temps dans le même process — deadlock ABBA compile×vjp possible (piège 20)

- **Où** : `TTS/Pipeline/VoxtralTTSPipeline.swift:425-452` (`enrollVoice` ne change jamais `state`, qui reste
  `.ready`) ; gardes d'inférence `:195`, `:309`, `:506` (`guard state.isReady`) ; démo
  `VoxtralTTSStreamingDemo/StreamingDemoViewModel.swift:392-444` (enrôlement en `Task.detached` via une
  `Box: @unchecked Sendable`, l. 406-411) et `:457` (`startStreaming` ne teste pas `isEnrolling`) ;
  `StreamingDemoView.swift:106` (Play désactivé seulement si `!isModelLoaded || isLoading`), `:42`
  (Load actif pendant l'enrôlement → `pipeline?.unload()` `StreamingDemoViewModel.swift:175` sous un
  enrôlement en cours) ; gradient `TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:558`.
- **Constat** : rien n'empêche une synthèse (ou un chargement) pendant les ~30 min d'un enrôlement, ni dans
  la bibliothèque, ni dans la démo. Or la fonction différenciée appelle une fonction **compilée** :
  le FFN du codec fait `w2(silu(w1(x)) * w3(x))` (`TTS/VoxtralCodecDecoder.swift:295-296`) et, dans
  mlx-swift, `silu` **est** un `compile(shapeless: true)` global (`MLXNN/Activations.swift:212-213`,
  `:1049`, identique en 0.31.6). La synthèse appelle le même `silu` (`VoxtralFlowMatching.swift:121`,
  `Models/VoxtralLlama.swift:194`). Le `scan.py` compte `compile` = 0 : c'est un faux négatif.
- **Preuve (amont 0.31.6, résolu par Voxtral)** : `CompiledFunction.call` prend d'abord son verrou
  d'instance (`Transforms+Compile.swift:12` `NSLock`, `:39-43` `lock.withLock { innerCall }`) puis
  `evalLock` (`:89`) ; `valueAndGradient` prend `evalLock` (`Transforms+Internal.swift:19`) et rappelle le
  Swift tracé, qui entre dans `compiledSilu.call` → ordre inverse. Fil A (synthèse) : `L_silu → evalLock` ;
  fil B (enrôlement) : `evalLock → L_silu` ⇒ interblocage. mlx-swift `main` @ `9019419` corrige l'ordre
  (`evalLock` le plus externe, doc `Transforms+Compile.swift:110-139`, `call` `:140`, PR #461 = commit
  `df9ae26`, qui cite l'issue #339 ; `Source/CompileLockRepro`), mais **aucun tag** ne le contient
  (`git tag --contains df9ae26` vide, dernier tag `0.31.6` = `0bb916c`), et mlx-swift-lm impose
  `.upToNextMinor(from: "0.31.6")`. *(Vérif. croisée : `evalLock` est un `NSRecursiveLock`,
  `Transforms+Eval.swift:9` @ 0.31.6, d'où la réentrance du fil B et l'inversion ; la démo atteint le cas par
  l'UI : Play actif pendant l'enrôlement. FluxForge intègre le TTS : réponse du mainteneur à l'issue #11.)*
- **Aggravants** : RNG global partagé — la synthèse fait `MLXRandom.seed(seed)`
  (`TTS/VoxtralTTSModeling.swift:440`, `:582`) pendant que l'enrôlement tire son bruit de Gumbel et son
  initialisation dans le même état global (`VoxtralVoiceEnrollment.swift:428`, `:516-517`) : les deux
  deviennent non reproductibles. Sens inverse (enrôler pendant une synthèse) : refusé par
  `guard state.isReady` (`:433`) mais avec le message trompeur « Model not loaded ».
- **Correction** : (1) sérialiser toute opération GPU du pipeline (verrou/acteur interne) et marquer
  l'enrôlement « occupé » **sans** ajouter de cas à l'`enum State` public (ajout de cas = cassant pour
  les `switch` exhaustifs des consommateurs) : drapeau interne + erreur explicite `busy` ; (2) démo :
  désactiver Play, Load et le sélecteur de modèle pendant `isEnrolling`, bouton Annuler branché sur
  `shouldContinue` ; (3) `withRandomState` pour l'enrôlement (API présente en 0.31.6,
  `State.swift:99`) ; (4) relever `from:` dès qu'un tag mlx-swift > 0.31.6 contient #461 (`df9ae26`).
- **Risque API** : additif (nouvelle erreur, pas de nouveau cas d'enum). **Effort** : M.
- **Statut** : VÉRIFIÉ (préconditions lues des deux côtés) ; reproduction du blocage À MESURER.
- **Fiche** : *Sérialiser enrôlement et inférence (+ démo)* — **porte** : test d'intégration qui lance
  `enrollVoice` (50 époques) puis `synthesizeStreaming` en parallèle : 20/20 exécutions sans blocage
  (timeout 120 s), la synthèse est refusée avec l'erreur `busy` en < 1 s ou s'exécute après ; le test
  doit échouer sans le correctif (piège 38 ; s'inspirer de `CompileLockRepro`) ; suite complète verte.
  **Cible** : macos-gpu.

### A-02 · haute · « Téléchargé » = un seul fichier `.json` (variantes de MLX-012), pas de SHA-256, révision non épinglée

- **Où** : `Utils/ModelDownloader.swift:537-581` (`findTTSModelPath` : présence de `params.json` suffit),
  `:589-592` (retour anticipé de `downloadTTSModel`), `:644-692` + `:699-702` (Realtime : `config.json` ou
  `params.json` suffit), `:312-335` (`verifyShardedModel` : **sans** index ⇒ `complete = true`, même si
  aucun `.safetensors` n'est présent), `:367-372` (`download` : téléchargement incomplet ⇒ simple `print`,
  l'URL est retournée), `:206-210` (`isModelDownloaded` public : `config.json` seul), `:81`
  (`revision: "main"`), `:85` (`TreeEntry` ignore `lfs.oid`), `:115-120` (saut sur la seule taille),
  `:136` (`URLSession.download` : reprise par fichier seulement, pas de `resumeData`).
- **Preuve** : listing Hub du 2026-09-27 de `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` et `-6bit` :
  `config.json`, `model.safetensors` (2,51 / 3,47 Go), `model.safetensors.index.json`, `params.json`,
  `tekken.json`, `voice_embedding/`. Une coupure pendant `tekken.json` ou `voice_embedding/*` (après
  `params.json`) laisse un modèle « téléchargé » pour toujours : les voix manquantes sont sautées en
  silence (`TTS/Pipeline/VoxtralTTSPipeline.swift:166-175`), le tokenizer manque. Le détecteur MLX-012
  (`contains { $0.hasSuffix(".safetensors") }`) n'en voit aucune : 0 occurrence dans `patterns-scan.md`.
  Aucun `sha256` / `CryptoKit` dans `Sources/` (grep).
- **Correction** : manifeste de complétude écrit en fin de `downloadRepoDirect` (liste, tailles,
  `lfs.oid` SHA-256 de l'API `tree`) + vérification SHA-256 en flux des fichiers LFS ; `find*Path`
  exigent le manifeste (ou, pour les téléchargements existants, l'index + `tekken.json` + voix, puis
  écrivent le manifeste) ; `download` lève une erreur si incomplet ; champ `revision` (commit) optionnel
  dans les trois registres ; reprise d'octets via `resumeData`.
- **Risque API** : additif (champs optionnels, comportement plus strict). **Effort** : M. **Statut** : VÉRIFIÉ.
- **Fiche** : *Complétude + SHA-256 des téléchargements* — **porte** : tests unitaires (dossiers
  temporaires) : `params.json` seul ⇒ non téléchargé puis reprise effective ; shard manquant ⇒ non
  téléchargé ; SHA incorrect ⇒ rejet ; modèle non shardé sans `.safetensors` ⇒ non téléchargé ;
  `ModelDownloaderSizeTests` et `ModelLoadingSymlinkedDirectoryTests` restent verts. **Cible** : macos-gpu
  (toolchain Swift requise).

### A-03 · moyenne · La conversion Core ML n'est pas reproductible depuis le dépôt (script cassé, README faux, dépendances non épinglées)

- **Où** : `Scripts/CoreMLConversion/convert.sh:74` (`--include-projector`), `:47` (`huggingface-cli`),
  `README.md:54`, `:77` (`--model-path`), `convert_to_coreml_ane.py:231-239` (argparse : `--weights`
  requis, pas de `--include-projector`), `:314-316` (renvoie vers `VoxtralCLI benchmark-coreml`, qui
  n'existe pas : sous-commandes `VoxtralCLI.swift:26-35`), `requirements.txt` (`torch>=2.0.0`,
  `coremltools>=7.0`, `huggingface_hub>=0.20.0`), `README.md:110-116` (« One Core ML model works with both
  Mini and Small ») contredit par les variantes 3072/5120 (`voxtral_encoder_ane.py:50-57`,
  `CoreML/VoxtralCoreMLEncoder.swift:52-62`, dépôts HF `voxtral-encoder-coreml-mini|small`).
- **Preuve (exécutée ici)** : les définitions `add_argument` des scripts, rejouées sans torch :
  `convert.sh` étape 5 ⇒ « arguments non reconnus ['--include-projector'] » (argparse sort en code 2,
  `set -e` arrête le script) ; README étape 3 ⇒ « the following arguments are required: --weights ».
  PyPI au 2026-09-27 : `huggingface_hub` 2.0.0 n'expose plus que la commande `hf` (pas de
  `huggingface-cli`) ⇒ l'étape 3 échoue aussi sur une machine neuve ; `torch` 2.14.0 alors que
  `coremltools` 9.0 déclare `_TORCH_MAX_VERSION = "2.7.0"`. Aucune révision HF épinglée
  (`convert.sh:47`, `convert_weights.py` `snapshot_download` sans `revision`).
- **Correction** : retirer le drapeau (le modèle ANE inclut toujours le projecteur :
  `ANEVoxtralEncoderWithProjector`), passer `--variant` ; README aligné ; `hf download --revision <sha>` ;
  verrou de dépendances (`coremltools==9.0`, `torch==2.7.*`, …) ; versions et révision écrites dans les
  métadonnées du modèle ; SHA-256 du `weight.bin` produit publié à côté du `.mlmodelc`.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : *Rendre la conversion Core ML reproductible* — **porte cloud** : contrôle argparse
  (script d'extraction AST) = 0 argument inconnu / 0 requis manquant pour `convert.sh` et le README ;
  `requirements.txt` sans `>=`. **Porte Mac** : `convert.sh` va au bout, puis parité embeddings Core ML
  (fp16) vs MLX : erreur relative L2 ≤ 1e-2 sur 3 fichiers réels. **Cible** : cloud (scripts/doc) puis
  macos-gpu (validation).

### A-04 · moyenne · Encodeur Core ML : cache jamais trouvé, hors ligne impossible, `customModelsDirectory` contourné, double stockage possible

- **Où** : `CoreML/VoxtralCoreMLEncoder.swift:456-459` (cache vérifié sous
  `modelsDirectory/VincentGOURBIN--voxtral-encoder-coreml-mini/…`), `:478-481`
  (`HubApi(downloadBase: modelsDirectory.deletingLastPathComponent(), useOfflineMode: false)`, `cache`
  par défaut), `:464-470` (complétude = `model.mil` **ou** dossier `weights`) ; swift-transformers 1.3.4
  `HubApi.swift:618-619` (`localRepoLocation = downloadBase/models/<org>/<repo>`), `:903-975` (hors ligne
  seulement si `useOfflineMode`, sinon `getFilenames` réseau), `:148` + `:83-90` (`cache: .default` =
  HubCache partagé) ; `Utils/ModelDownloader.swift:54-61` (le reste du dépôt passe `cache: nil` exprès) ;
  repli silencieux `Pipeline/VoxtralPipeline.swift:290-295` (log `VoxtralDebug` seulement).
- **Constat** : (a) le chemin testé (`org--repo`) n'est jamais celui où HubApi écrit (`org/repo`) ⇒
  chaque chargement hybride interroge le réseau ; hors ligne, `snapshot` lève et le pipeline retombe sur
  MLX sans le dire ; (b) le commentaire de #49 suppose que `modelsDirectory` finit par `models` : avec
  `customModelsDirectory = …/VoxtralModels`, l'encodeur (1,32 Go pour mini, 1,38 Go pour small, listing
  Hub) atterrit dans `…/models/…`, **hors** de la racine choisie par l'app (demande App Store #1/#49) ;
  (c) `HubCache.default` peut dupliquer le `weight.bin` (À MESURER) ; (d) un `weight.bin` absent passe la
  vérification si `model.mil` existe. Chemin emprunté par FluxForge : `VoxtralTranscriptionManager` crée
  le pipeline en `.auto` (`Pipeline/VoxtralTranscriptionManager.swift:90-94`).
- **Amendement (vérif. croisée)** : (d) est aujourd'hui **latent**. Le test `:462-470` ne porte que sur le
  chemin `org--repo`, que rien n'écrit, donc il ne s'exécute jamais sur un dossier réel ; (d) ne deviendra
  actif qu'une fois (a) corrigé, et la correction le couvre (manifeste). (b) est conditionnel : le
  contournement de l'issue #2 suppose un dossier nommé `models` à la casse près. FluxForge passe
  `…/FluxforgeStudio/Models/` (issue #2) et n'est donc pas touché sur un volume APFS insensible à la casse
  (le cas par défaut). Sont touchés : un volume sensible à la casse, et tout autre nom de dossier.
  (a) est confirmé : `HubApi(… useOfflineMode: false)` (`:478-481`) interdit le mode hors ligne
  (`HubApi.swift:904` @ 1.3.4), donc `getFilenames` passe par le réseau à chaque chargement hybride.
- **Correction** : télécharger via `ModelDownloader.downloadRepoDirect(repoId:, matching: ["<name>/*",
  "<name>/*/*"])` (le glob maison ne gère pas `**`, `ModelDownloader.swift:188-194`) ⇒
  `modelsDirectory/<org>/<repo>/<name>`, même manifeste de complétude que A-02, cache vérifié sur ce
  chemin, pas de HubCache ; journaliser le repli MLX dans `encoderStatus`.
- **Risque API** : additif. **Effort** : S-M. **Statut** : VÉRIFIÉ (lecture des deux côtés) ; duplication
  disque et comportement hors ligne À MESURER.
- **Fiche** : *Chemin unique pour l'encodeur Core ML* — **porte** : test avec
  `customModelsDirectory = <tmp>/VoxtralModels` ⇒ encodeur trouvé sous ce dossier ; 2e chargement
  hybride réussi réseau coupé (`Core ML available: true`) ; 0 octet écrit sous `~/.cache/huggingface`.
  **Cible** : macos-gpu.

### A-05 · moyenne · API de téléchargement publique factice ou morte

- **Où** : `Utils/VoxtralModelLoading.swift:15-46` (`public func downloadModel(modelId:revision:)`) et
  `:53-63` (`downloadFromHuggingFaceHub` ne fait qu'imprimer) ; `:149-154` (`loadVoxtralModel(modelPath:)`
  y tombe si le chemin n'existe pas) ; `Utils/ModelDownloader.swift:30-68` (`public static var hubApi`,
  `reconfigureHubApi()` : aucun appelant, aucun chemin de téléchargement ne les utilise — grep).
- **Constat** : `downloadModel` crée un dossier **vide** `Documents/models/<id>` et ne télécharge rien ; les
  appels suivants le voient et sautent ; `loadVoxtralModel("org/repo")` échoue ensuite sur « Config file
  not found ». `reconfigureHubApi()` est documentée « à appeler après avoir changé
  `customModelsDirectory` » mais n'a aucun effet sur les téléchargements (qui passent par `URLSession`).
- **Correction** : `@available(*, deprecated, message: "use ModelDownloader.resolveModel")`, et lever
  une erreur explicite au lieu de créer un dossier vide ; idem dépréciation de `hubApi` /
  `reconfigureHubApi`. Suppression = cassant ⇒ ASK (consommateurs : FluxForge ?).
- **Risque API** : additif (dépréciation) / cassant (suppression). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : rattachée à la fiche A-02 — **porte** : test « `downloadModel` sur id inconnu localement ⇒
  erreur, aucun dossier créé » ; build sans nouvel avertissement hors dépréciations voulues. **Cible** :
  macos-gpu.

### A-06 · basse (amendé, était moyenne) · Enrôlement : jusqu'à 3,8 G paramètres inutilisés **possiblement** résidents, aucune politique mémoire

- **Où** : l'enrôlement n'utilise que le codec et la table des codes audio
  (`VoxtralVoiceEnrollment.swift:104-106`, `:447`, `:512`, `:683`) mais exige le modèle TTS entier chargé
  (`VoxtralTTSPipeline.swift:433`) : 26 couches du LLM, transformeur acoustique (3 couches) et
  `tok_embeddings` (`TTS/VoxtralTTSModeling.swift:258-262`). Aucun `Memory.cacheLimit` / `clearCache`
  dans `VoiceCloning/` ni dans `enrollVoice` (grep) ; le cache MLX reste plein après l'enrôlement.
- **Estimation (à partir de `config.json` du pack 6-bit)** : couche = 3072×4096 + 2×3072×1024 + 4096×3072
  + 3×3072×9216 ≈ 116 M ; 26 couches ≈ 3,03 G ; acoustique ≈ 0,35 G ; embeddings 131 072×3 072 ≈ 0,40 G
  ⇒ ≈ 3,8 G paramètres inutiles, soit ≈ 7,6 Go en bf16 et ≈ 3 Go en 6 bits. **Estimation, pas une
  mesure** ; le pic réel de l'enrôlement n'est documenté nulle part (À MESURER).
- **Amendement (vérif. croisée, piège 18)** : la **résidence** de ces poids n'est pas établie.
  `loadVoxtralTTSModel` fait `MLX.loadArrays` puis `model.update(parameters:)` **sans `eval`**
  (`TTS/VoxtralTTSModelLoading.swift:70-74`), et `VoxtralTTSPipeline.loadModel` n'évalue rien non plus
  (`:125-185`). Les poids du LLM restent donc paresseux tant qu'aucune synthèse n'a tourné. Or le chemin
  CLI `voxtral enroll` charge puis enrôle directement (`VoxtralCLI.swift:646-662`) : les 26 couches n'y sont
  probablement **jamais matérialisées**, et le gain « ≈ −50 % » n'y est pas attendu. La résidence est
  plausible seulement quand une synthèse a précédé l'enrôlement dans le même pipeline (démo, FluxForge).
  Ce qui reste VÉRIFIÉ : aucun `cacheLimit` ni `clearCache` dans `VoiceCloning/` ni dans `enrollVoice`.
  Sévérité ramenée à basse.
- **Correction** : profil `enroll-lean` (§ 5) : chargement du seul sous-ensemble codec + table audio, ou
  libération des poids du LLM pendant l'enrôlement (T4, résidence par étape) ; `Memory.cacheLimit` borné
  pendant la boucle (formes constantes : un petit cache devrait suffire, T1/T2) ; `Memory.clearCache()`
  en fin d'`enrollVoice` (T3).
- **Risque API** : additif. **Effort** : M. **Statut** : À MESURER (absence de politique : VÉRIFIÉ).
- **Fiche** : *enroll-lean : résidence + cache* — **porte (amendée)** : étape 0 : mesurer le pic
  `phys_footprint` de l'enrôlement dans deux scénarios, (i) `voxtral enroll` (chargement paresseux) et
  (ii) synthèse puis enrôlement dans le même pipeline. Si (ii) − (i) < 5 %, le volet « résidence » est
  retiré sans code et la fiche se limite à `cacheLimit`/`clearCache`. Sinon, dans le scénario (ii) :
  `enroll-lean` −40 % minimum contre `enroll-fast`, temps/époque ±5 % (A/B/B/A), codes finaux identiques
  à graine fixe (A-08) ou perte finale ±1 %. Sinon code retiré. **Cible** : macos-gpu.

### A-07 · moyenne · Pas de reprise ni de porte GPU ; l'annulation existe mais aucun hôte ne l'expose

- **Où** : `VoxtralVoiceEnrollment.swift:495-626` (aucun point de contrôle) ; durée annoncée ≈ 30 min
  pour 5000 époques (`VoxtralTTSPipeline.swift:418`, `VoxtralCLI.swift:576`, non mesurée en banc) ;
  CLI : pas de `shouldContinue` (`VoxtralCLI.swift:654-662`) ; démo : pas de `shouldContinue`
  (`StreamingDemoViewModel.swift:417`) ni de bouton Annuler (`StreamingDemoView.swift:238-242`) ;
  VoxtralCore cible iOS 17 (`Package.swift:12`, PR #32) et `VoxtralVoiceEnrollment` est
  `@available(macOS 14.0, *)` donc disponible sur iOS, sans garde d'arrière-plan (piège 22).
- **Amendement (vérif. croisée)** : la ligne est `Package.swift:12` (`.iOS(.v17)`) et non `:11`
  (`.macOS(.v15)`). Pour la CLI, l'absence de `shouldContinue` n'est pas un défaut : Ctrl-C tue le
  processus, et rien de partiel n'est écrit puisque la sauvegarde n'a lieu qu'à la fin
  (`VoxtralTTSPipeline.swift:450`). Le manque réel est l'absence de point de contrôle (30 min perdues) et
  de bouton Annuler dans la démo. « Annulation exposée : SIGINT CLI » est retiré de la correction.
- **Correction** : `Config.checkpointURL` + `checkpointEvery` (paramètres, moments d'Adam, température,
  époque, meilleur instantané, état RNG via `RandomState`) ⇒ reprise bit-exacte (T23) ; annulation
  exposée dans la démo (bouton branché sur `shouldContinue`) ; sur iOS : refuser ou exiger
  `shouldContinue` (ASK Q2).
- **Risque API** : additif. **Effort** : M. **Statut** : VÉRIFIÉ (absences) ; bit-exactitude À MESURER.
- **Fiche** : *Checkpoint et reprise de l'enrôlement* — **porte** : arrêt à 2500/5000 puis reprise ⇒ codes
  finaux identiques bit à bit au run continu (même graine) ; surcoût de sauvegarde < 1 % du temps total.
  **Cible** : macos-gpu.

### A-08 · moyenne · Enrôlement non reproductible (aucune graine)

- **Où** : `VoxtralVoiceEnrollment.Config` (`:28-79`, pas de graine) ; CLI `Enroll` (`VoxtralCLI.swift:579-609`,
  pas de `--seed`) ; RNG global `:428`, `:516-517`.
- **Constat** : deux enrôlements identiques donnent des codes différents ; impossible de comparer deux
  réglages (ou deux profils) à graine égale, alors que la synthèse, elle, a une graine (`--seed`).
- **Correction** : `Config.seed: UInt64?` + `withRandomState(MLXRandom.RandomState(seed:))` autour de
  `optimizeCore` (API disponible en 0.31.6) ; `--seed` en CLI ; graine écrite dans les métadonnées du
  `.safetensors`.
- **Risque API** : additif. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : rattachée à A-07 — **porte** : 2 enrôlements (200 époques) même graine ⇒ codes [T, 37]
  identiques ; graines différentes ⇒ codes différents. **Cible** : macos-gpu.

### A-09 · basse · Surcharge publique non levante de `optimize` et garde NaN non testée

- **Où** : `VoxtralVoiceEnrollment.swift:464-469` (ignore `failed` ; *vérif. croisée* : `cancelled` n'y est
  pas atteignable puisque `shouldContinue: nil`, seul `failed` est perdu) ; `:536-537` (le meilleur
  instantané est initialisé aux paramètres aléatoires) ; appelants : sondes
  `TTSFundamentalDeficitProbeTests.swift:70`, `TTSReEnrollExperimentTests.swift:57`. Aucun test ne couvre
  `EnrollmentDivergedError`, `shouldContinue` ou le repli (grep dans `Tests/`).
- **Constat** : si la première perte est non finie, cette surcharge rend des codes aléatoires sans
  erreur. La garde de #44 (`:562-576`, `enrollVoice :444-449`) est correcte mais non protégée par un test.
- **Correction** : déprécier la surcharge non levante ; rendre la boucle testable (perte injectable ou
  mini-modèle à poids aléatoires, graine fixée, piège 40) et tester divergence / annulation / repli.
- **Risque API** : additif. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : rattachée à A-07 — **porte** : 3 tests (NaN à l'époque 0 ⇒ `EnrollmentDivergedError` ; NaN
  à l'époque k ⇒ codes du meilleur pas ; annulation ⇒ `CancellationError`) rouges sans la garde (revert
  local), verts avec. **Cible** : macos-gpu.

### A-10 · basse · Deux synchronisations GPU→CPU par époque

- **Où** : `VoxtralVoiceEnrollment.swift:566` (`values[0].item`) puis `:601` (`MLX.eval` des paramètres) ;
  `:613` en plus aux pas de log.
- **Constat** : la perte est lue avant la mise à jour, ce qui vide la file GPU une fois de plus par époque
  (5000 fois). La garde NaN pourrait lire la perte **après** l'`eval` commun et revenir à l'instantané.
- **Correction** : un seul `eval(total, params…)` par époque, test de finitude ensuite (T15 par analogie).
- **Risque API** : aucun. **Effort** : S. **Statut** : À MESURER (gain attendu faible).
- **Amendement (vérif. croisée)** : les deux synchronisations sont VÉRIFIÉES. Le **gain attendu**, lui,
  est chiffré par ordre de grandeur : le graphe arrière existe déjà à l'appel `valueAndGrad` (`:558`), si
  bien que la sync supplémentaire ne coûte que le trou GPU entre `.item` (`:566`) et `eval` (`:601`),
  c'est-à-dire la construction du graphe clip + Adam, soit au plus ≈ 1 ms. Rapporté à ≈ 360 ms/époque
  (30 min / 5000, valeur annoncée et non mesurée), cela donne **< 1 %**, un ordre de grandeur sous le
  seuil de 5 % (piège 25). Le constat devient une **note sans fiche propre** : on ne le mesure qu'en
  passager de la série A/B/B/A d'A-06, et on le retire sans code si l'écart reste sous 5 %. La correction
  doit aussi préserver la sémantique de la garde : l'instantané « meilleur » reste celui d'**avant** la mise
  à jour.
- **Fiche** : aucune fiche propre (amendé) ; passager d'A-06 — **porte** : temps/époque −5 % minimum
  (A/B/B/A), codes identiques à graine fixe ; sinon retirée. **Cible** : macos-gpu.

### A-11 · moyenne · Réglages d'enrôlement incohérents selon le point d'entrée (profil manquant)

- **Où** : bibliothèque `Config()` : 100 trames (8 s), 5000 époques (`VoxtralVoiceEnrollment.swift:29-30`) ;
  CLI : 16 s, 5000 (`VoxtralCLI.swift:590-594`) ; démo : 16 s, **3000** (`StreamingDemoViewModel.swift:31-32`) ;
  recherche Python : 8 s, 5000 (`enroll_voice.py:113-114`).
- **Preuve (mesures existantes)** : similarité ECAPA à 2000 époques : 4 s 0,67 · 8 s 0,69 · **16 s 0,72** ·
  24 s 0,72 (`docs/voice_cloning.md:36-45`). Un consommateur qui appelle `enrollVoice(config: .init())`
  obtient la configuration 8 s.
- **Correction** : type de profil `enroll-fast|lean` (§ 5) utilisé par CLI, démo et bibliothèque ; changer
  le défaut de `Config` change la sortie des consommateurs ⇒ ASK Q5.
- **Risque API** : additif (profil) / comportement (défaut). **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : *Profils d'enrôlement* — **porte** : `voxtral enroll --reference-profile <id>` et démo
  utilisent la même table ; mesure des deux profils (temps, pic, similarité) consignée. **Cible** :
  macos-gpu.

### A-12 · moyenne · Hybride Core ML : gain ANE annoncé non mesuré, unités de calcul contradictoires, aucune parité, 1,3 Go téléchargés par défaut

- **Où** : `Scripts/CoreMLConversion/README.md:33` (« ~150 ms ANE vs ~500 ms MLX ») ; défaut
  `.cpuAndGPU` « ~280 ms » (`CoreML/VoxtralCoreMLEncoder.swift:118-126`) mais valeur par défaut de l'`init`
  `.cpuAndNeuralEngine` (`:164-172`) ; conversion en `ComputeUnit.ALL`, cible iOS16
  (`convert_to_coreml_ane.py:194-210`) ; « test » = PyTorch vs PyTorch-ANE sur entrée aléatoire,
  simple avertissement au-delà de 1,0 (`:290-295`) ; backend par défaut `.auto`
  (`Pipeline/VoxtralPipeline.swift:196`).
- **Mesures existantes** : premier lancement « Encoder Setup » 1 min 09,6 s (71 % du temps STT) contre
  1,41 s à chaud (issue #16) ; encodage Core ML à 48 % GPU (issue #14). Aucune comparaison A/B/B/A
  encodeur MLX vs Core ML dans le dépôt, aucune parité des embeddings.
- **Amendement (vérif. croisée)** : ces chiffres sont des mesures **« en session »** au sens de
  `measurement.md` (traces du profileur d'avril 2026, révision antérieure, sans A/B/B/A ni
  refroidissement), pas des références. Le mainteneur a clos #16 comme intrinsèque à Core ML (« once per GPU
  architecture and is cached »). L'issue #22 ajoute 2 min 25 s pour Small. Le coût à froid ne se paie
  donc qu'une fois par architecture GPU, ce qu'il faut documenter plutôt que présenter comme un coût à
  chaque lancement.
- **Correction** : fiche de mesure (encodeur MLX, Core ML `.cpuAndGPU`, `.cpuAndNeuralEngine`, `.all`, à
  froid et à chaud) + parité ; règle R14 (Core ML > 2× MLX ou gain < 5 % ⇒ pas d'asset, `.mlx` par défaut) ;
  si gardé : reconversion `coremltools` 9 avec cible iOS18/macOS15 (SDPA fusionné, compression des poids
  8 bits par bloc) — gain attendu, À MESURER.
- **Risque API** : changement de défaut = comportement ⇒ ASK Q3. **Effort** : M. **Statut** : À MESURER.
- **Fiche** : *Décider l'hybride par la mesure* — **porte** : Core ML gardé seulement si ≥ 5 % plus rapide
  à chaud **et** parité (erreur rel. L2 embeddings ≤ 1e-2, transcriptions greedy identiques sur 5 fichiers)
  **et** coût du premier lancement documenté ; sinon `.mlx` par défaut. **Cible** : macos-gpu.

### A-13 · basse · Encodeur hybride : sorties non vérifiées, repli silencieux sur des poids non initialisés

- **Où** : noms « legacy » `VoxtralEncoderFull.mlmodelc` (mini, 3072) cherchés quelle que soit la variante
  (`VoxtralCoreMLEncoder.swift:236-248`) ; `encode` ne compare pas la forme de sortie à
  `config.outputShape` (`:348-374`) ; `createHybridEncoder(preferredBackend: .auto)` public sur un Small
  (5120) dans une app qui embarque `VoxtralEncoderFull` (VoxtralApp, `Package.swift:79-80`) ⇒ encodeur mini
  choisi (`VoxtralHybridEncoder.swift:470-507`). `encodeMLX` tourne sur des poids **non initialisés** avec un
  simple log (`:289-300`) et projecteur vide (`:275-277`). Le pont ignore `strides`
  (`MLXCoreMLBridge.swift:161-210`) et la conversion fp16 tronque et met les sous-normaux à zéro
  (`:22-44`). Chemins DEBUG codés en dur `/Users/vincent/…` (`VoxtralCoreMLEncoder.swift:305-311`).
- **Correction** : lever une erreur si la forme de sortie ne correspond pas ou si aucun poids n'est chargé ;
  noms legacy réservés à `.mini` ; respecter `strides` ; `Float16(x)` natif.
- **Risque API** : additif (erreurs là où le résultat était faux). **Effort** : S. **Statut** : VÉRIFIÉ en
  lecture (le chemin par défaut du pipeline passe `.mlx` au repli, donc non atteint par défaut).
- **Amendement (vérif. croisée)** : la ligne de ressource est `Package.swift:79-80` et non `:87-89`, où se
  trouve la cible `VoxtralTranscriptionTest`. La raison du non-atteint est corrigée. Les poids non
  initialisés ne sont pas évités par `.mlx` mais par `standardModel`, que le pipeline renseigne toujours
  (`VoxtralPipeline.swift:237-241` → `VoxtralModeling.swift:576`, puis `setMLXEncoderFromStandard`
  `VoxtralHybridEncoder.swift:499-503`). Le cas « poids non initialisés » n'arrive qu'à un consommateur qui
  construit `VoxtralForConditionalGeneration(config:)` (`VoxtralModeling.swift:456`) sans Core ML.
  « Small + mini » demande un appel direct à `createHybridEncoder(.auto|.coreML)`.
  `strides` ignorés et fp16 tronqué (le double décalage `>> 13` à `MLXCoreMLBridge.swift:33-34` met bien
  les sous-normaux à zéro) sont **latents** en production : entrée mel float32 et sortie float32
  contiguë [1, 375, 3072|5120]. Observation ajoutée, VÉRIFIÉE en lecture : au repli
  (`VoxtralPipeline.swift:293-295`), `VoxtralHybridEncoder.init` tente quand même l'auto-découverte
  (`VoxtralHybridEncoder.swift:120-128`). Dans VoxtralApp, qui fournit `Bundle.module`, le
  `VoxtralEncoderFull.mlmodelc` embarqué (≈ 1,3 Go) est donc **chargé pour rien** alors que `.mlx` est
  demandé ; son coût est À MESURER.
- **Fiche** : rattachée à A-12 — **porte** : tests « Small + encodeur mini ⇒ erreur », « encodeur non
  initialisé ⇒ erreur ». **Cible** : macos-gpu.

### A-14 · moyenne · Ressource `VoxtralEncoderFull.mlmodelc` déclarée mais ignorée par git ; empaquetage de l'app en Debug

- **Où** : `Package.swift:79-80` (`.copy("Resources/VoxtralEncoderFull.mlmodelc")` ; *amendé en
  vérif. croisée, le rapport citait `:87-89`*), `.gitignore:41-46`
  (ignorée), dossier absent du clone (`Sources/VoxtralApp/Resources/` ne contient que `Info.plist`) ;
  `VoxtralApp/VoxtralAppMain.swift:19` utilise `Bundle.module` ; `create_app_bundle.sh` copie
  l'exécutable **Debug** de `swift build` sans bundle de ressources ni `metallib`, alors que le mainteneur
  recommande `xcodebuild` pour Metal (réponse à l'issue #11).
- **Constat** : sur un clone neuf, le build de `VoxtralApp` dépend de la façon dont SwiftPM traite une
  ressource absente (avertissement « Invalid Resource … File not found » attendu, puis `Bundle.module`
  généré ou non) : À VÉRIFIER. L'app empaquetée par le script ne peut fonctionner que sur la machine de
  build.
- **Correction** : ne plus embarquer le `.mlmodelc` (téléchargement à l'exécution déjà supporté, A-04), ou
  le rendre conditionnel ; script d'empaquetage via `xcodebuild -configuration Release` + copie des bundles.
- **Risque API** : aucun (cibles exécutables). **Effort** : S. **Statut** : À MESURER (build).
- **Fiche** : *Build de VoxtralApp depuis un clone neuf* — **porte** : `xcodebuild -scheme VoxtralApp
  -configuration Release` réussit sur un clone neuf ; l'app empaquetée démarre et transcrit sur une
  autre machine. **Cible** : macos-gpu.

### A-15 · basse · `profile` et CLI : backend et graine non figés, pas d'enrôlement, pas de ligne JSON

- **Où** : `VoxtralTranscriptionTest/ProfileCommand.swift:177`, `:210` (pipeline STT/chat en backend
  `.auto` par défaut, non noté dans les métadonnées ; *amendé en vérif. croisée : le rapport citait
  `:175`/`:208`, qui sont les lignes de température*). S'y ajoute `:208` : le chat est profilé à
  `temperature 0.7` sans graine, donc la longueur de sortie varie d'un run à l'autre, ce qui rend l'A/A
  impossible ; `:256-259` (synthèse TTS sans graine ⇒ nombre de
  trames variable, cf. `docs/voice_cloning.md` section `--seed`) ; `:23-28` (pas de pipeline `enroll`) ;
  `VoxtralCLI.swift:210`, `:321` (`--backend` n'accepte que `mlx`/`hybrid`, alors que le README l. 266
  appelle « Auto mode (recommended) » un exemple `--backend hybrid`).
- **Correction** : `--backend`, `--seed`, pipeline `enroll`, une ligne JSON par mesure avec la révision
  résolue de mlx-swift-lm (branche mouvante, piège 21) — prérequis de toute baseline.
- **Risque API** : additif. **Effort** : S-M. **Statut** : VÉRIFIÉ.
- **Fiche** : *Instrument de mesure* — **porte** : A/A sur la même commande : dispersion ≤ 3 % (piège 33 :
  l'instrument se valide avant la baseline). **Cible** : macos-gpu.

### A-16 · basse · `VoxtralBenchmark` mesure un chemin que personne n'exécute (piège 33)

- **Où** : `VoxtralBenchmark/BenchmarkCLI.swift:1-8`, `:187-241` (copie « same as MLXCoreMLBridge » de la
  conversion fp16, sur données aléatoires) ; le modèle Core ML est déclaré en float32 en entrée et en
  sortie (`convert_to_coreml_ane.py:196-205`), donc le chemin fp16 n'est pas celui de la production.
- **Correction** : réorienter vers un vrai banc (encodeur MLX vs Core ML, TTS, enrôlement) ou retirer le
  produit (ASK Q6).
- **Risque API** : cassant si le produit exécutable est retiré. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : fusionnée avec A-15. **Cible** : macos-gpu.

### A-17 · basse · `Memory.cacheLimit = Int.max` présenté comme le « défaut » (code mort)

- **Où** : `VoxtralApp/TranscriptionManager.swift:285-297` ; aucun appelant de `aggressiveMemoryCleanup`
  (grep).
- **Preuve** : mlx-swift 0.31.6 `Memory.swift:232` : « The cache limit defaults to the memory limit » ;
  `:281` : la limite mémoire vaut 1,5 × le working set recommandé. `Int.max` supprimerait la borne.
- **Correction** : supprimer la fonction, ou sauver/restaurer la valeur précédente de `Memory.cacheLimit`.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : *Hygiène des annexes* — **porte** : build sans avertissement nouveau. **Cible** : macos-gpu.

### A-18 · basse · `FFmpeg.swift` : tubes lus seulement à la fin, pas d'annulation, fichiers temporaires jamais nettoyés

- **Où** : `VoxtralTTSStreamingDemo/FFmpeg.swift:76-95` (lecture de stdout/stderr dans
  `terminationHandler` : un processus qui écrit plus que le tampon du tube se bloque et ne termine jamais ;
  atténué par `-loglevel error`) ; aucune `terminate()` sur annulation ; fichiers `mic_*`, `part_*`,
  `reference_*` jamais supprimés (`StreamingDemoViewModel.swift:64-68`, `:235`, `:349`, `:353`).
  Non-constat : arguments passés en tableau (pas de shell) et liste `concat` construite sur des chemins
  internes (UUID) — pas d'injection.
- **Correction** : `readabilityHandler` ou sortie vers fichier ; `proc.terminate()` sur annulation ;
  nettoyage à la fin de l'enrôlement.
- **Risque API** : aucun (démo). **Effort** : S. **Statut** : VÉRIFIÉ en lecture (blocage : À MESURER).
- **Fiche** : *Hygiène des annexes*. **Cible** : macos-gpu.

### A-19 · basse · Nom de voix clonée non assaini dans la démo

- **Où** : `StreamingDemoViewModel.swift:395-403` (`cloneName` → `appendingPathComponent("\(name).safetensors")`).
- **Constat** : un nom avec `/` ou `..` écrit hors de `VoxtralClonedVoices` ; une voix existante est écrasée
  sans avertissement.
- **Amendement (vérif. croisée)** : seul `..` sort du dossier (`../x` → `<base>/x.safetensors`). Un `/`
  seul (`a/b`) vise un sous-dossier inexistant : `MLX.save` échoue **après** les ~30 min d'enrôlement, et
  ce travail est perdu. Le nom est saisi par l'utilisateur local : il n'y a pas d'enjeu de sécurité, c'est
  de la robustesse (sévérité basse maintenue). La validation doit se faire **avant** de lancer
  l'enrôlement.
- **Correction** : filtrer les caractères, refuser l'écrasement sans confirmation.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ. **Fiche** : *Hygiène des annexes*. **Cible** : macos-gpu.

### A-20 · basse · `RuntimeBeacon` : `update` concurrent d'`end` peut recréer le manifeste

- **Où** : `Utils/RuntimeBeacon.swift:185-195` (le verrou est relâché avant `write()`), `:198-207`, `:210-214`.
- **Constat** : l'API se dit « safe from any thread » ; si `end()` s'exécute entre le test `stillLive` et
  l'écriture d'un autre fil, le fichier réapparaît jusqu'à la mort du processus (ramassage seulement sur
  pid mort, `:105-115`). Les appelants actuels font `update` et `end` sur la même tâche : non observé.
- **Correction** : écrire sous le verrou, ou retester `ended` après l'écriture et supprimer.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ en lecture.
- **Fiche** : *Hygiène des annexes* — **porte** : test de course (1000 `update` concurrents d'un `end`) ⇒
  0 manifeste résiduel. **Cible** : macos-gpu.

### A-21 · basse · Annexe recherche Python non reproductible ; divergence du portage Swift non documentée

- **Où** : `Scripts/VoiceCloningResearch/README.md` (Setup : `git clone` de l'amont sans commit),
  `patches/upstream_fixes.patch` (épinglé à des blobs `index 50b812c..4d00f28`), `enroll_voice.py:47-65`
  (contrôle d'une chaîne, pas d'un commit), `requirements.txt` (non versionné sauf `torch>=2.12`),
  `enroll_voice.py:167` (`torch.load(..., weights_only=False)`), `:142` (`--speaker-weight 0.5`) vs pertes
  Swift L1 + STFT multi-résolution + mel seulement (`VoxtralVoiceEnrollment.swift:548-553`) alors que
  l'en-tête Swift n'annonce que deux différences (`:10-14`). Les similarités ECAPA de la doc n'ont aucun
  script dans le dépôt (grep `ecapa|speechbrain` vide hors `requirements`).
- **Amendement (vérif. croisée)** : il faut nuancer « divergence non documentée ». Les deux jeux de pertes
  sont bien décrits, mais séparément : Swift « L1 + multi-resolution STFT + log-mel »
  (`docs/voice_cloning.md:252-254`), Python « multi-resolution STFT + mel + MFCC + speaker losses »
  (`Scripts/VoiceCloningResearch/README.md:18`). Le défaut réel est double : aucun document ne les confronte,
  et l'en-tête Swift (`:10-14`) annonce « Two deliberate differences » sans les pertes retirées. La
  comparaison « le portage fait mieux : 0,72 contre 0,56-0,59 » est **écartée** car confondue : 0,72 est
  mesuré à 16 s / 2000 époques (`voice_cloning.md:36-45`), 0,56-0,59 à 100 trames = 8 s
  (README recherche l. 72-74, où la référence Swift à 8 s donne 0,69). `torch.load(weights_only=False)`
  (`:167`) ne charge qu'un fichier produit localement par le script amont : c'est de l'hygiène, pas une
  vulnérabilité exploitable.
- **Correction** : épingler le commit amont, verrou pip, `weights_only=True`, confronter explicitement
  les deux jeux de pertes (en-tête Swift + doc), sans conclusion de supériorité tant qu'une mesure à
  conditions égales manque (même référence, même durée, mêmes époques), et ajouter le script de similarité.
- **Risque API** : aucun. **Effort** : S. **Statut** : VÉRIFIÉ.
- **Fiche** : *Reproductibilité de l'annexe recherche* — **porte** : README et scripts sans dépendance non
  épinglée ; commande de similarité documentée. **Cible** : cloud.

### A-22 · moyenne · Aucune éval WER ni parité automatisée ; la campagne existante n'affirme rien

- **Où** : `Tests/.../TTS/TTSQuantizationCampaignTests.swift:60-149` (sautée sans `VOXTRAL_TTS_CAMPAIGN=1`,
  imprime la couverture de mots sans `XCTAssert`, 3 phrases françaises, ASR `VoxtralPipeline(model:
  .mini3b8bit)` en backend `.auto` `:112`) ; `TTSBlindValidationTests` = écoute humaine ; aucune donnée audio
  dans `Tests/` ; aucun test STT de transcription réelle, aucune parité fp16/8 bits/4 bits, aucune parité
  Core ML/MLX ; pas de CI suivie (aucun `.github/`).
- **Mesures existantes** : q6 99,4 % vs bf16 96,5 % de couverture, RTF 1,47 vs 3,44
  (`docs/voice_cloning.md:122-126`, issue #45 ; un seul locuteur, 3 phrases × 5 graines, « en session »).
  *Amendé en vérif. croisée* : « 3,5 vs 8 Go » est retiré des mesures. Ce sont les **tailles des packs**
  (`README.md:86-87`, `TTS/VoxtralTTSRegistry.swift:34,62`), pas une mémoire mesurée, et elles ne figurent
  pas aux lignes 122-126 citées.
- **Correction** : `voxtral eval` : WER STT sur un petit corpus public (sous-ensemble FLEURS fr/en,
  licence à vérifier) + aller-retour TTS→STT, graines fixées, backend `.mlx` explicite, une ligne JSON par
  point ; campagne avec seuils relatifs affirmés.
- **Risque API** : additif. **Effort** : M. **Statut** : VÉRIFIÉ (absence).
- **Fiche** : *Éval reproductible* — **porte** : deux exécutions à graine égale ⇒ scores identiques ; baseline
  WER de `mini-3b-8bit` enregistrée ; toute fiche perf ultérieure garde WER ≤ baseline + 0,5 point.
  **Cible** : macos-gpu.

### A-23 · basse (conditionnel) · Serveur absent : prérequis bloquants si l'option serveur est retenue

- **Où** : aucun serveur (MLX-011 : 0 occurrence) ; flux TTS sans `onTermination`
  (`TTS/Pipeline/VoxtralTTSPipeline.swift:556-557`, MLX-003 de `patterns-scan.md`) ; A-01 (pas de
  sérialisation) ; dépendances de `VoxtralCore` consommées par une app App Store (`Package.swift:57-71`).
- **Constat** : un client HTTP qui se déconnecte laisserait la génération TTS tourner jusqu'à
  `maxFrames` (2500 trames ≈ 200 s) ; un point d'entrée d'enrôlement exposé déclencherait A-01.
- **Correction** : voir § 4 ; ne rien construire avant la réponse à Q1.
- **Risque API** : aucun tant que rien n'est ajouté. **Effort** : L si retenu. **Statut** : VÉRIFIÉ (absence).
- **Fiche** : conditionnelle (§ 4). **Cible** : macos-gpu.

## 3. Synthèse par thème demandé

- **Enrôlement** : politique mémoire absente (A-06 ; résidence du LLM à mesurer, chargement paresseux) ;
  eval par pas présent (`:601`) mais double synchronisation (A-10, gain attendu < 1 %, note sans fiche) ; reprise/points de contrôle absents (A-07) ; garde NaN de #44 présente et
  correcte (`:562-576`, `enrollVoice :444-449`) mais non testée (A-09) ; ABBA : **aucun `compile` dans
  Voxtral, mais `silu` de MLXNN est compilé et appelé sous le vjp** ; inférence concurrente possible (A-01) ;
  porte de calcul iOS absente (A-07) ; profils `enroll-fast|lean` (§ 5).
- **Core ML** : reproductibilité (A-03), version coremltools non épinglée (A-03), ANE non mesuré et
  unités contradictoires (A-12), repli silencieux et chemin de cache (A-04, A-13), ressource d'app (A-14).
- **Recherche Python** : A-21.
- **Téléchargement / registres** : complétude (A-02), SHA-256 absent (A-02), reprise par fichier seulement
  (A-02), liens symboliques : #49 couvert par `ModelLoadingSymlinkedDirectoryTests` et
  `ModelDownloaderSizeTests` (MLX-015 : les 3 occurrences sont dans ces tests) ; `customModelsDirectory`
  contourné par l'encodeur Core ML (A-04) ; API factice (A-05). Registres : pas de révision, pas de
  SHA-256, taille en texte libre (`"~8 GB"`), pas de lien vers un profil. Nouveaux packs communautaires
  vus sur le Hub (ex. `MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder`, MLX, 23/09/2026) : à
  évaluer par l'audit profils, non recommandés ici.
- **CLI / bench / apps** : A-14 à A-19.
- **RuntimeBeacon** : A-20 ; sinon conforme (écriture atomique, ramassage des pid morts, opt-in).

## 4. Serveur OpenAI-compatible : attendu ou non ? → ASK Q1

Rien dans le dépôt, les issues (1-50) ni les plans `project:mlx-voxtral-swift` d'action-plans ne demande
un serveur ; les deux consommateurs connus (FluxForge, LipDub/LTX) sont **in-process**. Le cadrage signale un
`stt-mcp` dans le homelab optimOrin (non vérifié ici : dépôt non accessible). Le standard
(`references/server-standard.md`) est centré chat ; pour Voxtral la surface naturelle serait
`POST /v1/audio/transcriptions` (Mini/Small/Realtime), `POST /v1/audio/speech` (TTS, voix prédéfinies ou
embedding enrôlé), éventuellement `POST /v1/chat/completions` avec audio (mode chat), `GET /v1/models`,
`GET /healthz`. Options :

- **A. Pas de serveur** (statu quo) : aucun coût ; le `stt-mcp` du homelab reste sur sa pile actuelle.
- **B. Serveur dans un paquet imbriqué `Server/Package.swift`** (Hummingbird) dépendant de `..` par chemin,
  pour ne pas imposer swift-nio à FluxForge (constat SwiftPM du standard) ; `127.0.0.1` par défaut,
  `--host` explicite, `--api-key` obligatoire hors loopback ; file unique vers le modèle ; limites de taille
  et de durée audio ; **aucun** point d'entrée d'enrôlement (ou refus pendant le service) ; prérequis :
  A-01, MLX-003 (`VoxtralTTSPipeline.swift:556`), A-02.
- **C. Binaire MCP** (`voxtral-mcp`, stdio/SSE) : outils `transcribe`, `speak`, directement consommable
  par le `stt-mcp` du homelab ; mêmes prérequis que B, sans surface HTTP publique.

## 5. Profils d'enrôlement `enroll-fast|lean` (transposition de `lora-fast|lean`)

Règle du standard : chaque champ = un bouton existant ou créé par une fiche ; valeurs mesurées ou
« à mesurer ».

| Champ | `enroll-fast` | `enroll-lean` | Source / état |
|---|---|---|---|
| Poids | ceux de la synthèse visée (doc : même `--model` pour `enroll` et `tts`) ; défaut CLI `tts-4b-mlx` (bf16) | idem ; 6 bits candidat (synthèse q6 : 99,4 % couverture, RTF 1,47 « en session » ; pack ≈ 3,5 Go = taille, pas mémoire mesurée) | enrôlement **via** un pack quantifié : non mesuré ; codec quantifié ou non dans le pack : à vérifier (index) |
| Durée de référence | 16 s (200 trames) | 16 s | similarité 0,72 à 16 s (2000 époques) — mesuré |
| Époques | 5000 | 5000 (3000 à mesurer) | « 5000 good » (doc) ; ≈ 30 min annoncé, non mesuré en banc |
| Résidence | modèle entier (chargé paresseusement : résident seulement après une synthèse) | codec + table audio seulement | A-06 : mesurer d'abord (piège 18), coder seulement si l'écart ≥ 5 % |
| `cacheLimit` | défaut (= limite mémoire) | borné, valeur à mesurer | T1/T2 |
| `clearCache` en fin | oui | oui | T3 |
| Graine | fixée (A-08) | fixée | à coder |
| Checkpoint | toutes les 500 époques | toutes les 250 | A-07, à coder |
| Porte | temps total et pic mesurés, similarité ≥ 0,70 | pic ≤ 60 % de `fast` (scénario synthèse→enrôlement ; volet retiré si l'étape 0 d'A-06 montre < 5 %), similarité à ±0,01 de `fast`, temps ±5 % | à mesurer |

## 6. Fiches proposées (ordre du skill : stabilité → hygiène → baseline → leviers)

| Fiche | Constats | Porte chiffrée | Cible |
|---|---|---|---|
| F-A1 Sérialiser enrôlement/inférence + démo | A-01 | 20/20 sans blocage (timeout 120 s), refus `busy` < 1 s, test rouge sans correctif | macos-gpu |
| F-A2 Complétude + SHA-256 + API factice | A-02, A-05 | 5 tests de complétude/SHA verts, rouges sans correctif | macos-gpu |
| F-A3 Chemin unique encodeur Core ML | A-04, A-13 | encodeur sous `customModelsDirectory`, 2e chargement hors ligne OK, 0 octet sous `~/.cache/huggingface` | macos-gpu |
| F-A4 Conversion Core ML reproductible | A-03 | cloud : 0 argument inconnu/requis manquant, dépendances épinglées ; Mac : parité rel. L2 ≤ 1e-2 | cloud puis macos-gpu |
| F-A5 Annexe recherche reproductible | A-21 | aucune dépendance non épinglée, divergence documentée | cloud |
| F-A6 Instrument de mesure (profile/bench) | A-15, A-16 | A/A ≤ 3 % de dispersion, révision notée | macos-gpu |
| F-A7 Éval reproductible | A-22 | scores identiques à graine égale, baseline WER enregistrée | macos-gpu |
| F-A8 Graine, checkpoint, tests de garde | A-07, A-08, A-09 | reprise bit-exacte, même graine ⇒ mêmes codes, 3 tests de garde | macos-gpu |
| F-A9 `enroll-lean` (résidence + cache ; A-10 en passager) | A-06, A-10, A-11 | étape 0 : pic CLI vs synthèse→enrôlement (si écart < 5 %, volet résidence retiré) ; sinon pic −40 % min (scénario synthèse→enrôlement), temps ±5 %, codes identiques à graine fixe ; A-10 retiré si < 5 % | macos-gpu |
| F-A10 Décision hybride par la mesure | A-12 | ≥ 5 % à chaud + parité, sinon `.mlx` par défaut | macos-gpu |
| F-A11 Build app + hygiène annexes | A-14, A-17, A-18, A-19, A-20 | build Release sur clone neuf, 0 manifeste résiduel | macos-gpu |
| F-A12 Serveur (si Q1 = B ou C) | A-23 | 401/413, deux flux concurrents séquentiels, déconnexion ⇒ arrêt < 3 s | macos-gpu |

## 7. Questions (ASK)

1. **Serveur** : A (aucun), B (paquet imbriqué HTTP OpenAI-compatible) ou C (binaire MCP pour le homelab) ?
2. **Enrôlement sur iOS** : cible supportée (⇒ porte GPU + checkpoint obligatoires) ou refus explicite ?
3. **Backend par défaut** `.auto` (télécharge 1,32 Go et compile ≈ 70 s au premier lancement pour Mini,
   2 min 25 s pour Small, une fois par architecture GPU, d'après les issues #16/#22 mesurées en session) :
   passer à `.mlx` si la mesure F-A10 ne montre pas de gain ?
4. **API publique factice** (`downloadModel(modelId:)`, `ModelDownloader.hubApi`, `reconfigureHubApi`) :
   dépréciation (additif) ou suppression (cassant — FluxForge les appelle-t-il ?) ?
5. **Défaut d'enrôlement de la bibliothèque** (8 s → 16 s) : changement de sortie pour les consommateurs,
   accepté ?
6. **`VoxtralBenchmark`** : réorienter vers un vrai banc ou retirer le produit ?
7. **Ressource `VoxtralEncoderFull.mlmodelc`** de VoxtralApp : garder dans le bundle (fichier hors git) ou
   téléchargement à l'exécution uniquement ?

## 8. Plans d'action existants liés (action-plans, lecture seule)

- `#349` (issue #45, fermée le 2026-07-27, 5 points dispositionnés ; point 1 vérifié dans le code :
  `VoxtralVoiceEnrollment.swift:562-576`, `VoxtralTTSPipeline.swift:444-449`) : `ready-to-act`, source
  fermée ⇒ à clore après vérification humaine.
- `#307` (PR #41, fusionnée le 2026-07-20) et `#71` (PR #34, fusionnée le 2026-07-09 ; le portage Swift de
  la recherche existe) : `ready-to-act` depuis leur transition automatique, source fermée ⇒ à clore.

## Annexe — Constats écartés à la vérification croisée

**Constats entiers écartés : aucun.** Les 23 constats ont été relus contre le code à `9392ed1` et contre
l'amont réellement résolu. Aucun n'est faux dans son ensemble, et les 10 non listés ci-dessous sont gardés
tels quels, lignes vérifiées.

**Sous-affirmations écartées ou corrigées** (le constat parent est gardé et amendé dans le corps) :

| Constat | Sous-affirmation écartée | Raison |
|---|---|---|
| A-01 | « corrige l'ordre (… #339) » | La correction amont est la PR #461 (`df9ae26`) ; #339 est l'issue qu'elle cite. |
| A-06 | « ≈ 3,8 G paramètres **résidents** », « attendu ≈ −50 % » | Chargement paresseux sans `eval` (`VoxtralTTSModelLoading.swift:70-74`) : sur le chemin `voxtral enroll`, le LLM n'est probablement jamais matérialisé (piège 18). Sévérité moyenne → basse, porte à deux scénarios. |
| A-07 | `Package.swift:11` ; « annulation non exposée par la CLI » | La ligne iOS est `:12`. En CLI, Ctrl-C suffit (aucune écriture partielle, `VoxtralTTSPipeline.swift:450`). |
| A-10 | Levier perf à fiche propre | Gain attendu < 1 % (≈ 1 ms de trou GPU sur ≈ 360 ms/époque annoncés), sous le seuil de 5 % : note passagère d'A-06. |
| A-13 | « non atteint par défaut **parce que** `.mlx` au repli » ; `Package.swift:87-89` | C'est `standardModel`, toujours renseigné par le pipeline, qui évite les poids non initialisés. La ressource est à `:79-80`. `strides` et fp16 sont latents en production (float32 contigu). |
| A-14 | `Package.swift:87-89` | Ligne réelle `:79-80` (`:87-89` = cible `VoxtralTranscriptionTest`). |
| A-15 | `ProfileCommand.swift:175`, `:208` = construction du pipeline | Ce sont les lignes de température ; la construction est à `:177`/`:210`. |
| A-19 | « un nom avec `/` écrit hors du dossier » | Seul `..` sort du dossier ; `a/b` fait échouer la sauvegarde en fin d'enrôlement (≈ 30 min perdues). |
| A-21 | « divergence non documentée » ; « le portage fait mieux : 0,72 contre 0,56-0,59 » | Les deux jeux de pertes sont documentés, mais séparément (`voice_cloning.md:252-254`, README recherche l. 18). La comparaison est confondue : 16 s / 2000 époques contre 8 s. |
| A-22 | « 3,5 vs 8 Go » présenté comme mesure | Ce sont les tailles de packs (`README.md:86-87`), absentes des lignes citées. |

Nuances ajoutées sans écarter de sous-affirmation : A-04 ((d) latent ; (b) n'affecte pas FluxForge sur
APFS insensible à la casse, cf. issue #2), A-09 (seul `failed` est perdu, `cancelled` n'est pas atteignable
dans la surcharge), A-12 (issues #16/#14/#22 = mesures « en session », coût à froid une fois par
architecture GPU).
