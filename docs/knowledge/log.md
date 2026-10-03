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
- 2026-09-30 — **K-22 : `Package.resolved` suivi** (ASK-28 = A, mlx-swift-lm reste sur `main@604fae7`) et
  dépendances élaguées de `VoxtralCore` (`MLXLLM`, `MLXOptimizers`, `ArgumentParser`, `Transformers` → `Hub`) ;
  profiler 1.5.1. Build propre Release de la CLI : 122 s → 96 s.
- 2026-09-30 — **70 Go pour un modèle de 5 Go = cache de buffers MLX, pas le modèle** (K-2, swift-mlx-profiler) :
  sur C-long, MLX actif 8,6 Go stable, cache MLX 24,5 → 56,7 Go sans `cacheLimit`, compression de 45 Go et swap.
  Toujours lire « MLX Active » vs « MLX Cache » du profileur avant de conclure sur la mémoire ; `/usr/bin/time -l`
  (footprint) compte le cache. Conséquence : la politique mémoire (K-52) passe avant les baselines.
- 2026-09-30 — **K-52 : `cacheLimit` opt-in** : C-long STT 70,6 Go → 10,4 Go de pic processus (actif 8,6 Go + 2 Go), plus
  de swap, aucun coût en temps détecté (−0,5 % sur points voisins ; bruit machine jusqu'à 36 %). Après `unload()`, la
  mémoire revient en 1 à 5 s (pilote GPU asynchrone) : mesurer un footprint de déchargement après stabilisation.
  Mesures faites avec la balise active et une veille des balises d'autres runtimes (aucune pendant la série 2).
- 2026-09-30 — **K-5 : budget STT selon la durée** (6 jetons/s + 64, ≥ 500 ; parole mesurée 3,1 j/s EN, 4,0 j/s FR ;
  l'ancien 500 tronquait déjà C-moyen EN) et Realtime borné par les trames (budget texte). Constats : le Realtime
  **dégénère au-delà de ≈ 30 s** (P-62, octets NUL ; correctif K-13) ; un audio EN/FR alterné ne se transcrit pas avec
  une langue imposée (fin de séquence en FR, boucle en EN) ; les références de `docs/tts_benchmark.md` ne couvrent pas
  les clips C-moyen (K-33).
- 2026-09-30 — **K-32 : `VoxtralCLI bench` validé en A/A** (≤ 0,5 % sur STT, TTS, Realtime et chat, sorties
  identiques) : c'est désormais l'instrument de toutes les mesures (`BENCH {json}` → `BENCHMARKS.md`). Il refuse un
  binaire Debug et toute balise vivante d'un autre runtime. Piège trouvé en l'écrivant : lire la sortie d'un `Process`
  **avant** `waitUntilExit()` (sinon `ps` remplit le pipe et tout se bloque).
- 2026-10-01 — **K-13 : Realtime à fenêtres glissantes** (encodeur 750 par tranches + `RotatingKVCache`, décodeur
  `RotatingKVCache(8192)`) : la sortie ne dégénère plus après ≈ 30 s (C-moyen à 5 % / 2 % de mlx-audio, contre 83 % /
  85 %) ; pic d'encodage indépendant de la durée (+5,5 % de 3 à 12 min) ; mémoire plate après 8 192 pas. Piège de mesure :
  sur ce Mac, une charge GPU continue de plus de ≈ 5 min ralentit les pas jusqu'à ×1,6 (thermique : un clip court lancé à
  chaud est aussi lent) — ne pas attribuer au code une dérive de fin de run long.
- 2026-10-01 — **Vérification du planificateur** : 10 tâches `verified` (K-1, K-2, K-4, K-5, K-6, K-11, K-16, K-22,
  K-25, #599) ; portes amendées par Vincent (K-2, K-4, K-5, K-16, K-22 ; critères à références → K-33, Realtime C-long →
  K-13) ; ASK-7 = A ; `CHANGELOG.md` créé (2.3.0 à venir, ASK-9) ; à la fusion sur `main` : compiler FluxForge (K-22).
- 2026-10-01 — **K-13, complément** : les NUL de la sortie Realtime venaient de `decode`, qui ne sautait que BOS/EOS/PAD
  alors que le modèle émet `[STREAMING_PAD]` (32) / `[STREAMING_WORD]` (33) ; tout id < 1000 est désormais sauté.
  Écart restant avec mlx-audio : l'invite (pad 11 × 1 + délai côté Swift, 32 × 32 côté mlx-audio) → fiche à créer.
- 2026-10-01 — **K-12 : streaming TTS annulable** : le flux est rendu en 0,2 ms (677 s avant : tout se générait dans la
  closure de construction), l'annulation rend `.ready` en 8 ms à une frame près. Parité stream/batch : codes identiques,
  audio à 1,2e-6 près (re-décodage codec par chunk ; Vincent : tolérance 1e-5, l'identité exacte relève de K-43).
- 2026-10-01 — **K-26 : enrôlement reproductible et reprenable** (graine, point de contrôle, `--seed/--checkpoint`).
  Piège : un `MLX.take` sur des indices qui se chevauchent (découpage STFT) a un gradient non déterministe sur GPU
  (scatter-add atomique) ; à graine égale les codes divergeaient dès l'époque 1. Découper par blocs (reshape + tranches)
  rend le gradient bit à bit stable sans changer les valeurs.
- 2026-10-01 — **K-32b : Core ML sur GPU + MLX dans le même processus = interblocage** (1re prédiction bloquée dans
  `MTLCommandQueue commandBuffer`, 0 % CPU) ; `.auto`, défaut du pipeline, était touché. Préréglages Core ML sur l'ANE
  (Vincent). Instrument `bench` : phases en temps exclusif (le décodage STT était compté deux fois), `pad_fraction`
  (0,72 sur C-moyen EN), `--trace`.
- 2026-10-01 — **K-3 : masques construits par le cache** (booléens, forme des clés) : bf16 possible (prérequis K-40),
  décodeur hérité sans arrêt au 2ᵉ tronçon de préfill ; logits STT identiques (L2 0,0), temps inchangé (+0,07 %).
  Le chargeur hérité `loadVoxtralModel(modelPath:…)` ne charge pas le dossier HF bf16 (`keyNotFound audio_tower.conv2`).
- 2026-10-01 — **K-15 : annulation < 200 ms, calcul hors pool coopératif** (file dédiée + drapeau d'annulation lu par
  les boucles). Piège : une annulation n'est visible qu'aux points où le calcul s'arrête — un encodeur évalué d'un seul
  graphe (STT 23 fenêtres, conv Realtime sur tout l'audio) retardait l'arrêt de 3 à 5 s ; évaluer par couche/tronçon.
- 2026-10-01 — **K-24 : registres exacts et `consolidated` exclu** : `mini-3b` télécharge 9,37 Go au lieu de 18,7 Go
  (Small 24B : 48,5 au lieu de 97 Go) ; tailles et précisions des 13 entrées alignées sur `docs/Weights.md`.
- 2026-10-02 — **K-14 : plafond TTS = 70 + 10,4 × jetons** (3 × l'ajustement 23,5 + 3,48 × jetons sur 108 synthèses) :
  un emballement (517 frames pour « Bonjour, comment ça va ? », 6 bits, graine 2) s'arrête à 143 ; 107/108 sorties
  identiques. Rappel : la CLI `tts` ignore `--seed` avec `-v` (passer par `--voice-embedding`).
- 2026-10-02 — **K-23 : −915 lignes mortes, bibliothèque muette hors debug** (`print` → `VoxtralDebug`/`os.Logger`,
  plus d'écriture dans `/tmp`, plus de CWD ni de `/Users/…`). Le point d'entrée legacy `VoxtralGenerator` est cassé
  (arrêt `needModuleInfo` au chargement) : à déprécier par K-30.
- 2026-10-02 — **K-27 : API publique honnête** : `tokenCount` réel, 0 `as!`, souches dépréciées avec message exact ;
  les arrêts sur type de module non supporté passent par la boîte d'erreurs MLX (K-1) et les entrées qui lèvent
  valident les types d'emblée (décision de Vincent).
- 2026-10-02 — **K-29 : CI GitHub Actions verte** (`macos-26`, Xcode 26.6, 569 tests dont les tests MLX, 4 min 25 s) ;
  les tests « de logique » appellent maintenant les fonctions de production. Constat : le « top-p » du sampling STT est
  un top-k(1000).
- 2026-10-02 — **K-30 : famille legacy dépréciée (2.3)**, 37 symboles avec alternative, 0 appel interne. FluxForge : une
  ligne à retirer à la fusion (`ModelDownloader.reconfigureHubApi()`, sans effet depuis K-6/K-25).
- 2026-10-02 — **K-10 : une seule table de modèles STT** (registre) ; Small 8 bits = VincentGOURBIN, chargé hors ligne
  sans requête réseau. Leçon : ne jamais écrire un fichier de test avec `cat >` sans vérifier qu'il n'existe pas.
- 2026-10-02 — **K-9 : `realtime-4b` (Mistral original) chargeable** : 8,87 Go téléchargés (au lieu de 17,72), 0 clé
  manquante ou en trop, même texte que le pack fp16 ; un id Realtime inconnu lève.
- 2026-10-02 — **K-8 : quantification décodée comme MLXLMCommon** : les packs 2026 à `"mode"` (aufklarer, Markus) se
  chargent ; Markus Mini 8 b encodeur dense : 0 clé en écart, greedy identique à mlx-voxtral. Les packs mlx-voxtral
  sauvegardent `embed_tokens` deux fois (alias Python) : la vérification `[.all]` exige de retirer l'alias.
- 2026-10-02 — **K-28 : clone neuf constructible** (4 schémas Release, plus de ressource `.mlmodelc` déclarée) ; app
  empaquetée par `Scripts/package-app.sh` (bundles dans `Contents/Resources`, où MLX trouve son `metallib`) ;
  `RuntimeBeacon` : 50/50 tours laissaient un manifeste avant le correctif, 0 après.
- 2026-10-02 — **K-31 : API 3.0** : surface publique 1 110 → 338 lignes (façades seulement), types génériques
  préfixés `Voxtral…` (alias dépréciés), code hérité supprimé ; FluxForge à migrer par Vincent (retirer
  `reconfigureHubApi()` et ses typealias de contournement).
- 2026-10-03 — **K-33 : `voxtral eval`** (WER normalisé, corpus à SHA-256, reproductible 18/18). Les références
  « Full test texts » étaient condensées (163 mots pour 413 dits) : C-moyen régénéré depuis des textes exacts. STT
  `mini-3b-8bit` : 1,8 % EN / 2,2 % FR sur C-moyen ; l'auto-détection vaut la langue imposée. Deux limites du modèle :
  sur un audio long EN/FR alterné il **traduit le français en anglais**, et sur un clip FR de 17 s il saute la 1ʳᵉ
  phrase (le juge Realtime, lui, s'arrête après). Le juge Realtime n'est pas fiable sur du français court.
