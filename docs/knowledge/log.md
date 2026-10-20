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
- 2026-10-03 — **K-37 : enrôlement** ≈ 62–65 ms/époque (6 bits comme bf16) ; pic 2,7 Go seul contre 6,1 Go (6 bits) /
  11,6 Go (bf16) après une synthèse dans la même pipeline : libérer le LLM avant d'enrôler vaut le code (K-64), car il
  ne sert pas à l'enrôlement (voix identique quel que soit le pack). `bench enroll --seed` n'était pas appliqué avant.
- 2026-10-03 — **K-15, complément `.auto`** : la boucle de l'encodeur Core ML n'avait aucun point d'annulation (12,3 s
  d'attente) → 221 ms ; le `convStem` Realtime, évalué d'un bloc sur tout l'audio, retenait l'annulation ≈ 3 s une fois
  sur trois → évalué par tranches de 60 s (résultat identique, L2 0). Leçon : un test d'annulation qui ne passe qu'une
  fois ne prouve rien si l'instant d'annulation tombe dans une phase variable ; le relancer.
- 2026-10-03 — **R-4 : dépôt aligné sur la vérification du 2026-10-03** (décisions dans `ASK.md` §Dérogations,
  CHANGELOG 3.0.0 sans symbole interne, PLAN §3 = tracker). Parité streaming K-12 jugée sur la sortie brute (sans
  rognage) : codes identiques, audio à 1,24e-6 du batch (≤ 1e-5, décision de Vincent) ; K-43 tolère 1e-4. Les quatre
  classes de tests gardés restent vertes à HEAD.
- 2026-10-03 — **K-33 débloquée (Vincent, option A)** : « dernière phrase présente » = ≥ 90 % de ses mots
  (`last_sentence_coverage`), C-long jugé par langue. À retenir pour les hôtes : Voxtral Mini **ne sait pas transcrire
  d'un bloc un audio long qui alterne anglais et français** (il traduit le français) et ne sera pas corrigé : découper
  par langue avant de transcrire.
- 2026-10-05 — **K-36 : baseline Realtime**.
  - 4 bits : 25 à 29 ms par pas de 80 ms (RTF 0,36 à 0,40).
  - fp16 : 126 à 148 ms par pas (RTF 1,7 à 1,9), pas temps réel.
  - GPU occupé pendant tout le décodage : 92 à 99 % selon l'instrument, écart ≤ 7 pts. Le « 49 % » de #24 était un
    artefact de lecture.
  - Pièges d'instrument :
    - (1) une Metal System Trace sur 2 min d'audio écrit ≈ 20 Go dans le dossier temporaire du système, sans tenir
      compte de `TMPDIR` : tracer un clip court, ou surveiller l'espace libre ;
    - (2) sur une passe longue, le temps total suit la charge de fond et la chaleur : juger l'A/A sur la médiane par
      pas (et le p90) ;
    - (3) 120 s de repos ne suffisent pas après une passe de 10 min.
  - Constat : l'invite Realtime remplit avec `<pad>` (id 11) au lieu de `[STREAMING_PAD]` (32), avec 1 jeton à gauche
    au lieu de 32 (mlx-audio). Le WER reste bon (1 à 2 % sur C-moyen) ; mesure de l'alignement : K-84.
- 2026-10-05 — **K-34 : baseline STT**.
  - Mini 8 bits : 19 à 27 ms par pas ; 4 bits : 13 à 21 ; bf16 : 132 à 150, soit 6 fois plus lent que le 8 bits
    (même signature que le Realtime fp16).
  - Small 4 bits : 70 à 84 ms par pas.
  - Le préfill occupe le GPU à 99-100 % selon trois instruments.
  - Small sur 32 Go : seulement avec un plafond de cache MLX (19,0 Go contre 37,3 Go sur 8 min d'audio).
  - Small `.auto` × C-moyen EN est bimodal d'un processus à l'autre (75 ou 81-85 ms par pas), cause non isolée, ASK-32.
    Leçon : n'écrire une cause que si chaque passe lente la montre dans ses échantillons (la vérification a refusé une
    cause « ANE » vue dans 1 passe sur 5).
  - Mini en audio long EN/FR : traduit les segments français (WER FR 96-100 %, EN 9 %) ; jugé par langue.
  - Small `.auto` traduit le dernier segment français de l'audio long alterné (WER FR 51 % contre 1,5 % en `.mlx`).
- 2026-10-06 — **ASK-32 = A** (Vincent) : K-34 garde la cellule Small `.auto` × C-moyen EN sans A/A (référence 75,13 ms,
  médiane des passes rapides). **K-35** réduite (3 graines sur le texte court, 1 sur moyen et long). Piège : `voxtral
  enroll` exige 16 s de référence par défaut et l'erreur part sur stderr ; une file qui ne journalise que stdout ne
  la voit pas (toujours `2>&1` et vérifier le fichier produit).
- 2026-10-08 — **K-92** : le streaming TTS ré-émettait tout l'audio quand l'EOA tombait juste après une frontière de
  morceau (morceau final sans frame nouvelle), environ une synthèse sur six, toujours pour les mêmes graines. Vu dans
  K-35 par `audio_s` = 2 × frames × 80 ms. Corrigé dans `synthesizeStreaming` ; test gardé qui échoue avant (2,00 ×)
  et passe après. Leçon : un champ dérivé (`audio_s` / frames) qui vaut exactement un multiple entier signale un
  défaut de découpage, pas du bruit.
- 2026-10-09 — **K-35 : baseline TTS**.
  - Par pas : 4 bits 25 à 26 ms, 6 bits 38 à 39, bf16 129 à 137 (après la reprise du 2026-10-10). Le bf16 est plus lent que le temps réel même en
    batch (RTF 1,7 à 1,8).
  - Le streaming re-décode tout l'accumulé : 49 à 214 ms par frame sur le moyen et le long, RTF jusqu'à 3,0 en première passe, 3,25 en seconde (K-43).
  - Le codec décode d'un bloc : sur le texte long, 21 à 29 Go de pic MLX. Le plafond de cache retire 9 à 12 Go de
    `phys_footprint`, écart de vitesse sous 5 % en comparaison paire à paire (pas un A/B/B/A).
  - Aller-retour ASR : couverture 0,91 (court) à 0,98.
  - Leçon (1) : un premier relevé donnait au plafond un coût de 8 à 19 % ; il comparait deux protocoles et deux macOS.
    Une comparaison ne vaut que dans le même protocole, en A/B/B/A.
  - Leçon (2) : 5 cellules sur 9 ont échoué l'A/A un soir de compilations Xcode (charge relevée dans 3 des 5), puis 5 sur 5 sont passées le
    lendemain, machine calme. Un échec sans charge relevée se refait, sans cause écrite. (Une partie de ces échecs
    touchait le texte moyen, alors mesuré sans repos après l'amorçage : entrée du 2026-10-10.)
- 2026-10-09 — **K-35, vérification** : en bf16, `synthesize(voice:)` (préfixe de voix en cache) et
  `synthesize(voiceEmbedding:)` donnent un autre audio à même graine, alors que les 4 et 6 bits sont identiques.
  Les EVAL bf16 sont refaits par le chemin des BENCH. Leçon : un EVAL qui sert de référence de qualité pour une
  ligne BENCH doit avoir le même `out_sha256`, à vérifier pour chaque pack. Autre leçon : une file qui reprend une
  cellule écrase son `.log` ; la ligne reste dans `bench.jsonl`, à comparer aux journaux avant de clore.
- 2026-10-10 — **K-35, repos après l'amorçage** : `bench --cooldown 0` lance la passe mesurée juste après
  l'amorçage, sur un GPU chaud. Les cellules du texte moyen mesurées ainsi étaient 2,9 à 17 % plus lentes (A/B/B/A
  6 bits : 41,4 / 44,7 contre 38,52 / 38,50 ms/pas, même sortie). Elles faisaient croire à un 6 bits plus lent sur
  le moyen et à une ligne consommateur dispersée de 15 %. Leçon : en un processus par passe, toujours
  `--warmup 1 --cooldown 120`. Une valeur qui s'écarte de la matrice d'origine se compare d'abord au protocole
  (repos, amorçage, processus) avant de chercher une cause dans le code.
- 2026-10-10 — **Replanification des lots 4 à 6 (#590)** : 44 fiches tranchées (42 gardées, 2 retirées : K-67, K-71 ;
  K-69 réduite à son volet mémoire), 5 fiches de suite (K-93 à K-97), 56 tâches dispatchables. Leçon : avant de garder un levier, rapporter
  sa phase au temps total de la baseline. Sur le TTS, le post-traitement pèse au plus 0,05 % et le codec 0,2 à 4,4 %
  du batch : un levier sur ces phases ne peut pas atteindre 5 % du total. Autre leçon : l'occupation GPU du
  profileur (99,9 % sur la génération TTS) surestime de 2 à 7 points celle d'xctrace ; elle ne suffit pas à borner un
  levier de recouvrement.
- 2026-10-11 — **K-93 : part du flow matching dans le pas TTS** : en `--trace`, le FM (tête sémantique, 7 pas
  d'Euler avec CFG, FSQ) fait 68,0 % du pas en 4 bits, 71,6 % en 6 bits, 82,3 % en bf16 (107,6 ms par frame contre
  23,2 pour le LLM) ; LLM + FM = 99,86 à 99,96 % du pas. Hors trace, sortie et `step_ms_p50` inchangés (+ 0,98 %,
  A/B/B/A). Le streaming émet ses phases en trace. Borne des leviers FM (K-39, K-48, K-57) ; note
  [baseline TTS](benchmarks/m3max-tts-baseline-2026-10.md), PLAN.md §7.
- 2026-10-11 — **K-39 : FM bf16 pour le pack bf16, − 48,8 % sur le pas** : seules les couches du transformeur du
  flow matching passent en bf16 (tts-4b-mlx 129,8 → 66,5 ms/pas, A/B/B/A ; parité forcée 99,23 % Δ1, sémantique
  100 % ; couverture ASR sur 5 graines égale ou meilleure que l'ancien code, mais sous la référence K-35 en long : ASK-35). La tête sémantique en bf16 change 1,7 à 2,6 % des codes ;
  en 4 / 6 bits, le bf16 casse la parité sans gain : chemin fp32 gardé, sortie identique. Leçon : sur un pack
  quantifié, le FM n'est pas limité par le dtype de calcul. En attente de l'écoute de Vincent et d'ASK-35 / ASK-36 ;
  [baseline TTS](benchmarks/m3max-tts-baseline-2026-10.md), PLAN.md §7.
- 2026-10-11 — **K-68 : le codec TTS est causal, le porteur se coupe sur un préfixe** : un préfixe de codes se
  décode comme le début du décodage complet (écart ≤ 1,9·10⁻⁶), donc la coupe du porteur décidée sur le waveform
  partiel égale celle du batch (10/10 graines). Le streaming n'attend plus 3 s : TTFA cloné avec porteur − 42,6 % en
  6 bits (1,91 → 1,09 s). Mais la coupe lit 5 frames de référence : avant 7 frames décodées (`chunkSize` ≤ 3) elle
  se trompe (graine 1 : frame 4 au lieu de 19), d'où une garde de 7 frames. Leçon : une précaution « décider sur la vue
  complète » se teste avant d'être payée, sur tout le domaine d'un paramètre public ;
  écoute de Vincent (2026-10-16, 10 paires à l'aveugle) : A et B identiques, non inférieure ; mais la référence
  elle-même laisse entendre le porteur sur 2 graines sur 10 (02, 08 : coupe trop tôt, frames 3 et 4) et perd
  « FluxForge Studio » sur une (01, coupe à la frame 19), et la voix
  clonée 6 bits n'est bonne que sur 3 graines : deux fiches de suite ; PLAN.md §7.
- 2026-10-11 — **K-94 : en bf16, la matmul dense arrondit selon le nombre de lignes (sur le court, par élimination)** : le cache de préfixe de voix
  (préfixe préfillé seul) et le préfill d'un bloc donnent d'autres codes acoustiques dès la frame 0 en bf16 sur le court et le long (le moyen sans écart), aucun écart en
  4 / 6 bits sur 3 textes (invariance à M constatée, pas expliquée). Couverture ASR égale en moyenne sur 3 graines (≤ 1,8 pt par graine sur le long) : pas un défaut, documenté. « La voix clonée démarre
  plus vite » venait d'une voix plus courte (101 frames contre 218), pas d'un chemin : `bench tts` préfille toute voix
  d'un bloc. Leçon : avant d'expliquer un écart entre deux cellules, vérifier quel chemin de code chacune a pris ;
  conséquence pour K-50 dans sa fiche ; PLAN.md §7.
- 2026-10-11 — **K-97 : le 4 bits ne supporte pas la voix clonée** : couverture 0,63 à 0,77 sur le moyen (3 graines)
  et 0,237 sur le long, contre 0,940 / 0,955 en 6 bits ; pauses allongées (frames et pic `codec` doublés), le long
  quitte le texte. La fin est lue au bon endroit, et le LLM comme la tête 4 bits s'écartent du bf16 ; une voix
  prédéfinie de même longueur perd moins (0,90 à 0,93) : la voix clonée rendrait la trajectoire plus fragile (hypothèse).
  Leçons : une mesure de qualité faite sur une seule voix ne couvre pas un pack ; un taux d'écart par pas sans échelle
  (paire bf16) et sans séparation des modules ne désigne pas une cause ; les pauses courtes échappent à une règle
  « silence > 1 s » ; PLAN.md §7.
- 2026-10-13 — **K-96 : sans repos après l'amorçage, les passes STT C-moyen (mini-3b bf16, small-4bit) et chat étaient
  lentes ; à la reprise, small-4bit C-moyen, le chat `.mlx`, realtime-4b-fp16 C-moyen et le STT C-long `.mlx`
  bougent de plus de 3 % (ces deux derniers refaits par prudence, A/A en échec)** :
  A/B/B/A sur 10 témoins (A = `--warmup 1 --cooldown 0`, le protocole de K-34 / K-36 ; B = `--cooldown 120`, C-long
  sans amorçage) : B − A − 6,2 % sur le pas (mini-3b C-moyen EN), − 13,0 % sur le total (small-4bit C-moyen
  EN) et − 15,1 % sur le total du chat q4 ; A/A de A en échec sur le Realtime C-moyen et le STT C-long ; C-court et
  Realtime C-long sans effet. 25 cellules refaites au protocole actuel : la plupart tiennent à 3 % près ; ont bougé
  small-4bit C-moyen (`.mlx` − 4,7 à − 7,4 % sur le pas, − 10,7 à − 12,1 % sur le total ; `.auto` FR − 2,6 / − 7,0 %), chat `.mlx` q2 à q4 (total − 4,2 à
  − 19,1 %, TTFT ≈ 4,7 s à chaque question), realtime-4b-fp16 C-moyen (− 4,6 / − 4,2 %) et STT C-long `.mlx` (total
  − 10,2 % en 4 bits, − 5,3 % en 8 bits ; `.auto` 8 bits − 3,8 %) ; mêmes sorties. Traces GPUOCC refaites : occupation de la phase principale inchangée à 0,8 pt près (phases courtes
  davantage ; small-4bit refaite le 2026-10-16 avec un délai d'arrêt `xctrace` de 600 s au lieu des 120 s par défaut :
  Generation 96,6 → 96,8 %).
  Leçon : un repos après l'amorçage fait partie du protocole, aussi pour le STT et le chat ; la dispersion d'une
  paire A dont l'A/A échoue (jusqu'à 15 %, avec une charge de fond dans deux passes sur trois) ne permet pas de
  lire l'effet : on refait la cellule ; [baseline STT](benchmarks/m3max-stt-baseline-2026-10.md),
  [Realtime](decisions/realtime-diagnostics-23-25.md), PLAN.md §7.
- 2026-10-16 — **K-95 : le plafond de cache `--cache-limit-mb 2048` ne coûte rien au pas TTS et retire 26 à 33 % de
  l'empreinte** : A/B/B/A sur 5 cellules (6 et 4 bits, moyen et long, ligne consommateur `clone_fr`), A/A ≤ 0,31 %,
  mêmes sorties. `step_ms_p50` − 0,47 à + 0,05 % ; `total_ms` + 0,1 à + 0,9 %, le surcoût est dans la phase `codec`
  (+ 11 à + 28 %) ; `phys_footprint` − 3,6 à − 5,1 Go sur le moyen, − 11,1 Go sur le long, pic MLX actif inchangé (le
  long garde 22 Go). Le « 6 bits plafonné 2,6 à 2,9 % plus lent » de K-35 venait d'une comparaison paire à paire.
  Défaut public posé à Vincent (ASK-37). Leçon : un écart de 3 % entre deux jours ou deux binaires ne se lit pas
  comme un effet ; seul l'A/B/B/A le tranche ; [baseline TTS](benchmarks/m3max-tts-baseline-2026-10.md), PLAN.md §7.
- 2026-10-16 — **K-40 : le STT calculait tout en fp32, le dtype du décodeur divise le pas de mini-3b par 4,7** :
  mel fp32, positions F32 du checkpoint bf16 et `embed_tokens.scales` BF16 des packs F16 promouvaient tout le
  graphe (KV, logits) en fp32. Calcul au dtype de `norm.weight` : bf16 pour mini-3b, fp16 pour les packs 8 et 4 bits.
  A/B/B/A, 18 A/A PASS : mini-3b `.mlx` 135,72 → 28,75 ms par pas, − 4,4 Go ; 8 et 4 bits − 19 à − 30 % par pas,
  `ttft_ms` − 7 à − 23 %, `peak_mlx_mb` − 13 à − 28 %. Parité : C-court identique, corpus court identique, C-moyen
  2 meilleures et 2 moins bonnes d'un mot (FR `.auto`, + 0,25 pt contre un seuil de + 0,2) : ASK-38. Leçon : un
  dtype mixte (F16 et BF16, ou bf16 et F32) dans un graphe MLX promeut en silence tout ce qui suit en fp32 ; relever
  le dtype du cache KV après préfill (`VOXTRAL_DTYPE_AUDIT=1`) avant de mesurer un levier de dtype ; PLAN.md §7,
  `BENCHMARKS.md` §« 2026-10-16 — K-40 ».
- 2026-10-18 — **K-41 : le pic du codec TTS passe de 22 à 3,6 Go, mais l'attention par bandes seule ne suffit pas** :
  scores par blocs de 64 requêtes sur leurs W + 64 clés au lieu de (T, T) fp32 (exact : clés masquées à − 1e9).
  Seule, elle laisse le pic codec à 5,2 Go (6 bits long, 1 903 frames) : un graphe MLX paresseux ordonnance ensemble
  ses tranches indépendantes et leurs transitoires s'additionnent (sonde : graphe entier + 2,0 à 2,3 Go, tranches non
  évaluées + 1,7 Go, tranches évaluées une à une + 0,6 Go). Inférence par tranches de 2 048 positions évaluées : pic
  codec 3,6 à 3,9 Go (6 bits) et 3,0 Go (4 bits cloné), `phys_footprint` 33 → 8 Go et 39 → 7 Go, codec long − 78 %.
  Une évaluation par couche coûte 4 ms sur un graphe petit (texte court + 16 %) : tranches seulement au-delà de 256
  frames. Leçon : contre un pic de transitoires dans MLX, découper ne suffit pas, il faut `eval` chaque morceau ;
  et le garder hors d'un graphe différentiable (enrôlement). PLAN.md §7, `BENCHMARKS.md` §« 2026-10-18 — K-41 ».
- 2026-10-18 — **K-86 : les voix `.pt` du pack d'origine `tts-4b` se lisent sans Python** : un fichier `torch.save`
  est un zip à entrées stockées (non compressées) avec un pickle de 165 octets (`_rebuild_tensor_v2`, stockage
  `BFloat16Storage`, taille, pas) et le stockage brut ; un lecteur de 200 lignes (répertoire central du zip, machine
  pickle réduite aux opcodes d'un tenseur) suffit, et les 20 voix sont égales bit à bit à celles du pack MLX bf16.
  Même fiche : ZeroVoice (`--voice-xyz`) prend la graine, le streaming signale la troncature, un dossier de N shards
  sans index se charge. Leçon : un format « Python seulement » peut être un conteneur simple ; vérifier `unzip -lv`
  et `pickletools.dis` avant d'ajouter une dépendance ou un refus. PLAN.md §7, `BENCHMARKS.md` §« 2026-10-18 — K-86 ».
- 2026-10-18 — **K-38 : le Realtime calculait en fp32 à cause du pack, pas seulement du mel** : les packs
  mlx-community gardent normes, biais et MLP Ada en fp32 (300 tenseurs) ; `rms_norm` et `matmul` rendent le type
  promu, donc un seul poids fp32 remet le flux résiduel, le cache KV et les logits en fp32 et chaque `Linear` recopie
  ses poids à chaque appel. Caster ces paramètres au dtype de `tok_embeddings` au chargement, plus mel, tables RoPE,
  `tCond` et les zéros du conv causal (`MLX.zeros` est fp32 par défaut) : 4 bits 26,9 → 11,7 ms par pas, fp16
  129,7 → 26,9 ms (sous le temps réel), pic − 4,2 / − 5,3 Go, textes identiques. Leçon : lire le dtype des
  activations (`VOXTRAL_DTYPE_AUDIT=1`) après chaque cast, pas celui des poids ; inspecter les dtypes stockés du pack
  avant de choisir le tenseur qui fixe le dtype de calcul. PLAN.md §7, `BENCHMARKS.md` §« 2026-10-18 — K-38 ».
- 2026-10-19 — **K-43 : le streaming TTS re-décodait tout l'accumulé ; un contexte gauche de 24 frames suffit** : le
  codec est causal à champ récepteur borné, calculable depuis sa configuration en remontant les couches (conv de
  projection K − 1 ; chaque étage de transformer n × W ; conv de pas 1 r + K − 1 ; conv transposée causale de pas s
  ⌈(r + K − 1) / s⌉) : 22 frames pour le pack (21 exactement : le code arrondit au plafond) ; sur un codec à poids aléatoires l'erreur
  tombe dès 5 à 7 frames, ce qui ne dit rien des poids réels. Chaque chunk décode
  ses nouvelles frames + 24 de contexte : streaming = batch à 4,9e-6, codec par chunk 27-29 ms du début à la fin au
  lieu de 34 → 666 ms, streaming long − 36 % par frame (RTF 0,85 → 0,56, au niveau du batch 0,55), part du codec
  41 → 6 %. Le gain en temps (10,7×) reste loin du gain en frames (÷ 33) : le coût fixe d'un appel au codec domine.
  Leçon : un décodeur causal se découpe exactement si l'on calcule son champ récepteur ; mesurer le coût par chunk à
  plusieurs positions (trace), pas seulement la moyenne. PLAN.md §7, `BENCHMARKS.md` §« 2026-10-19 — K-43 ».
- 2026-10-19 — **K-58 : le codec TTS en bf16 gagne 20 % sur sa phase, rien de bout en bout, et perd en SNR** :
  les packs stockent le codec en bf16, mais il calculait en fp32 (centroïdes et niveaux FSQ en fp32 promeuvent tout).
  Le calculer dans le dtype de ses poids : codec batch long − 20 %, pic du codec − 8 % ; or le codec ne pèse que
  0,6 % du batch et 6 % du streaming depuis K-43 : streaming − 0,65 %, dans le bruit ; SNR ≈ 36 dB contre fp32.
  Les caches du weight norm et des centroïdes (P-39) laissent la sortie identique mais ne gagnent rien (− 0,6 %).
  Tout est retiré. Pièges vus en route : `Module.update(parameters:)` recopie les nouvelles valeurs dans les
  `MLXArray` existants (`_updateInternal`), si bien qu'un cache indexé sur l'identité ou le handle de l'array reste
  périmé ; il faut surcharger `update` pour l'invalider. Leçon : avant d'optimiser une phase, relire sa part du
  temps total après les fiches précédentes (K-43 avait ramené le codec de 41 à 6 % du streaming). PLAN.md §7,
  `BENCHMARKS.md` §« 2026-10-19 — K-58 ».
- 2026-10-19 — **K-46 : quantifier la tête liée Realtime au chargement gagne 10,6 % par pas sans changer un mot** :
  la table liée (131 072 × 3 072, fp16, 768 Mio) est relue en entière à chaque pas pour les logits ; quantifiée en
  `QuantizedEmbedding` (groupes de 64) et appliquée par `asLinear` (`quantizedMM`), elle pèse 408 Mio en 8 bits et
  216 en 4 bits : `load_active_mb` − 360 et − 552 Mo, exactement ces écarts ; pas − 10,6 % (8 bits) et − 16,6 %
  (4 bits) sur realtime-4b-4bit ; textes identiques sur C-court et C-moyen EN/FR avec les deux têtes. Derrière
  `VOXTRAL_REALTIME_HEAD_BITS`, défaut inchangé. Piège : une table quantifiée est `uint32` ; le dtype de calcul se lit
  sur `scales`, sinon un cast fait après la quantification serait sauté sans bruit (ici le cast de K-38 la précède :
  seul le test unitaire couvre cette branche). Leçon : en décodage borné
  par la mémoire, le poids d'un module lu à chaque pas prédit le gain (ici aux deux tiers). PLAN.md §7,
  `BENCHMARKS.md` §« 2026-10-19 — K-46 ».
- 2026-10-19 — **K-65 : la RoPE fusionnée de l'encodeur Realtime est exacte mais ne gagne que 2,5 %** :
  `MLXFast.RoPE(traditional: true)` sur `[1, H, L, D]` (offset = début de la tranche) remplace la forme explicite
  (tables cos/sin, ≈ 10 opérations par q et k) : embeddings à 8e-4 de l'ancienne forme, textes identiques, encodage
  C-moyen − 2,5 % (− 5 % exigé), pas inchangé. Retirée. Leçon : gain réel (40 fois la dispersion A/A) mais sous la
  porte ; la part de la RoPE dans l'encodeur n'a pas été profilée ; K-38 avait déjà retiré les tables fp32 visées par
  P-68 (encodage 2 801 → 2 208 ms), et l'encodage ne fait que 9 % du total : l'ordre de grandeur laissait prévoir
  moins de 5 %, à lire avant de fixer la porte d'une fiche « opération élémentaire fusionnée ».
  PLAN.md §7, `BENCHMARKS.md` §« 2026-10-19 — K-65 ».
- 2026-10-19 — **K-50 : le cache de préfixe TTS sert toute voix, retrouvé par empreinte de l'embedding** : forme,
  dtype et SHA-256 des octets de l'embedding identifient la voix (trois entrées, LRU, ≈ 27 Mo chacune) ; la surcharge
  `voiceEmbedding:` en profite : préfill de la 2ᵉ synthèse de `clone_fr` − 52,6 % en 6 bits, − 55 % en 4 bits, audio
  identique en 6 et 4 bits ; en bf16, tout appel par embedding change d'arrondi (K-94), couverture égale en moyenne (court, moyen, long ; une graine du long, déjà dégénérée avant, l'est davantage). Leçon : une clé
  fournie par l'appelant ne garantit pas le contenu ; l'empreinte coûte une copie et un SHA-256 de l'embedding à chaque appel (≈ 1,2 Mo pour `clone_fr` en fp32,
  [101, 3072]) ; elle seule distingue les voix, l'étiquette n'identifie rien.
  Et `machine-check` ne voit un débranchement que s'il précède son passage : lire aussi le champ `power` de chaque
  ligne `BENCH`. Et en streaming, tout calcul MLX fait avant la `Task` productrice est hors de la frontière K-1 :
  un embedding mal formé y arrêtait l'hôte. Défaut gardé par décision de Vincent (2026-10-19, A).
  PLAN.md §7, `BENCHMARKS.md` §« 2026-10-19 — K-50 ».
- 2026-10-20 — **K-47 : le décodage Realtime en pipeline (`asyncEval`) gagne 12,1 % par pas, même texte** : le jeton
  reste un `MLXArray`, le pas suivant part avant la lecture `.item()` du jeton courant ; occupation GPU du décodage
  86,9 → 98,3 %. Leçon : un plafond de recouvrement se lit sur l'occupation du jour du A (86,9 %), pas sur une trace
  plus ancienne (93,1 %, K-96). Et une passe sur batterie se voit au champ `power` de sa ligne `BENCH`, pas au
  machine-check : A1 écartée. PLAN.md §7, `BENCHMARKS.md` §« 2026-10-20 — K-47 ».
