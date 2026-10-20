# Décisions à prendre (ASK) — mlx-voxtral-swift — 2026-09-27

> 31 questions fermées, regroupées par thème, issues des rapports d'audit de `9392ed1` (`v2.2.2`). Chaque question
> suit le gabarit du skill : **Contexte** (2 lignes) · **Ce qui a été constaté** · **Question** (fermée) ·
> **Options** A) B) (C quand le rapport source en proposait une troisième). La ligne **Proposé** est l'avis de
> l'audit, pas une décision. Réponses à inscrire dans la section [Réponses](#réponses) en fin de fichier : une
> fiche marquée ⛔ dans [`PLAN.md`](PLAN.md) lit ce fichier et s'arrête en `blocked` tant que sa question n'a pas
> de réponse.

| Groupe | ASK | Fiches qui attendent (⛔) | Fiches informées |
|---|---|---|---|
| A. Périmètre produit | 1, 2, 3, 4 | K-76 | K-77, K-78 ; hors plan (serveur, iOS, streaming Realtime) |
| B. Défauts publics et comportement | 5, 6, 7, 8, 9, 10 | K-5, K-42, K-61, K-68, K-79 ; K-2 (après mesure) | K-14, K-51, K-64 |
| C. Qualité et parité | 11, 12, 13, 14 | K-13, K-38, K-39, K-48, K-58 | K-33, K-40, K-46, K-65, K-77, K-79, K-80 |
| D. Poids, packs, licences | 15 … 22 | K-9, K-10, K-80 ; K-79 (licence) | K-8, K-78 |
| E. API publique | 23, 24, 25, 26, 27 | K-28, K-30, K-31, K-32, K-74, K-75 | K-59 |
| F. Dépendances amont | 28, 29 | K-22 | K-21 (plan de suivi) |
| G. Dépôt et tracker | 30, 31 | K-20, K-21 | — |

Rappels valables pour toutes les questions : tout gain est **attendu** (jamais obtenu) tant qu'une fiche ne l'a pas
mesuré ; une suppression ou un renommage public est **cassant** ; un changement de défaut est un changement de
**comportement** pour FluxForge Studio (App Store) et SongAnalysisDb, qui importent `VoxtralCore`
(faits-et-actions.md §1.4).

---

## A. Périmètre produit

### ASK-1 — Serveur d'inférence

- **Contexte** : aucun serveur n'existe (0 occurrence du motif MLX-011). Le standard serveur du skill en décrit un
  (HTTP compatible OpenAI ou binaire MCP pour le homelab), mais aucun consommateur connu ne le demande.
- **Ce qui a été constaté** : trois prérequis bloquent un serveur tant qu'ils ne sont pas corrigés. (1) Un client qui
  se déconnecte laisse la synthèse TTS tourner jusqu'à `maxFrames` (2 500 frames ≈ 200 s) : le flux n'a pas
  d'`onTermination` (S-08, MLX-003). (2) Un enrôlement exposé déclencherait le deadlock A-01. (3) Les
  téléchargements ne sont pas vérifiés (S-03). Sources : audit-annexes-serveur.md A-23 et §4.
- **Question** : faut-il un serveur d'inférence pour Voxtral ?
- **Options** : A) non, hors périmètre (documenté) ; B) oui, paquet imbriqué HTTP compatible OpenAI ;
  C) oui, binaire MCP pour le homelab.
- **Proposé** : A. Si B ou C : une fiche L est écrite depuis audit-annexes-serveur.md §4, après K-6, K-11 et K-12.
- **Fiches** : aucune ; hors plan (PLAN.md §6).

### ASK-2 — iOS : cible réelle ou seulement compilée

- **Contexte** : `Package.swift:12` déclare `.iOS(.v17)` depuis la PR #32 ; la compilation passe au simulateur,
  mais rien n'a jamais tourné sur un appareil.
- **Ce qui a été constaté** : les pics annoncés contredisent le README (FV-20). Le pic de 7 949 Mo cité pour le TTS
  est incompatible avec le jetsam d'un iPhone de 8 Go. Il n'existe ni porte GPU en arrière-plan (T23), ni profil
  `lean` iOS, ni point de contrôle d'enrôlement (A-07). Sources : FA-09, T23, audit-performance-stt.md ASK 5,
  Realtime Q5, annexes Q2.
- **Question** : iOS est-il une cible d'exécution supportée (STT, TTS, Realtime, enrôlement) ?
- **Options** : A) oui : une fiche appareil (iPhone 8 Go, profils `lean`, porte GPU en arrière-plan, pic ≤ 4 Go sans
  jetsam) entre dans le plan, et l'enrôlement iOS exige le point de contrôle de K-26 ; B) non : le README et
  `Package.swift` disent « compile, non supporté à l'exécution ».
- **Proposé** : B tant qu'aucun consommateur iOS n'existe.
- **Fiches** : hors plan (PLAN.md §6) ; K-40 (fp16 si iPhone), K-26.

### ASK-3 — API de streaming Realtime

- **Contexte** : le modèle « Realtime » est un décodeur synchrone à la trame, mais Voxtral n'expose que
  `transcribe(url)` : tout le fichier est encodé avant le premier jeton.
- **Ce qui a été constaté** : P-71, aucune API de flux (`VoxtralRealtimePipeline.swift:118-154`,
  `VoxtralRealtimeModel.swift:57-157`). L'encodeur par tranches de K-13 est le prérequis technique. En streaming,
  l'encodeur et le décodeur alternent : le budget mémoire est leur somme (profils.md §4).
- **Question** : un flux temps réel (micro → texte au fil de l'eau) est-il un objectif produit (FluxForge, démo) ?
- **Options** : A) oui : fiche L après K-13, qui réutilise l'encodeur par tranches ; B) non : « Realtime » est
  documenté comme transcription de fichier à faible coût par pas.
- **Proposé** : B pour ce plan ; A se réévalue après K-13 et K-78.
- **Fiches** : aucune ; hors plan (PLAN.md §6).

### ASK-4 — Périmètre de la matrice de profils

- **Contexte** : le standard impose `<bits>bit-fast|lean` par modèle ; Voxtral a quatre familles (STT Mini 3B,
  STT Small 24B, Realtime 4B, TTS 4B) et un enrôlement.
- **Ce qui a été constaté** : le TTS n'existe qu'en 4, 6 et 16 bits ; aucun pack 8 bits valable n'existe
  (profils.md §5). Le Realtime 8 bits voxmlx se charge **faux en silence** (P-70). Small 8 bits a besoin d'un Mac
  d'au moins 48 Go. Sur audio conversationnel, Small n'est pas plus précis que Mini 8 bits dense
  (modeles-2026-09.md §6).
- **Question** : quelle matrice K-76 doit-il typer et K-77…K-79 mesurer ?
- **Options** : A) complète : STT Mini et Small en 4/8/16, Realtime en 4/16 (8 si le pack PK-1 est publié), TTS en
  4/6/16 (6 déclaré hors standard, voir ASK-19), `enroll-fast|lean` ; B) réduite : STT Mini seul, Realtime 4 bits
  seul, TTS 4/6, enrôlement.
- **Proposé** : A. Les profils Small restent « non mesurés » sur les machines de moins de 48 Go.
- **Fiches** : ⛔ K-76 ; K-77, K-78.

---

## B. Défauts publics et comportement

### ASK-5 — Modèle TTS par défaut

- **Contexte** : le registre, la CLI, `profile` et `VoxtralTTSSynthesisManager` utilisent le bf16 par défaut ;
  FluxForge charge explicitement le 6 bits. Le commentaire de #27 affirme l'inverse du code.
- **Ce qui a été constaté** : ce défaut bf16 est délibéré et documenté (P-34, sévérité basse), mais incohérent
  entre les surfaces (FA-03 : `VoxtralTTSRegistry.swift:37`, `VoxtralCLI.swift:400`, `:588`,
  `StreamingDemoViewModel.swift:14`). Le bf16 est aussi le chemin le plus lent : RTF bf16 6,50/6,32 contre 6 bits
  1,15/1,07, en session et avant K-39 (profils.md §5). La campagne q6 existante contient un détecteur de fuite
  défectueux (K-79).
- **Question** : le défaut TTS passe-t-il au 6 bits si la campagne K-79 le justifie (couverture ASR q6 ≥ bf16 −1 pt
  **et** RTF q6 ≤ 0,5 × bf16) ?
- **Options** : A) oui, sur les 4 surfaces (registre, CLI, démo, manager) ; B) non : le bf16 reste le défaut,
  documenté partout comme « référence de qualité ».
- **Proposé** : A, conditionné à la porte et à ASK-18.
- **Fiches** : ⛔ K-79.

### ASK-6 — Backend de l'encodeur STT par défaut (`.auto` Core ML ou `.mlx`)

- **Contexte** : `.auto` télécharge un encodeur Core ML de 1,32 Go et le compile au premier lancement ; FluxForge
  passe par `.auto`.
- **Ce qui a été constaté** : le gain ANE annoncé (« ~150 ms ANE contre ~500 ms MLX ») n'est pas mesuré. Les unités
  de calcul se contredisent (`.cpuAndGPU` documenté, `.cpuAndNeuralEngine` dans l'`init`). Il n'existe aucune
  parité des embeddings. Coût à froid : 1 min 09,6 s pour Mini et 2 min 25 s pour Small, une fois par architecture
  GPU (issues #16/#22, en session). Gain publié du chemin hybride : −4,3 % (FV-12). Enfin, les bits de l'encodeur
  du pack MLX sont sans effet sous Core ML (M-07). Sources : A-12, A-13, P-14.
- **Question** : si la matrice K-42 montre que Core ML n'est pas ≥ 1,2× plus rapide à chaud avec parité et
  WER ≤ +0,3 pt, le défaut passe-t-il à `.mlx` ?
- **Options** : A) oui, la règle de K-42 décide (changement de comportement noté dans le CHANGELOG, FluxForge
  prévenu) ; B) non, `.auto` est gardé quoi qu'il arrive et le coût du premier lancement est documenté.
- **Proposé** : A.
- **Fiches** : ⛔ K-42 ; conditionne l'utilité du pack PK-3 (ASK-22).

### ASK-7 — Cache KV STT : contexte complet et budget mémoire

- **Contexte** : tous les préréglages `MemoryOptimizationConfig` posent `maxKVCacheSize`, ce qui donne un
  `RotatingKVCache` et un masque maison de mauvaise forme : `fatalError` au préfill à partir de 6, 11, 17 ou
  22 fenêtres de 30 s selon le préréglage ; l'app impose 8 192.
- **Ce qui a été constaté** : S-02 = P-03, simulation exacte en annexe A d'audit-performance-stt.md. Le LM n'a pas de
  fenêtre glissante (131 072 positions). Le coût est d'environ 12,5 jetons audio par seconde, soit ≈ 8 500 jetons
  pour C-long (11 min 22 s). Le cache KV fp32 de Mini fait 240 Kio par jeton, d'où ≈ 2 Gio (calcul ; 120 Kio en
  bf16 après K-40). Le pic réel est **à mesurer**. K-2 livre `KVCacheSimple` par défaut plus un garde-fou explicite,
  puis s'arrête si le pic sur C-long dépasse 12 Go.
- **Question** : si le pic mesuré par K-2 sur C-long dépasse 12 Go, garde-t-on le contexte complet ?
- **Options** : A) oui, le contexte complet prime : le profil `lean` borne la **durée** d'audio acceptée, avec une
  erreur typée au-delà ; B) non : on garde une fenêtre bornée, mais un dépassement lève une erreur Swift et aucun
  audio plus long n'est transcrit.
- **Proposé** : A (la troncature silencieuse et l'arrêt du processus sont les deux défauts à éliminer).
- **Fiches** : K-2 (⛔ après la mesure) ; K-55 (KV 8 bits `lean`), K-77.

### ASK-8 — Sémantique de `maxTokens`

- **Contexte** : en STT, `maxTokens = 500` par défaut, soit environ 3 minutes de parole ; en Realtime, le même champ
  compte des **trames** (4 096 ≈ 5 min 27 s).
- **Ce qui a été constaté** : la troncature est silencieuse dans les deux cas : aucun champ ni aucune erreur ne la
  signale (P-11, P-64). Le Realtime hérite cette sémantique de mlx-audio. Preuves : `VoxtralPipeline.swift:109`,
  `VoxtralRealtimeModel.swift:115-124`.
- **Question** : quel sens donner à `maxTokens` ?
- **Options** : A) défaut `nil` : budget proportionnel à la durée d'audio en STT, boucle bornée par l'audio en
  Realtime ; une valeur explicite reste un plafond de jetons **texte**, signalé (`truncated`) quand il est
  atteint ; B) défaut inchangé (500), avec seulement le signalement `truncated` ; le champ Realtime est déprécié.
- **Proposé** : A.
- **Fiches** : ⛔ K-5.

### ASK-9 — Politique des défauts publics qui changent quand une porte l'emporte

- **Contexte** : plusieurs fiches améliorent un défaut public, ce qui change la sortie observée par les
  consommateurs même quand la signature ne change pas.
- **Ce qui a été constaté** : cinq défauts sont concernés.
  - `maxTokens` (K-5).
  - Plafond effectif de frames TTS, plus bas pour les textes courts (K-14).
  - Rythme `eval`/`clearCache` des préréglages mémoire (K-51).
  - Pénalité de répétition STT de 1,2 à 1,0 : une carte externe mesure −27 % de virgules à 1,2 (P-12, K-61).
  - Durée d'enrôlement par défaut de 8 à 16 s : similarité ECAPA 0,69 contre 0,72, en session (K-64).
- **Question** : un défaut public change-t-il quand la porte chiffrée de sa fiche l'emporte ?
- **Options** : A) oui : version mineure, entrée CHANGELOG, FluxForge prévenu ; B) non : défauts gelés, et le
  nouveau comportement n'est accessible que par un profil ou un paramètre explicite.
- **Proposé** : A pour K-5 et K-14 (stabilité), fiche par fiche pour K-51, K-61 et K-64.
- **Fiches** : ⛔ K-61 ; K-5, ⛔ K-14, K-51, K-64.

### ASK-10 — Warm-up des voix clonées : qualité contre latence

- **Contexte** : pour les voix clonées, le vocalise recommandé est généré, décodé puis jeté à chaque synthèse ; en
  streaming, rien n'est émis avant 3 s d'audio accumulé.
- **Ce qui a été constaté** : le premier chunk utile arrive à 43 frames, soit 1,2 à 1,6 s en 4 bits et ≈ 9 s en bf16
  (calcul, une fois S-08 corrigé). Réutiliser le cache après le porteur est **réfuté** (#45). Raccourcir le porteur
  risque de réintroduire les fuites ou les coupes (1/8 dans #45). Source : P-47.
- **Question** : accepte-t-on un porteur plus court, ou une attente adaptée à sa durée, si K-68 obtient une TTFA
  des voix clonées au moins 30 % plus basse, avec des fuites ≤ la référence (0/10 sur 10 prises) ?
- **Options** : A) oui, la porte décide ; B) non : la qualité du warm-up prime, la TTFA des voix clonées reste
  inchangée.
- **Proposé** : A (l'écoute des 10 prises fait partie de la porte).
- **Fiches** : ⛔ K-68.

---

## C. Qualité et parité

### ASK-11 — Tolérance de parité des leviers numériques (STT, Realtime)

- **Contexte** : les leviers de dtype (bf16 au lieu de fp32), de tête quantifiée et de RoPE fusionnée changent les
  arrondis. Une transcription greedy peut alors différer d'un jeton sans perte de qualité.
- **Ce qui a été constaté** : la référence FR est sans accents et le WER doit être normalisé (casse, ponctuation,
  accents). La référence mlx-audio garde le mel en fp32 : elle n'est pas bit-comparable pour P-60. Source :
  Realtime Q1.
- **Question** : quelle tolérance vaut pour K-13, K-38, K-39 (partie ASR), K-40, K-46 et K-65 ?
- **Options** : A) texte identique sur C-court **et** WER normalisé ≤ référence +0,2 pt sur C-moyen EN/FR, en 16 et
  8 bits (+0,5 pt pour la tête 4 bits de K-46) ; B) texte strictement identique partout : un levier qui change un
  seul jeton est rejeté.
- **Proposé** : A.
- **Fiches** : ⛔ K-13, K-38 ; K-39, K-40, K-46, K-65.

### ASK-12 — mlx-audio comme référence de parité des fenêtres Realtime

- **Contexte** : au-delà de 15 s, la sortie Swift actuelle est fausse : les fenêtres glissantes déclarées
  (encodeur 750, décodeur 8 192) sont ignorées. Elle ne peut donc pas servir de référence pour K-13.
- **Ce qui a été constaté** : mlx-audio `main` implémente les deux fenêtres (`encoder.py:188-219`,
  `decoder.py:226-229`), le `RotatingKVCache(8192)`, `as_linear` et la tête quantifiée (P-62, P-63). Les fichiers
  Realtime de Voxtral citent mlx-audio en en-tête.
- **Question** : accepte-t-on la sortie de mlx-audio (Python, Mac, commit épinglé dans la fiche) comme référence de
  WER pour les audios de plus de 15 s ?
- **Options** : A) oui ; B) non : seules les portes internes s'appliquent (embeddings identiques ≤ 15 s, texte
  identique sur C-moyen), sans comparaison externe.
- **Proposé** : A.
- **Fiches** : ⛔ K-13.

### ASK-13 — Écoute à l'aveugle pour les leviers numériques TTS

- **Contexte** : le flow matching est stochastique et à rétroaction : une comparaison libre avant/après diverge par
  construction. Les portes objectives sont la parité forcée par l'enseignant, la couverture ASR et le SNR.
- **Ce qui a été constaté** : toutes les parités TTS antérieures à `07e6317` ont été jugées à l'écoute, pas au bit
  près (faits-et-actions.md §2.1). Quatre fiches touchent la précision audio : FM en bf16 (K-39), pas de flow
  matching au-dessous de 8 (K-48), codec en bf16 (K-58), défaut q6 (K-79).
- **Question** : ces fiches exigent-elles, en plus de la porte objective, une écoute A/B à l'aveugle par Vincent
  avant de conclure ?
- **Options** : A) oui : la fiche prépare les paires (fichiers anonymisés, grille), passe en `blocked` jusqu'à
  l'écoute, puis conclut ; B) non : la porte objective suffit (parité forcée, couverture ASR ≥ référence −0,5 pt,
  SNR ≥ 40 dB, 0 `maxFrames`).
- **Proposé** : A pour K-48 et K-79 (changement audible possible), B pour K-39 et K-58.
- **Fiches** : ⛔ K-39, K-48, K-58 ; K-79, K-80.

### ASK-14 — Corpus réel long et modèle de référence du WER

- **Contexte** : le corpus disponible est synthétique : clips FluxForge générés par le TTS en 6 bits, C-long et
  C-xlong obtenus par concaténation. Une parole de synthèse biaise le WER.
- **Ce qui a été constaté** : aucun enregistrement réel long avec transcription de référence n'existe dans le dépôt
  (P-19, P-78, FA-07). Le texte de référence extrait de `docs/tts_benchmark.md` compte peut-être moins de mots que
  les « ~350 mots » annoncés ; K-33 le vérifie. Le choix entre Mini et Small est une question de corpus
  (profils.md §2).
- **Question** : un enregistrement réel d'au moins 12 min, avec sa transcription de référence, est-il fourni ?
- **Options** : A) oui, dans le dépôt, avec son SHA-256 dans `docs/Benchmarks.md` ; B) oui, hors dépôt (chemin
  local et SHA-256 notés, non versionné) ; C) non : corpus synthétique seul, biais noté à chaque ligne de mesure.
  Question liée : le WER de référence se mesure-t-il sur Mini 3B 2507 (STT) ou aussi sur Realtime 4B 2602 ?
- **Proposé** : B, avec les deux modèles mesurés.
- **Fiches** : K-33, K-77.

---

## D. Poids, packs, licences

### ASK-15 — Dépôt retenu pour `small-24b-8bit`

- **Contexte** : le même id de modèle pointe vers deux dépôts : l'enum vers `mzbac/…`, le registre et le README vers
  `VincentGOURBIN/…`. L'app télécharge l'un, `loadModel` cherche l'autre.
- **Ce qui a été constaté** : `mzbac/Voxtral-Small-24B-2507-8bit` pèse 28 056 927 031 o et
  `VincentGOURBIN/voxtral-small-8bit` 26 499 134 369 o, pour une config identique ; l'écart n'est pas expliqué.
  Une troisième option existe : `MarkusKaemmerer/…-8bit-dense-encoder`, 27 138 066 384 o, encodeur bf16,
  chargeable après K-8. Sources : `VoxtralPipeline.swift:47-48`, `ModelRegistry.swift:90-91`, S-06, Q-M1.
- **Question** : quel dépôt `small-24b-8bit` désigne-t-il ?
- **Options** : A) `VincentGOURBIN/voxtral-small-8bit`, cohérent avec le README ; B) `mzbac/…` ;
  C) `MarkusKaemmerer/…-8bit-dense-encoder`, épinglé par révision et SHA-256.
- **Proposé** : A dans K-10 (aucun changement pour les utilisateurs actuels) ; C évalué dans K-77.
- **Fiches** : ⛔ K-10.

### ASK-16 — Packs tiers : référencer ou republier

- **Contexte** : les meilleurs packs STT de 2026 (encodeur dense, tête 6 bits) sont publiés par des tiers sous
  Apache-2.0 (MarkusKaemmerer, noScribe).
- **Ce qui a été constaté** : un pack externe mesure un WER de 4,27 % contre 4,74 % en 8 bits uniforme, sur un
  passage difficile (M1 Max, Python, externe). Un pack tiers peut disparaître ou changer. Le standard exige révision,
  SHA-256 et recette reproductible (modeles-2026-09.md §7, PK-4).
- **Question** : les profils peuvent-ils pointer vers des dépôts tiers ?
- **Options** : A) oui, épinglés par révision et SHA-256 (vérifiés au téléchargement par K-6) ; B) non : miroir sous
  `VincentGOURBIN`, avec la carte d'origine et la licence reprises.
- **Proposé** : A.
- **Fiches** : K-80.

### ASK-17 — `realtime-4b` (original Mistral) : corriger ou retirer l'entrée

- **Contexte** : `mistralai/Voxtral-Mini-4B-Realtime-2602` a reçu après coup un `config.json` au format
  transformers et un `model.safetensors`. Le chargeur Voxtral sonde `config.json` en premier et casse.
- **Ce qui a été constaté** : l'entrée `realtime-4b` n'est plus chargeable. Le glob `*.safetensors` télécharge
  17,72 Go au lieu de 8,87. Un `modelId` inconnu retombe en silence sur le défaut (M-01, M-05,
  `VoxtralRealtimePipeline.swift:78-79`).
- **Question** : corrige-t-on l'entrée ou la retire-t-on ?
- **Options** : A) corriger : liste de fichiers fixée par entrée, config Mistral lue depuis `params.json`, id strict
  (changement additif) ; B) retirer l'entrée (**cassant** pour qui passe cet id).
- **Proposé** : A.
- **Fiches** : ⛔ K-9.

### ASK-18 — Licence des poids TTS (CC BY-NC 4.0) dans un usage commercial

- **Contexte** : d'après la carte Mistral, les poids `Voxtral-4B-TTS-2603` sont sous CC BY-NC 4.0. `VoxtralCore` est
  intégré à FluxForge Studio (App Store) et à la chaîne LipDub/LTX.
- **Ce qui a été constaté** : le dépôt ne documente pas cette licence à l'endroit où le modèle est recommandé
  (profils, README). Un pack tiers (`majentik/…-TurboQuant-MLX-8bit`) déclare Apache-2.0 sur une base
  CC BY-NC ; il est rejeté. Ce point relève d'une **vérification juridique**, pas d'une décision technique
  (Q-M4).
- **Question** : le TTS est-il exposé dans un usage commercial ?
- **Options** : A) non, ou une licence commerciale a été obtenue : les profils TTS peuvent être recommandés sans
  réserve et PK-2 peut être publié (dérivé, attribution) ; B) oui, ou c'est incertain : chaque profil et chaque
  page TTS porte la mention « non commercial », aucune recommandation commerciale, PK-2 n'est pas publié.
- **Proposé** : aucune proposition (question juridique) ; B par défaut tant qu'il n'y a pas de réponse.
- **Fiches** : ⛔ K-79 (recommandation), ⛔ K-80 (pack TTS).

### ASK-19 — Largeurs TTS : 6 bits hors standard, 8 bits à publier

- **Contexte** : le standard prévoit des largeurs de 4, 8 et 16 bits. Le TTS existe en 4, 6 et 16 bits ; le 6 bits
  est le seul pack livré par FluxForge.
- **Ce qui a été constaté** : la couverture ASR des voix clonées est de 99,4 % en 6 bits contre 96,5 % en bf16
  (n = 15, une voix, en session). Aucun pack 8 bits valable n'existe. PK-2 (LLM et FM en 8 bits, codec bf16) est
  estimé à 4,37 Go (profils.md §5, Q-M5).
- **Question** : quelles largeurs TTS déclarer ?
- **Options** : A) 4/6/16, avec 6 bits déclaré comme largeur intermédiaire hors standard, et publication d'un 8 bits
  (PK-2, sous réserve d'ASK-18) ; B) 4/6/16 sans 8 bits.
- **Proposé** : B tant qu'ASK-18 n'est pas tranchée, puis A si la licence le permet.
- **Fiches** : K-80 ; K-76 (types).

### ASK-20 — Realtime 8 bits : format voxmlx ou pack propre

- **Contexte** : les packs Realtime 6 et 8 bits de la communauté sont au format voxmlx. Voxtral les charge
  **faux en silence** : poids aléatoires, sans plantage (P-70).
- **Ce qui a été constaté** : tout le modèle voxmlx, décodeur `language_model.*` compris, resterait aléatoire après
  chargement. Un pack 8 bits au format mlx-audio (PK-1, encodeur et décodeur 8 bits g64, `tok_embeddings` 8 bits)
  est estimé à 4,73 Go (Realtime Q4).
- **Question** : comment offrir le Realtime 8 bits ?
- **Options** : A) écrire un assainisseur voxmlx (lecture des packs existants) ; B) exporter un pack propre au
  format mlx-audio (PK-1), avec SHA-256 et `Weights.md`, et refuser explicitement le format voxmlx.
- **Proposé** : B.
- **Fiches** : K-78, K-80.

### ASK-21 — Modes de quantification non affines (mxfp4, mxfp8, nvfp4)

- **Contexte** : les convertisseurs de 2026 écrivent `"mode"` dans `quantization`. Le décodeur maison de Voxtral
  refuse tout le fichier, et un chargeur qui force `.affine` chargerait un mxfp4 faux.
- **Ce qui a été constaté** : sur Voxtral (externe), mxfp4 fait moins bien qu'affine en 4 bits, et nvfp4 sans échelle
  globale casse le modèle. mlx-swift 0.31.6 n'a pas d'échelle globale NVFP4. Il n'existe aucune mesure MLX Swift
  (M-02, M-04, Q-M8).
- **Question** : confirme-t-on ces modes **hors profils**, refusés explicitement au chargement ?
- **Options** : A) oui : erreur explicite (K-8) ; B) non : chargement accepté en expérimental, sans profil.
- **Proposé** : A.
- **Fiches** : ⛔ K-8.

### ASK-22 — Packs à publier et compte de publication

- **Contexte** : aucun pack n'est publié par ce plan sans décision. Quatre packs sont préparés et chiffrés
  (modeles-2026-09.md §7).
- **Ce qui a été constaté** : quatre packs sont préparés.
  - PK-1 : Realtime 8 bits au format mlx-audio, 4,73 Go.
  - PK-2 : TTS 8 bits, 4,37 Go, CC BY-NC.
  - PK-3 : Mini « LM 4 bits, `lm_head` 6 bits, encodeur 8 bits ou bf16 », 3,05 ou 3,67 Go ; utile seulement si
    K-42 retient `.mlx`.
  - PK-4 : Small 4 et 8 bits à encodeur dense ; réutiliser les dépôts Markus.
- **Question** : quels packs K-80 prépare-t-il pour publication sous `VincentGOURBIN`, une fois les portes de parité
  passées ?
- **Options** : A) PK-1 ; PK-3 si ASK-6 = A et que K-42 retient `.mlx` ; PK-2 seulement si ASK-18 = A ; PK-4 par
  référence (ASK-16) ; B) aucun : les profils ne référencent que les packs existants.
- **Proposé** : A. Le script `Scripts/publish-packs.sh` prépare le staging et n'uploade rien : l'envoi reste un
  geste de Vincent.
- **Fiches** : ⛔ K-80.

---

## E. API publique

### ASK-23 — API legacy, code mort public, API factice et modules factices

- **Contexte** : `VoxtralCore` expose une famille legacy parallèle au pipeline : 216 appels, `loadVoxtralModel`,
  `VoxtralGenerator`, `LlamaModel`, `VoxtralConfig`, `PythonVoxtralConfig`. Il expose aussi une API de
  téléchargement factice et des modules publics chargés pour rien.
- **Ce qui a été constaté** : quatre ensembles sont concernés.
  - Code mort public (S-13 lot 2) et famille legacy (S-14).
  - `downloadModel(modelId:)`, `ModelDownloader.hubApi`, `reconfigureHubApi` : ne font rien ou trompent (A-05).
  - `audioTower` et `multiModalProjector` chargés inutilement (P-24).
  - Types du double chargeur (S-16).

  Consommateurs connus : FluxForge utilise `VoxtralPipeline`, `ModelRegistry`,
  `ModelDownloader.customModelsDirectory`, `RuntimeBeacon.isEnabled` et `VoxtralTTSPipeline`. On relève 0 usage de
  `loadVoxtralModel`, `VoxtralGenerator` et `TekkenTokenizer`, d'après une recherche de code GitHub **non
  re-vérifiable** ici.
- **Question** : quel calendrier pour ces symboles ?
- **Options** : A) dépréciation annotée en 2.3 (additif, K-30), suppression en 3.0 ; B) suppression directe en 3.0
  (**cassant**).
- **Proposé** : A. K-59 vide les modules factices sans les supprimer ; K-74 remplace les types publics du double
  chargeur, avec des typealias dépréciés.
- **Fiches** : ⛔ K-30, ⛔ K-74 ; K-59.

### ASK-24 — Conformance `LanguageModel` (cause de la casse #50)

- **Contexte** : les deux classes de modèle STT se conforment à `LanguageModel` et `KVCacheDimensionProvider`.
  Aucun code ne s'en sert, et le `prepare` passe des embeddings comme des ids.
- **Ce qui a été constaté** : le changement de signature en amont (`prepare(_:cache:state:prefill:)`) a cassé le
  build (#50, FV-03). Aucun consommateur connu n'utilise cette conformance (S-17, P-18, MLX-006).
- **Question** : que fait-on de la conformance ?
- **Options** : A) la retirer (cassant théorique, 0 consommateur connu, fin du couplage à `main`) ; B) la rendre
  juste : `prepare` conforme, génération par `TokenIterator` (effort L, gains P-06/P-07/P-12/P-16 attendus).
- **Proposé** : A, sauf si ASK-28 = B et qu'un gain amont est démontré.
- **Fiches** : ⛔ K-75.

### ASK-25 — Revue de la surface publique pour la 3.0

- **Contexte** : environ 1 100 déclarations publiques et 45 fonctions libres (39 noms distincts), aux noms
  génériques (`ModelRegistry`, `RuntimeBeacon`…) déjà en collision chez FluxForge.
- **Ce qui a été constaté** : FluxForge contourne ces collisions par des typealias (`ModelManager.swift`,
  `LTXBeaconBridge.swift`). `applyChatTemplate` rend un type non typé. La faute de frappe
  `applyTranscritionRequest` est publique. Source : S-21.
- **Question** : valide-t-on, pour la 3.0, la liste de façades proposée par K-31 (3 pipelines, 3 managers,
  registres, `RuntimeBeacon`, enrôlement, `WAVWriter`), avec noms préfixés et typealias dépréciés ?
- **Options** : A) oui : la liste est relue et amendée par Vincent, puis appliquée ; B) non : pas de revue 3.0,
  seulement l'alias corrigé de la faute de frappe.
- **Proposé** : A, après K-30.
- **Fiches** : ⛔ K-31.

### ASK-26 — Produit `VoxtralBenchmark`

- **Contexte** : le produit exécutable `VoxtralBenchmark` mesure un chemin que personne n'exécute. L'instrument
  réel (`VoxtralCLI bench`, K-32) est à créer.
- **Ce qui a été constaté** : `VoxtralBenchmark` fait des conversions Float16 sur des données aléatoires, copiées
  « same as MLXCoreMLBridge ». La constante `typicalInferenceMs = 30000` est codée en dur. Il n'y a ni sortie JSON
  ni révision notée (A-16, piège 33).
- **Question** : que devient `VoxtralBenchmark` ?
- **Options** : A) réorienté : le produit est gardé et appelle le banc `bench` ; B) retiré du `Package.swift`
  (**cassant** pour le seul produit exécutable).
- **Proposé** : B, puisque `bench` le remplace.
- **Fiches** : ⛔ K-32.

### ASK-27 — Ressource `VoxtralEncoderFull.mlmodelc` de VoxtralApp

- **Contexte** : `Package.swift:79-80` copie `Resources/VoxtralEncoderFull.mlmodelc`, mais le dossier est ignoré par
  git et absent d'un clone neuf.
- **Ce qui a été constaté** : le comportement de SwiftPM face à cette ressource absente reste **à vérifier**, par un
  build sur clone neuf. `create_app_bundle.sh` empaquette l'exécutable Debug sans bundle de ressources. Le
  téléchargement de l'encodeur à l'exécution est déjà supporté (A-04, A-14).
- **Question** : l'encodeur Core ML reste-t-il embarqué dans l'app ?
- **Options** : A) non : il est téléchargé à l'exécution uniquement et la ligne `.copy` est retirée ; B) oui : un
  script de récupération est lancé avant le build et documenté.
- **Proposé** : A.
- **Fiches** : ⛔ K-28.

---

## F. Dépendances amont

### ASK-28 — Épinglage de `mlx-swift-lm`

- **Contexte** : `mlx-swift-lm` est suivi sur `branch: "main"` (tête `ee673d6`, dernier tag 3.31.4) et
  `Package.resolved` est ignoré par git : deux clones peuvent résoudre deux révisions différentes.
- **Ce qui a été constaté** : la casse #50 est avérée. Le commentaire « Revisit once ml-explore cuts a tag beyond
  3.31.4 » n'a pas de plan de suivi. FluxForge impose aujourd'hui la même branche (S-18, FA-02, ACT-55).
- **Question** : comment épingler ?
- **Options** : A) rester sur `main` tant que FluxForge l'impose, avec `Package.resolved` suivi dans git ; B) passer
  ensemble (FluxForge et Voxtral) au prochain tag > 3.31.4 avec `from:`, dès qu'il existe.
- **Proposé** : A tout de suite (K-22), B au premier tag. Le plan `upstream-blocker` de K-21 surveille ce tag.
- **Fiches** : ⛔ K-22 ; K-21.

### ASK-29 — mlx-swift 0.31.6 sans le correctif du deadlock compile × vjp

- **Contexte** : Voxtral résout mlx-swift 0.31.6 (`0bb916c`). Le correctif `df9ae26` (#461) n'est dans aucun tag,
  et mlx-swift-lm impose `upToNextMinor(from: "0.31.6")`.
- **Ce qui a été constaté** : un épinglage sur branche dans Voxtral entrerait en conflit avec la contrainte de
  mlx-swift-lm. Le deadlock A-01 est évitable côté Voxtral par l'exclusion de K-11, indépendamment du tag
  (Q-M7).
- **Question** : quelle attitude vis-à-vis de l'amont ?
- **Options** : A) accepter 0.31.6 jusqu'au prochain tag, avec K-11 comme parade et un plan `upstream-blocker`
  créé avec celui de K-21 ; B) demander un tag en amont (issue sur `ml-explore/mlx-swift`, écrite par Vincent).
- **Proposé** : A, B en complément si le tag tarde.
- **Fiches** : aucune (PLAN.md §6) ; K-11, K-21.

---

## G. Dépôt et tracker

### ASK-30 — WAV suivis malgré `.gitignore`

- **Contexte** : 24 fichiers sont suivis malgré `.gitignore` : 22 WAV (70,2 Mio) et le cache `.serena/*.pkl`
  (1 790 543 o). Les 4 clips `fluxforge_{short,long}_{en,fr}_6bit.wav` servent de corpus de mesure.
- **Ce qui a été constaté** : 14 WAV sont référencés par la doc ; 8 ne le sont pas (18 274 912 o). Réduire le poids
  du dépôt pour les clones existants exige une réécriture d'historique (S-25).
- **Question** : que fait-on des WAV ?
- **Options** : A) garder les WAV référencés (exceptions explicites dans `.gitignore`), retirer les 8 autres de
  l'arbre, **sans** réécrire l'historique ; B) déplacer les WAV en assets de Release GitHub ou en Git LFS
  (liens de doc mis à jour). Réécrire l'historique exige un « oui » explicite, en plus de A ou B.
- **Proposé** : A, sans réécriture.
- **Fiches** : ⛔ K-20.

### ASK-31 — Écritures dans le tracker `action-plans`

- **Contexte** : trois plans `ready-to-act` (#71, #307, #349) ont une source close depuis 61 à 79 jours.
  `Package.swift` dépend d'une branche amont sans plan de suivi.
- **Ce qui a été constaté** : les sources sont vérifiées en lecture seule (PR #34 et #41 fusionnées, #45 fermée) et
  les commentaires de preuve sont prêts (faits-et-actions.md §3.9). Les actions FluxForge (déchargement, HubApi,
  réenrôlement, doc de stockage) sont hors de ce dépôt. Cet audit n'a rien écrit sur GitHub.
- **Question** : l'agent peut-il écrire dans `VincentGourbin/action-plans` ?
- **Options** : A) oui : clôture `verified` de #71, #307 et #349 avec preuve, création du plan `upstream-blocker`
  (tag mlx-swift-lm > 3.31.4) et d'un plan `manual` pour FluxForge ; B) non : Vincent le fait lui-même à partir des
  textes prêts.
- **Proposé** : A.
- **Fiches** : ⛔ K-21.

---

### ASK-32 — K-34 : cellule small-4bit `.auto` × C-moyen EN sans A/A

- **Contexte** : la porte de K-34 exige A/A ≤ 3 % pour chaque ligne. La fiche dit : « refaire, sinon `blocked` ».
  31 cellules sur 32 et les 2 lignes de chat passent.
- **Ce qui a été constaté** : 7 séries, 11 passes en processus séparés (2026-10-04 et 05), dont une écartée pour
  charge le 2026-10-04. `step_ms_p50` vaut 74,99 à 75,72 ms dans 6 passes et 80,56 à 85,38 dans 5 ; `out_sha256`,
  WER (1,32 %) et pic (≈ 19,0 Go) sont identiques partout.
  - Les ralentissements ne suivent pas l'ordre des passes (p1 lente 3 fois, p2 lente 2 fois).
  - Deux passes lentes avaient une charge visible : `ANECompilerService` 100 % (p1 de la 6e série), Time Machine
    397 % (p2 de la 6e série).
  - Dans la 7e série, Time Machine était arrêté et aucun processus ne dépassait 50 % CPU, mais la p2 tournait à
    80,56 ms.
  - Un processus d'amorçage à froid décode à 190,7 ms.
  - Les autres cellules de Small sont stables : `.auto` C-moyen FR 74,77 / 74,71 ; `.mlx` C-moyen EN 0,0x %.
  - Cause non isolée.
- **Question** : comment traiter cette cellule dans la baseline K-34 ?
- **Options** :
  - A) baseline sans A/A pour cette cellule : régime bimodal consigné (75,0 / 80,6–85,4 ms, 11 passes), valeur de
    référence = médiane des passes rapides. Fiche de suivi pour en trouver la cause (Core ML et Neural Engine de
    Small `.auto`) ; K-34 `verified` avec cette réserve.
  - B) K-34 reste `blocked` jusqu'à ce que la fiche de suivi isole la cause et que la cellule passe l'A/A.
  - C) retirer la cellule `.auto` × C-moyen EN de Small de la porte (Small `.auto` reste mesuré sur les 3 autres
    clips).
- **Proposé** : A. Les fiches qui s'appuieront sur cette cellule (K-42, `.auto`) comparent des A/B/B/A, qui
  verront le régime bimodal.
- **Fiches** : ⛔ K-34 (#587).

---

### ASK-33 — Plafond de cache par défaut pour Small sur un Mac de 32 Go ou moins

- **Contexte** : K-34 (`docs/knowledge/decisions/small-32gb.md`) : small-4bit sur 8 min d'audio, pic 37 313 Mo sans
  plafond, 19 013 Mo avec `--cache-limit-mb 2048`.
- **Question** : faut-il un plafond de cache MLX par défaut pour `small-*` sur les machines de 32 Go ou moins ?
- **Options** : A) oui, Small seulement, sur ≤ 32 Go ; B) non, documenter et avertir ; C) plafond pour tous.
- **Proposé** : A, si K-52 confirme un temps à ± 5 %.
- **Fiches** : K-52.

---

### ASK-34 — K-35 : cellules streaming moyen et long sans A/A

- **Contexte** : le streaming TTS re-décode tout l'accumulé à chaque morceau (`VoxtralTTSPipeline.swift`,
  `decodeToWaveform(chunk.accumulatedCodes)`), donc le coût par frame croît avec la longueur. Mesures : 41 ms par
  frame (texte court), environ 80 (moyen), 140 à 166 (long), contre 38 à 39 ms par pas en batch.
- **Ce qui a été constaté** : 4 cellules streaming moyen ou long varient de 3 à 25 % d'un processus à l'autre, même
  sur la médiane par frame (`stream_frame_ms_p50`).
- **Question** : comment traiter ces cellules dans la baseline K-35 ?
- **Options** : A) les garder sans A/A, cause consignée, référence « avant » de K-43 ; B) K-35 bloquée jusqu'à K-43 ;
  C) les retirer de la porte.
- **Proposé** : A.
- **Fiches** : ⛔ K-35 (#588) ; K-43.

### ASK-35 — K-39 : référence de la couverture ASR du FM bf16

- **Contexte** : K-39 (#610) passe les couches du FM en bf16 pour tts-4b-mlx : pas − 48,8 %, parité forcée 99,23 %
  Δ1. Porte recalée : couverture ASR (3 textes × 5 graines) ≥ référence K-35 − 0,5 pt, référence prise à la graine 1
  (0,909 / 0,952 / 0,955).
- **Ce qui a été constaté** (`BENCHMARKS.md` §« 2026-10-11 — K-39 ») : texte long, B graine 1 = 0,949 (− 0,6 pt,
  deux mots sur 334) et moyenne 5 graines 0,945 (− 1,0 pt) : la clause échoue à la lettre. Mais A (code d'avant, même
  jour, mêmes graines) a une moyenne de 0,923 sur le long (0,826 à 0,955 selon la graine) : A échoue aussi à cette
  lecture, la référence d'une seule graine ne représente pas cinq graines. B ≥ A en moyenne sur les trois textes
  (0,909 / 0,950 / 0,945 contre 0,909 / 0,950 / 0,923).
- **Question** : quelle référence juge la couverture de K-39 (et des autres leviers TTS numériques) ?
- **Options** : A) B ≥ A − 0,5 pt, moyenne des 5 graines, A mesuré le même jour (K-39 passe) ; B) référence K-35 −
  0,5 pt sur la moyenne des 5 graines (K-39 échoue en long, code retiré) ; C) remesurer le long sur 10 graines A et B
  (≈ 2 h de GPU), puis A).
- **Proposé** : A.
- **Fiches** : ⛔ K-39 (#610) ; mêmes clauses dans K-48, K-58, K-79.

### ASK-36 — K-39 : clause « pas 4 bits − 5 % » non tenue

- **Contexte** : la fiche dit « gain sous la porte ⇒ code retiré, fiche retirée ». Le changement gardé ne touche pas
  les packs quantifiés (sorties 4 et 6 bits identiques bit à bit, + 0,20 % / − 0,13 %).
- **Ce qui a été constaté** : trois variantes bf16 pour les packs quantifiés échouent à la parité (98,48 % Δ2 à
  98,09 % Δ3) ; la seule rapide (tout le FM en bf16, environ − 9 % en 4 bits, essai d'une passe hors protocole)
  est la moins fidèle.
- **Question** : que faire de la clause 4 bits ?
- **Options** : A) clause retirée avec la mesure, K-39 conclut sur le bf16 seul, le gain 4 bits reste à K-57 ; B)
  fiche entière retirée (code bf16 retiré aussi) ; C) garder la variante rapide en 4 bits et assouplir la parité.
- **Proposé** : A.
- **Fiches** : ⛔ K-39 (#610) ; K-57.

### ASK-37 — Plafond de cache MLX par défaut en TTS

- **Contexte** : K-95 (#635, 2026-10-16), A/B/B/A sur M3 Max 96 Go, profil par défaut contre
  `--cache-limit-mb 2048` (`BENCHMARKS.md` §« 2026-10-16 — K-95 »). Cinq cellules : 6 bits et 4 bits sur les textes
  moyen et long, plus la ligne consommateur (6 bits, `clone_fr`, moyen). Les 10 A/A passent (0,00 à 0,31 %).
  Aujourd'hui, `VoxtralTTSPipeline.Configuration.cacheLimitBytes` vaut `nil` par défaut, c'est-à-dire sans plafond.
- **Ce qui a été constaté** :
  - `step_ms_p50` : de − 0,47 à + 0,05 %.
  - `total_ms` : de + 0,1 à + 0,9 %, soit + 50 à + 486 ms. Le surcoût vient de la phase `codec`.
  - Pic `phys_footprint` : − 3,6 à − 5,1 Go sur le texte moyen (− 26 à − 32 %), − 11,1 Go sur le texte long
    (− 33 %).
  - Sorties identiques à l'octet.
  - Le pic MLX actif ne change pas : le texte long garde 22 Go plafonné.
  - Le seuil de K-79 pour un profil `lean` est tenu largement : pic − 20 % pour au plus + 5 % de temps.
- **Question** : quel plafond de cache MLX par défaut pour `VoxtralTTSPipeline` ?
- **Options** :
  - **A)** Pas de plafond par défaut, comme aujourd'hui. On documente `cacheLimitBytes` (2048 Mo conseillés) et
    K-79 ne l'applique qu'aux profils `lean`.
  - **B)** Plafond de 2048 Mo par défaut pour toutes les machines, désactivable par l'hôte. C'est un défaut public :
    version mineure, CHANGELOG, FluxForge prévenu (règle d'ASK-9). Il faut aussi un moyen additif de dire « sans
    plafond », puisque `nil` deviendrait le défaut plafonné : par exemple une valeur 0 documentée ou une constante.
  - **C)** Plafond par défaut seulement sous N Go de RAM, désactivable, sur le modèle de la réponse à ASK-33 pour
    Small. Je propose N = 32.
- **Proposé** : B. Le coût mesuré est sous 1 % de temps total sur cette machine, et la baisse de 26 à 33 % de
  l'empreinte profite aussi aux hôtes qui partagent la mémoire (FluxForge). Limites : une seule machine a été
  mesurée, et le streaming n'est pas comparé (la baseline K-35 le mesure déjà plafonné). Si tu préfères ne changer
  le défaut que là où il manque de mémoire, prends C.
- **Fiches** : ⛔ K-95 (#635, `blocked` jusqu'à la réponse) ; application dans K-79.

### ASK-38 — K-40 : parité « WER ≤ réf + 0,2 pt » manquée d'un mot sur deux cellules `.auto` FR

- **Contexte** : K-40 (#611, 2026-10-16) fait calculer le STT dans le dtype du décodeur : bf16 pour mini-3b, fp16 pour
  les packs 8 et 4 bits, au lieu de tout promouvoir en fp32. A/B/B/A sur M3 Max 96 Go, A = `be7f8df9`, B = `215452fa`
  (`BENCHMARKS.md` §« 2026-10-16 — K-40 »). La porte de parité est celle d'ASK-11 = A : texte identique sur C-court et
  WER normalisé ≤ référence + 0,2 pt sur C-moyen EN/FR. Sur C-moyen FR (406 mots), un mot vaut 0,25 pt.
- **Ce qui a été constaté** :
  - Vitesse et mémoire tenues partout, 18 A/A sous 3 % : mini-3b `.mlx` C-moyen EN `step_ms_p50` 135,7 → 28,7 ms
    (− 79 %), pic `phys_footprint` − 4,4 Go ; packs 8 et 4 bits, `.mlx` et `.auto`, moyen et long : `step_ms_p50`
    − 19 à − 30 %, `ttft_ms` − 7 à − 23 %, `peak_mlx_mb` − 13 à − 28 %.
  - C-court : texte identique dans les 6 cas.
  - C-moyen (12 cellules) : 8 égales, 2 meilleures (mini-3b EN : `.mlx` 1,05 → 0,53, `.auto` 0,79 → 0,53), 2 moins
    bonnes d'un mot, donc au-delà du seuil : mini-3b `.auto` FR et 8 bits `.auto` FR, 1,97 → 2,22 (« explorer » →
    « explorez », « l'exporte » → « l'export »).
  - Corpus élargi (c_20s EN/FR, es_1 à es_3 : 30 cas) : WER identique partout. C-long exact (WER de 52 à 56 % dans
    les deux versions, le modèle saute des passages) : 8 bits `.auto` 53,75 → 53,05, 4 bits `.auto` 55,73 → 55,53,
    mini-3b `.auto` 52,54 → 52,67, mini-3b `.mlx` 52,16 → 54,13 ; les quatre autres cellules sont identiques.
  - Repli prévu par la fiche, encodeur fp32 et décodeur bf16 (sonde non commitée, `.mlx`) : il ne règle rien. Les
    packs quantifiés retrouvent la sortie de A, mini-3b perd le gain du moyen EN (1,05) et reste à 53,75 sur le long.
    Il ne touche pas `.auto`, dont l'encodeur est Core ML : les deux cellules en échec ne changent pas.
- **Question** : K-40 est-elle acceptée malgré ces deux écarts d'un mot ?
- **Options** :
  - **A)** Oui, telle quelle : le bilan sur C-moyen est net positif (− 0,78 pt en EN, + 0,50 pt en FR sur 12
    cellules), C-court et les 30 cas courts sont identiques, et l'écart est un mot par clip. La fiche est vérifiée
    sur cette exception, écrite dans `ASK.md` §Dérogations ; le seuil d'ASK-11 reste le même pour les autres fiches.
  - **B)** Oui, mais K-40 ne s'applique qu'au backend `.mlx` : `.auto` garde son décodeur en fp32. La parité tient
    alors à la lettre sur C-moyen ; `.auto` perd ses gains (− 20 à − 30 % par pas, − 15 à − 28 % de pic MLX).
  - **C)** Non : K-40 est retirée (revert de `215452fa`) et le calcul reste en fp32.
- **Proposé** : A. Un mot sur un clip de 406 mots est sous la résolution du seuil, et les écarts vont dans les deux
  sens. Limite : un seul clip FR long de référence (C-moyen FR) ; le C-long exact varie de 2 pt selon les arrondis
  dans les deux versions et ne départage pas.
- **Fiches** : ⛔ K-40 (#611, `blocked` jusqu'à la réponse) ; en attente derrière elle : K-42, K-45, K-53, K-54,
  K-55, K-56, K-60.

### ASK-39 — K-68 : « 0 fuite sur 10 prises » (ASK-10) est-il un seuil absolu ou « pas pire que la référence » ?

- **Contexte** : K-68 (#618) coupe le porteur de warm-up sur le waveform partiel en streaming : TTFA des voix clonées
  avec porteur − 42,6 % en 6 bits (1,91 → 1,09 s, A/B/B/A, `BENCHMARKS.md` §« 2026-10-11 — K-68 »). La porte
  (ASK-10 = A) demande « fuites du porteur ≤ la référence (0/10) sur 10 prises », écoutées à l'aveugle par Vincent.
- **Ce qui a été constaté** : à l'écoute du 2026-10-16, les prises A (avant K-68) et B (après) sont identiques à
  1 pas 16 bits près, avec la même coupe sur 19/19 cas ; mais la référence A elle-même laisse entendre le porteur sur
  2 prises sur 10 (graines 02 et 08, coupées trop tôt, frames 3 et 4) et perd « FluxForge Studio » sur une (graine 01,
  coupe à la frame 19). Le « 0/10 » venait de K-35 (ASR en batch), jamais réécouté. K-68 ne change ni le nombre ni la
  place des fuites ; la retirer ne rendrait pas le 0/10.
- **Question** : la clause « fuites » de K-68 se juge-t-elle par rapport à la référence (tenue) ou au 0/10 absolu
  (manquée, avant comme après) ?
- **Options** :
  - **A)** Par rapport à la référence : K-68 est validée (gain de TTFA gardé) ; les 2/10 fuites et la graine 01 vont
    à une fiche de suite sur la coupe du porteur, avec sa propre porte d'écoute.
  - **B)** 0/10 absolu : K-68 reste `blocked` jusqu'à la fiche de suite sur la coupe, puis l'écoute est refaite sur
    les 10 graines.
  - **C)** 0/10 absolu, et K-68 est retirée : le streaming attend de nouveau 3 s (TTFA 1,91 s), sans changer les fuites.
- **Proposé** : A. La garantie « ne pas aggraver » est tenue ; le défaut est antérieur et mérite sa propre fiche.
- **Fiches** : ⛔ K-68 (#618, `blocked` jusqu'à la réponse ; vérification : vérificateur `verified` au 3e passage,
  contradicteur `not verified` sur ce seul point).

---

## Réponses

À remplir par Vincent (une ligne par ASK ; « A », « B », « C » ou texte libre). Une fiche ⛔ ne démarre qu'avec une
réponse datée ici.

| ASK | Réponse | Date | Remarque |
|---|---|---|---|
| ASK-1 | | | |
| ASK-2 | | | |
| ASK-3 | | | |
| ASK-4 | A | 2026-10-06 | Vincent, session Mac : matrice complète. |
| ASK-5 | A | 2026-10-06 | Vincent, session Mac : le défaut TTS passe au 6 bits si la porte de K-79 le justifie. |
| ASK-6 | A | 2026-10-06 | Vincent, session Mac : la règle de K-42 décide ; `.mlx` par défaut si `.auto` ne gagne pas (CHANGELOG, FluxForge prévenu). Chiffres K-34 : `.auto` + 35 % de TTFT sur Mini, même décodage et même WER ; Small `.auto` traduit le dernier segment FR. |
| ASK-7 | A | 2026-10-01 | Donnée par Vincent au planificateur, 2026-10-01 : contexte complet ; le profil lean borne la durée d'audio acceptée (erreur typée au-delà) ; le seuil de 12 Go se lit avec le plafond de cache MLX (10,99 Go de pic avec K-52, pour 8,6 Go de mémoire MLX active). |
| ASK-8 | A | 2026-09-30 | Donnée par Vincent dans la session Mac : défaut `nil`, budget proportionnel à la durée (STT), boucle bornée par l'audio (Realtime) ; valeur explicite = plafond de jetons texte signalé `truncated` (K-5). |
| ASK-9 | A | 2026-09-30 | Donnée par Vincent dans la session Mac : un défaut public change quand la porte l'emporte (version mineure, CHANGELOG, FluxForge prévenu) ; A pour K-5 et K-14, fiche par fiche pour K-51, K-61, K-64. K-5 et K-14 partent dans la 3.0.0. |
| ASK-10 | A | 2026-10-07 | Vincent, session Mac : la porte de K-68 décide (TTFA des voix clonées −30 % au moins, 0 fuite sur 10 prises, écoute ASK-13). |
| ASK-11 | A | 2026-10-01 | Donnée par Vincent dans la session Mac : texte identique sur C-court et WER normalisé ≤ référence +0,2 pt sur C-moyen EN/FR (16 et 8 bits ; +0,5 pt pour la tête 4 bits de K-46). |
| ASK-12 | A, sous condition | 2026-10-01 | Donnée par Vincent dans la session Mac : mlx-audio accepté **comme référence seulement** — sortie capturée une fois (environnement Python temporaire hors du dépôt, commit épinglé) et figée en fichier texte ; **aucune dépendance** du code, des tests ou du build à un backend Python. |
| ASK-13 | A | 2026-10-06 | Vincent, session Mac : écoute à l'aveugle pour toutes les fiches (K-39, K-48, K-58, K-79, K-80). |
| ASK-14 | | | |
| ASK-15 | A | 2026-10-02 | Donnée par Vincent dans la session Mac : `small-24b-8bit` = `VincentGOURBIN/voxtral-small-8bit` (registre et README) ; l'enum `VoxtralPipeline.Model.small24b8bit` s'aligne dessus (K-10). |
| ASK-16 | A | 2026-10-06 | Vincent, session Mac : pointer vers les dépôts tiers quand c'est possible, épinglés par révision et SHA-256. |
| ASK-17 | A | 2026-10-02 | Donnée par Vincent dans la session Mac : corriger `realtime-4b` (liste de fichiers fixée, config lue depuis `params.json`, id strict) (K-9). |
| ASK-18 | A | 2026-10-02 | Donnée par Vincent dans la session Mac : « FluxForge est gratuit et le restera » ; usage non commercial, les profils TTS peuvent être recommandés et PK-2 publié (dérivé CC BY-NC 4.0 : attribution, licence reprise, mention « non commercial » sur chaque page TTS). À rouvrir si FluxForge devient payant (prix, achat intégré, publicité). |
| ASK-19 | A | 2026-10-07 | Vincent, session Mac : fabriquer un pack TTS 8 bits pré-quantifié et l'ajouter aux benchs (même matrice que K-35) ; publié s'il n'existe pas ailleurs (ASK-22). |
| ASK-20 | A | 2026-10-06 | Vincent, session Mac : apprendre à lire le format voxmlx ; l'agent ajoute un refus explicite de tout pack mal lu (fini le chargement silencieux en poids aléatoires). |
| ASK-21 | B | 2026-10-02 | Donnée par Vincent dans la session Mac : les modes non affines (mxfp4, mxfp8, nvfp4) sont acceptés au chargement en **expérimental**, sans profil ni garantie de qualité (K-8 : décodage correct du mode, avertissement, hors profils). |
| ASK-22 | autre | 2026-10-06 | Vincent, session Mac : publier sur Hugging Face (jeton disponible) tout ce qui n'est pas trouvable ailleurs, en dépôt complet comme pour yue2 et gemma4 ; le reste par référence (ASK-16). |
| ASK-23 | A | 2026-10-02 | Donnée par Vincent dans la session Mac : dépréciation annotée en 2.3 (additif, K-30), suppression en 3.0. 3.0.0 directe, sans 2.3 publiée (décision de Vincent du 2026-10-03, donnée au planificateur). |
| ASK-24 | A | 2026-10-06 | Vincent, session Mac : la conformance `LanguageModel` est retirée. |
| ASK-25 | A | 2026-10-02 | Donnée par Vincent dans la session Mac : revue 3.0 sur une liste proposée par K-31, relue et amendée par Vincent avant application. |
| ASK-26 | B | 2026-09-30 | Donnée par Vincent dans la session Mac : `VoxtralBenchmark` retiré du `Package.swift`, remplacé par `VoxtralCLI bench` (K-32). |
| ASK-27 | A | 2026-10-02 | Donnée par Vincent dans la session Mac : `VoxtralEncoderFull.mlmodelc` n'est plus embarqué ; VoxtralApp le télécharge à l'exécution (ligne `.copy` retirée, K-28). |
| ASK-28 | A | 2026-09-30 | Donnée par Vincent dans la session Mac : rester sur `main` de mlx-swift-lm, `Package.resolved` suivi dans git (K-22) ; premier tag ensuite (plan `upstream-blocker`). |
| ASK-29 | | | |
| ASK-30 | | | |
| ASK-31 | A (déduite) | 2026-09-28 | Déduite par la session cloud de la demande initiale de Vincent (« ne perds pas les actions sur Voxtral », tester le concept du tracker) ; écritures faites : #71, #307, #349 fermés `verified`, #556 et #557 créés (PLAN §7, K-21). À confirmer ou infirmer par Vincent. |
| ASK-32 | A | 2026-10-06 | Vincent, session Mac (« À » en réponse à la question fermée) : la cellule small-4bit `.auto` × C-moyen EN entre dans la baseline K-34 sans A/A, régime bimodal consigné, référence = médiane des passes rapides (75,13 ms), fiche de suivi K-85. |
| ASK-33 | A, désactivable | 2026-10-06 | Vincent, session Mac : plafond par défaut pour Small sur ≤ 32 Go, que l'hôte peut désactiver. |
| ASK-34 | A | 2026-10-07 | Vincent, session Mac : les cellules streaming moyen et long sans A/A restent dans la baseline K-35, cause consignée (re-décodage complet), référence « avant » de K-43. |
| ASK-35 | | | |
| ASK-36 | | | |
| ASK-37 | | | |
| ASK-38 | | | |
| ASK-39 | | | |

### Dérogations

- **Rôles (2026-10-03, Vincent)** : session Voxtral du Mac autonome (plan, dispatch, exécution, vérification,
  replanification) ; remplace la répartition du 2026-09-28 pour ce dépôt. Protocole : `VERIFY.md`.
- **K-14 (#564), 2026-10-03** (décision de Vincent, donnée au planificateur ; porte reformulée) : « 0 troncature » —
  une sortie déjà au-delà de 3 × l'attendu peut être coupée par le plafond, même si le texte arrive plus tard (cas
  fr01 `tts-4b-6bit` graine 2, 517 → 143 frames, accepté) ; reproducteur #45 tenu si la génération s'arrête au plafond
  ou en dessous (≤ 70 + ⌈10,4 × jetons⌉ ; observé 133 = plafond à 6 jetons, avant : 209).

- **K-31, 2026-10-02** (Vincent, session Mac) : liste validée telle quelle, appliquée sur la branche d'audit (3.0
  directe) ; 338 lignes `public` acceptées au lieu de ≤ 300 (`docs/audit/2026-09-27/K-31-liste-api.md` §E). La clause
  FluxForge sort de la porte Voxtral : Vincent la traite à la fusion (entrée FluxForge du 2026-10-03 ci-dessous).

Décisions de Vincent du 2026-10-03, données au planificateur :

- **K-12 (#570), 2026-10-03** : parité streaming = codes identiques bit à bit ; audio concaténé brut
  (`trimLeadIn=false`, `trimTail=false`) à max|Δ| ≤ 1e-5 du batch, même graine (observé 1,24e-6) ; aucune fiche
  n'exige l'identité exacte (K-43 : 1e-4).
- **K-32b (#604), 2026-10-03** : préréglages Core ML default/mini/small sur `.cpuAndNeuralEngine`, ce qui change le
  défaut public `.auto` ; ASK-6 / K-42 tranchent ensuite `.auto` contre `.mlx`.
- **K-13 (#582), 2026-10-03** : clause WER jugée contre la référence mlx-audio figée (`docs/eval/realtime-reference/`)
  avec la tolérance d'ASK-11, provisoire jusqu'au WER exact de K-33 repris par K-36 ; dérive ms/pas au-delà de
  ≈ 9 000 pas (jusqu'à +63 %) jugée thermique : coût par pas constant après la rotation 8 192, mémoire plate
  6 392,7 Mo, tranche 8 250–8 450 à +1,6 % ; le protocole de K-36 refroidit et ordonne les clips.
- **K-27 (#575), 2026-10-03** : arrêt `fatalError`/`precondition` conservé pour un appel direct au modèle hors
  frontière MLX (`MLXErrorBoundary.swift:43`), `prepare(_:cache:state:prefill:)` et appels directs à
  `VoxtralTTSModel` compris ; non atteignable depuis l'API publique depuis K-31 (`58917f3`).
- **FluxForge (K-22, K-30, K-31), 2026-10-03** : toute clause FluxForge sort des portes Voxtral (K-30 : 0 usage hors
  dépôt, 0 avertissement de dépréciation dans FluxForge, `reconfigureHubApi` ; K-31 : FluxForge compile sans ses
  typealias) ; Vincent les traite à la fusion.
- **K-15 (#580), 2026-10-03** : la porte d'annulation couvre le backend par défaut `.auto` ; point d'annulation dans
  la boucle Core ML (`VoxtralHybridEncoder.swift`, `encodeCoreML`) et test C-long en `.auto`, par le commit
  `fix(K-15)` (`a4beb715`).

- **K-33 (#585), 2026-10-03** (Vincent, session Mac) : option A. « Dernière phrase présente » = au moins 90 % de ses
  mots retrouvés dans l'ordre (`last_sentence_coverage` ≥ 0,9 ; même règle pour les phrases exigées) ; C-long évalué
  par segment de langue (= C-moyen EN et FR). La traduction du français en anglais sur un audio long EN/FR alterné,
  transcrit d'un bloc, est une limite du modèle qui ne sera pas corrigée : aucune fiche de correctif.

- **K-35 (#588), 2026-10-03, décision de l'agent** — clause « textes : court, 60 s = premier paragraphe du long, long
  complet ». Raison : le premier paragraphe du texte long fait 40 mots (≈ 15 s d'audio), pas 60 s ; le texte long
  complet (163 mots) dure ≈ 75 s. Décision : court = `docs/eval/tts/short_en.txt` (11 mots), « 60 s » =
  `medium_en.txt` (Long EN complet, 163 mots), long = `long_en.txt` (Long EN × 2, 326 mots, texte de K-12) ; copies
  octet pour octet des fichiers déjà mesurés (`.local-runs/corpus/tts_*`), SHA-256 dans `docs/eval/tts/README.md`.
  Preuve : `wc -w` 11 / 163 / 326 ; audio mesuré 7,4–7,8 s / 74–84 s (lignes `prof-*` de `BENCHMARKS.md`).

Décisions du planificateur du 2026-10-03 :

- Clauses sur du code retiré ou rendu `internal` par K-31 (liste ASK-25 = A, 3.0 directe), sans objet à HEAD : K-3
  « invite de 600 positions via `loadVoxtralModel` (chemin hérité) » ; K-23 « 0 écriture dans /tmp pendant une
  génération par `VoxtralGenerator` » ; K-30 annotations S-14 de `writeDebugToDump` et `loadQuantizedVoxtral` ; K-27
  classes de modèle publiques.
- K-7 : traduction `attention.wq/wk/wv/wo` → `q/k/v/o_proj` du sanitizer Realtime (`26f3d83`), rapportée comme choix
  de Vincent (rapport K-7 du 2026-09-30), retenue ; référence de K-9 pour `realtime-4b-fp16` à partir de `26f3d83`.
- K-31 : « ≤ 300 lignes public » remplacé par « liste validée par Vincent (ASK-25 = A) appliquée ; compte consigné :
  338 (1 110 avant) ».
- K-14 : la clause « 0 troncature » reste telle qu'écrite ; la sortie fr01 6 bits graine 2 (517 → 143 frames) n'est
  requalifiée « dégénérée » que sur preuves recopiées dans #564 : transcriptions avant/après de ce cas et
  reproducteur #45 dans ses propres conditions.

- **GPU et services de fond (2026-10-05, Vincent)** : « prends le GPU, peu importe qui d'autre l'utilise » ; les
  processus de fond (`mediaanalysisd`, Time Machine, Spotlight) ne bloquent jamais une mesure. Les simulateurs iOS
  sont arrêtés avant et pendant chaque passe ; une charge GPU d'une autre app MLX invalide toujours la mesure.
- **Protocole A/A des baselines K-34 / K-36 (2026-10-05, décision de l'agent)**. Clause « A/A ≤ 3 % sur 2 passes »
  jugée ainsi :
  - `step_ms_p50` (`epoch_ms_p50` pour l'enrôlement) ≤ 3 % ;
  - `step_ms_p90` ≤ 10 % ;
  - `total_ms` ≤ 3 % pour les passes de moins de 60 s ;
  - `out_sha256` identique ;
  - au-delà de 60 s, le total et le TTFT sont consignés, sans faire échouer la cellule.

  Raison : sur une passe longue, le total suit la charge de fond et la dérive thermique, pas le code ; la médiane par
  pas, elle, reste stable. Données : la règle « pire de total, pas, TTFT ≤ 3 % » faisait échouer 10 cellules sur 42,
  et toutes celles du fp16, même quand la médiane restait à moins de 3 %. Le p90 est ajouté après une cellule
  (fp16 C-moyen EN) à médiane stable mais p90 136 → 233 ms ; elle a été refaite (p90 0,20 %).
  Preuve : `.local-runs/tools/aa.py`, lignes `AA` de `BENCHMARKS.md` §2026-10-05.
- **Conditions de passe des baselines (2026-10-05, décision de l'agent)** :
  - Chaque passe tourne dans un processus séparé, avec un amorçage, `machine-check --cooldown 120` avant elle et les
    simulateurs arrêtés.
  - C-long passe sans amorçage : l'amorçage doublait les ≈ 10 min de GPU continu et ralentissait la passe mesurée
    (`step_ms_p50` 11,9 % → 0,19 % pour mini-3b `.auto`).
  - Les reprises ajoutent 180 s de repos, car 120 s ne suffisent pas après une passe C-long. Dans toutes les cellules
    courtes en échec, p1 suivait une C-long et était seule lente : par exemple small-4bit C-court 77,5 contre 70,7 ms,
    puis 0,09 % après repos.
- **Clips des baselines K-34 et K-36 (2026-10-05, décision de l'agent)** : les cellules « C-moyen EN/FR » et
  « C-long » sont mesurées sur C-moyen exact (`docs/eval/clips/c_moyen_{en,fr}.wav`, 146,1 / 130,9 s) et C-long exact
  (`c_long_exact`, 9 min 14 s ; recette dans `docs/eval/README.md`). Les C-moyen et C-long du corpus sont
  `fluxforge_long_*`, 167,0 / 173,8 s, et ≈ 11 min 22 s.
  - Raison : la porte exige le WER (K-33) pour chaque ligne. Les anciens C-moyen n'ont qu'une référence condensée
    (163 / 202 mots pour 413 / 451 dits), donc pas de WER juste. Les clips exacts mettent le WER et la performance
    sur la même ligne. Pour K-36, les EVAL ont le même `out_sha256` que les BENCH de chaque cellule. Pour K-34, il
    faut ceux du complément de vérification (`--language-mode auto`, 32 sur 32) : les premiers EVAL, en langue
    imposée, différaient dans 13 cellules.
  - Le WER de la référence Realtime figée (`rt_ref_c_moyen_{en,fr}`, amendement du 2026-10-03) est porté par les
    EVAL de ces mêmes lignes.
  - Les anciens clips restent témoins de performance hors baseline. Leurs chiffres ne se comparent pas à ces lignes
    (durée et texte différents).
  - Le chat (K-34, porte « C-moyen EN ») reste sur `fluxforge_long_en_6bit.wav`, comme l'écrit la fiche (étape 5 bis).
- **K-34 (#587), 2026-10-05 — dérogation retirée** (vérification du 2026-10-05). L'agent avait déclaré
  « A/A non atteignable » pour la cellule small-4bit `.auto` × C-moyen EN, avec une cause non prouvée et un décompte
  faux. La question va à Vincent : ASK-32.
- **K-35 (#588), 2026-10-06 (Vincent)** : matrice réduite pour la durée. Le texte court garde 3 graines (1 à 3),
  les textes moyen et long une seule (graine 1) ; les 3 packs, 2 voix, 2 modes et l'aller-retour ASR sont inchangés.
- **K-35 (#588), 2026-10-09, décisions de l'agent** (conditions de la baseline TTS, en plus du protocole du
  2026-10-05) :
  - **Texte court** : les 36 cellules gardent leurs deux passes dans un même processus (`--passes 2`, matrice du
    2026-10-03/06), A/A 0,00 à 0,74 %. Ce régime ne montre pas l'écart d'un processus à l'autre. Contrôle : 4
    cellules courtes refaites un processus par passe, sous macOS 27.0.1 (écart − 0,6 à + 0,3 %, sortie identique bit à bit, `BENCHMARKS.md` §K-35 « Contrôle macOS »).
  - **macOS** : cellules courtes sous 27.0.0, le reste sous 27.0.1 (mise à jour du système pendant la campagne ; la
    6 bits prédéfinie moyen en streaming, d'abord sous 27.0.0, a été refaite le 2026-10-09 au soir) ; le même contrôle
    couvre l'écart.
  - **Streaming moyen et long avec `--cache-limit-mb 2048`** : sans plafond, le `phys_footprint` plafonne à 74,1
    Go (75 899 Mo) sur 96 (swap non exclu). Ce plafond est une condition de mesure : le défaut public reste sans plafond.
  - **Batch moyen et long et ligne consommateur en profil par défaut** (sans plafond, chemin de FluxForge). Le profil
    plafonné, mesuré aussi, ne change pas `step_ms_p50` de façon mesurable sur les 6 cellules batch long (− 1,0 à
    + 2,9 %, moyenne des deux passes) et retire 9,3 à 12,3 Go ; ses lignes sont consignées comme comparaison. Le
    texte moyen n'est pas comparé (paires plafonnées mesurées sans repos après l'amorçage, 2026-10-10).
  - **Streaming court jugé sur `total_ms`** (passe de moins de 60 s) : il a été mesuré avant `stream_frame_ms_p50`
    (`0e7588a9`). Le streaming moyen et long est jugé sur `stream_frame_ms_p50` / `p90`.
  - **Reprises** : une cellule en échec d'A/A est refaite (`-r2`, `-r3`, `-default-r2`, `-cd120-r2`, `-cd120-r3`) sur
    machine calme. Pour une cellule streaming qui reste hors A/A, la paire retenue est celle dont l'écart est le plus
    faible. La cause
    n'est écrite que si l'échantillonneur de charge la montre dans la passe lente ; sinon « cause non isolée »
    (régime de K-85). Les passes écartées restent dans `BENCHMARKS.md`, chacune avec son fichier et sa cause.
  - **Cellules à audio doublé (K-92)** : 1 cellule streaming retenue (4 avant les reprises du 2026-10-10) garde son temps par frame ; ses
    `audio_s`, `rtf` et `out_sha256` sont marqués faux, et un contrôle après correctif est consigné.
  - **Machine-check** : 4 cellules retenues ont une ligne `KO indexation/sauvegarde en cours` avant une passe (7
    avant la reprise du texte moyen ; liste dans `BENCHMARKS.md`, complément de vérification K-35), comme 1 EVAL retenu (`RT-4bit-medium_en-s1`),
    2 passes du contrôle macOS et la p2 plafonnée de bf16 clonée long batch (comparaison).
    Décision de Vincent du 2026-10-05 : les services de fond ne bloquent jamais une mesure ; `quiet.sh` n'attend que
    pour les autres `KO` (autre inférence MLX, binaire Debug, batterie).
  - **TTFT et TTFA** : consignés, pas jugés (protocole du 2026-10-05). Clause de la fiche « AA ≤ 3 % (fps, ttft) »
    reformulée : le TTFT ou TTFA d'une cellule ne sert de référence que si ses deux passes sont à 3 % ou moins
    (47 cellules sur 60 après la reprise du 2026-10-10) ; les 13 autres sont listées. `fps` n'est pas écrit par l'instrument :
    fps = 1 000 / `step_ms_p50` (l'audio en demande 12,5). K-66 (porte TTFT) refait sa référence en A/B/B/A.
  - **« ms par frame LLM et FM (run de diagnostic `--trace`) »** : deux runs `bench tts --trace` faits le
    2026-10-09 (ligne consommateur ; 6 bits prédéfinie moyen en streaming). L'instrument ne sépare pas LLM et flow
    matching : `recordStep` enveloppe le pas entier (un span « Step i/n » par frame), et le chemin streaming n'émet
    aucune phase. Ce qui est mesuré : ms par frame (LLM + FM, `step_ms_p50`), phases du batch (préfill 344 ms,
    génération 43,05 s à 99,9 % de GPU, décodage codec 465 ms à 98,8 %). Ces deux runs ont été faits sans repos
    après l'amorçage (passes chaudes, 41,22 et 67,61 ms/frame contre 38,47 et 58,09 reposées, 2026-10-10) : leurs
    durées ne sont pas des références. Séparer LLM et FM demande un span par
    sous-étape dans `VoxtralTTSModeling.swift` : fiche de suite à la replanification (#590).
  - **Enrôlement de la voix de test** : `--duration 8` ajouté à la commande de la fiche
    (`enroll docs/examples/clone_fr.wav -m tts-4b-6bit --epochs 2000 --duration 8 --seed 7`). Raison : `clone_fr.wav`
    dure 8,6 s et `enroll` exige 16 s par défaut (`Reference too short: 8.6s < 16.0s`,
    `.local-runs/voices/enroll.log`). 8 s est la valeur de `bench enroll`. SHA-256 de la voix :
    `43ff457c1077be9929e5f8d03428c05403129ec02e7d8c451a22defc1362645b`.
  - **Aller-retour ASR du bf16** : les EVAL bf16 de la matrice (`eval tts-roundtrip --voice
    neutral_female`) passaient par `synthesize(voice:)` (cache de préfixe de la voix prédéfinie), les BENCH par
    `synthesize(voiceEmbedding:)` ; en bf16 seulement, les deux chemins donnent un autre audio à même graine (4 textes
    sur 5 ; 4 et 6 bits identiques). Les 5 EVAL bf16 sont refaits avec `--voice-embedding` (fichier
    `neutral_female.safetensors` du pack) : même `out_sha256` et mêmes frames que les BENCH, 5 sur 5. Couvertures
    retenues : 0,909 / 0,909 / 0,909 (court), 0,952 (moyen), 0,955 (long ; 0,949 par l'ancien chemin). L'écart entre
    les deux chemins en bf16 va à une fiche de suite (K-94, #634).
  - **ASK-34** : la question décrivait 4 cellules ; 6 cellules streaming moyen ou long restent sans A/A après les
    reprises du 2026-10-10 (9 avant). La réponse
    de Vincent (« les cellules streaming moyen et long sans A/A restent dans la baseline ») porte sur la classe
    entière.
- **K-35 (#588), 2026-10-10, décision de l'agent** — **repos de 120 s après l'amorçage** dans toute mesure en un
  processus par passe (`--warmup 1 --cooldown 120`). Raison : avec `--cooldown 0`, la passe mesurée démarre sur un
  GPU chaud, juste après l'amorçage. A/B/B/A sur tts-4b-6bit moyen : 41,40 / 44,70 ms/pas sans repos, 38,52 / 38,50
  avec, même `out_sha256`. Le texte moyen et la ligne consommateur, mesurés sans repos du 2026-10-07 au 2026-10-09,
  ont été refaits (`BENCHMARKS.md` §« 2026-10-10 — K-35, reprise ») ; leurs anciennes lignes sont écartées. Les
  autres baselines faites en un processus par passe avec amorçage et `--cooldown 0` (K-34, K-36, K-37) sont à
  vérifier à la replanification (#590). Vérifié le 2026-10-10 (#590, corrigé après le vérificateur) : K-34 et K-36 ont passé
  `--warmup 1 --cooldown 0` en un processus par passe (`.local-runs/queue2b.sh`, `queue4.sh` à `queue16.sh` ; `queue2.sh`
  pour mini-3b-8bit `.mlx` C-moyen EN et FR, p1 avec amorçage et p2 sans), y compris les cellules chat
  (`CHAT-mlx`, `queue9.sh` ; `CHAT-auto`, `queue16.sh` : un amorçage par question puis la mesure, sans repos,
  `BenchCommand.swift:764-770`) et 5 des 10 cellules C-long retenues (mini-3b-8bit et mini-3b-4bit en `.mlx` et `.auto`, realtime-4b-fp16 : journaux
  « warm-up 1/1 done (excluded) ») ; C-long sans amorçage seulement à partir de `queue7.sh`. K-37 : la p1 des séries
  `A4-6bit-cli`, `A4-mlx-cli` et `A4-mlx-after_synthesis` a tourné avec amorçage puis `--cooldown 0` (`queue2.sh:30`),
  la p2 sans amorçage ; leur A/A p1 / p2 de 0,03 à 0,26 % montre un effet non mesurable sur l'enrôlement : K-37 n'est
  pas reprise. Le contrôle des cellules K-34 / K-36 est confié à K-96 ; d'ici là, une porte en pourcentage se juge
  contre le A de chaque fiche, mesuré avec repos. Issue (K-96, 2026-10-13) : effet mesuré sur les témoins STT C-moyen
  mini-3b bf16 et small-4bit et sur le chat (mini-3b-8bit C-moyen sans effet) ; Realtime C-moyen jugé au-delà et STT C-long refait, par prudence (A/A en échec, effet non établi) ; 25 cellules refaites avec repos (`BENCHMARKS.md` §« 2026-10-13 — K-96 »), la plupart à 3 % près
  des anciennes valeurs ; C-court et Realtime C-long sans effet, gardés.
- **K-35 (#588), 2026-10-10, décision de l'agent** — **préalables de l'amendement du 2026-10-03 et contrôle macOS
  gardés malgré l'absence de repos après l'amorçage** (un processus par passe, `--warmup 1` sans `--cooldown`, au lieu
  d'une invocation `--passes 2 --warmup 1 --cooldown 120`). Raison : sur le texte court, l'amorçage dure environ 3 à 12 s
  selon le pack (4 s en 6 bits), contre 45 s sur le moyen ; les valeurs (38,04 à 38,22 ms/pas, contrôle macOS à 0,6 % près) recoupent la matrice
  reposée (38,15 à 38,20). Preuve : `BENCHMARKS.md` §« Complément du 2026-10-10 (contradicteur) ».
- **K-36 (#589), 2026-10-05, décision de l'agent** :
  - Clause « Metal System Trace sur C-moyen complet » : la trace Realtime est faite sur `c_20s_en`. Sur C-moyen,
    xctrace écrit ≈ 20 Go de données brutes dans le dossier temporaire du système, sans tenir compte de `TMPDIR`. Le
    disque s'est rempli deux fois (passes perdues, données brutes supprimées). Le décodage Realtime est un régime
    stationnaire (un pas par trame de 80 ms) : 20 s suffisent pour son occupation.
    Équivalence vérifiée en 4 bits : la trace sur C-moyen EN de la 1re série (`trace-realtime-2026-10-05T135409Z`,
    complète, les GPUPHASE perdus avec le disque plein) donne 93,0 % d'occupation (union des intervalles GPU de
    VoxtralCLI, profondeur 0, 4 premières secondes d'encodage et de préfill exclues ; 93,3 % sur toute la passe),
    contre 92,3 % sur `c_20s_en`. La trace fp16 sur C-moyen a échoué : verrou `NSDistributedLock` de xctrace pendant
    que le disque se remplissait. Outil : `.local-runs/tools/xct_busy.py` (calcul en ligne dans le rapport).
  - Les traces STT de K-34 (préfill Mini et Small) sont, elles, faites sur C-moyen EN complet.
  - Clause « relevé powermetrics » : `powermetrics` exige root, non disponible pour l'agent ; `pmset -g therm` ne
    rapporte rien sur cette machine. Remplacé par l'effet thermique mesuré (ralentissement de p1 après C-long,
    ci-dessus) et `top_process` de chaque ligne.
- **K-22, 2026-10-01** (Vincent, au planificateur) : la clause « FluxForge compile (si présent sur la machine) » est
  vérifiée à la fusion de la branche sur `main` (FluxForge suit `main`), pas sur la branche d'audit.
- **#590 (replanification des lots 4 à 6), 2026-10-06 au 2026-10-10, décisions de l'agent** (`VERIFY.md` §2 ; détail
  dans la section « Replanification #590 » de chaque fiche) :
  - K-52 : clause « max/médiane des pas » remplacée par `step_ms_p90` / `step_ms_p50` ≤ 1,3 (l'instrument ne rapporte
    pas le maximum) ;
  - K-54 : « 5 min » remplacé par C-moyen EN et C-long exact (clips existants) ; K-53, K-55, K-56 : « 20 / 30 min »
    construits par concaténation de C-long exact, versionnés par SHA-256 ;
  - K-72 : variante `.mlx` retirée le 2026-10-06, rétablie le 2026-10-10 (contradicteur : l'encodeur `.mlx` n'est
    pas mesuré et K-42 peut faire de `.mlx` le défaut) ; variante du défaut retenu par K-42 obligatoire ; K-40 : tolérance de parité
    alignée sur ASK-11 (+ 0,2 pt) ;
  - K-69 : volet vitesse retiré (phase audio 0,06 à 0,11 % du total, K-34), volet mémoire gardé (tampon d'un fichier
    stéréo 48 kHz de 30 min non mesuré ; retrait sans code si l'extraction pèse moins de 5 % du pic) ;
  - retirées par la mesure : K-67 (post-traitement au plus 0,05 % du total,
    K-35), K-71 (codec 0,2 à 4,4 % du total en batch, gain attendu ≤ 10 % de la phase, K-35 ; rouverte si K-43 mesure
    le codec au-dessus de 50 % (10 % de gain sur la phase = 5 % du total) du temps par frame en streaming) ;
  - K-44 : gardée sous condition d'une mesure xctrace de l'occupation de la génération TTS (≥ 95,3 % → volet retiré sans code ; streaming jugé de même après K-93),
    le 99,9 % de K-35 venant du profileur ;
  - K-41 : porte recentrée sur le pic du codec ; K-58 : porte d'origine (codec − 20 %) gardée et complétée par
    `stream_frame_ms_p50` − 5 % sur le long après K-43 ; K-56 : valeur de départ mesurée par la fiche (le champ
    `audio` cité était la phase mel) ; K-53, K-55, K-56 : « 20 / 30 min » = C-long exact × 2 / × 3 (18,5 / 27,7 min) ;
    K-87 : repli `.auto` → `.mlx` hors ligne gardé et annoncé par un log, au lieu d'une erreur qui changerait un
    comportement public ; K-94 scindée (qualité de la voix clonée en 4 bits → K-97) ; K-43 ← K-93 ; K-50 ← K-94 ; K-77, K-78 ← K-96 ; K-79 ← K-95 ;
  - le relevé « `--cache-limit-mb 2048` coûte ≈ + 15 % au TTS 6 bits moyen » (2026-10-07), passe chaude, est retiré de
    K-52 et K-79 ; le défaut TTS du plafond est mesuré par K-95 et reste à Vincent ;
  - fiches de suite : K-83, K-84, K-85 (2026-10-06, constats de K-34 et K-36), K-86 à K-89 (2026-10-06, entrées de
    #590), K-90, K-91 (2026-10-07, ASK-19 et ASK-20), K-93 à K-97 (2026-10-10, constats de K-35) ;
  - prérequis changés (ordre des tâches) : 2026-10-06 : K-49 et K-70 ← K-34 (au lieu de K-45) ; K-52 ← K-34, K-36 (K-51
    retiré) ; K-64 ← K-37 (K-52 retiré) ; K-65 ← + K-38 ; K-72 ← K-42 (K-56 retiré) ; K-73 ← + K-84 ; K-42 ← + K-85 ;
    K-56, K-60 ← + K-89 ; 2026-10-07 : K-77 ← + K-52 ; K-78 ← + K-91 ; K-79 ← + K-90 ; K-80 ← + K-90, K-91 ; K-75 ←
    K-22, K-31 ; K-43 ← K-12, K-35 ; 2026-10-10 : K-43, K-44 ← + K-93 ; K-50 ← + K-94 ; K-58 ← + K-43 ; K-72 ← + K-89 ;
    K-77, K-78 ← + K-96 ; K-79 ← + K-95 ; K-89 ← #604 (K-32b) au lieu de K-32.
- **K-93 (#633), 2026-10-11, décision de l'agent** (`VERIFY.md` §1 et §2) : les huit lignes `BENCH` portent
  `"commit":"794d9eb14","dirty":true` (champs lus dans le dossier courant au lancement, l'arbre contenant les
  modifications non construites de K-39 ; `794d9eb1` amendé en `a02db97e` avant le push, sources identiques). Preuve
  indirecte acceptée : A construit dans un worktree propre à `16dd4088` (chemin dans `top_process`), compilation de B à
  23:20:24 sans source recompilée, modification de K-39 datée de 23:21:42, `out_sha256` identique A/B. Le détecteur du
  mode trace a été remplacé après les mesures (`exportChromeTrace` → `VOXTRAL_TTS_SPLIT_TRACE`), chemin identique sous
  `bench --trace`, mesures non refaites ; contrôle après correctif sur un binaire `dirty` (arbre avec K-39 en cours) : il prouve la présence des spans et des phases ; son `out_sha256` `e64d2131…`, celui d'avant K-39, montre que le FM de K-39 n'y était pas (K-39 change les codes du 6 bits : parité acoustique 98,17 %). Correctif de l'instrument (`commit` du binaire, pas du dossier) : note pour K-89.
- **K-39 (#610), 2026-10-11, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) objectif de la fiche réduit,
  parité oblige : tête sémantique gardée en fp32 (en bf16 elle change 1,7 à 2,6 % des codes sémantiques, porte 100 %) ;
  embedding temporel et projections d'entrée gardés en fp32 (le transformeur seul passe en bf16) ; packs quantifiés
  sur le chemin fp32 (variantes bf16 : 98,48 % Δ2 à 98,09 % Δ3). (2) Couverture ASR et clause 4 bits : pas des
  décisions de l'agent, posées à Vincent (ASK-35, ASK-36), fiche `blocked`. (3) Champs `commit` / `dirty` lus dans le
  dossier courant (K-93) : binaire de A attesté par la file (`queue-k39.sh`, `$OLD` = worktree `k93-a` à
  `a8820cbb`), par `top_process` quand VoxtralCLI y figure et, en bf16, par `out_sha256` ; `ABBA-6bit-4-A` (`top_process`
  sysmond, sortie identique à B) ne l'est que par la file. (4) Streaming non mesuré : il appelle le même
  `decodeOneFrame` (`VoxtralTTSModeling.swift:744`), donc change aussi en bf16 ; mesure laissée à K-43.
- **K-68 (#618), 2026-10-11, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) la clause « fuites ≤ la
  référence (0/10) » est jugée sur la coupe : le test du chemin streaming donne la même coupe en A et en B sur 19/19
  cas, donc autant de fuites ; le « 0/10 » de la porte (valeur de K-35) n'est pas mesuré en valeur absolue, l'écoute
  tranche. (2) `out_sha256` différent A / B sur 8 graines sur 10 : décodage d'un préfixe, au plus 1 pas de
  quantification 16 bits sur 0 à 44 échantillons. (3) tts-4b-4bit B/B 28,6 % (temps du premier morceau, sortie
  identique) : consigné, la porte se juge sur le 6 bits (ligne consommateur). (4) La mesure A/B/B/A porte sur
  `a9d3c5cf` ; la garde de `7cea4cb5` ne change rien au `chunkSize` 10 (contrôle : même sortie, même TTFA).
  Complément du 2026-10-18, après l'écoute de Vincent (2026-10-16) : (5) l'écoute mesure la référence A elle-même à
  2 fuites audibles sur 10 (graines 02 et 08, coupées trop tôt, frames 3 et 4) et 1 coupe qui emporte « FluxForge Studio »
  (graine 01, frame 19 : coupe trop tard ou omission du modèle), identiques en B ; la clause est donc
  jugée sous sa forme relative (« ≤ la référence », tenue : B = A à 1 pas 16 bits près) et le « (0/10) » de la porte
  et d'ASK-10, valeur de K-35 non réécoutée, est réfuté pour la référence : défaut antérieur à K-68, renvoyé à une
  fiche de suite (coupe du porteur). K-68 ne change ni le nombre ni la place des fuites, et la retirer ne rendrait pas
  le 0/10 ; mais le « 0/10 » est un seuil de qualité validé par Vincent (ASK-10) : la lecture relative lui revient,
  posée en ASK-39 (contradicteur du 2026-10-18), K-68 `blocked` jusqu'à la réponse. (6) La préférence « = » de la grille est déduite par l'agent de
  l'écart A − B (≤ 1 LSB sur 0 à 44 échantillons) et des remarques de Vincent, qui décrivent les mêmes défauts dans
  les deux prises. (7) Le machine-check, d'abord seulement résumé, est recopié le 2026-10-18 dans `BENCHMARKS.md`
  §« 2026-10-18 — K-68 : sorties du machine-check » (41 sorties ; aucun `KO` avant les séries de la porte ; un `KO`
  « indexation » avant ABBA-4bit-3-B, hors porte, couvert par la décision du 2026-10-05).
- **K-94 (#634), 2026-10-11, décision de l'agent** (`VERIFY.md` §2, inscrite le 2026-10-16) : la porte cite les
  couvertures ASR de référence K-35 (0,955 / 0,949, graine 1) ; K-39 ayant changé la sortie bf16 entre-temps, la
  couverture des deux chemins est comparée sur 3 graines au même binaire (`c7935f73`, après K-39) au lieu de la
  référence K-35, qui ne mesure plus le même code. Mesure plus stricte (même binaire, 3 graines au lieu d'une) ; la
  branche « correctif » de la porte est sans objet (aucun correctif). Le machine-check n'est pas recopié dans
  `BENCHMARKS.md`, seulement résumé, écart à `VERIFY.md` §1 (depuis K-37, la sortie du machine-check est recopiée) : brut dans `.local-runs/k94/mc_*.log`.
- **K-96 (#636), 2026-10-13, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) le chat compte dans le pipeline STT,
  comme le dit la porte (« STT, chat compris »), mais ses cellules sont jugées comme un clip à part : le témoin chat q2
  (`total_ms` B − A − 6,14 %, A/A 1,88 %) déclenche la reprise du chat, pas celle d'un clip STT. (2) « Un témoin dont
  l'A/A échoue deux fois est jugé au-delà du seuil » est lu par pipeline et longueur : le Realtime C-moyen, dont l'A/A
  de A échoue sur ses deux témoins (4 bits 4,12 %, fp16 12,95 % sur `step_ms_p50`), est jugé au-delà par prudence ;
  le STT C-long (A/A en échec une fois, 11,04 %, écart B − A − 7,26 / − 11,36 % juste sous la dispersion) est aussi
  refait par prudence. Le Realtime C-long et tous les C-court ne sont pas refaits : leurs témoins sont sans effet
  (|B − A| ≤ 1,64 %, A/A ≤ 1,51 %). (3) Valeurs de départ des fiches = moyenne des deux passes retenues de la reprise ;
  seuils absolus dérivés recalculés, pourcentages des portes inchangés (une porte en pourcentage reste jugée contre le
  A de chaque fiche). (4) Trace GPUOCC prefill-small-4bit : quatre essais (2026-10-13 19:50 ; 2026-10-16 06:33,
  07:53, 08:10, le dernier écrit sur le SSD interne) finissent par `xctrace record failed (exit -1)`, `xctrace` ne
  finalisant pas la trace dans le délai par défaut de 120 s de `MetalSystemTrace.Recorder.stop(timeout:)`. Ce délai
  est un paramètre public de swift-mlx-profiler 1.5.1 ; seul l'appel de `bench` (`BenchCommand.swift:127`) garde la
  valeur par défaut (correction du contradicteur, 2026-10-16 : la première rédaction le disait fixe dans la
  dépendance). Décision : la trace est refaite après la file K-95 (le binaire et l'arbre doivent rester inchangés
  pendant ses passes), avec un délai d'arrêt plus long passé par `bench` (outil interne, pas d'API publique, aucune
  dépendance modifiée) ; en attendant, ses chiffres d'origine (prefill 99,8 %, decode 96,6 / 97,6 %) restent ceux
  cités. Fait le 2026-10-16 à 10:44 (`stop(timeout: 600)`, commit `135260c1`) : trace finalisée, prefill 99,7 %, decode
  96,8 / 97,5 % (xctrace), écart ≤ 0,2 pt. `queue-k96-trace3.sh` a été réécrit pour cette reprise : l'essai de 07:53
  n'a plus son script ni son `mc_` archivés (son journal reste dans `trace/old/`). (5) K-56, nommée dans la liste de la porte, est
  déclarée sans objet (2026-10-16) : sa porte recalée du 2026-10-10 mesure la valeur de départ dans son propre A ; les
  chiffres K-34 qu'elle cite encore (4 840 Mo, ≈ 1 735 ms) sont annotés comme historiques dans la fiche, avec les
  valeurs refaites (89,6 / 68,7 Mo, ≈ 1 729 ms). (6) Écarts aux règles de mesure de `VERIFY.md` §1, sans effet sur
  les verdicts : les deux traces Realtime du 2026-10-16 (06:40, 06:51) et les quatre essais small-4bit en échec tournent
  en `"dirty":true` (docs K-96 non commitées, `Sources/` et `Package.resolved` inchangés) ; prefill-mini-3b-8bit
  (2026-10-13) et la trace small-4bit retenue (2026-10-16 10:44, `135260c1`) sont en `"dirty":false` ; seules les
  lignes GPUOCC des traces sont retenues, pas leurs lignes `BENCH` ;
  les 10 séries de témoins partagent les étiquettes `A` / `B` (chaque série reste distinguée par son nom de journal
  `.local-runs/k96/<témoin>-{A1,B1,B2,A2}.log`), et la reprise réutilise les étiquettes de K-34 / K-36 parce que la
  porte impose la même commande. (7) Retirée (2026-10-16) : l'ASK à Vincent sur un délai réglable dans swift-mlx-profiler
  était sans objet, le délai étant déjà un paramètre (voir (4)).
- **K-41 (#612), 2026-10-18, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) trois commits `perf(K-41)` au lieu
  d'un (`d10653c0` bandes, `97878621` tranches évaluées, `c6b13d3f` seuil de 256 frames) : chaque série de mesure a
  conduit au suivant, chacun mesuré. (2) Blocs de 64 requêtes au lieu de 128 à 256 (fiche) : avec W ≤ 16, le bloc ne
  touche que la taille du transitoire, exactitude testée. (3) Les tranches de temps évaluées dépassent l'objectif
  « attention par bandes » : sans elles, la clause du pic de la porte recalée est manquée (série 1 : 5 160 > 4 537 Mo),
  elles restent internes à `decode` (aucune API publique), l'enrôlement garde le graphe entier. (4) Les cellules
  longues sont mesurées sur `97878621`, le court sur `c6b13d3f` : pour 998 et 1 903 frames, `c6b13d3f` exécute le même
  chemin que `97878621` (seuil franchi), la parité est refaite sur `c6b13d3f`. (5) Série 1 : champ `"commit":"e89b59602"`
  pour B (commits de documentation K-68 après `d10653c0`, `Sources/` identiques). (6) Clause d'origine « décodage court
  ± 5 % » remplacée par la porte recalée (2026-10-10) ; mesurée quand même (série 3 : − 4,4 %). Ajouts après le
  vérificateur (2026-10-18) : (7) A reste `c903e109` (état avant K-41) dans les trois séries, au lieu du parent de
  chaque commit B (`e89b5960`, `97878621`) : comparaison à l'état d'origine, plus stricte pour la porte, qui juge
  K-41 entière. (8) Les tags ne changent pas d'une série à l'autre (`K41-<cellule>-A|B`, `VERIFY.md` : un tag distinct
  par série) : séries distinguées par leur dossier (`.local-runs/k41/seriesN`), leur date et le champ `commit` de B.
  (9) Les A/A jugés sont ceux de `step_ms_p50` (`aa.py`) ; sur `phases_ms.codec` et le pic codec, la dispersion
  monte à 23 % (B 6 bits, série 1) : la porte est en seuils absolus, tenus par chaque passe, et l'effet (− 78 à
  − 88 %) dépasse toute dispersion.
- **K-86 (#627), 2026-10-18, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) « `out_sha256` » de la porte lu comme
  le SHA-256 du WAV écrit par `VoxtralCLI tts` : `tts` n'émet pas de ligne `BENCH` et `bench tts` n'a pas
  `--voice-xyz` ; graine vérifiée sur tts-4b-6bit, et non sur le défaut du CLI (`tts-4b-mlx`) que prendrait la commande de la porte, sans `-m` : le chemin ZeroVoice ne dépend pas du pack (graine transmise à `generate`, pas de cache de préfixe) (correction du contradicteur). (2) API publique : une surcharge
  `synthesize(text:voiceCoordinate:seed:)` ajoutée (additive, prévue par la fiche), l'ancienne inchangée et déléguée ;
  pas de nouveau cas dans `VoxtralTTSError` (un cas ajouté casserait les `switch` exhaustifs des consommateurs) :
  `modelLoadingFailed` nomme le `.pt`. (3) « Rouge avant » prouvé, pour les tests, par mutation du correctif dans un
  worktree jetable (les tests appellent `loadVoiceEmbeddings` et `TorchTensorReader`, absents avant) ; `mut2` saute la
  recherche des shards et reproduit l'erreur d'avant K-86 ; pour le CLI, par le binaire de `c6b13d3f` (même
  `Sources/` que le parent). (4) Le test « frame par frame » est le générateur streaming avec `chunkSize` 1 (graine 3,
  texte court) ; K-12 le compare déjà au batch par morceaux de 10. (5) Cinq commits : deux `fix(K-86)` (`4d93fcb4`, `0bd4d410` après le vérificateur) et trois `docs(K-86)`, plus celui de la vérification.
  (6) `voiceEmbeddings` est désormais remplacé au chargement au lieu d'être complété (un rechargement d'un autre pack
  ne garde plus les voix du précédent). Ajouts après le vérificateur (2026-10-18) : (7) la clause « 0 clé manquante
  (`verify` de K-7) » est prouvée par la fixture (`loadAllTTSWeights`, clés synthétiques) et par le pack réel
  `tts-4b-mlx` (2 shards sans index, `updateVerified`) ; pas de pack réel à 3 shards sans index dans le registre.
  (8) `lastSynthesisTruncated` n'est pas remis à zéro quand `synthesizeStreaming` refuse de démarrer (pipeline occupé,
  modèle absent, module non pris en charge : le flux finit en erreur, le drapeau garde la valeur précédente) ; après une
  annulation en streaming il vaut `false`. (9) Un `model.safetensors` accompagné de shards résiduels sans index lève
  désormais une erreur (shards incomplets nommés) au lieu de charger le seul shard 1.
  Ajouts après le contradicteur (2026-10-18) : (10) un `.pt` illisible fait échouer tout `loadModel` (porte en
  `.error`) ; avant, `tts-4b` se chargeait avec 0 voix et restait utilisable par `synthesize(text:voiceEmbedding:)`.
  Refus permis par la fiche (« erreur typée nommant le format ») ; `tts-4b` suit `main` sans révision épinglée : une
  republication des voix dans un autre format `torch.save` le rendrait inchargeable, à surveiller (CHANGELOG).
  (11) « Appliquer la graine partout » couvre le CLI (`-v`, `--blend`, `--voice-embedding`, `--voice-xyz`, `bench`) ;
  la façade `VoxtralTTSSynthesisManager` n'a toujours pas de graine (hors porte), noté pour une fiche de suite.
- **K-38 (#609), 2026-10-18, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) dtype de calcul lu sur
  `tok_embeddings` et non sur `norm` ou `ada_down.weight` (fiche, comme mlx-audio) : dans les packs mlx-community ces
  tenseurs sont fp32 ; la table liée n'est jamais quantifiée et porte le dtype 16 bits des matrices. (2) Paramètres
  fp32 du pack castés en 16 bits au chargement (hors fiche, qui ne castait que mel, RoPE et `tCond`) : sans ce cast,
  ces trois casts ne changent pas le dtype des activations ; parité prouvée (textes identiques). Un modèle fp32 reste
  fp32 (test). (3) 4e source fp32 corrigée, absente de l'audit : zéros du remplissage du conv causal de l'encodeur.
  (4) `mel` reste rapporté float32 par l'audit (casté dans l'encodeur, après le point d'audit) ; la porte nomme
  embedding, KV et logits, tous fp16. (5) `--passes 1` par processus, quatre processus A1/B1/B2/A2, au lieu de
  `--passes 2` (fiche) : protocole K-96 (un processus par passe, repos et `quiet.sh` avant chacune). (6) Rouge du test
  montré sur l'arbre du correctif sans le cast des zéros (3 échecs, float32) : au parent, `castRealtimeParameters`
  n'existe pas et le test ne compile pas ; l'audit A (float32) fait le rouge sur les vrais packs. (7) `perf(K-38)` et
  `docs(K-38)` : deux commits. (8) Le dtype des embeddings rendus par `extractAudioEmbeddings` passe de float32 au
  dtype du modèle (0 consommateur connu ; CHANGELOG).
- **K-38 (#609), 2026-10-19, ajouts après le vérificateur** : (9) le checkpoint d'origine `realtime-4b` (bf16) calcule
  désormais en bf16 au lieu de fp32 (mel et `tCond` fp32 promouvaient tout) ; hors porte recalée (4 bits et fp16),
  mesuré après coup (`BENCHMARKS.md` §« 2026-10-19 — K-38, complément ») : WER identiques (9,09 / 1,05 / 1,97 %),
  texte C-court identique, un trait d'union de différence sur C-moyen EN. (10) Passe 4 bits A1 démarrée sur un `KO`
  « indexation/sauvegarde » (mediaanalysisd 29 %), toléré par la décision du 2026-10-05, désormais nommé ; charge de
  fond des passes fp16 (`spotlightknowledged`) consignée, des deux côtés, A/A ≤ 0,41 %.
- **K-38 (#609), 2026-10-19, après le contradicteur** : (11) laissés en suite, hors porte : `logits * (1 / temperature)`
  se calcule en 16 bits (`VoxtralRealtimeModel.swift`), une température très basse pourrait dépasser 65 504 en fp16
  (défaut 0 = argmax, non concerné ; cast fp32 avant la division à faire avec K-47) ; pas d'`EVAL` C-long après K-38
  (K-78 le mesure) ; tests gardés Realtime non rejoués (`RealtimeOriginalRepoTests`,
  `CancellationTests.testCancelRealtimeOnCLong`, `UnloadFootprintTests.testRealtimeUnloadReturnsFootprint`).
- **K-43 (#639), 2026-10-19, décision de l'agent** (`VERIFY.md` §1 et §2) : (1) pas de champ `newCodes` sur
  `GenerationChunk` (fiche : additif) : le pipeline découpe `accumulatedCodes` à partir de la première frame neuve,
  aucune API publique ne change. (2) Contexte = champ récepteur calculé (22) + 2 = 24 frames, et non le minimal
  mesuré (7 frames sur le codec aléatoire) : borne exacte par construction, la marge coûte peu (coût fixe par appel).
  (3) `--cache-limit-mb 2048` pour l'A/B/B/A streaming (condition des cellules streaming de K-35) ; clauses pic et
  RTF lues en profil par défaut sur B seul (une passe streaming, une batch, sans A/A) ; RTF + 4,86 % pour un seuil de
  + 5 %, tenu de peu. (4) `--passes 1` par processus, quatre processus par cellule, au lieu de `--passes 2` (fiche) :
  protocole K-96. (5) Pas de « rouge avant » pour la parité : l'ancien chemin était exact aussi (il re-décodait
  tout) ; les tests unitaires utilisent des API nouvelles (`receptiveFieldFrames`, `decode(_:from:context:)`). (6) En
  configuration par défaut, le streaming garde le silence de tête que le batch rogne (0 à 0,32 s) : écart documenté
  (le streaming émet avant de connaître la fin du silence), aucune correction. (7) `out_sha256` du streaming change
  A → B (arrondis du décodage par fenêtres), borné par la parité 4,9e-6. (8) `peak_mlx_mb` du streaming long
  + 13,6 % sur le batch (4 420 contre 3 891 Mo), hors porte (champ de la porte : `peak_footprint_mb`, − 9,8 %).
  (9) Passes `stream-medium-B1` et `B2` démarrées sur un `KO` « indexation/sauvegarde » (corespotlightd), toléré par
  la décision du 2026-10-05 ; valeurs égales aux graines 2 et 3 sans `KO`. (10) `perf(K-43)` et `docs(K-43)` : deux
  commits (trois avec `9af2c6ac`, remarques du vérificateur ; quatre avec la vérification).
- **K-43 (#639), 2026-10-19, après le vérificateur** (`verified`, 7 remarques non bloquantes) : (11) chemin avec
  amorçage (`warmUpText`, tampon `head` du pipeline) non couvert par les tests de K-43 : rejoué après coup sur
  `69bb2727`, tests gardés `TTSCarrierEarlyCutTests` (2/2, `STREAMSUM cut mismatches 0/19, late first chunk 0/19`) et
  `TTSStreamingSeedReproTests` (2/2, dont `testWarmUpCarrierTrimmedLongText`), voix `clone_fr`
  (`.local-runs/k43/gated2.log`, `gated3.log` ; le premier essai `carrier.log` échoue sur `XCTUnwrap`, variable
  `_EMB` absente, aucune assertion jugée). (12) Le test à poids aléatoires ne distingue pas un champ récepteur
  sous-estimé entre 7 et 21 frames : seule l'assertion `receptiveFieldFrames == 22` protège le calcul ; le recul exact
  est 21 (plafond conservateur). (13) `VoxtralCodecDecoder.swift` est modifié (ajouts : `windowSizes`,
  `samplesPerFrame`, `receptiveFieldFrames`, `decode(_:from:context:)`) alors que la note de replanification ne
  l'annonçait pas. (14) Le pic sans plafond « 75 899 → 7 192 Mo » cumule K-41 et K-43 (A n'a pas été mesuré sans
  plafond) ; CHANGELOG corrigé. (15) La trace dure plus que `total_ms` de sa passe (A 144 contre 128 s, B 91 contre
  84 s, coût du profileur) ; rapportée à `total_ms`, la part du codec vaut 45,8 → 6,6 %, toujours sous 50 %.
- **K-43 (#639), 2026-10-19, après le contradicteur** (`verified`, 9 remarques non bloquantes) : (16) avec amorçage,
  les tests rejoués (11) contrôlent la position de la coupe par la longueur, le déterminisme et l'absence de silence,
  pas l'égalité des échantillons au batch : un décalage de `head` / `fullStart` à longueur constante passerait
  (relecture : aucun). (17) `frameSize = sampleRate * 2 / 25` du pipeline (`VoxtralTTSPipeline.swift:693`) doit
  égaler `samplesPerFrame` du codec (1 920 pour tous les packs) : couplage nouveau, non vérifié par le code ; à
  aligner (lire `samplesPerFrame`) avec la prochaine fiche qui touche le pipeline TTS. (18) Les fenêtres [2, 4, 8, 16]
  sont codées en dur comme dans le forward (`attn_sliding_window_size` ignoré des deux côtés) : « calculé depuis la
  configuration » vaut pour noyaux, pas et longueurs ; CHANGELOG précisé. (19) Clause RTF : marge 0,14 point sous la
  dispersion B/B du même texte plafonné (2,7 %) : tenue, non robuste à une nouvelle mesure (3). (20) K-58 (codec bf16)
  doit relancer `IncrementalCodecDecodeTests` gardé et revoir la tolérance 1e-4 (note dans sa fiche). (21) Passe
  `stream-short-A1` : `top_process` corespotlightd 169,7 % sans `KO` au machine-check (A1 40,84 contre A2 41,03 ms) ;
  `trace-stream-long-B` : Music 30,2 %. (22) CHANGELOG : RTF plafonné et batch par défaut côte à côte, corrigé.


- **K-58 (#656), 2026-10-19** (retirée) : (1) passes `--passes 1 --warmup 0 --cooldown 0`, un processus par passe,
  au lieu de `--passes 2 --warmup 1 --cooldown 120` de l'étape 4 : texte long sans amorçage (porte recalée), repos
  de 180 s et refroidissement de 120 s du machine-check avant chaque passe (conduite de K-43). (2) SNR calculé sur
  les WAV 16 bits de `eval tts-roundtrip` (plancher de quantification de deux arrondis 16 bits au niveau réel des
  WAV : 58,4 / 50,1 / 48,5 dB, corrigé après le contradicteur ; le SNR du seul codec n'en monte que d'≈ 0,25 dB), mêmes codes
  des deux côtés (même graine, mêmes longueurs, mêmes transcriptions). (3) Pas d'A/A batch distinct : la dispersion
  A1/A2 (5,3 % en file 1, 0,5 % en file 2) en tient lieu. (4) `stream-long-B2` (file 1) démarré sur un `KO
  indexation/sauvegarde` toléré (décision du 2026-10-05), valeur égale à B1. (5) `IncrementalCodecDecodeTests` gardé
  non rejoué : code retiré, codec inchangé depuis la vérification de K-43. (6) Deux commits de code (`9a1ce03f`,
  `06aaaa64`) avant le retrait au lieu d'un commit unique : les mesures exigent un commit propre par variante.
  Après le vérificateur (`verified`, 7 remarques) : (7) A/A de la file 1 au-dessus des 3 % de VERIFY.md (codec batch
  5,3 %, streaming p50 3,3 %) : le « − 20 % » du codec en bf16 est indicatif, non prouvé ; sans effet sur le retrait
  (SNR déterministe 3 à 4 dB sous le seuil ; en streaming, B dans l'intervalle de A et au-dessus de l'A de la file 2,
  A/A 0,1 %). (8) Le machine-check vérifie le binaire du dépôt principal, y compris avant les passes A lancées depuis
  le worktree (`top_process` de `batch-long-A1` le confirme) : formulation de BENCHMARKS précisée.
  Après le contradicteur (`verified`, 9 remarques) : (9) décision de l'agent : la clause « sinon ne garder que les
  invariants (P-39) s'ils laissent la sortie identique » est lue comme une borne (au plus les invariants), non comme
  une obligation : les invariants, sortie identique, gagnent − 0,6 % / − 0,5 % (< 5 %, PLAN.md §0 ; étape 4 de la
  fiche « ou < 5 % ⇒ code retiré ») et leur pic est dans la dispersion (+ 0,7 %) : retirés. (10) Décision de l'agent :
  fiche retirée après mesure rapportée `applied` avec la ligne de rejet (et non `blocked`, qui appelle une question à
  Vincent : aucune décision réservée n'est en jeu ; « un rejet documenté vaut une technique retenue ») ; convention
  pour les fiches suivantes. (11) Clause streaming inatteignable par construction depuis K-43 (codec 6,1 % du
  streaming long : un codec gratuit donnerait − 6 %) : leçon à reporter au prochain replanning des fiches codec / TTS.
  (12) SNR par segments d'une seconde (long : médiane 31,1 dB, minimum 20,9 ; moyen : 36,0 / 31,8) : plus bas que le
  SNR global, renforce le retrait. (13) Pic streaming − 12 % du bf16 non exploité : le garder pour un profil « lean »
  demanderait d'accepter un SNR de 36 dB (garantie de qualité, décision de Vincent) ; non proposé.
- **K-46 (#642), 2026-10-19, décision de l'agent** (`VERIFY.md` §1 et §2 ; appliquée) : (1) passes `--passes 1 --warmup 1 --cooldown 120`, un processus par
  passe, au lieu de `--passes 2` de l'étape 4 (conduite de K-38 et K-43 ; repos de 180 s et machine-check avant
  chacune). (2) Clause mémoire lue entre B sans option (B0) et B tête 8 bits (B8) du même binaire, et non contre A :
  le parent n'écrit pas `load_active_mb` (champ ajouté par `b43ce756`) ; B0 égale A en texte (`out_sha256`), en pic
  (4 247,2 Mo) et en pas (+ 0,1 %). Pas d'A/A distinct pour `load_active_mb`, valeur déterministe (deux fois la même
  valeur pour B0, B8 et B4). (3) Tête 4 bits mesurée contre B0 (B0, B4, B4, B0), pas en A/B/B/A contre A : la porte
  ne lui demande que le WER. (4) Commit de code et commit de mesures séparés (`perf(K-46)` puis `docs(K-46)`), comme
  K-38 et K-43. (5) Ajout hors fiche, dans le seul outil de mesure : option `--head-bits` de `bench realtime` et `eval
  realtime` et champ `load_active_mb` ; la variable `VOXTRAL_REALTIME_HEAD_BITS` est lue par le chargeur de
  `VoxtralCore` (interne, aucun symbole public). (6) `head8-A2` : `backupd` (Time Machine) sur 11 des 81 relevés de
  charge, sans `KO` au machine-check ; A2 sous A1, le gain n'en est pas grossi. (7) Audit de dtype fait hors file sur
  C-court (`.local-runs/k46/dtype-audit.log`), pas une mesure. (8) Le titre de la fiche (« 8 bits en fast, 4 bits en
  lean ») n'est pas appliqué ici : aucun profil ne l'active, c'est K-76 à K-78 et un défaut public (question à
  Vincent).
  Après le vérificateur (`verified`, 7 remarques) : (9) le cast de K-38 précède la quantification : dans le chargeur,
  la branche `scales` de `realtimeComputeDType` n'est pas atteinte et l'audit de dtype aurait été fp16 sans elle ;
  le piège n'est couvert que par `RealtimeQuantizedHeadTests` (assertion `realtimeComputeDType == .float16` sur une
  table `uint32`, rouge si l'on relit `weight.dtype`, non exécuté en rouge). (10) Charge de fond complétée dans
  BENCHMARKS (photoanalysisd `b0-2`, mediaanalysisd `head4-1`), sans effet sur les valeurs ; « jamais une autre charge
  GPU » remplacé (le machine-check ne voit que les processus MLX). (11) CHANGELOG : bases des deux pourcentages
  précisées, Mio, la variable d'environnement n'y est plus annoncée (interne, pas un réglage) ; commentaire périmé
  de `realtimeComputeDType` corrigé (commentaire seul, aucune mesure à refaire).
  Après le contradicteur (`verified`, 9 remarques) : (12) le champ `head_bits` des lignes `EVAL` vient de l'option,
  pas de l'état du modèle ; preuve indirecte que la tête était quantifiée en `eval` : même mécanisme que `bench`
  (où `load_active_mb` baisse) et durées C-moyen EN 22 / 20 / 18 s (B0 / B8 / B4) ; à dériver de l'état du modèle
  dans une prochaine fiche qui touche `eval`. (13) Rouge non exécuté pour la branche `scales` (voir (9)) ; phrase de
  log.md corrigée. (14) CHANGELOG : « −360 MiB » lu entre B0 et B8, non contre le parent (contre le parent : pic
  MLX 4 247,2 → 3 887,2) ; corrigé. (15) La bibliothèque lit `VOXTRAL_REALTIME_HEAD_BITS` (comme
  `VOXTRAL_DTYPE_AUDIT`) : un hôte qui en hériterait changerait de tête sans signal ; CHANGELOG le dit de nouveau ;
  le choix d'un réglage public est dans la question à Vincent. (16) `load_ms` + 15 % avec la tête quantifiée (A 205,4
  / 207,3, B0 206,3 / 208,2, B8 238,0 / 238,6, B4 235,2 / 238,3) ; pic transitoire du chargement non mesuré (la table
  fp16 et la quantifiée coexistent ≈ + 408 Mio pendant `quantizeRealtimeHead`). (17) Seul realtime-4b-4bit mesuré
  avec l'option ; fp16 et bf16 (`realtime-4b`) par raisonnement seulement : à mesurer par K-78. (18) Tête 4 bits non
  rejetée sur trois clips (porte : C-moyen EN/FR) ; C-long et autres langues à reprendre avant K-76 à K-78.
  (19) `load_active_mb` est en Mio, les lignes de porte écrivent « Mo » : 360 Mio = 377 Mo, porte tenue dans les deux
  unités. (20) « Les deux tiers » repose sur le trafic estimé par P-61 (2,5 Go par pas), non mesuré ici.
- **K-65 (#650), 2026-10-19, décision de l'agent** (`VERIFY.md` §1 et §2 ; retirée) : (1) passes `--passes 1
  --warmup 1 --cooldown 120`, un processus par passe, au lieu de `--passes 2` de l'étape 4 (conduite de K-38, K-43,
  K-46). (2) Parité mesurée par un test Debug gardé (`RealtimeFusedRoPETests`, forme explicite reconstruite depuis
  les mêmes modules dans le même processus) et non entre deux binaires : seule la RoPE diffère. (3) Mesuré sur
  realtime-4b-4bit seul (pack de la porte, ASK-11) ; parité aussi sur realtime-4b-fp16. (4) Suite Debug complète non
  rejouée sur `445439e6` (tests ciblés `RealtimeFusedRoPETests` et `RealtimeSlidingWindowTests` verts) : code
  retiré, `git diff da4cbfff 89570bdb -- Sources Tests` vide. (5) Tolérance fp16 du test aléatoire resserrée de
  1e-2 à 2e-3 après le rouge (à 1e-2, l'offset faux passait en fp16 : 4,2e-3) ; le rouge n'a pas été rejoué à 2e-3.
  (6) Deux commits de code (`perf(K-65)` essai, `perf(K-65)` retrait) et un de mesures, comme K-58 ; rapportée
  `applied` avec la ligne de rejet (convention de K-58, dérogation 10).
  Après le vérificateur (`verified`, 8 remarques) et le contradicteur (`verified`, 10 remarques) : (7) charge de
  fond de `eval-B` (backupd) ajoutée à BENCHMARKS. (8) Encodeur aléatoire du test non reproductible d'un processus à
  l'autre malgré la graine (7,99e-8 / 2,6e-4 puis 7,25e-8 / 2,97e-4 ; ordre de `parameters().mapValues`), marge
  6,7× sous 2e-3. (9) `encodeFull` sur les packs couvert par `out_sha256` de C-court seulement ; dans l'essai,
  `bandedReference` de `RealtimeSlidingWindowTests` passait par la nouvelle forme. (10) Tests et rouge joués dans
  l'arbre principal avant le commit de l'essai, pas dans un worktree jetable. (11) `phases_ms.encode` inclut mel,
  convolution d'entrée et adaptateur (BENCHMARKS précisé) ; chiffre fp16 ajouté. (12) Leçon de log.md reformulée
  (aucun profil de l'encodeur, gain réel mais sous la porte). (13) Porte quasi inatteignable par construction, comme
  la clause streaming de K-58 : à reporter au prochain replanning. (14) Tags `K65-A` / `K65-B` partagés par A1/A2 et
  B1/B2 (VERIFY.md §1 : un tag par série) ; passes distinguées par la date et l'ordre de la file. (15) Le corps de
  #650 cite encore 2 860 ms (K-36), l'amendement K-96 (2 801 ms) prime ; sans effet, la porte se juge sur A.
- **K-50 (#643), 2026-10-19, décision de l'agent** (`VERIFY.md` §1 et §2 ; validée) : (1) passes `--passes 1
  --warmup 1 --cooldown 120`, un processus par passe, au lieu de `--passes 2` de l'étape 4 (conduite de K-38 à K-65) :
  l'amorçage est la 1ʳᵉ synthèse de la voix, la passe mesurée la 2ᵉ. (2) Pas de paramètre `voiceKey:` sur la
  surcharge batch (prévu par la fiche, « additif ») : l'empreinte du contenu suffit, aucune signature publique ne
  change. (3) Rouge prouvé en remplaçant la correspondance par empreinte par une correspondance sur la clé seule
  (`.local-runs/k50-red.log`), et non par `git stash` des fichiers du correctif (étape 5) : sans `VoicePrefixCache`,
  le test ne compile pas. (4) Parité ASR mesurée sur la 1ʳᵉ synthèse (un processus par `eval tts-roundtrip`) : B y
  prefille voix puis texte, A d'un bloc ; la 2ᵉ synthèse est couverte par `out_sha256` identique dans les A/B/B/A
  (graine 1 seule, trois cellules). (5) bf16 sur le texte court seul (3 graines), pas 3 textes : conformité à K-94
  lue sur la couverture. (6) Passe `6bit-medium-B2` (information) sur batterie (`"power":"Battery Power"`, Mac
  débranché 13:54 → 14:13 ; son machine-check teste le secteur avant ses 120 s de refroidissement, vers 13:52:22, environ 2 min avant le débranchement : « sur secteur ») :
  écartée et refaite (`6bit-medium-B2r`, après la cellule 4 bits, hors de l'ordre A/B/B/A) ; ligne recopiée telle
  quelle. (7) `KO  indexation/sauvegarde` avant `4bit-short-B2` et `6bit-medium-A2`, tolérés (décision du
  2026-10-05). (8) Évaluations (non chronométrées) pendant une sauvegarde Time Machine. (9) Binaire B construit à
  `d8b457d8` (12:43:09) ; ses lignes portent `86899785`, le HEAD lu à l'exécution (`BenchCommand.swift:262`), un commit
  `docs(K-65)` de plus, sans changement de `Sources`. (10) L'empreinte (copie et
  SHA-256 de l'embedding) et, au premier appel, le calcul du préfixe se font hors de la phase « Prefill »
  du modèle (`VoxtralTTSModeling.swift:482`) : `phases_ms.prefill` ne les compte pas ; `total_ms` les compte (− 2,0 %, cohérent avec − 66 ms de
  préfill), coût de l'empreinte non mesuré seul. (11) Tags `K50-<cellule>-A|B` partagés par A1/A2 et B1/B2 ;
  passes distinguées par la date et l'ordre de la file.
  Après le vérificateur : (12) défaut corrigé après la mesure (commit `fix(K-50)`) : un échec de recherche retirait
  l'entrée de même étiquette, et toutes les voix `voiceEmbedding:` portent l'étiquette « embedding » : deux voix
  alternées se chassaient (une seule entrée utile). Le retrait par étiquette est supprimé (l'empreinte suffit contre
  un préfixe périmé ; un `voiceKey` réutilisé laisse l'ancienne entrée jusqu'à son éviction LRU) ; test
  `testVoicesSharingAKeyKeepTheirEntries` rouge sur l'ancien code (worktree jetable, `.local-runs/k50-red2.log` et
  `k50-red2.diff.txt` : 4 calculs au lieu de 2), vert avec (`.local-runs/k50-unit2.log`) ; suite Debug rejouée
  (`.local-runs/k50/suite2.log` : 628 tests, 49 sautés, 0 échec). Cellules mesurées à une voix : chemin non emprunté, mesures non rejouées.
  (13) Le rouge de la mutation « clé seule » (dérogation 3) a été fait dans l'arbre principal et son diff n'est pas
  archivé (`VERIFY.md` §1 : worktree jetable) ; celui de (12) l'est. (14) La surcharge streaming sans `voiceKey`
  calcule désormais le préfixe (et l'empreinte) de façon synchrone sur le thread de l'appelant, hors de
  `withMLXErrors` (`VoxtralTTSPipeline.swift:643`, avant le `Task`), comme les voix prédéfinies et un `voiceKey`
  fourni le faisaient déjà : ≈ 100 à 300 ms au premier appel d'une voix, à reprendre par une fiche de suite.
  (15) Les amendements de #643 (2026-10-16) lisent la porte « sur le chemin caché `synthesize(voice:)` » : mesurée
  par `voiceEmbedding:`, le chemin que K-50 met en cache (`clone_fr` n'est pas une voix prédéfinie). (16) Câblage
  streaming et correction « Fixed » du `voiceKey` couverts par les tests unitaires de `VoicePrefixCache` seuls, sans
  test de pipeline ni mesure du streaming ; coût de l'empreinte non mesuré seul (≈ 1,2 Mo copiés et hachés par appel
  pour `clone_fr` en fp32). (17) `unload()` vide le cache hors du `gate` (`:864` à `6580961d`, `:869` à `9004f580`) ;
  depuis `9004f580`, la course est plus large qu'avant K-50 (dérogation 28). (18) A/A de la cellule d'information 6 bits moyen 4,0 %, au-dessus du seuil de 3 % : lue à
  titre indicatif (BENCHMARKS précisé).
  Après le second vérificateur : (19) correctif de frontière (commit `fix(K-50)`) : `voicePrefix` tourne dans la
  `Task` productrice du streaming, sous `withMLXErrors`, et lève l'erreur MLX interceptée ; `precomputeVoicePrefixCache`
  s'arrête avant les couches sur une erreur interceptée ; un préfixe en échec n'est pas gardé
  (`testFailedPrefixIsNotKept`, rouge par compilation seulement sur l'ancien code : le paramètre `failed` n'existait
  pas). Test gardé `TTSStreamingPrefixBoundaryTests` (embedding de largeur 100 en streaming puis en batch) : rouge
  sur `6580961d` (worktree jetable, `.local-runs/k50-red3.log`, pile `k50-red3.stack.txt` : `fatalError` du
  gestionnaire par défaut dans `buildInputEmbeddings`), vert à HEAD (`.local-runs/k50-unit3.log`, code de sortie 0) ;
  suite Debug `.local-runs/k50/suite3.log` : 630 tests, 50 sautés, 0 échec, code de sortie 0. Avant K-50, les voix
  prédéfinies et un `voiceKey` fourni calculaient déjà le préfixe hors de la frontière : corrigé aussi (CHANGELOG
  Fixed). Le préfixe batch était déjà dans la frontière ; il lève maintenant l'erreur avant `generate`. Chemin
  batch mesuré inchangé en succès (aucune erreur) : mesures non rejouées ; la dérogation (14) est levée.
  (20) Charge de fond pendant `6bit-short-B2` (cellule de la porte) : corespotlightd jusqu'à 220 % et mediaanalysisd
  249 % (`6bit-short-B2.samples.load`) ; B1/B2 5,4 % d'écart, au détriment de B, sans effet sur la porte.
  (21) Logs de test de `d8b457d8` (`k50-unit1`, `k50-red`, `suite`) sans code de sortie (`VERIFY.md` §1) ; ceux de
  `6580961d` (`k50-red2`, `k50-unit2`, `suite2`) non plus ; ceux de `9004f580` l'ont. (22) Depuis K-50, `bench tts --warmup ≥ 1` mesure le chemin en cache pour toute voix : les
  valeurs de départ K-35 de préfill et de TTFT citées par d'autres portes (K-66 : TTFT 252 à 258 ms `neutral_female`,
  K-79) ne sont plus comparables : à recaler au prochain replanning. (23) Capacité fixée à 3, sans réglage ;
  `docs/References.md:107` prévoit pour `lean` une entrée : à porter par K-79. (24) Prérequis K-35 encore `open`
  dans le tableau (antérieur à K-50, comme K-41 et K-43). (25) Cinq commits pour la fiche (`perf`, `docs`, deux
  `fix`, `docs` de vérification), contre « une fiche = un commit » ; précédents K-58, K-65. (26) Coût de l'empreinte
  payé à chaque appel, voix prédéfinies comprises (avant : comparaison de chaînes), non mesuré seul.
  Après le troisième vérificateur (verified) : (27) rouges partiels de `9004f580` : `testFailedPrefixIsNotKept` rouge
  par compilation seulement ; la garde `hasError` de `precomputeVoicePrefixCache` (`VoxtralTTSModeling.swift:438`)
  sans rouge par mutation ; la moitié batch de `TTSStreamingPrefixBoundaryTests` jamais exercée en rouge (le test
  plante avant, sur le streaming). (28) Entrée périmée possible : depuis que le préfixe se calcule dans la `Task`,
  `synthesizeStreaming(…)` ; `unload()` ; `loadModel(autre pack)` en séquence peut voir le `removeAll` du chargement
  passer avant que la `Task` ajoute l'entrée de l'ancien modèle (`VoxtralTTSPipeline.swift:707` contre `:191`,
  `:869`) ; l'annulation ne l'empêche pas. Impact faible (mêmes formes, pas de plantage), cas rare : fiche de suite
  (n'ajouter l'entrée que si la génération du `gate` n'a pas changé). (29) En streaming, après un raté de cache,
  `chunk.elapsed` inclut maintenant le calcul du préfixe ; le TTFA de `bench` (mesuré côté appelant) n'est pas
  touché. (30) `TTSConsecutiveSynthesisReproTests` (gardé) non rejoué ; l'invariant « entrée jamais mutée » est
  couvert sur `voiceEmbedding:` par les `out_sha256` identiques de la 2ᵉ synthèse. (31) Le machine-check des passes A
  vérifie le binaire B ; le binaire A n'est attesté que par `top_process` ; `k50B-build.log` (12:44:13) est une
  reconstruction à vide à `86899785`. (32) Références de ligne des dérogations (10) et (17) écrites à leur commit
  (`:482` → `:485`, `:864` → `:869` à HEAD). (33) « Gain attendu (non obtenu) » de la fiche est le gabarit commun,
  laissé tel quel.
  Après le contradicteur (not verified) : (34) Défaut : le cache est désormais toujours actif pour les voix par
  embedding (batch, et streaming sans `voiceKey`), sans moyen de le couper ; c'est l'objectif écrit de la fiche (« la
  surcharge `voiceEmbedding:` utilise le cache… LRU de 2 à 4 entrées ») et sa porte prévoit l'écart bf16 (« sortie
  conforme à K-94 ») ; mais c'est un comportement par défaut visible de FluxForge (audio bf16 à graine égale, ≈ 27 à
  81 Mo gardés jusqu'à `unload()`) : question posée à Vincent (`.local-runs/A-DECIDER.md`, K-50 ; A garder, proposé),
  code gardé en attendant ; la dérogation (2), sur les signatures seules, ne suffisait pas. (35) La dérogation (5)
  (bf16 sur le court seul) est levée : moyen et long × graines 1 à 3 mesurés (`.local-runs/queue-k50c.sh`,
  BENCHMARKS §K-50, complément) : couverture moyenne égale (moyen) ou meilleure (long) ; graine 3 du long hors des
  1,8 point de K-94 (A et B au plafond de 2 500 frames, B WER 252,7 contre 61,7), ajouté à la question de (34).
  (36) Cellule de la porte : 6 bits court seul, choisie avant la mesure (`queue-k50.sh`, ligne de validation à une
  cellule) ; la porte recalée de #643 citait aussi 295 ms en 6 bits moyen : celui-ci fait − 26,4 % (toutes les paires
  A×B entre − 23 et − 29,6 %), sous − 30 % ; consigné, affirmations publiques corrigées (CHANGELOG : 26 % sur le
  moyen). (37) Garde ajoutée : `voicePrefix` évalue l'embedding et lève une erreur MLX déjà interceptée avant
  l'empreinte (un mélange paresseux en échec laisserait un tableau vide à lire) ; non testé (aucun moyen simple de
  fabriquer un tableau dont seule l'évaluation échoue) ; tests ciblés et gardé verts (`.local-runs/k50-unit4.log`,
  10 tests, code 0), suite `.local-runs/k50/suite4.log` (630 tests, 50 sautés, 0 échec, code 0). (38) Taille d'une entrée : ≈ 27 Mo jusqu'à 254 frames de voix,
  ≈ 54 Mo jusqu'à 510 (le KV croît par pas de 256 positions ; `Config.numFrames` de l'enrôlement est réglable) ;
  commentaire et CHANGELOG précisés. (39) Glossaire TTFT-frame (`docs/Benchmarks.md`), README, `docs/tts_benchmark.md`
  et en-tête de `VoicePathParityTests` mis à jour : depuis K-50, `ttft_ms` et `phases_ms.prefill` excluent toujours le
  préfill de la voix, `--warmup 0` compris (complète la dérogation 22). (40) Aucune API publique pour vider le cache
  ou s'en passer (rejoint (23) et K-79). (41) La couverture sur 11 mots (texte court) discrimine mal ; l'écoute
  anonymisée (ASK-13) est proposée en option C de la question de (34).
  (42) Question de (34) tranchée par Vincent le 2026-10-19 : A, le cache reste actif par défaut pour toute voix ;
  dérogations (34) et (35) levées, prévenir FluxForge reste à Vincent (version).
- **K-47 (#652), 2026-10-20, décision de l'agent** (`VERIFY.md` §1 et §2 ; appliquée) : (1) passes `--passes 1 --warmup 1
  --cooldown 120`, un processus par passe, au lieu de `--passes 2` de l'étape 4 (conduite de K-38, K-43, K-46).
  (2) A1 sur batterie écartée ; A1r refaite 10 h après B1/B2 ; la porte est lue sur un second A/B/B/A complet
  (`queue-k47c.sh`), la file 1 donnée pour information (A/A 3,04 %, au-dessus du seuil de 3 % : invalide comme porte). (3) Aucun test unitaire nouveau : le
  pipeline n'est pas séparable du modèle sans réécrire `generate` ; la parité se prouve par `out_sha256` sur les trois
  clips (porte) et la branche asynchrone par l'occupation xctrace (porte), pas par un test rouge. (4) `KO
  indexation/sauvegarde` avant A1r (29 %) et B3 (75 %), tolérés (décision du 2026-10-05). (5) En `temperature > 0`,
  un tirage `MLXRandom` de plus est consommé après un arrêt anticipé (pas jeté) : l'état global du générateur après
  l'appel diffère, la transcription non ; non mesuré (le défaut est 0).
  (6) Après le vérificateur (`verified`) : une erreur MLX levée en construisant le pas qui sera jeté (sur EOS) fait
  échouer la transcription, ce que l'ancienne boucle évitait en ne le construisant pas ; théorique (une erreur de
  construction est déterministe et serait survenue aux pas précédents), non testé. Occupation relue sur le décodage
  seul (≈ 86,1 → 98,8 %), citation `REFUSED` complétée (second `VoxtralCLI` non identifié), `rtf` file 1 − 10,3 %,
  commentaire de `decodeLoop` mis à jour.
  (7) Après le contradicteur (`verified`) : le cast fp32 des logits avant la division par la température, renvoyé à
  K-47 par K-38 (dérogation de K-38, après son contradicteur), n'est pas fait : `sampleToken` garde `logits * (1 /
  temperature)` dans le dtype de calcul ; défaut `temperature = 0` (argMax, non concerné), risque en fp16 pour une
  température d'environ 5e-4 seulement ; renvoyé à une fiche de suite (PLAN.md §7, K-47). (8) La parité couvre la
  lecture finale (C-moyen va au bout de l'audio : 1 835 pas, `truncated:false`), pas l'arrêt anticipé sur EOS ou budget
  avec un pas jeté, prouvé par lecture du code seulement. (9) L'occupation de 92,3 / 93,1 % est un plafond de gain
  (PLAN.md §5), non un seuil : 1 − 86,9 / 98,3 = 11,6 % de temps GPU inactif retiré, pour − 11,7 % (c_20s) et − 12,1 %
  (C-moyen) mesurés. (10) `embedToken(Int)` ne sert plus qu'aux tests (`RealtimeDTypeTests`) ; la production passe par
  `embedTokens` sur un `MLXArray` [1] uint32 (même ligne de la table, `RealtimeQuantizedHeadTests`).

- **K-91 (#632), 2026-10-20, décision de l'agent** (`VERIFY.md` §1 et §2 ; appliquée) : (1) refus et lecture dans un
  seul commit `feat(K-91)` (`f1e4cc44`) au lieu de `fix(K-91)` puis `feat(K-91)` de l'étape 7 : la détection du format
  sert au refus et à la lecture, les séparer aurait demandé un état intermédiaire non testé ; le commit `fix(K-91)`
  (`de78332d`) corrige le conditionnement temporel trouvé par les mesures. (2) Test rouge prouvé dans un worktree
  jetable sur l'ancien chargeur, avec des bouchons pour les symboles nouveaux (`.local-runs/k91-red.diff.txt`) : 5/6 en
  échec, mlx-audio en témoin vert (`.local-runs/k91-red.log`). (3) Packs téléchargés sur le disque interne
  (`~/Library/Caches/models`), pas sur le disque Lexar (94 % plein). (4) Mesures en B seul, sans binaire A : A ne lit pas
  voxmlx ; la parité des trois packs déjà lus est jugée sur leurs `out_sha256` consignés (K-38, K-47). (5) Une passe
  `bench` par pack (`--passes 1`), sans A/A : la porte demande une ligne par pack, la matrice reste à K-78. (6) Première
  file écartée (texte vide sur C-moyen en 6 et 8 bits), lignes gardées dans `BENCHMARKS.md` ; file refaite à
  l'identique après le correctif. (7) `KO indexation/sauvegarde (103 %)` avant une passe de la file 1, toléré
  (décision du 2026-10-05). (8) Aide de la CLI (`realtime --model`, `profile`) complétée des deux ids. (9) Suite
  notée : `load_ms` / `load_active_mb` des packs voxmlx ne mesurent pas le chargement (rien n'est évalué avant le
  premier encodage) ; évaluer les paramètres en fin de chargement rendrait la mesure comparable (à mesurer, K-78).
  (10) Tags identiques entre la file 1 (écartée) et la file 2 (`K91EVAL-<pack>`, `K91-<pack>`, contre « un tag
  distinct par série » de `VERIFY.md` §1) : la file 2 refait la file 1 à l'identique ; les séries se distinguent par
  `commit` (`f1e4cc44f` / `de78332da`), `date` et le bloc de `BENCHMARKS.md`. (11) Diagnostic de la file 1 : la
  comparaison des poids 6 bits au fp16 est conservée (`.local-runs/k91r/weights-vs-fp16-6bit.txt`, refaite après coup,
  script `.local-runs/tools/k91_cmp_weights.py` ; convolutions voxmlx au format MLX, comparées sans transposition :
  `SHAPE`) ; l'essai sur 8 s de C-moyen (texte vide en 6 bits, texte en 4 bits) a été lu à l'écran, sans log conservé.
  (12) Charge de fond de la file 2 enregistrée (`.samples.load`) : `photoanalysisd` / `photolibraryd` pendant `bench`
  4 bits, Mail pendant `bench` 6 bits, `backupd` pendant `eval` 6 bits (Time Machine relancé malgré `tmutil
  stopbackup`) ; sans effet sur le texte, et les `step_ms_p50` sont une passe sans A/A (annoncé). (13) Prémisse de la
  fiche inexacte : avant K-91, un pack voxmlx complet levait `VoxtralError.missingWeights` (sans format ni clé en
  trop nommés), il ne se chargeait pas en poids aléatoires ; le rouge porte sur l'erreur dédiée (format, clés en
  écart) et sur les clés en trop ignorées. (14) Suite `k91/suite3.log`
  lancée sur l'arbre de `de78332d` avant son commit (mêmes fichiers, commit 8 s après la fin de la suite).
  (15) Vérification (2026-10-20 ; vérificateur rejeté puis vérifié, contradicteur vérifié) : dérogation (12) complétée
  (Mail pendant `bench` 4 bits, `textunderstandingd` et `mediaanalysisd` pendant `eval` 8 bits) ; écart maximal des
  poids 6 bits au fp16 2,45 % (et non ≤ 2,4 %) ; `encode` 2 387 ms (6 bits) contre 2 213 ms (4 bits), part de la
  matérialisation non mesurée. Suites notées, sans effet sur les deux packs du registre : surcharge de quantification
  par couche lue sous le nom interne (un pack voxmlx mixte écrit en noms voxmlx serait mal lu, erreur de forme) ;
  `sanitizeVoxmlxWeights` ne filtre pas `rotary_emb` / `position_ids` ; une couche privée de `.scales` et `.biases`
  lève `loadingFailed` (forme) et non l'erreur dédiée ; `findRealtimeModelPath` ignore la révision épinglée ;
  message « loaded without error » des tests trompeur pour une erreur d'un autre type ; aucun journal du build
  Release gardé (correctif prouvé par le comportement).
