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
  - **macOS** : cellules courtes sous 27.0.0, moyen et long sous 27.0.1 (mise à jour du système pendant la
    campagne) ; le même contrôle couvre l'écart.
  - **Streaming moyen et long avec `--cache-limit-mb 2048`** : sans plafond, le `phys_footprint` atteint 74 à 75,9
    Go sur 96 (swap non exclu). Ce plafond est une condition de mesure : le défaut public reste sans plafond.
  - **Batch moyen et long et ligne consommateur en profil par défaut** (sans plafond, chemin de FluxForge). Le profil
    plafonné, mesuré aussi, ne change pas `step_ms_p50` au-delà du seuil de bruit (− 5,0 à + 4,1 %, dans les deux
    sens) ; ses lignes sont consignées comme comparaison.
  - **Streaming court jugé sur `total_ms`** (passe de moins de 60 s) : il a été mesuré avant `stream_frame_ms_p50`
    (`0e7588a9`). Le streaming moyen et long est jugé sur `stream_frame_ms_p50` / `p90`.
  - **Reprises** : une cellule en échec d'A/A est refaite (`-r2`, `-r3`, `-default-r2`) sur machine calme. La cause
    n'est écrite que si l'échantillonneur de charge la montre dans la passe lente ; sinon « cause non isolée »
    (régime de K-85). Les passes écartées restent dans `BENCHMARKS.md`, chacune avec son fichier et sa cause.
  - **Cellules à audio doublé (K-92)** : 4 cellules streaming retenues gardent leur temps par frame ; leurs
    `audio_s`, `rtf` et `out_sha256` sont marqués faux, et un contrôle après correctif est consigné.
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
