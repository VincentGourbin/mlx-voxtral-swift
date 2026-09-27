# Poids et profils de référence au 2026-09-27 — état de l'art Voxtral (phase 3)

> Audit `mlx-swift-audit`, phase 3 (« standard des profils »). Dépôt `mlx-voxtral-swift`, révision **`9392ed1`**
> (branche `claude/action-plan-skills-beta-wifgmu`). Rédigé le 2026-09-27 dans une session cloud Linux : pas de Mac,
> pas de toolchain Swift, pas de GPU. **Aucun chiffre de performance n'est produit ici.** Chaque valeur est soit
> « mesurée (source) », soit « à mesurer ». Un gain est « attendu », jamais « obtenu ». Rien n'a été téléchargé, rien
> n'a été publié sur le Hub ni sur GitHub.
>
> Constats de ce rapport : **M-01 … M-07**. Ils complètent, sans les répéter, `audit-stabilite.md` (S-xx),
> `audit-annexes-serveur.md` (A-xx), `audit-performance-stt.md` (P-01…P-29), `audit-performance-tts.md` (P-30…P-49),
> `audit-performance-realtime-instruments.md` (P-60…P-79) et `patterns-verdicts.md` (MLX-0xx).

## 0. Sources et méthode

| Source | Ce qui en est tiré | Limite |
|---|---|---|
| Code à `9392ed1` : `Utils/ModelRegistry.swift`, `TTS/VoxtralTTSRegistry.swift`, `Realtime/VoxtralRealtimeRegistry.swift`, `Utils/VoxtralStandardLoader.swift`, `TTS/VoxtralTTSModelLoading.swift`, `Realtime/VoxtralRealtime{ModelLoading,Configuration}.swift`, `Utils/ModelDownloader.swift`, `CoreML/*`, `VoxtralQuantization.swift`, `Pipeline/VoxtralPipeline.swift` | inventaire, formats acceptés, constats | lecture seule |
| Connecteur Hugging Face (`hf_fs search/ls/cat/stat`, `hub_repo_details`), 2026-09-27 | dépôts, dates, tailles exactes en octets, `config.json`, index, cartes de modèle, licences | **SHA-256 non exposés** par le connecteur ; l'API `tree` (`lfs.oid`) est bloquée par le proxy de la session (HTTP 403) → tous les SHA-256 sont « à relever » |
| `ml-explore/mlx-swift-lm` `main` @ `ee673d6` (2026-09-22), historique approfondi à 120 commits ; tag `3.31.4` = `bd4b743` (2026-06-29) | écarts amont (§4) | clone initial superficiel (1 commit) : `git fetch --deepen` nécessaire |
| `ml-explore/mlx-swift` `main` @ `9019419` (2026-09-17) ; tag `0.31.6` = `0bb916c` (**2026-07-01**) | écarts amont (§4) | sous-module `mlx` (cœur C++) non extrait : contenu de mlx 0.32.2 non lu |
| Rapports de la phase 2 du même audit | mesures déjà relevées (F-xx), constats liés | — |
| Issues GitHub `VincentGourbin/mlx-voxtral-swift` (22, toutes fermées) | aucune demande liée aux poids | lecture seule |

Unités : `Go` = 10⁹ octets (tailles du Hub, décimales), `Gio`/`Kio` = puissances de 2. Les « estimations » de
résidence (§6) sont calculées ainsi : paramètres (dérivés des `config.json`, recoupés avec les compteurs du Hub à
0,1 M près) × bits effectifs (affine g64 : b + 0,5 ; g128 : b + 0,25). L'écart avec la taille du fichier est ≤ 9 %.
Ce sont des estimations, **pas des mesures**.

## 1. Synthèse

1. **Aucun nouveau poids Voxtral chez Mistral depuis mars 2026.** La famille ouverte compte 4 dépôts : Mini 3B et
   Small 24B (2507, Apache-2.0), Realtime Mini 4B (2602, Apache-2.0) et TTS 4B (2603, **CC BY-NC 4.0**). Le code les
   couvre tous. « Voxtral Mini Transcribe 2.0 », cité par la carte Realtime comme référence hors ligne, n'a pas de
   dépôt sur le Hub. L'état de l'art de septembre 2026 se joue donc sur les **packs MLX**, pas sur de nouveaux
   checkpoints.
2. **Tous les dépôts du registre existent encore, mais l'un d'eux a dérivé.** `mistralai/Voxtral-Mini-4B-Realtime-2602`
   publie désormais un `config.json` et un `model.safetensors` au format transformers. Le chargeur lit ce
   `config.json` en premier et ne sait pas le décoder : l'entrée `realtime-4b` ne se charge plus, et son
   téléchargement double à 17,7 Go (**M-01**).
3. **Les meilleurs packs STT publiés en 2026 ne se chargent pas.** Le chargeur STT refuse toute config de
   quantification qui porte la clé `"mode"`. Or c'est le cas de tous les convertisseurs actuels :
   `aufklarer/…-MLX-8bit` (2026-07-23) et les trois packs `MarkusKaemmerer/…-dense-encoder` (juillet-septembre 2026).
   Les chargeurs TTS et Realtime ignorent ce mode, et chargeraient donc un pack `mxfp4` ou `nvfp4` faux sans erreur
   (**M-02**).
4. **L'axe qualité qui compte le plus pour l'ASR est la précision de l'encodeur audio, pas le nombre de bits du
   LM.** Des mesures externes publiées (noScribe, M1 Max, Python) sur Voxtral Small donnent un CER de
   4,87 / 4,28 / 2,46 / 2,41 % pour un encodeur 4 / 6 / 8 bits / bf16. `lm_head` 6/8/bf16 donne des taux identiques.
   Les deux packs 4 bits du registre ont un encodeur en 6 bits.
   Mais en backend `.auto` (défaut), l'encodeur et le projecteur viennent du modèle Core ML fp16 dense : les bits
   d'encodeur du pack ne jouent qu'en backend `.mlx`. Un profil STT doit donc figer le couple (backend, pack)
   (**M-07**, lié à P-14).
5. **Amont** :
   - Voxtral résout mlx-swift **0.31.6 (2026-07-01)**, car mlx-swift-lm `main` l'épingle en `upToNextMinor`. Ce tag
     ne contient ni le correctif du deadlock `CompiledFunction.lock` × `evalLock` (`df9ae26`, lié à A-01), ni
     l'échelle globale NVFP4 (`09051ed`), ni mlx 0.32.2.
   - mlx-swift-lm `main` offre ce qu'il faut pour corriger M-02 : `BaseConfiguration` public décode `mode`, le
     per-layer et `false`.
   - Il offre aussi des stratégies de cache KV typées (affine 8/4 bits, TurboQuant, variance normalisée) et une
     conversion Swift avec modes et prédicat par couche.
   - Il n'a **aucun modèle audio natif**.
6. **Matrice candidate (§6)** :
   - Largeurs chargeables aujourd'hui : STT 4/8/16 ; Realtime 4/16 (le 8 bits est bloqué, P-70) ; TTS 4/6/16 (aucun
     8 bits valable).
   - Presque tous les réglages qui comptent (dtype de calcul, KV quantifié, `cacheLimit`, tranche de préfill,
     libération de l'encodeur) **n'ont pas encore de bouton**. Un profil v0 ne peut figer que les poids, le backend,
     `MemoryOptimizationConfig`, `maxTokens`, la température, la pénalité de répétition et le retard Realtime.
   - Packs à publier (ASK) : Realtime 8 bits, TTS 8 bits, et éventuellement Mini 4 bits à encodeur 8 bits/bf16.

## 2. Ce que le code sait charger (inventaire à `9392ed1`)

### 2.1 Registres

Tailles Hub relevées le 2026-09-27 (somme des `*.safetensors` chargés ; `consolidated.safetensors` compté à part).

| Registre · id | Dépôt | Taille déclarée | Taille réelle (Hub) | Précision déclarée → réelle | Charge aujourd'hui ? |
|---|---|---|---|---|---|
| STT · `mini-3b` (`ModelRegistry.swift:49-56`) | `mistralai/Voxtral-Mini-3B-2507` | « ~6 GB » | shards 9 356 474 312 o (**+ consolidated 9 348 806 528 o téléchargé aussi, S-07**) | « float16 » → **bf16** (`torch_dtype`) | oui ; 16 bits inutilisable en perf avant P-01/P-05 |
| STT · `small-24b` (`:58-65`) | `mistralai/Voxtral-Small-24B-2507` | « ~48 GB » | shards 48 527 546 144 o (+ consolidated 48 519 877 672 o, S-07) | « float16 » → **bf16** | oui (idem) |
| STT · `mini-3b-8bit` ★ (`:68-77`) | `mzbac/voxtral-mini-3b-8bit` | « ~3.5 GB » | 5 404 054 476 o (2 shards) | 8 bits g64 uniforme, encodeur compris ; `embed_tokens`/`lm_head` 8 bits | oui |
| STT · `mini-3b-4bit` (`:78-86`) | `mzbac/voxtral-mini-3b-4bit-mixed` | « ~2 GB » | 3 195 753 212 o | LM 4 b g64, MLP des couches 0-1/28-29 en 6 b, **encodeur + projecteur 6 b**, `embed_tokens` 4 b, `lm_head` 6 b g128 | oui |
| STT · `small-24b-8bit` (`:89-97`) | `VincentGOURBIN/voxtral-small-8bit` (l'enum `VoxtralPipeline.Model` pointe vers `mzbac/Voxtral-Small-24B-2507-8bit`, S-06) | « ~25 GB » | 26 499 134 369 o (5 shards) ; `mzbac/…` 28 056 927 031 o (6 shards) | 8 b uniforme (config **identique** octet pour octet entre les deux dépôts : 80 122 o ; écart de taille 1,56 Go non expliqué, À VÉRIFIER) | oui |
| STT · `small-4bit` (`:98-106`) | `VincentGOURBIN/voxtral-small-4bit-mixed` | « ~12 GB » | 14 857 318 962 o (3 shards) | même prédicat que le Mini 4 bits mixte (encodeur 6 b, `lm_head` 6 b g128) | oui |
| Core ML (encodeur + projecteur, backend `.auto`/`.hybrid`) | `VincentGOURBIN/voxtral-encoder-coreml-{mini,small}` (`CoreML/VoxtralCoreMLEncoder.swift:65-69`) | — | `weight.bin` 1 324 309 760 o / 1 378 839 808 o | fp16, converti depuis `mistralai/…` (`Scripts/CoreMLConversion/convert.sh:47`, `convert_to_coreml_ane.py:162`) | oui |
| Realtime · `realtime-4b-4bit` ★ (`VoxtralRealtimeRegistry.swift:29-38`) | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | « ~3 GB » | 3 133 798 126 o | 4 b g64 `affine` ; `tok_embeddings` (tête liée) et adaptateur **non quantifiés** | oui |
| Realtime · `realtime-4b-fp16` (`:39-47`) | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` | « ~8 GB » | 8 870 608 794 o | fp16 (≠ bf16 d'entraînement) | oui |
| Realtime · `realtime-4b` (`:48-56`) | `mistralai/Voxtral-Mini-4B-Realtime-2602` | « ~8 GB » | consolidated 8 859 462 744 o **+ model.safetensors 8 859 446 848 o** (tous deux téléchargés) | bf16 | **non (M-01)** |
| TTS · `tts-4b-mlx` ★ (`VoxtralTTSRegistry.swift:29-38`) | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | « ~8 GB » | 8 004 759 170 o (2 shards) | bf16 | oui (défaut le plus lent, P-34) |
| TTS · `tts-4b` (`:39-47`) | `mistralai/Voxtral-4B-TTS-2603` | « ~8 GB » | consolidated 8 004 752 248 o + 20 voix `.pt` | bf16 | oui |
| TTS · `tts-4b-4bit` (`:48-56`) | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | « ~2.5 GB » | 2 509 879 373 o | 4 b g64 `affine` ; codec (116 tenseurs) bf16 (F-12 du rapport TTS) | oui |
| TTS · `tts-4b-6bit` (`:57-65`) | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | « ~3.5 GB » | 3 465 520 393 o | 6 b g64 `affine` | oui |

★ = défaut du registre. Paramètres (Hub, recoupés par le calcul) : Mini 4 676,3 M, Small 24 261,8 M, Realtime
4 429,7 M, TTS ≈ 4 002 M.

### 2.2 Chargeurs : ce qu'ils acceptent

| Aspect | STT (`VoxtralStandardLoader.swift`) | TTS (`VoxtralTTSModelLoading.swift`) | Realtime (`VoxtralRealtimeModelLoading.swift`) |
|---|---|---|---|
| Config lue | `config.json` décodé strictement en `VoxtralStandardConfiguration` (`:1293-1295`) | `params.json` puis `config.json` (`:22-34`) ; quantification relue dans `config.json["quantization"]` seulement (`:91-102`) | `config.json` **d'abord**, puis `params.json` (`:63-77`) ; deux formes : mlx-community puis `params.json` Mistral (`VoxtralRealtimeConfiguration.swift:195-205`) |
| Fichiers de poids | tous les `*.safetensors` sauf `consolidated*` (`:986-1024`) | `model-00001-of-00002` + `-00002-of-00002`, sinon `model.safetensors`, sinon `consolidated.safetensors` (`:118-145`) | index, sinon `model.safetensors`, sinon `consolidated` (`:81-111`) |
| Quantification par couche | oui : `true` / `false` / `{group_size, bits}` (`:87-114`, `:1092-1217`) | non : bits et groupe globaux pour toutes les couches à `.scales` (`:61-64`) | non : globaux, liste de clés exclues codée en dur (`:34-41`) |
| Mode (`affine`/`mxfp4`/`mxfp8`/`nvfp4`) | **clé `mode` = échec du décodage** ; `.affine` en dur (`:1245`) — M-02 | lu puis ignoré (`.affine` dans les deux branches, `:58`) — P-48 | non lu (`VoxtralRealtimeConfiguration.swift:101-109`) — M-02 |
| Vérification des clés | `update(parameters:)` sans `verify` (`:1323`, `:1338`) | `verify: .none` (`:71`) | `verify: .none` (`:55`) |
| Format de clés | HF « transformers » Voxtral (`audio_tower.*`, `language_model.*`, `multi_modal_projector.*`, `lm_head`) | Mistral `consolidated` ou mlx-community (`language_model.model.model.*`) (`:160-210`) | Mistral `mm_streams_embeddings.*` ou mlx-audio (`encoder.*`, `decoder.*`) (`:123-224`) ; **ni voxmlx, ni transformers** (P-70, M-01) |

Chaîne de **production** de packs : aucune. `saveQuantizedModel` n'écrit que `config.json`
(`VoxtralQuantization.swift:503-524`). `saveModel` n'écrit ni bloc `quantization` ni métadonnées (`:85-107`).
`getQuantizationStats` cherche des poids `int8`/`uint8` (`:16-36`), alors que les poids quantifiés MLX sont `uint32`
packés (piège n° 2 du catalogue) ; elle compterait donc 0 couche quantifiée. Tout cela est du code mort
(tableau de `audit-stabilite.md` l. 346). Tout pack Voxtral est donc produit hors du dépôt : mlx-voxtral / noScribe
pour le STT, mlx-audio pour le Realtime et le TTS.

## 3. Le Hub au 2026-09-27

### 3.1 Famille officielle `mistralai`

| Dépôt | Mise à jour | Paramètres | Licence | Fichiers | Remarque |
|---|---|---|---|---|---|
| `Voxtral-Mini-3B-2507` | 2025-07-28 | 4 676,3 M | Apache-2.0 | `consolidated` + 2 shards HF + `params.json` + `config.json` | inchangé depuis la version portée |
| `Voxtral-Small-24B-2507` | 2025-12-20 | 24 261,8 M | Apache-2.0 | `consolidated` + 11 shards HF | inchangé côté poids |
| `Voxtral-Mini-4B-Realtime-2602` | **2026-03-11** | 4 429,7 M | Apache-2.0 | `consolidated` + **`model.safetensors` + `config.json` transformers** (`VoxtralRealtimeForConditionalGeneration`, `transformers_version` 5.2.0.dev0) + `params.json` | dérive de format → M-01 ; retard 480 ms recommandé ; 13 langues |
| `Voxtral-4B-TTS-2603` | 2026-03-31 | ≈ 4 B | **CC BY-NC 4.0** (héritée des voix de référence) | `consolidated` + `params.json` + 20 voix `.pt` | usage commercial exclu par la licence (ASK Q-M4) |

Aucun autre dépôt Voxtral chez `mistralai` : la recherche `hf://models/mistralai Voxtral` en renvoie 4. Parmi les 40
créations les plus récentes de l'organisation, les derniers dépôts audio sont `Voxtral-4B-TTS-2603` et
`Voxtral-Mini-4B-Realtime-2602`, et les plus récentes (Shieldstral, 2026-08) ne sont pas audio.
`mistral-experimental` publie seulement des exports GGUF et ExecuTorch du Realtime.

### 3.2 Variantes MLX (et apparentées) utiles au choix

| Dépôt | MAJ | Modèle · format | Octets | Convertisseur | Chargeable par le code actuel ? | Mesures publiées (externes) |
|---|---|---|---|---|---|---|
| `mlx-community/Voxtral-Mini-3B-2507-bf16` | 2026-01-13 | Mini · bf16, 2 shards de même taille que ceux de `mistralai` | 9 356 474 312 | mlx-audio | oui (config sans quantification, lecture) ; **évite le `consolidated`** (S-07) | — |
| `aufklarer/Voxtral-Mini-3B-2507-MLX-8bit` | 2026-07-23 | Mini · 8 b g64 uniforme, `"mode": "affine"` | 5 566 018 003 | inconnu | **non (M-02)** | — |
| `aufklarer/…-MLX-5bit`, `…-MLX-FP16` | 2026-07-23 | Mini | non inspectés | — | — | — |
| `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` | 2026-07-29 | Mini · LM + `lm_head` 8 b g64, **encodeur + projecteur bf16**, `"mode": "affine"` | 6 017 427 099 | noScribe `tools/quantize_voxtral.py` (sans données, reproductible bit à bit) | **non (M-02)** | 6,60× temps réel contre 6,68× (8 b uniforme) et 1,46× (bf16) ; pic 7,7 / 7,2 / 13,2 Go (2 min) ; WER 4,27 % contre 4,74 % (8 b uniforme) sur un passage difficile de 422 mots ; FLEURS DE 4,81 % contre 4,81 % (M1 Max, mlx-voxtral 0.0.4) |
| `MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder` | 2026-09-23 | Small · LM + `lm_head` 4 b g64, encodeur + projecteur bf16, `"mode"` | 15 016 527 526 | idem | **non (M-02)** | FLEURS DE WER 2,78 % contre 2,61 % (bf16, Open ASR Leaderboard) ; pic 19,4 Go sur une passe de 10 min ; ≈ 2× temps réel (M1 Max). Balayage de l'encodeur (Small, allemand difficile, LM 4 b) : WER 10,19 / 9,72 / 8,29 / 8,29 %, CER 4,87 / 4,28 / 2,46 / 2,41 % pour 4 / 6 / 8 / bf16. `lm_head` 6/8/bf16 : erreurs identiques, bf16 −27 % de débit. **mxfp4 moins bon qu'affine 4 b ; nvfp4 sans échelle globale : modèle cassé** (« ist, ist, ist ») |
| `MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder` | 2026-09-23 | Small · 8 b + encodeur bf16, `"mode"` | 27 138 066 384 | idem | **non (M-02)** | « ~34 Go nécessaires » (carte) |
| `mlx-community/Voxtral-Mini-4B-Realtime-6bit` | 2026-02-08 | Realtime · 6 b, format **voxmlx** | 3 609 304 614 | voxmlx | non (P-70) | — |
| `ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx` | 2026-02-20 | Realtime · 8 b, format voxmlx | 4 714 618 595 | voxmlx | non (P-70) | — |
| `T0mSIlver/Voxtral-Mini-4B-Realtime-2602-4bit-qhead` | 2026-07-20 | Realtime · 4 b **tête liée quantifiée** (−0,58 Go), format mlx-audio | 2 554 984 165 | mlx-audio | **non** : `tok_embeddings` exclu de la quantification (`VoxtralRealtimeModelLoading.swift:39`), `verify: .none` → table packée chargée dans un `Embedding` plein, sans erreur (À MESURER) | — |
| `shreyask/voxtral-mini-4b-realtime-mlx-mixed-4-6` | 2026-02-07 | Realtime · mixte 4/6 par couche, `"mode"` | 3 279 615 774 | mlx-audio 0.3.2 | **non** : per-layer ignoré (`:34-41`) | — |
| `majentik/Voxtral-4B-TTS-2603-TurboQuant-MLX-8bit` (et 2/4 bits, et variantes Mini/Realtime) | 2026-07-20 | TTS · 8 b, seulement `quantization_config` (pas de `quantization`) | 4 267 207 958 | inconnu | **non** : la quantification est lue dans `quantization` (`VoxtralTTSModelLoading.swift:95`) → poids packés chargés comme des `Linear` pleins, sans erreur (À MESURER) | **rejeté** : licence déclarée Apache-2.0 alors que la base est CC BY-NC 4.0 ; exemple d'usage `mlx_lm.load` incohérent pour un TTS |
| `jburtoft/Voxtral-Mini-3B-2507-draft-4layer` | 2026-07-31 | Mini · **drafter** 4 couches (0, 10, 20, 29) pour décodage spéculatif, format transformers | 3 794 476 992 (encodeur et tête partagés inclus) | distillation | non (pas de spéculatif dans Voxtral) | accord top-1 66,67 % ; accélération analytique 1,416× à K = 2 (L40S, anglais seulement) |

Écosystème non MLX, hors périmètre mais indicatif de l'usage : GGUF (`handy-computer/…-Realtime-2602-gguf`, 430 k
téléchargements), ONNX, ExecuTorch (Metal, XNNPACK), Core AI (`harshav/…-CoreAI`).

### 3.3 Dépôts du registre : introuvables, renommés, dérivés

- **Introuvables ou renommés** : aucun. Les 13 dépôts cités par les registres, l'enum `VoxtralPipeline.Model` et
  le Core ML existent.
- **Dérive de format** : `mistralai/Voxtral-Mini-4B-Realtime-2602` (M-01).
- **Cartes trompeuses** (dépôts de Vincent) :
  - `VincentGOURBIN/voxtral-small-8bit` est titré « voxtral-small-8bit-mixed » alors que la config est uniforme ;
  - les deux cartes Small montrent un usage `AutoModelForCausalLM` (transformers) impossible sur un pack MLX ;
  - aucune licence n'est reprise dans le corps des cartes, ni aucune parité ou SHA-256.
  - À corriger lors d'une republication (ASK).

## 4. Amont : ce que Voxtral utilise contre ce qui existe

Résolution effective (aucun `Package.resolved` suivi) :
- **mlx-swift** : `from: "0.31.6"` (`Package.swift:42`), alors que mlx-swift-lm `main` exige
  `.upToNextMinor(from: "0.31.6")` (mlx-swift-lm `Package.swift:64`). Le seul tag dans cette plage est **0.31.6**
  (`0bb916c`, 2026-07-01).
- **mlx-swift-lm** : `branch: "main"` (`Package.swift:52`) ; tête au 2026-09-22 = `ee673d6`, plus de 120 commits
  après `3.31.4`.

| Changement amont | Réf. | Dans la résolution Voxtral ? | Pertinence pour les poids et profils | Suite |
|---|---|---|---|---|
| `prepare(_:cache:state:prefill:)` + `PrefillParameters` (tranchage équilibré, ≈ 9 % à 32 k) | lm `4c7874b` (#470, 2026-08-07) | oui | Conformances adaptées dans `9392ed1`. `VoxtralForConditionalGeneration.prepare` fait un seul forward et ignore `stepSize` (MLX-006, chemin mort) ; la boucle Voxtral tranche elle-même à 512 (P-22) | rien de plus ici |
| Configuration KV typée (#453), limites effectives (#514), KV affine 8/4 bits, **TurboQuant** (#232), **variance normalisée** (#329) | lm `38927f5`, `5d8def8`, `fd0f13b`, `6745899` | oui | Voxtral construit ses caches lui-même (`KVCacheSimple`/`RotatingKVCache`). Le KV 8 bits (`QuantizedKVCache` / `AffineKVCacheConfiguration.eightBit`) est le candidat `lean` (P-16). TurboQuant et variance normalisée n'ont **aucune mesure Voxtral** : hors profils tant qu'une parité WER n'existe pas | fiche P-16 |
| `BaseConfiguration.Quantization` / `PerLayerQuantization` : décode `mode`, le per-layer, `false`, `quant_method`… | lm `BaseConfiguration.swift:22-56`, `:139-170`, `:214-217` ; tests mixtes #395 | oui | **Correctif clé en main pour M-02** (public, déjà une dépendance) | K-M02 |
| Conversion Swift `convert(modelDirectory:model:to:options:)` : modes, prédicat par couche, shards, index, `config.json` réécrit ; calibration q4_0 | lm `ModelConversion.swift:224-336` ; #507 | oui | Production de packs en Swift possible **si** les modèles Voxtral implémentent `sanitize(weights:)`. Aujourd'hui c'est l'identité par défaut (`LanguageModel.swift:49`) ; le vrai assainissement est une extension `sanitize(_:)` hors protocole (`VoxtralModelLoading.swift:420`) | À VÉRIFIER (effort M) ; sinon Python |
| Chargement parallèle des poids (≈ 1,8×), fichiers hors index, suspension coopérative | lm `e36d8ce` (#575), `d661402` (#562), `1c1b257` (#579) | oui | Seulement via `loadWeights(modelDirectory:model:…)` amont. Les 3 chargeurs Voxtral utilisent `MLX.loadArrays` en série : pas de bénéfice | option de migration, gain À MESURER |
| Vider le cache MLX au premier jeton (#620), extraction du cache de modèles (#603), boucle non bloquante (#611) | lm `c6446cf`, `ee673d6`, `9ce5b4f` | oui | Chemins `generate`/`ModelFactory` amont, non utilisés par Voxtral (MLX-005 sans objet, P-18) | — |
| Modèles audio natifs | — | — | **Aucun** (ni Voxtral, ni Whisper) ; seulement `UserInput.Audio` pour Gemma 4 | Voxtral garde ses modèles |
| Modes `affine`/`mxfp4`/`mxfp8`/`nvfp4`, `quantizedQuantizedMM` (qqmm) | mlx-swift `Ops.swift` @0.31.6 l. 1115-1123, 2483 | **oui** | Disponibles. qqmm (activations quantifiées) : aucun gain démontré sur GPU Apple | non candidat |
| `QuantizedLinear` transmet le mode à `MLX.quantized` | mlx-swift `3b11207` (#384, 2026-04-06) | oui (avant le tag) | condition pour tout pack non affine | — |
| **Échelle globale NVFP4** (`global_scale`) | mlx-swift `09051ed` (#426, 2026-07-14) | **non** | Un pack nvfp4 à échelle globale n'est pas chargeable avec 0.31.6. Preuve externe : nvfp4 sans échelle globale casse Voxtral Small | nvfp4 **exclu** des profils |
| Correctif du deadlock `CompiledFunction.lock` × `evalLock` et course sur le cache du compilateur | mlx-swift `df9ae26` (#461, 2026-08-24) | **non** | Lié à A-01 (`silu` compilé de MLXNN appelé sous vjp, inférences concurrentes) | ASK Q-M7 (attendre un tag > 0.31.6) |
| mlx cœur 0.32.2, cycle de vie des tickets de mémoire câblée, pool de `Stream` | mlx-swift `ab924c8`, `ea8a179`, `84d9eae` | **non** | contenu de mlx 0.32.2 non lu (sous-module) ; noyaux quantifiés éventuellement plus rapides : À VÉRIFIER | à relire au prochain tag |

Conséquence pour les profils : **seule la quantification affine g64 est mesurée**, dans ce dépôt comme dans le
catalogue (Y, Q). Les preuves externes jouent contre mxfp4 et nvfp4 sur Voxtral (§3.2), et l'échelle globale manque
à 0.31.6. mxfp4, mxfp8 et nvfp4 ne sont donc **pas candidats**. Il faut quand même que le chargeur les **refuse
explicitement** au lieu de les charger faux (M-02).

## 5. Constats

### M-01 · haute · `Realtime/VoxtralRealtimeModelLoading.swift:63-77`, `:98-102`, `:124`, `:55` ; `Realtime/VoxtralRealtimeConfiguration.swift:195-205`, `:208-251` ; `Realtime/VoxtralRealtimeRegistry.swift:48-56` ; `Utils/ModelDownloader.swift:707-711` — L'entrée `realtime-4b` (original Mistral) ne se charge plus depuis la dérive du dépôt

- **Constat** : `mistralai/Voxtral-Mini-4B-Realtime-2602` contient maintenant (maj. 2026-03-11) :
  - un `config.json` **transformers** : `architectures: VoxtralRealtimeForConditionalGeneration`, `audio_config`,
    `text_config`, `transformers_version: 5.2.0.dev0` ;
  - un `model.safetensors` de 8 859 446 848 o, à côté de `consolidated.safetensors` (8 859 462 744 o) et de
    `params.json`.

  Le chargeur :
  1. lit `config.json` **avant** `params.json` (`:65-68`) ;
  2. `VoxtralRealtimeConfiguration.load` essaie la forme mlx-community, qui exige `decoder` et `encoder_args`
     (`VoxtralRealtimeConfiguration.swift:111-123`), puis la forme `params.json`, qui exige `dim`, `n_layers`,
     `multimodal`… (`:256-288`). Aucune n'existe dans ce `config.json` → `DecodingError`, chargement impossible ;
  3. même avec `params.json`, les poids seraient pris dans `model.safetensors` avant `consolidated` (`:98-102`).
     Leurs clés transformers ne sont pas reconnues par `sanitizeRealtimeWeights` : la détection du format Mistral
     repose sur le préfixe `mm_streams_embeddings.` (`:124`), et `verify: .none` (`:55`) laisserait le modèle à son
     initialisation aléatoire, sans erreur ;
  4. `downloadRealtimeModel` télécharge `*.safetensors`, donc **les deux fichiers : 17,72 Go** au lieu de 8,86
     (`ModelDownloader.swift:707-711`).

  Le rapport Realtime (P-70, tableau §7) recommande justement cet original bf16 pour les profils 16 bits : c'est
  impossible en l'état.
- **Preuve** : listing et `config.json` du Hub (2026-09-27) ; lecture des deux décodeurs.
- **Correction** (additive) :
  - un champ optionnel `files: [String]?` sur `VoxtralRealtimeModelInfo`, qui restreint le téléchargement de
    l'entrée `realtime-4b` à `consolidated.safetensors`, `params.json` et `tekken.json` ;
  - dans `loadRealtimeConfig`, ne prendre `config.json` que s'il a la forme mlx-community (clé `decoder` ou
    `quantization` + `encoder_args`), sinon `params.json` ;
  - dans `loadAllRealtimeWeights`, préférer `consolidated.safetensors` quand la config vient de `params.json` ;
  - `verify: [.all]` (S-04).
  - Variante : retirer l'entrée du registre. C'est **cassant** pour tout consommateur qui passe
    `modelId: "realtime-4b"` (ASK Q-M3).
- **Risque API** : additif (correction) / cassant (retrait). **Effort** : S.
- **Statut** : **VÉRIFIÉ** (lecture du code + contenu du Hub). L'échec est déterministe (clés requises absentes) ; sa
  reproduction reste À MESURER sur Mac.
- **Fiche proposée — K-M01 « Realtime original chargeable »** :
  - **Porte** : test de décodage sur 2 fixtures (`config.json` transformers + `params.json` du dépôt) → config
    Mistral, `quantization == nil` ; téléchargement de `realtime-4b` = 8,87 Go ± 1 % ; chargement avec
    `verify: [.all]` : 0 clé manquante ou en trop ; transcription du corpus C-court identique (ou WER ≤ +0,2 pt) au
    pack `fp16`.
  - **Cible** : code et tests écrits en **cloud**, validation **macos-gpu**.

### M-02 · haute · `Utils/VoxtralStandardLoader.swift:87-114`, `:1293-1295`, `:1238-1247` ; `TTS/VoxtralTTSModelLoading.swift:58`, `:61-64`, `:91-102` ; `Realtime/VoxtralRealtimeConfiguration.swift:101-109` ; `Realtime/VoxtralRealtimeModelLoading.swift:31-42` — Mode de quantification : le STT refuse toute config qui le déclare, le TTS et le Realtime l'ignorent

- **Constat** :
  - **STT** : `quantization` est décodé en `[String: QuantizationValue]`, et `QuantizationValue` n'accepte que
    `Bool`, `Int` ou `{group_size, bits}` (`:102-113`). La valeur `"affine"` de la clé `"mode"` (une chaîne) lève
    `typeMismatch` ; le décodage de toute la config échoue (`:1295`). Or **tous les packs STT publiés en 2026
    portent cette clé** (`"mode": "affine"`) :
    - `aufklarer/Voxtral-Mini-3B-2507-MLX-8bit` ;
    - `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` ;
    - `MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder` ;
    - `MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder`.

    Seuls les packs de 2025 (mzbac, VincentGOURBIN, sans `mode`) passent. Même décodé, le mode serait écrasé par
    `.affine` (`:1245`).
  - **TTS** : `mode` est lu puis ignoré (`quantConfig.mode == "affine" ? .affine : .affine`, `:58`, déjà relevé par
    P-48). La quantification n'est cherchée que sous `quantization` (`:95`).
  - **Realtime** : `RealtimeQuantizationConfig` n'a pas de champ `mode` (`:101-109`), et `quantize` est appelé sans
    mode (`:34-41`). Un pack `mxfp4` (g32, échelles E8M0, sans biais) ou `nvfp4` serait donc chargé comme de
    l'affine, avec `verify: .none` : sortie fausse **sans erreur**.
- **Preuve** : lecture du code ; `config.json` des 4 packs cités (Hub, 2026-09-27) ; modes disponibles dans
  mlx-swift 0.31.6 (`Ops.swift:1115-1123`).
- **Correction** (additive) :
  - décoder `quantization` avec `MLXLMCommon.BaseConfiguration`, public et déjà en dépendance : `mode`, per-layer,
    `false`, `quant_method` sont gérés (`BaseConfiguration.swift:139-170` @`ee673d6`) ;
  - passer `(groupSize, bits, mode)` au filtre `quantize(model:filter:)` ;
  - **refuser explicitement** (erreur Swift) tout mode autre qu'`affine` tant qu'aucune parité n'existe, ainsi que
    toute clé `global_scale` avec mlx-swift 0.31.6 ;
  - TTS : repli sur `quantization_config` si `quantization` est absent.
- **Risque API** : additif. **Effort** : S (STT, décodeur) à M (trois chargeurs + tests).
- **Statut** : **VÉRIFIÉ** (lecture). Le chargement effectif des packs Markus et aufklarer une fois la config
  décodée est À MESURER : per-layer `language_model.embed_tokens` en plus de `embed_tokens`, clés `lm_head.*` à la
  racine (index Hub), même disposition que les packs mzbac.
- **Fiche proposée — K-M02 « Lire la quantification comme l'amont »** :
  - **Porte** : tests de décodage sur 5 fixtures `config.json` (mzbac mixte, VincentGOURBIN 8 b, aufklarer `mode`,
    Markus per-layer + `mode`, `mxfp4` synthétique → erreur explicite) : 5/5 verts ;
  - `MarkusKaemmerer/…-Mini-…-8bit-dense-encoder` se charge avec `verify: [.all]` : 0 clé manquante ou en trop ;
  - transcription greedy identique à mlx-voxtral (Python, même pack, même audio) sur C-court.
  - **Cible** : **cloud** (décodeur + tests), puis **macos-gpu** (chargement et parité).

### M-03 · moyenne · `Utils/ModelRegistry.swift:53-54`, `:62-63`, `:73`, `:83`, `:94`, `:103` ; `Realtime/VoxtralRealtimeRegistry.swift:53` ; `README.md:207-217` ; affichage `VoxtralApp/ContentView.swift:250`, `:877` — Tailles et précisions des registres fausses, et affichées à l'utilisateur

- **Constat** : l'app affiche `model.size` et `model.quantization` du registre. Écarts avec le Hub :

  | id | Déclaré | Réel | Écart |
  |---|---|---|---|
  | `mini-3b` | ~6 GB | 9,36 Go ; 18,7 Go téléchargés avec S-07 | −36 % (téléchargé : −68 %) |
  | `mini-3b-8bit` | ~3.5 GB | 5,40 Go | −35 % |
  | `mini-3b-4bit` | ~2 GB | 3,20 Go | −37 % |
  | `small-24b-8bit` | ~25 GB | 26,50 Go | −6 % |
  | `small-4bit` | ~12 GB | 14,86 Go | −19 % |
  | `realtime-4b` | ~8 GB | 17,7 Go téléchargés (M-01) | −55 % |

  Les deux originaux STT sont annoncés « float16 » alors que `torch_dtype` vaut `bfloat16`. Le README reprend les
  mêmes chiffres. A-02 notait déjà une « taille en texte libre » sans la contrôler.
- **Correction** (additive) :
  - champ `approximateBytes: Int64` (comme `YuE2Pack`), rempli depuis le Hub et affiché via `ModelDownloader.formatSize` ;
  - `size` dérivé de ce champ ; précision réelle ;
  - `docs/Weights.md` généré depuis le registre ;
  - test « registre ⊂ Weights.md, tailles ± 5 % ».
- **Risque API** : additif (`size` garde son type). **Effort** : S.
- **Statut** : **VÉRIFIÉ**.
- **Fiche — K-M03 « Registre exact + Weights.md »** :
  - **Porte** : test vert ; chaque entrée est à ± 5 % des octets Hub listés dans `Weights.md` (date + révision du
    dépôt).
  - **Cible** : **cloud**.

### M-04 · basse · `TTS/VoxtralTTSModelLoading.swift:118-145`, `:91-102`, `:160-163` — Chargeur TTS limité à trois dispositions de fichiers

- **Constat** :
  - les shards sont codés en dur (`model-00001-of-00002`, `-00002-of-00002`) et l'index est ignoré. Un pack à 1 ou
    ≥ 3 shards (par exemple un export 8 bits coupé à 2 Go) échoue en `fileNotFound` (échec franc) ;
  - un pack dont la quantification n'est que dans `quantization_config` (cas `majentik`) est chargé sans structure
    quantifiée, avec `verify: .none` (`:71`) : échec **silencieux** ;
  - la détection du format mlx-community repose sur le préfixe `language_model.model.model.` (`:162`).
- **Correction** : charger d'après `model.safetensors.index.json` (ou tous les `model*.safetensors`), lire la
  quantification comme en M-02, `verify: [.all]`.
- **Risque API** : aucun. **Effort** : S. **Statut** : **VÉRIFIÉ**.
- **Fiche** : fusionnée dans K-M02, partie TTS.
  - **Porte** : fixture « 3 shards + index » chargée ; fixture `quantization_config` seule → erreur explicite ou
    chargement correct.
  - **Cible** : **cloud**, puis **macos-gpu**.

### M-05 · basse · `Realtime/Pipeline/VoxtralRealtimePipeline.swift:78-79` ; appelants `VoxtralTranscriptionTest/VoxtralCLI.swift:727`, `ProfileCommand.swift:292` — Un id Realtime inconnu charge silencieusement le 4 bits

- **Constat** : `VoxtralRealtimeRegistry.model(withId:) ?? defaultModel`. Une faute de frappe, ou un futur id
  `realtime-4b-8bit` absent du registre installé, charge le pack 4 bits sans erreur. Une mesure de profil
  `8bit-*` serait alors attribuée au mauvais pack (piège n° 12 du catalogue : repli silencieux).
- **Correction** : lever `VoxtralRealtimeError.invalidConfiguration("unknown model id")` quand `modelId != nil` et
  introuvable ; le défaut ne s'applique qu'à `nil`.
- **Risque API** : cassant **seulement** pour un appelant qui passerait un id invalide (comportement devenu
  explicite). **Effort** : S. **Statut** : **VÉRIFIÉ**.
- **Fiche** — *Id Realtime strict*.
  - **Porte** : test « id inconnu → erreur ; nil → défaut ».
  - **Cible** : **cloud**.

### M-06 · basse · `CoreML/VoxtralCoreMLEncoder.swift:81-88`, `:530`, `:564` ; `CoreML/VoxtralHybridEncoder.swift:445-462` — Variante Core ML déduite du nom du dépôt dans l'API publique

- **Constat** : `fromMLXModelRepoId` renvoie `.small` si le nom contient « small » ou « 24b », `.mini` sinon. Le
  chemin du pipeline déduit correctement la variante de `hiddenSize` (`VoxtralHybridEncoder.swift:518-519`), mais
  les aides publiques `forMLXModel(mlxModelRepoId:)` et `fromHuggingFace(mlxModelRepoId:)` utilisent le nom. Un
  pack Small local, ou republié sous un nom sans ces sous-chaînes, recevrait l'encodeur Mini (sortie 3 072 au lieu
  de 5 120).
- **Correction** : aide additionnelle `variant(forConfigAt:)` fondée sur `text_config.hidden_size` ; dépréciation
  douce de l'aide par nom.
- **Risque API** : additif. **Effort** : S. **Statut** : **VÉRIFIÉ** (lecture) ; aucun consommateur connu de ces
  aides (FluxForge et LipDub passent par `VoxtralPipeline`, selon `audit-stabilite.md`).
- **Fiche** — *Variante Core ML par config*.
  - **Porte** : test « dossier Small nommé `x/model` → `.small` ».
  - **Cible** : **cloud**.

### M-07 · moyenne · `Utils/ModelRegistry.swift:78-106` (packs 4 bits à encodeur 6 bits) ; `Pipeline/VoxtralPipeline.swift:283-296`, `:346-360` ; `CoreML/VoxtralHybridEncoder.swift:223-257` — La précision de l'encodeur, premier levier de qualité ASR, n'est ni exposée ni mesurée ; elle dépend du backend

- **Constat** :
  - les deux packs 4 bits du registre quantifient l'encodeur audio et le projecteur en **6 bits** (configs Hub) ;
  - la seule mesure publiée qui isole ce facteur (Small, allemand conversationnel, LM 4 bits, externe) donne un CER
    de **4,28 % en 6 bits contre 2,46 % en 8 bits et 2,41 % en bf16** (×1,74) ;
  - en backend `.auto` ou `.hybrid` (défaut de `VoxtralPipeline.init`, `:196`), l'encodeur **et** le projecteur
    viennent du modèle Core ML **fp16 dense**, converti depuis l'original (`:346-360` ;
    `VoxtralHybridEncoder.swift:223-257` : sortie `[1, 375, 3072|5120]`, déjà projetée). Les bits d'encodeur du pack
    n'y jouent **aucun rôle** ;
  - ils ne comptent qu'en backend `.mlx`, ou quand Core ML échoue et que le pipeline retombe sur MLX
    (`VoxtralPipeline.swift:290-295`).

  Conséquences :
  - (a) les chiffres du README (F-11) ne disent pas le backend ;
  - (b) choisir un pack « encodeur dense » n'a d'intérêt que si `.mlx` devient le défaut (décision P-14) ;
  - (c) un profil STT doit figer le **couple (backend, pack)**.
- **Correction** :
  - campagne de mesure sur Mini : matrice encodeur {Core ML fp16, MLX 6 b, MLX 8 b, MLX bf16} × LM {4 b, 8 b},
    WER/CER sur corpus réel (ASK du rapport STT : corpus), temps et pic ;
  - puis choix des packs et, si besoin, publication (ASK Q-M2).
- **Risque API** : aucun. **Effort** : M. **Statut** : **À MESURER** (le mécanisme est VÉRIFIÉ en lecture ; l'effet
  qualité n'est prouvé qu'en externe, sur Small).
- **Fiche — K-M07 « Matrice encodeur × LM »** :
  - **Porte** : pour chaque largeur, le pack retenu a un WER ≤ celui du bf16 + 0,3 pt sur le corpus ; le temps
    d'encodage et le pic sont notés dans `BENCHMARKS.md` ; la décision (dense ou non, backend) est écrite dans
    `docs/knowledge/decisions/reference-profiles.md`.
  - **Cible** : **macos-gpu**.

## 6. Matrice candidate des profils `<bits>bit-fast|lean`

Règles du standard (`profiles-standard.md` §1) : chaque champ correspond à un **bouton existant**, et chaque valeur
est « mesurée (source) » ou « à mesurer ». Les boutons qui existent aujourd'hui sont les suivants.

| Pipeline | Boutons existants (figeables en profil v0) | Boutons à créer (profil v1, par fiche) |
|---|---|---|
| STT | poids (`VoxtralPipeline.Model`) ; `Backend` `.mlx`/`.hybrid`/`.auto` ; `MemoryOptimizationConfig` (`evalFrequency`, `clearCacheOnEval`, `resetPeakMemory`, `maxKVCacheSize`) ; `maxTokens`, `temperature`, `topP`, `repetitionPenalty` | dtype de calcul (P-01/P-05) ; `cacheLimit`/`memoryLimit` (P-09, MLX-010) ; tranche de préfill (P-22) ; KV 8 bits (P-16) ; libération de la tour audio (P-23) ; lot encodeur (P-13) |
| Realtime | poids (id du registre) ; `transcriptionDelayMs` ; `temperature` ; `maxTokens` (sémantique « trames », P-64) | dtype (P-60) ; tête liée 8 bits (P-61) ; fenêtres 750 et 8 192 (P-62/P-63) ; `cacheLimit` (P-67) ; encodeur seul (P-66) |
| TTS | poids (`VoxtralTTSModelInfo`) ; `maxFrames`, `temperature`, `cfgAlpha`, `flowSteps` (effet à vérifier, P-35), `sanitizeText`, `trimLeadIn`, `trimTail` | dtype FM/codec (P-30/P-38) ; `asyncEval` (P-31) ; codec par fenêtres (P-32/P-33) ; cache de préfixe des voix clonées (P-40) ; politique mémoire (P-42) |

Aucun décodage spéculatif n'existe dans Voxtral. Le seul drafter publié (Mini, 4 couches, externe, L40S, anglais)
est un candidat de R&D, pas un champ de profil.

### 6.1 STT — Voxtral Mini 3B 2507 (Apache-2.0)

**Poids**

| Profil | Pack chargeable aujourd'hui (repo · fichiers · octets) | Candidat après K-M02 / K-M07 | SHA-256 |
|---|---|---|---|
| `4bit-fast` / `4bit-lean` | `mzbac/voxtral-mini-3b-4bit-mixed` · `model.safetensors` · 3 195 753 212 | si le backend `.mlx` est retenu : pack « LM 4 b + `lm_head` 6 b g128 + encodeur 8 b ou bf16 » **à publier** (estimation 3,05 / 3,67 Go) | à relever |
| `8bit-fast` / `8bit-lean` | `mzbac/voxtral-mini-3b-8bit` · 2 shards · 5 404 054 476 | `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` · 2 shards · 6 017 427 099 (Apache-2.0, recette publique), épinglé par révision + SHA-256 | à relever |
| `16bit-fast` / `16bit-lean` | `mlx-community/Voxtral-Mini-3B-2507-bf16` · 2 shards · 9 356 474 312 (hors registre ; aucun `consolidated`) — ou `mistralai/…` avec exclusion S-07 | — | à relever |

**Réglages** (v = valeur proposée ; ✓ = bouton existant ; ✗ = à créer ; statut)

| Réglage | `fast` | `lean` | Statut |
|---|---|---|---|
| Backend encodeur | vainqueur de la matrice P-14/K-M07 (défaut actuel `.auto` = Core ML fp16) ✓ | `.hybrid` (Core ML ; ANE à mesurer, A-12) ✓ | à mesurer |
| Précision de calcul | bf16 ✗ (aujourd'hui **fp32** sur tout le chemin, P-01) | bf16 ; fp16 si cible iPhone (ASK du rapport STT) ✗ | à mesurer |
| KV | `KVCacheSimple` bf16, 120 Kio/position ; **`maxKVCacheSize: nil`** ✓ (tout préréglage non nul arrête le processus au-delà de la limite, P-03) | 8 bits (`QuantizedKVCache`, P-16) ✗ ; `nil` en attendant | à mesurer |
| Tranche de préfill | 512 (figée dans le code, P-22) | 256 ✗ | à mesurer |
| Limites mémoire | `cacheLimit` quelques Go ✗ (P-09) | `min(1 Go, max(256 Mo, dispo/6))` / `memoryLimit = dispo − 2 Go` ✗ (T2) | à mesurer |
| `MemoryOptimizationConfig` | `.disabled` ✓ | `evalFrequency: 8, clearCacheOnEval: false, maxKVCacheSize: nil` ✓ (les préréglages `aggressive`/`ultra` vident le cache tous les 2-4 jetons, P-08) | à mesurer |
| Résidence par étape | tout résident | Core ML ; tour audio MLX non matérialisée (chargement paresseux, P-04) ou libérée après l'encodage (P-23) ✗ | à mesurer |
| `maxTokens` | fonction de la durée (P-11 ; défaut 500 = troncature) ✓ | idem ✓ | à mesurer |
| `repetitionPenalty` | 1,0 proposé (P-12) ✓ | idem ✓ | à mesurer ; **externe** : 1,2 fait perdre 27 % des virgules sur un podcast de 10 min |
| `temperature` | 0 ✓ | 0 ✓ | valeur Mistral |

**Résidence estimée par étape** (Go ; estimation, pas une mesure)

| Pack | Encodeur audio | Projecteur | Décodeur (couches) | `embed_tokens` | `lm_head` | Total estimé | Fichier Hub |
|---|---|---|---|---|---|---|---|
| bf16 | 1,27 | 0,05 | 6,42 | 0,81 | 0,81 | 9,35 | 9,36 |
| mzbac 4 b mixte | 0,52 | 0,02 | 1,89 | 0,23 | 0,31 | 2,96 | 3,20 |
| mzbac 8 b | 0,68 | 0,03 | 3,41 | 0,43 | 0,43 | 4,97 | 5,40 |
| Markus 8 b encodeur dense | 1,27 | 0,05 | 3,41 | 0,43 | 0,43 | 5,59 | 6,02 |
| Core ML (backend `.auto`) | 1,32 (fp16, hors MLX) | inclus | — | — | — | — | 1,32 |

KV bf16 : 120 Kio par position. 10 min d'audio ≈ 7 500 jetons audio + transcription ≈ 9 500 positions ≈ 1,1 Gio
(le double aujourd'hui, en fp32).

**Valeurs connues**
- **Dépôt** (M3 Max 96 Go, conditions non documentées = « en session », F-11 du rapport STT) : fp16 / 8 b / 4 b
  mixte = 90,1 / 34,6 / 28,2 s ; 5,6 / 14,5 / 17,7 tok/s ; pic GPU 15,26 / 10,05 / 8,31 Go.
- **Dépôt** (issues) : génération 30,6 → 33,5 tok/s pour le 8 b (F-01, F-03) ; préfill tranché 512 : pic −20 % (F-05).
- **Externe** (M1 Max, Python) : table de §3.2.
- **Ce ne sont pas des références** : il n'existe aucune baseline (P-79).

### 6.2 STT — Voxtral Small 24B 2507 (Apache-2.0)

| Profil | Pack chargeable aujourd'hui | Candidat | Classe de machine (indicative) | SHA-256 |
|---|---|---|---|---|
| `4bit-fast` / `4bit-lean` | `VincentGOURBIN/voxtral-small-4bit-mixed` · 3 shards · 14 857 318 962 (encodeur 6 b) | `MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder` · 3 shards · 15 016 527 526 (après K-M02 ; externe : FLEURS DE 2,78 % contre 2,61 % en bf16, pic 19,4 Go sur 10 min) | Mac 32 Go (lean) ; issue #21 : 22 Go de pic process | à relever |
| `8bit-fast` / `8bit-lean` | `VincentGOURBIN/voxtral-small-8bit` · 26 499 134 369 **ou** `mzbac/Voxtral-Small-24B-2507-8bit` · 28 056 927 031 (S-06, ASK) | `MarkusKaemmerer/…-8bit-dense-encoder` · 27 138 066 384 (« ~34 Go nécessaires ») | Mac ≥ 48 Go | à relever |
| `16bit-fast` / `16bit-lean` | `mistralai/Voxtral-Small-24B-2507` · 11 shards · 48 527 546 144 (exclure `consolidated`, S-07) | — | Mac ≥ 64 Go ; README : pic GPU ≈ 56 Go | à relever |

- **Réglages** : ceux du Mini (§6.1). En `lean`, le KV 8 bits (P-16) et la libération de l'encodeur pèsent plus :
  KV bf16 à 160 Kio par position ≈ 1,5 Gio pour 10 min d'audio.
- **Résidence estimée** (Go) :

  | Pack | Encodeur | Projecteur | Décodeur | `embed_tokens` | `lm_head` | Total |
  |---|---|---|---|---|---|---|
  | bf16 | 1,27 | 0,10 | 44,46 | 1,34 | 1,34 | 48,52 |
  | VG 4 b mixte | 0,52 | 0,04 | 12,88 | 0,38 | 0,52 | 14,34 (fichier : 14,86) |
  | Markus 4 b encodeur dense | 1,27 | 0,10 | 12,50 | 0,38 | 0,38 | 14,64 (fichier : 15,02) |
  | 8 b uniforme | 0,68 | 0,06 | 23,62 | 0,71 | 0,71 | 25,78 |
- **Valeurs connues (dépôt, en session)** :
  - préfill 4 b 19,13 s ; décodage 11,1 tok/s (F-07) ;
  - pic process 22,0 / 20,0 Go (F-04) ;
  - le README annonce 0,5 / 0,7 / 1,0 tok/s, ce que contredit F-07 → à mesurer.
- **Externe** : §3.2.
  - Carte Markus : sur de l'audio conversationnel, le Small n'a pas été plus précis que le Mini 8 b dense sur
    quatre passages, et boucle plus volontiers.
  - Le choix entre Mini et Small est donc une question de corpus (ASK du rapport STT).

### 6.3 Realtime — Voxtral Mini 4B Realtime 2602 (Apache-2.0)

| Profil | Pack chargeable aujourd'hui | Candidat | SHA-256 |
|---|---|---|---|
| `4bit-fast` / `4bit-lean` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` · `model.safetensors` · 3 133 798 126 | variante « tête liée 8 bits » (P-61) : via `QuantizedEmbedding.asLinear` sur ce pack, ou pack à publier | à relever |
| `8bit-fast` / `8bit-lean` | **aucun** (P-70 : packs voxmlx `ellamind` 4 714 618 595 et `mlx-community …-6bit` 3 609 304 614 non chargeables) | pack 8 bits au format mlx-audio **à publier** (estimation 4,73 Go), ou assainisseur voxmlx (ASK Q4 du rapport Realtime) | — |
| `16bit-fast` / `16bit-lean` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` · 8 870 608 794 (fp16) | original bf16 `mistralai/…` · `consolidated.safetensors` 8 859 462 744, **après K-M01** | à relever |

| Réglage | `fast` | `lean` | Statut |
|---|---|---|---|
| Retard de transcription | 480 ms ✓ (recommandation Mistral : multiples de 80 ms, 80 à 2 400) | 480 ms ✓ | valeur Mistral ; WER FLEURS 8,72 % à 480 ms (carte, externe) |
| Précision de calcul | bf16 ✗ (aujourd'hui fp32, P-60) | bf16 ✗ | à mesurer |
| Tête liée | 8 bits ✗ (P-61 ; 16 bits dans le pack 4 b, recopiée en fp32 à chaque pas) | 4 bits si la parité tient ✗ | à mesurer |
| KV décodeur | `RotatingKVCache(8192)` ✗ (P-63) ; 104 Kio par position en bf16 → ≤ 0,85 Gio | idem, `kvBits` 8 à qualifier ✗ | à mesurer |
| Encodeur | fenêtre 750 par tranches ✗ (P-62) ; ≈ 256 Kio par position → ≤ 0,19 Gio | idem, « encodeur seul » pour l'extraction ✗ (P-66) | à mesurer |
| Limites mémoire | `cacheLimit` encodage 1 Go / décodage 2 Go ✗ (P-67) | adaptatives (T2) ✗ | à mesurer |
| `maxTokens` | sémantique à corriger (P-64) | idem | ASK Q3 du rapport Realtime |

- **Résidence estimée** (Go) :

  | Pack | Encodeur | Adaptateur | Décodeur | `tok_embeddings` | Total |
  |---|---|---|---|---|---|
  | bf16/fp16 | 1,94 | 0,05 | 6,06 | 0,81 | 8,86 |
  | mlx-community 4 b | 0,55 | 0,05 (16 b) | 1,71 | 0,81 (16 b) | 3,11 (fichier : 3,13) |
  | 4 b qhead | 0,55 | 0,05 | 1,71 | 0,23 | 2,53 (fichier : 2,55) |
  | 8 b (proposé) | 1,03 | 0,05 | 3,22 | 0,43 | 4,73 (ellamind : 4,71) |

  En streaming vrai (P-71), encodeur et décodeur tournent en alternance : pas de résidence par étape possible, le
  budget est la **somme**. Avec l'API actuelle, qui encode tout le fichier d'abord, l'encodeur pourrait être libéré
  avant le décodage (P-66).
- **Valeurs connues (dépôt, en session)** :
  - F-R1 (4 b) : encodage 5,44 s, préfill 448 ms, 501 pas à 33,7 ms en moyenne ;
  - pic MLX 4 619 Mo, pic process 7 949 Mo ;
  - instrument contesté (P-73).

### 6.4 TTS — Voxtral 4B TTS 2603 (**CC BY-NC 4.0**)

Le standard ne prévoit que 4, 8 et 16 bits. Le TTS existe en **4, 6 et 16 bits**, et il n'existe aucun pack 8 bits
valable (§3.2). Proposition : garder `6bit-*` comme largeur intermédiaire déclarée (écart au standard, ASK Q-M5).

| Profil | Pack | Octets | Statut |
|---|---|---|---|
| `4bit-fast` / `4bit-lean` | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` · `model.safetensors` | 2 509 879 373 | chargeable |
| `6bit-fast` / `6bit-lean` (hors standard) | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | 3 465 520 393 | chargeable |
| `8bit-*` | **aucun** ; pack « LLM + FM 8 b, codec bf16 » **à publier** (estimation 4,37 Go ; `majentik` rejeté) | — | ASK Q-M5 |
| `16bit-fast` / `16bit-lean` | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` (2 shards) ou `mistralai/…` `consolidated` | 8 004 759 170 / 8 004 752 248 | chargeable ; inutilisable en perf avant P-30 |

- **Réglages** : `temperature` 0 ✓, `cfgAlpha` 1,2 ✓, `flowSteps` 8 ✓ (effet réel à vérifier, P-35), `maxFrames`
  fonction du texte (P-41, aujourd'hui 2 500 fixe). Calcul FM et codec en bf16 ✗ (P-30/P-38) ; `asyncEval` ✗
  (P-31) ; codec par fenêtres ✗ (P-32/P-33) ; `lean` : `unload()` + `clearCache` entre deux synthèses d'une chaîne
  multi-modèles (FluxForge → LTX) ✗ (P-42). KV 104 Kio par position, ≤ 0,3 Go : pas de KV quantifié (T10 non
  prioritaire).
- **Résidence estimée** (Go) :

  | Pack | LLM (26 couches) | `embed_tokens` | FM + têtes | Codec (bf16 partout) | Total |
  |---|---|---|---|---|---|
  | bf16 | 6,05 | 0,81 | 0,79 | 0,30 | 7,95 (fichier : 8,00) |
  | 4 b | 1,70 | 0,23 | 0,22 | 0,30 | 2,45 (fichier : 2,51) |
  | 6 b | 2,46 | 0,33 | 0,32 | 0,30 | 3,41 (fichier : 3,47) |
  | 8 b | 3,22 | 0,43 | 0,42 | 0,30 | 4,37 |

  Les trois étages servent à chaque synthèse : pas de résidence par étape (T4 non applicable, rapport TTS §2).
- **Valeurs connues (dépôt, en session)** :
  - pas AR 42,6 ms (4 b) contre 240,7 ms (bf16) ; pic MLX 2,4 contre 7,7 Go (F-1, F-2) ;
  - CFG batché : 31,5 fps et TTFT ≈ 280 ms (F-6) ;
  - voix clonée en 6 b : couverture ASR 99,4 %, RTF 1,47 (F-9) ;
  - banc 2026-04-02 (F-8).
- **Externe** (carte mlx-community, mlx-audio, machine non précisée) : RTF 4 b 0,97 / 0,74, 6 b 1,15 / 1,07,
  bf16 6,50 / 6,32 (court / long).

### 6.5 Ébauche Swift (v0, boutons existants seulement)

Le gabarit `ReferenceProfiles.swift.tmpl` est pensé pour un LLM (vision, audio, spéculatif). Pour Voxtral, il faut un
type par pipeline. En v0, sans nouveau câblage :

```swift
public struct VoxtralSTTReferenceProfile: Sendable, Identifiable, Equatable {
    public enum Bits: String, CaseIterable, Sendable { case four = "4", eight = "8", sixteen = "16" }
    public enum Kind: String, CaseIterable, Sendable { case fast, lean }
    public let bits: Bits; public let kind: Kind
    public let model: VoxtralPipeline.Model          // pack (≡ repo)
    public let backend: VoxtralPipeline.Backend      // M-07 : fige le couple (backend, pack)
    public let memory: MemoryOptimizationConfig      // maxKVCacheSize: nil tant que P-03 n'est pas corrigé
    public let maxTokensPerAudioSecond: Double       // P-11, calculé par l'appelant
    public let repetitionPenalty: Float              // P-12
    public let summary: String                       // + mesure de référence en commentaire
    public var id: String { "\(bits.rawValue)bit-\(kind.rawValue)" }
}
// Realtime : model id + transcriptionDelayMs. TTS : model + flowSteps/cfgAlpha/maxFrames (+ Bits.six).
```

Les champs v1 (dtype, `cacheLimit`, KV, tranche, résidence) s'ajoutent **fiche par fiche**, une fois leur bouton
créé et mesuré.

## 7. Packs à publier (ASK — rien n'est publié ici)

| # | Pack | Contenu | Taille estimée | Recette | Prérequis | Licence |
|---|---|---|---|---|---|---|
| PK-1 | Realtime `int8-head8` (format mlx-audio) | encodeur + décodeur 8 b g64, `tok_embeddings` 8 b, adaptateur bf16 | 4,73 Go | mlx-audio `convert` (Python, Mac) | K-M02 ; décision voxmlx (ASK Q4 Realtime) | Apache-2.0 |
| PK-2 | TTS `int8` | LLM + FM 8 b g64, codec bf16 (comme les packs 4/6 b) | 4,37 Go | mlx-audio | K-M02 (partie TTS) ; ASK Q-M5 | **CC BY-NC 4.0 reprise** (dérivé non commercial, attribution) |
| PK-3 | Mini `int4-enc8-head6` (ou `-encbf16`) | LM 4 b, `lm_head` 6 b g128, encodeur + projecteur 8 b ou bf16 | 3,05 / 3,67 Go | noScribe `quantize_voxtral.py` (sans données, reproductible) ou mlx-voxtral avec prédicat | K-M07 conclut en faveur de `.mlx` | Apache-2.0 |
| PK-4 | Small 4 b et 8 b à encodeur dense | — | 15,02 / 27,14 Go | **réutiliser les dépôts Markus**, épinglés (révision + SHA-256), plutôt que republier ; sinon miroir `VincentGOURBIN` | K-M02 ; ASK Q-M2 | Apache-2.0 |

Pour tous les packs, suivre le standard (`profiles-standard.md` §1.8) :
- nom `<quant>[-head]`, avec un suffixe d'encodeur pour les modèles audio ;
- sidecar `model.safetensors.sha256` ;
- `docs/Weights.md` ;
- `ModelDownloader` qui **vérifie** le SHA-256 (A-02) ;
- carte de modèle avec licence et parité (WER contre bf16), sans exemple `transformers` erroné (§3.3).

Voxtral n'a pas d'export (§2.2). L'export Swift amont (`convert`) exige un `sanitize(weights:)` réel (À VÉRIFIER).

## 8. Fiches proposées (ordre du skill)

| Fiche | Objet | Porte chiffrée | Cible | Rang |
|---|---|---|---|---|
| K-M01 | Realtime original chargeable (M-01) | 2 fixtures de décodage vertes ; téléchargement 8,87 Go ± 1 % ; `verify: [.all]` : 0 écart ; transcription = pack fp16 (WER ≤ +0,2 pt) | cloud (code, tests) → macos-gpu | 1 (stabilité bloquante) |
| K-M02 | Quantification lue comme l'amont (M-02, M-04) | 5/5 fixtures de décodage ; pack Markus Mini 8 b chargé avec 0 clé en écart ; greedy identique à mlx-voxtral sur C-court | cloud → macos-gpu | 1 |
| K-M05 | Id Realtime strict (M-05) | test « id inconnu → erreur » | cloud | 2 (hygiène) |
| K-M06 | Variante Core ML par config (M-06) | test Small nommé sans « small » → `.small` | cloud | 2 |
| K-M03 | Registre exact + `Weights.md` (M-03) | toutes les entrées à ± 5 % des octets Hub | cloud | 2 |
| K-M07 | Matrice encodeur × LM (M-07) | pack retenu : WER ≤ bf16 + 0,3 pt ; ligne `BENCHMARKS.md` par cellule | macos-gpu | 3-4 (après baseline P-79) |
| K-M08 | Publication PK-1…PK-4 (ASK) | SHA-256 publiés ; téléchargement vérifié ; parité WER du pack ≤ bf16 + 0,3 pt (STT, Realtime) ; écoute contre bf16 + couverture ASR ≥ 99 % (TTS) | macos-gpu | 6 |
| K-M09 | Types de profils + CLI `references` / `--reference` (v0 puis v1) | 6 profils STT, 4 Realtime, 6 TTS listés ; chaque profil a une ligne mesurée dans `References.md` | cloud (type, CLI) → macos-gpu (mesures) | 5-6 |

## 9. Décisions à prendre (ASK)

1. **Q-M1 — Small 8 bits** (complète S-06) :
   - A) `VincentGOURBIN/voxtral-small-8bit` (26,50 Go, cohérent avec le README) ;
   - B) `mzbac/…` (28,06 Go ; config identique, écart non expliqué) ;
   - C) `MarkusKaemmerer/…-8bit-dense-encoder` (27,14 Go, encodeur bf16, après K-M02).
2. **Q-M2 — Packs tiers** : référencer des dépôts tiers Apache-2.0 (Markus) épinglés par révision et SHA-256, ou
   republier sous `VincentGOURBIN` (miroir) ?
3. **Q-M3 — `realtime-4b`** :
   - corriger le chargeur (additif, K-M01) ;
   - ou retirer l'entrée (**cassant** pour qui passe cet id).
4. **Q-M4 — Licence TTS** : les poids TTS sont CC BY-NC 4.0 (carte Mistral). VoxtralCore est intégré dans FluxForge
   Studio (App Store) et la chaîne LipDub/LTX. Le TTS est-il exposé dans un usage commercial ? Ce point relève d'une
   vérification juridique, **pas** d'une décision technique. Il conditionne aussi la publication de PK-2.
5. **Q-M5 — Largeurs TTS** : garder `6bit-*` comme largeur intermédiaire déclarée et publier un 8 bits (PK-2) ? Ou
   rester en 4/6/16 ?
6. **Q-M6 — Backend STT par défaut** (lié à P-14 et M-07) : si `.auto` (Core ML) reste le défaut, les packs à
   encodeur dense n'apportent rien au chemin par défaut, et PK-3 est inutile.
7. **Q-M7 — mlx-swift** : accepter 0.31.6 (sans le correctif du deadlock compile × eval, `df9ae26`) jusqu'au
   prochain tag, ou demander un tag en amont ? Un épinglage sur branche dans Voxtral entrerait en conflit avec
   `upToNextMinor` de mlx-swift-lm.
8. **Q-M8 — Modes non affines** : confirmer que mxfp4, mxfp8 et nvfp4 sont **hors profils**. Preuve externe négative
   sur Voxtral ; aucune mesure MLX Swift ; pas d'échelle globale dans 0.31.6. Le chargeur les refusera
   explicitement (K-M02).

## 10. Capitalisation proposée (phase 6)

Tout ce qui suit est vérifié en lecture. Les effets sont À MESURER, sauf mention « externe ».

- **Piège — clé `"mode"` dans `quantization`** : les convertisseurs de 2026 (mlx-lm, mlx-audio, noScribe) écrivent
  `"mode": "affine"`, et souvent un doublon `quantization_config`. Un décodeur `Codable` maison (Bool/Int/objet)
  **refuse tout le fichier**. Un chargeur qui passe `.affine` en dur charge un mxfp4 faux. Règle : décoder avec
  `MLXLMCommon.BaseConfiguration`, refuser explicitement les modes non qualifiés. Source : Voxtral
  `VoxtralStandardLoader.swift:87-114`, `TTS/VoxtralTTSModelLoading.swift:58` ; packs aufklarer et Markus.
- **Piège — dérive des dépôts officiels** : un dépôt source peut gagner après coup un `config.json` et un
  `model.safetensors` d'un autre format (transformers). Un chargeur qui sonde `config.json` d'abord casse, et un
  téléchargeur par glob double la taille. Règle : à chaque audit, relister chaque dépôt du registre et relire son
  `config.json` ; fixer la liste de fichiers par entrée. Source : `mistralai/Voxtral-Mini-4B-Realtime-2602` (M-01).
- **Technique (externe, mesurée)** — *précision de l'encodeur d'abord* : pour un LLM audio, garder l'encodeur en
  8 bits ou dense ; la tête en 6/8 bits suffit.
  - Chiffres : Voxtral Small, allemand, LM 4 b : CER 4,87 / 4,28 / 2,46 / 2,41 % pour un encodeur 4 / 6 / 8 / bf16 ;
    `lm_head` 6/8/bf16 identiques, bf16 −27 % de débit.
  - Source : `MarkusKaemmerer/Voxtral-*-dense-encoder`, noScribe `docs/voxtral-quantisation.md`.
  - Complète T13.
- **Rejet (externe)** — *mxfp4 et nvfp4 sur Voxtral* : mxfp4 fait moins bien qu'affine 4 b ; nvfp4 sans échelle
  globale casse le modèle (boucle « ist, ist, ist »). Même source.
- **Piège — backend qui masque la quantification** : quand un étage (encodeur) passe par un autre moteur (Core ML
  fp16), les bits de cet étage dans le pack MLX sont sans effet. Un profil doit figer le couple (backend, pack), et
  une mesure qui ne note pas le backend est inexploitable. Source : Voxtral `VoxtralPipeline.swift:346-360` (M-07).
- **Piège — pack tiers à provenance douteuse** : licence déclarée incompatible avec la base (Apache-2.0 sur une base
  CC BY-NC), quantification seulement dans `quantization_config`. Règle : licence, révision, SHA-256 et recette
  reproductible exigés avant qu'un pack tiers entre dans un profil. Source : `majentik/Voxtral-4B-TTS-2603-TurboQuant-MLX-8bit`.
- **Rappel (externe)** — `repetition_penalty` 1,2 en transcription verbatim : −27 % de virgules, répétitions réelles
  avalées. Recoupe P-12. Source : carte `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder`.
