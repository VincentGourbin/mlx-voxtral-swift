# Profils de référence `<bits>bit-fast|lean` — mlx-voxtral-swift (phase 3)

> Skill `mlx-swift-audit`, phase 3 (standard des profils, `references/profiles-standard.md`). Révision `9392ed1`,
> 2026-09-27. Sources : [`modeles-2026-09.md`](modeles-2026-09.md) (poids du Hub, octets exacts, matrice candidate),
> rapports perf [`audit-performance.md`](audit-performance.md) (consolidé), [`audit-annexes-serveur.md`](audit-annexes-serveur.md)
> §5 (enrôlement), [`faits-et-actions.md`](faits-et-actions.md) §2 (mesures existantes).
>
> **Règles du standard appliquées** :
> 1. **Chaque champ = un bouton qui existe déjà dans le code** (✓, avec son emplacement). Un réglage sans bouton est
>    marqué **« bouton à créer »** avec la fiche qui le crée (✗ K-n) ; il n'entre dans le type Swift qu'une fois
>    créé **et** mesuré (profil v1).
> 2. **Chaque valeur est « mesurée (source) » ou « à mesurer »**. Aucune mesure n'a été faite dans cette session
>    (cloud Linux) ; toutes les valeurs connues sont des mesures **« en session »** (révision ancienne, premier passage
>    froid, pas d'A/B/B/A) : elles orientent, elles ne sont pas des références.
> 3. Unités : `Go` = 10⁹ octets (tailles du Hub, 2026-09-27) ; `Gio`/`Kio` = puissances de 2.

## 0. Matrice complète (contrôle de la critique de complétude du 2026-09-27)

Quatre modèles × largeurs × `fast|lean`, plus l'enrôlement. Chaque case est un profil (« à mesurer » par la fiche
citée) ou une case vide **justifiée**.

| Modèle | 4 bits | 6 bits (hors standard) | 8 bits | 16 bits | Profils déclarés (K-76) |
|---|---|---|---|---|---|
| STT Mini 3B 2507 | `4bit-fast`, `4bit-lean` (K-77) | — : aucun pack MLX 6 bits sur le Hub (recherche du 2026-09-27 ; seul intermédiaire : `aufklarer/…-MLX-5bit`, non inspecté, `mode` refusé avant K-8) ; largeur absente du standard | `8bit-fast`, `8bit-lean` (K-77) | `16bit-fast`, `16bit-lean` (K-77, après K-40) | 6 |
| STT Small 24B 2507 | `4bit-fast`, `4bit-lean` (K-77) | — : aucun pack 6 bits publié ; largeur absente du standard | `8bit-fast`, `8bit-lean` (K-77, dépôt selon ASK-15) | `16bit-fast`, `16bit-lean` (K-77 ; Mac ≥ 64 Go) | 6 |
| Realtime Mini 4B 2602 | `4bit-fast`, `4bit-lean` (K-78) | — : `mlx-community/…-Realtime-6bit` chargé faux en silence (P-70), non retenu | **vide** : aucun pack chargeable (P-70) ; PK-1 ou assainisseur voxmlx selon ASK-20 → K-80 | `16bit-fast`, `16bit-lean` (K-78, fp16 puis bf16 après K-9) | 4 |
| TTS 4B 2603 | `4bit-fast`, `4bit-lean` (K-79) | `6bit-fast`, `6bit-lean` (K-79 ; seul pack livré par FluxForge ; ASK-19) | **vide** : aucun pack valable (`majentik` rejeté) ; PK-2 selon ASK-18, ASK-19 → K-80 | `16bit-fast`, `16bit-lean` (K-79, après K-39) | 6 |
| Enrôlement (sur le pack TTS) | — (la largeur est celle du pack TTS : champ « Poids ») | | | | `enroll-fast`, `enroll-lean` (K-64) : 2 |

Aucun profil `8bit-*` Realtime ou TTS n'est déclaré en v0 : il apparaît avec son pack (K-80) et sa mesure. Le champ
« spéculatif » est « — » partout (aucun drafter intégré). Chaque champ des tableaux ci-dessous est un bouton existant
(✓, emplacement) ou « bouton à créer » (✗, fiche).

## 1. État de l'art au 2026-09-27 : ce qui change le choix des poids

- **Aucun nouveau checkpoint Voxtral chez Mistral depuis mars 2026** : Mini 3B et Small 24B (2507, Apache-2.0),
  Realtime Mini 4B (2602, Apache-2.0), TTS 4B (2603, **CC BY-NC 4.0**). L'optimisation de septembre 2026 passe par
  les **packs MLX** et les **réglages**, pas par un nouveau modèle.
- **Les meilleurs packs STT de 2026 ne se chargent pas aujourd'hui** : ils portent `"mode": "affine"` que le décodeur
  STT refuse (M-02). Ce sont les packs `MarkusKaemmerer/*-dense-encoder` (encodeur en bf16, mesures externes
  publiées) et `aufklarer/…-MLX-8bit`. Après **K-8**, ils deviennent candidats.
- **Le levier de qualité ASR n°1 est la précision de l'encodeur audio** (externe, Small, LM 4 bits : CER 4,87 /
  4,28 / 2,46 / 2,41 % pour un encodeur 4 / 6 / 8 bits / bf16). Or en backend `.auto` (défaut), l'encodeur **et** le
  projecteur viennent du modèle Core ML fp16 dense : les bits d'encodeur du pack n'y jouent aucun rôle. **Un profil
  STT fige donc le couple (backend, pack)** (M-07) ; K-42 tranche le backend.
- **Largeurs chargeables aujourd'hui** : STT 4 / 8 / 16 ; Realtime 4 / 16 (8 bits : packs voxmlx non chargeables,
  P-70 ; l'original bf16 ne se charge plus, M-01 → K-9) ; TTS 4 / 6 / 16 (aucun 8 bits valable).
- **Aucun profil n'est sûr avant le lot 1** : tout préréglage mémoire actuel pose une fenêtre KV qui arrête le
  processus au-delà de 2 min 30 à 10 min 30 d'audio (S-02/P-03) ; les profils v0 imposent `maxKVCacheSize: nil`.
- **Modes non affines** (mxfp4, mxfp8, nvfp4) : hors profils (preuve externe négative sur Voxtral, pas d'échelle
  globale NVFP4 dans mlx-swift 0.31.6) ; refusés explicitement après K-8 (ASK-21).

## 2. STT — Voxtral Mini 3B 2507 (Apache-2.0)

### 2.1 Poids recommandés

| Largeur | Chargeable aujourd'hui (dépôt · fichiers · octets) | Recommandé à septembre 2026 | SHA-256 |
|---|---|---|---|
| 4 bits | `mzbac/voxtral-mini-3b-4bit-mixed` · `model.safetensors` · 3 195 753 212 (LM 4 b, MLP extrêmes 6 b, **encodeur + projecteur 6 b**, `lm_head` 6 b g128) | celui-ci en `.auto` ; si K-42 retient `.mlx` : pack « LM 4 b + `lm_head` 6 b + encodeur 8 b ou bf16 » **à publier** (PK-3, estimation 3,05 / 3,67 Go, ASK-22) | à relever (K-82) |
| 8 bits | `mzbac/voxtral-mini-3b-8bit` · 2 shards · 5 404 054 476 (8 b uniforme, encodeur compris) — **défaut du registre** | après K-8, si `.mlx` : `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` · 2 shards · 6 017 427 099 (externe : WER 4,27 % contre 4,74 % en 8 b uniforme sur un passage difficile ; 6,60× temps réel ; M1 Max, Python), épinglé par révision + SHA-256 (ASK-16) | à relever |
| 16 bits | `mistralai/Voxtral-Mini-3B-2507` · shards · 9 356 474 312 (bf16, pas « float16 » ; **+ consolidated 9 348 806 528 téléchargé en double** jusqu'à K-24) | le même après K-24, ou `mlx-community/Voxtral-Mini-3B-2507-bf16` (mêmes shards, sans `consolidated`, à ajouter au registre) ; inutilisable en perf avant K-40 (copie fp32 de chaque poids, P-05) | à relever |

### 2.2 Réglages figés (boutons)

| Réglage | Bouton | `fast` | `lean` | Statut |
|---|---|---|---|---|
| Poids | ✓ `VoxtralPipeline.Model` (`Pipeline/VoxtralPipeline.swift:27-66`) | pack de la largeur (§2.1) | idem | — |
| Backend encodeur | ✓ `VoxtralPipeline.Backend` `.mlx`/`.hybrid`/`.auto` (`:73-85`) | vainqueur de la matrice K-42 (défaut actuel `.auto` = Core ML fp16) | `.hybrid` (tour MLX jamais matérialisée) ou `.mlx` + libération (K-63) | à mesurer (K-42) |
| Fenêtre KV | ✓ `MemoryOptimizationConfig.maxKVCacheSize` | **`nil`** (obligatoire ; tout non-nil arrête le processus avant K-2/K-3) | **`nil`** | décision (S-02/P-03) |
| Rythme `eval`/`clearCache`/pic | ✓ `MemoryOptimizationConfig.evalFrequency`, `clearCacheOnEval`, `resetPeakMemory` | `.disabled` (0 / false / false) | `evalFrequency 8`, `clearCacheOnEval false`, `resetPeakMemory false` | à mesurer (K-51 retire ces opérations de la boucle) |
| `maxTokens` | ✓ `Configuration.maxTokens` (`:94-129`) | ⌈durée × taux⌉ + marge (500 par défaut = troncature > ≈ 3 min, P-11 ; K-5) | idem | taux à mesurer (K-5) |
| Pénalité de répétition | ✓ `Configuration.repetitionPenalty` | 1,0 proposé (1,2 aujourd'hui) | idem | à mesurer (K-61) ; externe : 1,2 fait perdre 27 % des virgules sur 10 min |
| Température / top-p | ✓ `Configuration.temperature`, `topP` | 0 / 0,95 (inutilisé à T = 0) | idem | valeur Mistral |
| Langue | ✓ paramètre `language` de `transcribe` | explicite quand connue ; `nil` (auto) seulement après validation K-33 | idem | à mesurer (K-33) |
| Dtype de calcul | ✗ **bouton à créer** (K-40) | bf16 (aujourd'hui fp32 partout, P-01) | bf16 ; fp16 si iPhone (ASK-2) | à mesurer |
| `cacheLimit` / `memoryLimit` | ✗ **bouton à créer** (K-52, opt-in) | quelques Go / `nil` | `min(1 Go, max(256 Mo, dispo/6))` / `dispo − 2 Go` (T2) | à mesurer |
| KV quantifié | ✗ **bouton à créer** (K-55) | `nil` (bf16, 120 Kio/position) | 8 bits (`QuantizedKVCache` g64) | à mesurer |
| Tranche de préfill | ✗ **bouton à créer** (K-62 ; 512 codé en dur) | 512 | 256 | à mesurer (balayage) |
| Lot d'encodeur | ✗ **bouton à créer** (K-56) | K à balayer | K petit | à mesurer |
| Encodeur dé-quantifié | ✗ **bouton à créer** (K-60) | oui si ≥ 5 % | non | à mesurer |
| Libération de la tour audio | ✗ **bouton à créer** (K-63) | non | oui (`.mlx`) | à mesurer |
| Spéculatif | — (aucun drafter intégré ; drafter externe `jburtoft/…-draft-4layer`, L40S, anglais, hors plan) | — | — | — |
| Compile | — : aucun bouton ; seules les activations de MLXNN sont compilées par l'amont (T18) | — | — | — |

### 2.3 Résidence estimée par étape (Go ; estimation depuis `config.json`, pas une mesure)

| Pack | Encodeur | Projecteur | Décodeur | `embed_tokens` | `lm_head` | Total | Fichier Hub |
|---|---|---|---|---|---|---|---|
| bf16 | 1,27 | 0,05 | 6,42 | 0,81 | 0,81 | 9,35 | 9,36 |
| mzbac 4 b mixte | 0,52 | 0,02 | 1,89 | 0,23 | 0,31 | 2,96 | 3,20 |
| mzbac 8 b | 0,68 | 0,03 | 3,41 | 0,43 | 0,43 | 4,97 | 5,40 |
| Markus 8 b encodeur dense | 1,27 | 0,05 | 3,41 | 0,43 | 0,43 | 5,59 | 6,02 |
| Core ML (`.auto`) | 1,32 (fp16, hors MLX, projecteur inclus) | — | — | — | — | — | 1,32 |

KV bf16 : 120 Kio par position ; 10 min d'audio ≈ 9 500 positions ≈ 1,1 Gio (le double aujourd'hui, en fp32).

### 2.4 Matrice des 6 profils Mini

| Id | Poids | Valeurs connues (toutes « en session ») | Mesure de référence |
|---|---|---|---|
| `4bit-fast` | mzbac 4 b mixte | README `e376f05` (M3 Max 96 Go, hybride, 500 jetons ≈ 8,5 min) : 28,2 s, 17,7 tok/s (jetons / temps total), pic GPU 8,31 Go | à mesurer (K-77) |
| `4bit-lean` | mzbac 4 b mixte | — | à mesurer (K-77) |
| `8bit-fast` | mzbac 8 b (ou Markus dense si `.mlx`) | README : 34,6 s, 14,5 tok/s, 10,05 Go ; issues #13-#18 (Core ML) : préfill 3,59 s, décodage 30,6 → 33,5 tok/s (`c1942ee`), pic MLX 6 116 → 4 878 Mo (`1eb2cc9`), process 11,1 Go | à mesurer (K-77) |
| `8bit-lean` | idem | — | à mesurer (K-77) |
| `16bit-fast` | mistralai shards / mlx-community bf16 | README (« fp16 ») : 90,1 s, 5,6 tok/s, 15,26 Go — avec la copie fp32 de P-05 | à mesurer (K-77, après K-40) |
| `16bit-lean` | idem | — | à mesurer (K-77) |

## 3. STT — Voxtral Small 24B 2507 (Apache-2.0)

| Largeur | Chargeable aujourd'hui | Recommandé à septembre 2026 | Classe de machine (indicative) | SHA-256 |
|---|---|---|---|---|
| 4 bits | `VincentGOURBIN/voxtral-small-4bit-mixed` · 3 shards · 14 857 318 962 (encodeur 6 b) | après K-8 : `MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder` · 15 016 527 526 (externe : FLEURS DE WER 2,78 % contre 2,61 % en bf16 ; pic 19,4 Go sur 10 min ; ≈ 2× temps réel, M1 Max) si `.mlx` retenu | Mac 32 Go en lean, **à vérifier** (K-34 : pic ≤ 24 Go ?) | à relever |
| 8 bits | `VincentGOURBIN/voxtral-small-8bit` · 26 499 134 369 **ou** `mzbac/Voxtral-Small-24B-2507-8bit` · 28 056 927 031 (deux dépôts pour un modèle, S-06) | ASK-15 : A) VincentGOURBIN, B) mzbac, C) `MarkusKaemmerer/…-8bit-dense-encoder` · 27 138 066 384 (« ~34 Go nécessaires ») | Mac ≥ 48 Go | à relever |
| 16 bits | `mistralai/Voxtral-Small-24B-2507` · 11 shards · 48 527 546 144 (+ consolidated 48 519 877 672 en double jusqu'à K-24) | le même après K-24 et K-40 | Mac ≥ 64 Go (README : pic GPU ≈ 56 Go) | à relever |

- **Réglages** : ceux du Mini (§2.2) ; en `lean`, KV 8 bits (K-55) et libération de l'encodeur (K-63) pèsent plus
  (KV bf16 160 Kio par position ≈ 1,5 Gio pour 10 min).
- **Résidence estimée** (Go) : bf16 48,52 · VG 4 b mixte 14,34 (fichier 14,86) · Markus 4 b dense 14,64 (15,02) ·
  8 b uniforme 25,78.
- **Valeurs connues (en session)** : small-4bit préfill 19,13 s à « 49 % GPU » (instrument contesté, FA-08), décodage
  11,1 tok/s (#19/#20) ; pic MLX 15 575 → 14 364 Mo (`1eb2cc9`) ; pic process 22,0 Go (#21) ; README : chat 0,54 /
  0,74 / 1,00 tok/s (contredit par 11,1 tok/s en STT et 11,5 tok/s en chat après `41ce59d`) → **à mesurer** (K-34, K-77).
| Id | Poids | Réglages propres | Mesure de référence |
|---|---|---|---|
| `4bit-fast` | VincentGOURBIN 4 b mixte (Markus 4 b dense si K-42 retient `.mlx`) | ceux de `4bit-fast` Mini | à mesurer (K-77) |
| `4bit-lean` | idem | KV 8 bits (K-55), libération de l'encodeur (K-63) ; porte : pic ≤ 24 Go à 10 min d'audio (ASK-7) | à mesurer (K-77) |
| `8bit-fast` | dépôt retenu par ASK-15 | ceux de `8bit-fast` Mini | à mesurer (K-77) |
| `8bit-lean` | idem | KV 8 bits, libération de l'encodeur | à mesurer (K-77) |
| `16bit-fast` | `mistralai/Voxtral-Small-24B-2507` (shards seuls après K-24) | ceux de `16bit-fast` Mini ; après K-40 | à mesurer (K-77) |
| `16bit-lean` | idem | KV 8 bits, libération de l'encodeur | à mesurer (K-77) |

- Externe (carte Markus) : sur audio conversationnel, le Small n'a pas été plus précis que le Mini 8 b dense et
  boucle plus volontiers ⇒ le choix Mini/Small est une question de corpus (ASK-14).

## 4. Realtime — Voxtral Mini 4B Realtime 2602 (Apache-2.0)

| Largeur | Chargeable aujourd'hui | Recommandé à septembre 2026 | SHA-256 |
|---|---|---|---|
| 4 bits | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` · 3 133 798 126 (tête liée et adaptateur **non quantifiés**) — défaut | le même ; tête quantifiée **au chargement** (K-46) plutôt que le pack tiers `T0mSIlver/…-4bit-qhead` (2 554 984 165, non chargeable : `tok_embeddings` exclu de la quantification) | à relever |
| 8 bits | **aucun** (packs voxmlx `ellamind/…-8bit-mlx` 4 714 618 595 et `mlx-community/…-Realtime-6bit` 3 609 304 614 chargés **faux en silence**, P-70) | pack 8 bits au format mlx-audio **à publier** (PK-1, estimation 4,73 Go) ou assainisseur voxmlx (ASK-20) | — |
| 16 bits | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` · 8 870 608 794 (fp16 ≠ bf16 d'entraînement) | **original bf16** `mistralai/Voxtral-Mini-4B-Realtime-2602` `consolidated.safetensors` 8 859 462 744, **après K-9** (inchargeable aujourd'hui, M-01) | à relever |

| Réglage | Bouton | `fast` | `lean` | Statut |
|---|---|---|---|---|
| Poids | ✓ id du registre (`VoxtralRealtimeRegistry.swift:28-57`) ; id strict après K-9 | §4 | idem | — |
| Retard de transcription | ✓ `Configuration.transcriptionDelayMs` (`VoxtralRealtimePipeline.swift:22-36`) | 480 ms | 480 ms | valeur Mistral ; externe : WER FLEURS 8,72 % à 480 ms (carte) |
| Température | ✓ `Configuration.temperature` | 0 | 0 | — |
| `maxTokens` | ✓ `Configuration.maxTokens` (compte des **trames** aujourd'hui, P-64) | borné par l'audio (K-5) | idem | ASK-8 |
| Dtype de calcul | ✗ **bouton à créer** (K-38) | bf16 (aujourd'hui fp32) | bf16 | à mesurer |
| Tête liée quantifiée | ✗ **bouton à créer** (K-46) | 8 bits | 4 bits si la parité tient (WER ≤ +0,5 pt) | à mesurer |
| Fenêtres encodeur 750 / décodeur 8 192 | ✗ correctif (K-13), pas un réglage | toujours | toujours | — |
| `cacheLimit` par étape | ✗ **bouton à créer** (K-52) | encodage ≈ 1 Go, décodage ≈ 2 Go (valeurs de départ Y/Q) | adaptatives (T2), `clearCache` après transcription | à mesurer |
| Encodeur seul (extraction) | ✗ **bouton à créer** (K-59, `.encoderOnly`) | — | oui | à mesurer |
| Encodeur dé-quantifié | ✗ **bouton à créer** (K-60) | oui si ≥ 5 % | non | à mesurer |
| KV quantifié | — : non retenu (T10, faible priorité une fois la fenêtre 8 192 de K-13 posée) | — | — | — |
| Tranche de préfill | — : N/A (préfill ≤ 32 jetons, T9) | — | — | — |
| Compile | — : aucun bouton ; RoPE par le noyau fusionné (K-65), sans `compile` | — | — | — |

- **Résidence estimée** (Go) : bf16/fp16 8,86 · 4 b 3,11 (fichier 3,13 ; tête 16 bits 0,81) · 4 b tête quantifiée
  2,53 · 8 b 4,73. En streaming vrai (hors plan, ASK-3), encodeur et décodeur alternent : le budget est la somme.
- **Valeurs connues (en session, instrument contesté P-73)** : realtime-4b-4bit, encodage 5,44 s, préfill 448 ms,
  501 pas à 33,7 ms (run probablement tronqué), pic MLX 4 619 Mo, process 7 949 Mo (#23-#25). Mesure de référence :
  **à mesurer** (K-36, K-78).

| Id | Poids | Mesure de référence |
|---|---|---|
| `4bit-fast` / `4bit-lean` | mlx-community 4 b (+ tête 8 / 4 bits au chargement) | à mesurer (K-78) |
| `8bit-fast` / `8bit-lean` | non disponible (ASK-20, K-80) | — |
| `16bit-fast` / `16bit-lean` | mistralai bf16 (après K-9) ; fp16 en attendant | à mesurer (K-78), **après K-38** (sinon chaque `Linear` recopie ses poids en fp32) |

## 5. TTS — Voxtral 4B TTS 2603 (**CC BY-NC 4.0**)

**Écart au standard** : le standard ne prévoit que 4 / 8 / 16 bits ; le TTS n'existe qu'en **4 / 6 / 16 bits** et
aucun pack 8 bits valable n'existe (`majentik/…-TurboQuant-MLX-8bit` rejeté : licence Apache-2.0 déclarée sur une
base CC BY-NC, quantification seulement dans `quantization_config`). Proposition : `6bit-*` déclaré comme largeur
intermédiaire (ASK-19) ; `8bit-*` « non disponible » tant que PK-2 n'est pas publié. **Licence** : l'usage dans une
app commerciale (FluxForge Studio, App Store) relève d'une vérification juridique (ASK-18).

| Largeur | Pack | Octets | Statut | Recommandé pour |
|---|---|---|---|---|
| 4 bits | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` · `model.safetensors` | 2 509 879 373 | chargeable (codec bf16, 0 `.scales` sur ses 116 tenseurs) | anglais, textes courts ; rate l'EOA sur le FR long (1 prise, sans graine) |
| 6 bits (hors standard) | `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` | 3 465 520 393 | chargeable ; **seul pack livré par FluxForge** | voix clonées (couverture 99,4 % contre 96,5 % bf16, n = 15, une voix) ; FR long sans `maxFrames` |
| 8 bits | aucun ; « LLM + FM 8 b, codec bf16 » **à publier** (PK-2, estimation 4,37 Go) | — | non disponible (ASK-19) | — |
| 16 bits | `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` (2 shards) ou `mistralai/Voxtral-4B-TTS-2603` `consolidated` | 8 004 759 170 / 8 004 752 248 | chargeable ; **défaut du registre** (décision ASK-5) ; inutilisable en perf avant K-39 | référence de qualité |

| Réglage | Bouton | `fast` | `lean` | Statut |
|---|---|---|---|---|
| Poids | ✓ `VoxtralTTSModelInfo` / `loadModel(modelInfo:)` (manager : K-76) | §5 | idem | ASK-5 (défaut) |
| Plafond de frames | ✓ `Configuration.maxFrames` (2 500) ; proportionnel au texte après K-14 | min(2 500, a + b × jetons) | idem | a, b à mesurer (K-14) |
| `temperature` | ✓ `Configuration.temperature` (jamais lu aujourd'hui, P-35) | 0 | 0 | K-48 : ≠ 0 ⇒ erreur |
| `cfgAlpha` | ✓ `Configuration.cfgAlpha` (jamais lu, P-35) | 1,2 | 1,2 | effectif après K-48 |
| Pas de flow matching | ✓ `Configuration.flowSteps` (jamais lu, P-35) | 8 (défaut Mistral) ; < 8 seulement si K-48 le retient | idem | à mesurer (K-48, écoute ASK-13) |
| Assainissement / coupes | ✓ `sanitizeText`, `trimLeadIn`, `trimTail` | true / true / false (true pour LipDub) | idem | — |
| Graine | ✓ paramètre `seed:` (batch et streaming) | fixée pour toute mesure | idem | — |
| Warm-up (voix clonées) | ✓ `warmUpText:` (`recommendedWarmUpVocalise`) | oui pour les voix clonées | idem | coût : K-68 |
| Amorce gardée après le porteur | ✓ `warmUpLeadInFrames:` (`VoxtralTTSPipeline.swift:307`, `:478`, `:504` ; 0 = coupe serrée recommandée par la doc du code) | 0 | 0 | — |
| Taille des chunks (streaming) | ✓ `chunkSize:` de `synthesizeStreaming` (10 trames = 800 ms, `VoxtralTTSPipeline.swift:475`, `:500`) | 10 | 10 | TTFA et coût du re-décodage (P-33) : à mesurer (K-35, K-43) |
| Compile du codec | ✗ **bouton à créer** (K-71 : coupe-circuit, désactivé sous gradient) | selon K-71 (retiré si < 5 %) | non | à mesurer |
| KV quantifié / tranche de préfill | — : N/A (KV ≤ 0,3 Go ; préfill court, pas de `lm_head`) | — | — | — |
| Cache de préfixe | ✓ voix prédéfinies, streaming avec `voiceKey` ; ✗ **bouton à créer** pour le batch cloné (K-50) | LRU 2-4 entrées | 1 entrée | à mesurer |
| Dtype FM / codec | ✗ **bouton à créer** (K-39 / K-58) | bf16 | bf16 | à mesurer (parité forcée, écoute) |
| Politique mémoire | ✗ **bouton à créer** (K-52, opt-in) | — | `clearCache` après décodage et au `unload()`, `cacheLimit` adaptatif | à mesurer |

- **Résidence estimée** (Go) : bf16 7,95 · 4 b 2,45 · 6 b 3,41 · 8 b 4,37 (LLM, `embed_tokens`, FM, codec bf16 0,30) ;
  pas de résidence par étape (les 3 étages servent à chaque synthèse). KV 104 Kio par position, ≤ 0,3 Go : pas de KV
  quantifié.
- **Valeurs connues (en session)** : banc `6ad4e56` (M3 Max 96 Go, 2026-04-02 ; RTF = génération / audio) : court EN
  4 b RTF 1,17 · 6 b 1,88 · bf16 6,86 ; long EN 4 b 120,61 s pour 181,28 s d'audio (0,67) · 6 b 0,93 · bf16 4,88 ;
  long FR 4 b et bf16 atteignent `maxFrames`. `0be05af` : 4 b 31,5 fps, TTFT-frame ≈ 280 ms. Campagne `e83778a` : q6
  RTF 1,47 contre bf16 3,44 (Debug probable, P-76). Externe (carte mlx-community) : RTF 4 b 0,97/0,74, 6 b 1,15/1,07,
  bf16 6,50/6,32.

| Id | Poids | Mesure de référence |
|---|---|---|
| `4bit-fast` / `4bit-lean` | mlx-community 4 b | à mesurer (K-79) |
| `6bit-fast` / `6bit-lean` | mlx-community 6 b | à mesurer (K-79) |
| `8bit-fast` / `8bit-lean` | non disponible (PK-2, ASK-19) | — |
| `16bit-fast` / `16bit-lean` | mlx-community bf16 | à mesurer (K-79, après K-39) |

## 6. Enrôlement de voix — `enroll-fast|lean` (transposition de `lora-fast|lean`)

| Champ | Bouton | `enroll-fast` | `enroll-lean` | Statut |
|---|---|---|---|---|
| Poids | ✓ pack TTS de la synthèse visée (codec et table audio non quantifiés dans les packs 4/6 b : même embedding, FV-40) | bf16 (défaut CLI `tts-4b-mlx`) | 6 bits | à mesurer (K-37) |
| Durée de référence | ✓ `Config.numFrames` (100 = 8 s par défaut ; CLI 16 s ; démo 16 s) | 200 trames (16 s) | 200 | **mesurée** : similarité ECAPA 0,67 / 0,69 / 0,72 / 0,72 pour 4 / 8 / 16 / 24 s (2 000 époques, `docs/voice_cloning.md:36-45`, en session) ; défaut 8 → 16 s = ASK-9 |
| Époques | ✓ `Config.epochs` | 5 000 | 5 000 (3 000 à mesurer) | « ≈ 30 min » annoncé, non mesuré ; 1 500 époques en 1 min 46 s (`bd59931`, en session) |
| Préparation de la référence | ✓ `referenceHighPassHz` 70 (50 pour voix graves, ACT-15), `gateReference`, `gateAttenuationDB` −24, `referenceTargetRMSdB` −20 | défauts | défauts | mesurés (`f63e2a8`, `1c7b57e`, `4fb44b7`) |
| Hyperparamètres d'optimisation | ✓ `Config.learningRate` 0,1, `reconstructionWeight` 0,5, `perceptualWeight` 1,0, `melWeight` 1,0, `temperature` 2,0 / `temperatureDecay` 0,99 / `minTemperature` 0,3, `gradClip` 1,0 (`VoxtralVoiceEnrollment.swift:26-40`) | défauts figés | défauts figés | alignés sur la référence Python (commentaires du code) ; pondérations seulement explorées par le harnais `TTSReEnrollExperimentTests` (variables `VOXTRAL_ENROLL_*`, en session) |
| Graine | ✗ **bouton à créer** (K-26) | fixée | fixée | — |
| Point de contrôle | ✗ **bouton à créer** (K-26) | toutes les 500 époques | toutes les 250 | — |
| Résidence | ✗ **bouton à créer** (K-64, seulement si K-37 montre ≥ 5 %) | modèle entier (paresseux) | codec + table audio | à mesurer |
| `cacheLimit` / `clearCache` final | ✗ **bouton à créer** (K-64) | défaut / oui | borné / oui | à mesurer |
| Porte qualité | — | similarité ≥ 0,70 | similarité à ±0,01 de `fast`, pic ≤ 60 % de `fast`, temps ±5 % | à mesurer |

## 7. Type Swift proposé (esquisse, **pas** dans `Sources/`)

Le gabarit `templates/ReferenceProfiles.swift.tmpl` suppose un seul LLM (vision, audio, spéculatif). Voxtral a quatre
pipelines aux boutons disjoints : **un type par pipeline**, même forme (`Bits`, `Kind`, `id`, `all`, `named`),
**v0 = boutons existants seulement** (K-76). Les champs v1 (dtype, `cacheLimit`, KV 8 bits, tranche, résidence) sont
ajoutés fiche par fiche, une fois leur bouton créé **et** mesuré.

```swift
// Esquisse v0 (boutons existants). Fichier cible : Sources/VoxtralCore/Configuration/ReferenceProfiles.swift (K-76).
import Foundation

public struct VoxtralSTTReferenceProfile: Sendable, Identifiable {
    public enum Bits: String, CaseIterable, Sendable { case four = "4", eight = "8", sixteen = "16" }
    /// `fast` : tout résident, rythme mémoire désactivé. `lean` : rythme d'éviction modéré, backend hybride.
    public enum Kind: String, CaseIterable, Sendable { case fast, lean }
    public enum Family: String, Sendable { case mini, small }

    public let family: Family
    public let bits: Bits
    public let kind: Kind
    // — Poids + backend (figés ensemble : M-07, le backend `.auto` masque la précision de l'encodeur du pack) —
    public let model: VoxtralPipeline.Model
    public let backend: VoxtralPipeline.Backend
    // — Génération (boutons de VoxtralPipeline.Configuration) —
    /// Budget de jetons par seconde d'audio ; l'appelant calcule `maxTokens` (K-5 le rendra automatique).
    public let tokensPerAudioSecond: Double
    public let repetitionPenalty: Float          // 1,2 aujourd'hui ; décision K-61
    // — Mémoire (bouton MemoryOptimizationConfig) : maxKVCacheSize toujours nil (S-02/P-03) —
    public let memory: MemoryOptimizationConfig
    public let summary: String                   // + mesure de référence en commentaire de chaque entrée (K-77)

    public var id: String { "\(bits.rawValue)bit-\(kind.rawValue)" }

    public func pipelineConfiguration(audioSeconds: Double) -> VoxtralPipeline.Configuration {
        VoxtralPipeline.Configuration(
            maxTokens: Int((audioSeconds * tokensPerAudioSecond).rounded(.up)) + 64,
            temperature: 0, topP: 0.95, repetitionPenalty: repetitionPenalty,
            memoryOptimization: memory)
    }

    public static func named(_ id: String, family: Family) -> Self? {
        all.first { $0.id == id && $0.family == family }
    }

    /// Six entrées Mini + six Small ; chaque entrée porte en commentaire sa mesure (temps, pic, WER, machine, date).
    public static let all: [Self] = [ /* K-76, valeurs de profils.md §2-§3 */ ]
}

// Mêmes formes, boutons propres :
// VoxtralRealtimeReferenceProfile : Bits (4, 16 ; 8 si pack) ; modelId ; transcriptionDelayMs ; maxTokens (après K-5).
// VoxtralTTSReferenceProfile      : Bits (4, 6, 16 — six déclaré hors standard, ASK-19) ; VoxtralTTSModelInfo ;
//                                   VoxtralTTSPipeline.Configuration (maxFrames, cfgAlpha, flowSteps effectifs après K-48) ;
//                                   warm-up pour voix clonées.
// VoxtralEnrollmentReferenceProfile : Kind (fast, lean) ; modèle TTS ; VoxtralVoiceEnrollment.Config (numFrames, epochs…).
//
// Pas de applyGlobalPolicy() en v0 : aucun réglage process-wide n'existe encore. En v1 (K-52), il sera OPT-IN et
// restaurera la valeur précédente à unload() : une bibliothèque embarquée (FluxForge) ne doit pas imposer
// Memory.cacheLimit à son hôte (P-42, MLX-010).
```

## 8. Brouillon de `docs/References.md` (d'après `templates/References.md.tmpl`)

> Le brouillon est en anglais, comme la documentation du dépôt ; il est recopié par K-81 (squelette sourcé) puis
> rempli par K-82 (mesures). Toutes les cellules chiffrées sont « to measure » tant que K-77…K-79 n'ont pas conclu.

````markdown
# The reference configurations

`voxtral references` lists them, `voxtral <transcribe|realtime|tts|enroll|bench> --reference <id>` applies one,
`Voxtral{STT,Realtime,TTS,Enrollment}ReferenceProfile.all` exposes them to an app. Each one pins every setting that
matters (weights, encoder backend, compute precision, KV cache, token budget, memory limits, residency).
Same seed + same profile = same output, comparable time on comparable hardware.

Source: `Sources/VoxtralCore/Configuration/ReferenceProfiles.swift`. Measurements: [Benchmarks.md](Benchmarks.md)
and `BENCHMARKS.md`. Weights: [Weights.md](Weights.md).

## Speech-to-text (Voxtral Mini 3B 2507, Small 24B 2507)

Workload: `fluxforge_long_{en,fr}_6bit.wav` (167 s / 174 s) and a 11 min 22 s concatenation, <machine>, idle GPU,
two passes after a 120 s cool-down, Release binary, mlx-swift 0.31.6, mlx-swift-lm main@<sha>.

| Id | Weights (HF) | Encoder backend | Compute | Prefill tok/s | Decode tok/s | TTFT | Peak process | WER EN/FR | Who it is for |
|---|---|---|---|---|---|---|---|---|---|
| mini `4bit-fast` | mzbac/voxtral-mini-3b-4bit-mixed (3.20 GB) | to decide (K-42) | bf16 | to measure | to measure | to measure | to measure | to measure | 16 GB Macs |
| mini `4bit-lean` | same | hybrid | bf16 | to measure | to measure | to measure | to measure | to measure | 8 GB Macs, long audio |
| mini `8bit-fast` | mzbac/voxtral-mini-3b-8bit (5.40 GB) | to decide | bf16 | to measure | to measure | to measure | to measure | to measure | default |
| mini `8bit-lean` | same | hybrid | bf16 | to measure | to measure | to measure | to measure | to measure | |
| mini `16bit-fast` | mistralai/Voxtral-Mini-3B-2507 (9.36 GB) | to decide | bf16 | to measure | to measure | to measure | to measure | to measure | quality reference |
| mini `16bit-lean` | same | hybrid | bf16 | to measure | to measure | to measure | to measure | to measure | |
| small `4bit-fast` | VincentGOURBIN/voxtral-small-4bit-mixed (14.86 GB) | to decide | bf16 | to measure | to measure | to measure | to measure | to measure | 32 GB+ (to verify) |
| small `4bit-lean` | same | hybrid | bf16 | to measure | to measure | to measure | to measure | to measure | |
| small `8bit-fast` | to decide (ASK-15) | to decide | bf16 | to measure | to measure | to measure | to measure | to measure | 48 GB+ |
| small `8bit-lean` | same | hybrid | bf16 | to measure | to measure | to measure | to measure | to measure | |
| small `16bit-fast` | mistralai/Voxtral-Small-24B-2507 (48.5 GB) | to decide | bf16 | to measure | to measure | to measure | to measure | to measure | 64 GB+ |
| small `16bit-lean` | same | hybrid | bf16 | to measure | to measure | to measure | to measure | to measure | |

## Realtime (Voxtral Mini 4B Realtime 2602)

| Id | Weights (HF) | Compute | Tied head | ms/step p50 / p90 (budget 80) | First text token | Peak process | WER EN/FR | Who it is for |
|---|---|---|---|---|---|---|---|---|
| `4bit-fast` | mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit (3.13 GB) | bf16 | 8-bit | to measure | to measure | to measure | to measure | default |
| `4bit-lean` | same | bf16 | 4-bit (if parity holds) | to measure | to measure | to measure | to measure | small Macs |
| `8bit-fast` / `8bit-lean` | not available yet | — | — | — | — | — | — | — |
| `16bit-fast` | mistralai/Voxtral-Mini-4B-Realtime-2602 (8.86 GB, bf16) | bf16 | 16-bit | to measure | to measure | to measure | to measure | quality reference |
| `16bit-lean` | same | bf16 | 8-bit | to measure | to measure | to measure | to measure | |

## Text-to-speech (Voxtral 4B TTS 2603, CC BY-NC 4.0)

| Id | Weights (HF) | fps | TTFA (streaming) | RTF (gen / audio) | Peak process | ASR coverage | Who it is for |
|---|---|---|---|---|---|---|---|
| `4bit-fast` / `4bit-lean` | mlx-community/Voxtral-4B-TTS-2603-mlx-4bit (2.51 GB) | to measure | to measure | to measure | to measure | to measure | English, short texts |
| `6bit-fast` / `6bit-lean` | mlx-community/Voxtral-4B-TTS-2603-mlx-6bit (3.47 GB) | to measure | to measure | to measure | to measure | to measure | cloned voices, French |
| `8bit-*` | not available | — | — | — | — | — | — |
| `16bit-fast` / `16bit-lean` | mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 (8.00 GB) | to measure | to measure | to measure | to measure | to measure | quality reference |

## Voice enrollment

| Id | Weights | s/epoch | Peak process | ECAPA similarity | Who it is for |
|---|---|---|---|---|---|
| `enroll-fast` | TTS pack of the target synthesis | to measure | to measure | to measure (0.72 at 16 s, 2,000 epochs, in session) | 32 GB+ |
| `enroll-lean` | 6-bit TTS pack | to measure | to measure | to measure | 16 GB Macs |

## What each setting does

| Setting | Values | Effect (measured) |
|---|---|---|
| Encoder backend | `.mlx` / `.hybrid` / `.auto` | to measure (K-42); `.auto` ignores the pack's encoder precision |
| `maxKVCacheSize` | always `nil` | any window crashed the process beyond 2.5-10.5 min of audio before 2.3 |
| Token budget | tokens per audio second | to measure (K-5) |
| Repetition penalty | 1.0 / 1.2 | to measure (K-61) |
| Flow-matching steps (TTS) | 8 / 6 / 5 / 4 | to measure (K-48) |

## Choosing

- **Mac, 32 GB and up**: to write after measurement (K-77…K-79).
- **Mac, 16-24 GB**: to write after measurement.
- **8 GB Mac, iPhone**: pending the iOS decision.
- **Even faster, memory no object**: to write after measurement.

## Adding or changing a profile

1. Add the entry to `.all` (every field is an existing knob, nothing new to wire).
2. Measure with `voxtral bench <pipeline> --reference <id> --passes 2 --warmup 1 --cooldown 120` (A/B/B/A, protocol in Benchmarks.md).
3. Quality gate: WER (STT, Realtime) or ASR coverage + blind listening (TTS) against the `16bit-fast` profile, same seed.
4. Add the row to `BENCHMARKS.md` and the decision to `docs/knowledge/decisions/reference-profiles.md`.

## Command-line equivalents

`--reference 8bit-fast` (mini) = `-m mini-3b-8bit -b <backend> --max-tokens <ceil(duration × rate) + 64>` (+ memory preset `.disabled`).
````

## 9. Écarts au standard (proposés au skill)

1. **Un type par pipeline** (STT, Realtime, TTS, enrôlement) au lieu d'un profil unique « LLM multimodal ».
2. **Largeur intermédiaire déclarée** (`6bit-*` pour le TTS, et le Realtime 6 bits voxmlx) quand le 8 bits n'existe pas.
3. **Backend dans le profil** : pour un pipeline hybride (Core ML + MLX), le couple (backend, pack) est un seul
   choix ; une mesure qui ne note pas le backend est inexploitable.
4. **Nom de pack audio** : `<quant>[-head][-enc<bits>]` (ex. `int4-enc8-head6`), la précision de l'encodeur étant le
   premier levier de qualité ASR.
5. **Réglages process-wide opt-in** et restaurés (bibliothèque embarquée) : `applyGlobalPolicy()` du gabarit doit
   sauver la valeur précédente de `Memory.cacheLimit` et la rendre à `unload()`.
6. **Pas de `fatalError("à implémenter")`** dans le gabarit : il contredit le constat S-28 (API publique qui plante).
