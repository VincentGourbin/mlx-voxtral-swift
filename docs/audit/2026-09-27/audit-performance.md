# Audit « Performance » consolidé — mlx-voxtral-swift (STT, TTS, Realtime, instruments)

> Skill `mlx-swift-audit`, phase 2, rapport **consolidé** attendu par le skill (`audit-performance.md`).
> Révision auditée : `9392ed1` (= tag `v2.2.2`, branche `claude/action-plan-skills-beta-wifgmu`). Date : 2026-09-27.
> Ce rapport ne refait aucune vérification : il **agrège** les trois rapports détaillés, vérifiés et relus en
> vérification croisée, et les relie aux fiches du plan.
>
> | Rapport détaillé | Constats | Après vérification croisée |
> |---|---|---|
> | [`audit-performance-stt.md`](audit-performance-stt.md) | P-01…P-29 (STT, chat) | 28 gardés (15 amendés), 1 écarté (P-25) |
> | [`audit-performance-tts.md`](audit-performance-tts.md) | P-30…P-49 (TTS 4B) | 20 gardés (15 amendés) |
> | [`audit-performance-realtime-instruments.md`](audit-performance-realtime-instruments.md) | P-60…P-79 (Realtime, instruments) | 20 gardés (10 amendés) |
>
> Constats perf portés par d'autres rapports : A-06, A-10, A-12 ([`audit-annexes-serveur.md`](audit-annexes-serveur.md)),
> M-07 ([`modeles-2026-09.md`](modeles-2026-09.md)), MLX-002/003/004/010/016…020
> ([`patterns-verdicts.md`](patterns-verdicts.md)). Plan : [`PLAN.md`](PLAN.md) ; fiches : [`fiches/`](fiches/).

## 0. Cadre

- **Aucune mesure dans cette session** (Linux, sans Mac, sans toolchain Swift, sans GPU). Tout gain est **attendu**,
  jamais obtenu. Les chiffres cités viennent du dépôt (README, docs, issues, commits) : ce sont des mesures
  **« en session »** au sens de `references/measurement.md` (révision ancienne, premier passage froid, pas
  d'A/B/B/A, deux conventions de RTF, TTFT ≠ TTFA), **aucune n'est une référence** (faits-et-actions.md §2.1, §2.8).
- **Amont relu dans la source réellement résolue** : mlx-swift `0.31.6` (= `0bb916c`, MLX C++ `ce45c52`),
  mlx-swift-lm `main@ee673d6`, référence mlx-audio `main` pour le Realtime et le TTS.
- **Règle transverse découverte** : la fuite fp32 est la cause racine commune des trois chemins (STT P-01/P-05,
  TTS P-30/P-38, Realtime P-60/P-61). Sur MLX, un seul opérande fp32 (features mel, table RoPE, embedding temporel,
  scalaire `MLXArray(Float(x))`) promeut tout le graphe ; `matmul` recopie alors **chaque poids** en fp32 à chaque
  appel (`ops.cpp:3069-3082`), et le cache KV est alloué en fp32. Le détecteur MLX-002 ne voit que les constantes
  littérales : la preuve de référence est un **audit de dtype** (cache KV et logits après préfill), prévu dans
  l'instrument K-32 (`VOXTRAL_DTYPE_AUDIT=1`).

## 1. Synthèse

| Chemin | Hauts | Moyens | Bas | Levier principal attendu | Fiche |
|---|---|---|---|---|---|
| STT (P-01…P-29) | 4 (P-01, P-02, P-03, P-05) | 16 | 8 | P-03 : arrêt du processus au-delà de la fenêtre KV (stabilité) ; P-01/P-05 : calcul fp32 de bout en bout | K-2, K-40 |
| TTS (P-30…P-49) | 3 (P-30, P-31, P-32) | 8 | 9 | P-30 : FM fp32 (bf16 ÷ 2 à 2,8 attendu) ; P-32 : attention du codec T×T (≈ 2 × 10,5 Go à 2 266 frames) | K-39, K-41 |
| Realtime + instruments (P-60…P-79) | 5 (P-60, P-61, P-62, P-73, P-79) | 9 | 6 | P-60/P-61 : tête liée recopiée en fp32 à chaque pas (≈ ×2,2 à ×3 attendu) ; P-79 : aucun instrument de baseline | K-38, K-32 |
| **Total** | **12** | **33** | **23** | 68 constats gardés, 1 écarté | 82 fiches |

**Ordre de lecture recommandé** : P-79 (sans instrument, rien ne conclut) → P-03 (arrêt de l'hôte) → P-60/P-61,
P-30, P-01/P-05 (fuites fp32) → P-32 (mémoire du codec) → le reste par gain attendu (PLAN.md §3, lot 4).

## 2. Catalogue T1…T23 × chemin

Légende : **appliquée** · **partielle** · **absente** · **N/A** (non applicable) ; entre parenthèses le constat, puis
la fiche. La colonne « Enrôlement » (entraînement par vjp) vient de `audit-annexes-serveur.md`.
Critique de complétude du 2026-09-27 : chaque cellule porte désormais un statut explicite (les 14 cellules « — »
de la colonne Enrôlement ont été tranchées à la lecture de `VoxtralVoiceEnrollment.swift` à `9392ed1`) ; les chemins
secondaires (chat, TTS en streaming, encodeur hybride Core ML) sont traités au §2.0 bis.

| T | Technique | STT | TTS | Realtime | Enrôlement |
|---|---|---|---|---|---|
| T1 | `Memory.cacheLimit` par étape | **absente** : 0 pose dans `VoxtralCore`, seule pose `0` puis `Int.max` dans une fonction morte de l'app (P-09) → K-52 | **absente** : 0 dans `TTS/` (P-42) → K-52 | **absente** ; substitut `clearCache` tous les 256 pas (P-67) → K-52 | **absente** (A-06) → K-64 |
| T2 | Limites adaptatives | **absente** ; les préréglages par RAM règlent eval/clearCache/KV, jamais `cacheLimit`/`memoryLimit` (P-08, P-09) → K-51, K-52 | **absente** (P-42) → K-52 | **absente** (P-67) → K-52 | **absente** (A-06) → K-64 |
| T3 | `clearCache` après réponse / entre étapes | **appliquée mais mal placée** : aussi **dans** la boucle tous les 2-4 jetons sur ≤ 31 Go (P-08) → K-51 | **absente** ; `unload()` sans `clearCache` (P-42) → K-52 | **partielle** : périodique seulement, rien en fin ni au `unload()` (P-67) → K-52 | **absente** (A-06) → K-64 |
| T4 | Résidence des poids par étape | **partielle, implicite** : paresseux en hybride ; tour audio MLX résidente pendant tout le décodage en `.mlx` (P-23, P-24, P-04) → K-63, K-59 | **N/A** en synthèse répétée (les 3 étages servent à chaque synthèse) | **absente**, masquée par le chargement paresseux ; pas d'« encodeur seul » (P-66) → K-59 | **absente** (probable ; effet à mesurer : LLM probablement jamais matérialisé en CLI, A-06 amendé) → K-37, K-64 |
| T5 | Variante sans tour | **N/A** (audio obligatoire ; l'hybride = « sans tour MLX », P-14) | **appliquée de fait** (pas de `lm_head`, encodeur de codec jamais instancié) | **absente** : `extractAudioEmbeddings` exige le modèle entier (P-66) → K-59 | **absente** : `enrollVoice` exige le pipeline TTS entier chargé (`VoxtralTTSPipeline.swift:433-436`) alors que la boucle ne traverse que le codec et la table audio (`VoxtralVoiceEnrollment.swift:447`, `:683`) (A-06) → K-64, si K-37 montre ≥ 5 % |
| T6 | Réutilisation du préfixe KV | **N/A** en transcription ; **absente** en chat (P-15) → K-49 | **partielle** : préréglages et streaming avec `voiceKey` ; absente pour les voix clonées, ZeroVoice, mélanges (P-40) → K-50 | **N/A** (un passage par fichier) ; l'équivalent est le streaming (P-71, hors plan) | **N/A** : ni LLM ni cache KV dans la boucle d'optimisation (`VoxtralVoiceEnrollment.swift:539-615`) |
| T7 | Médias nouveaux seulement | **absente** en chat (P-15) → K-49 | N/A | N/A | **N/A** (idem T6) |
| T8 | Budget de jetons média | **N/A** : 375 jetons / 30 s fixés par le modèle ; ne pas couper la dernière fenêtre | N/A (longueur de voix fixée par le preset ou l'enrôlement) | **appliquée (analogue)** : retard exposé (`transcriptionDelayMs`, `--delay`) | **appliquée (analogue)** : durée de référence exposée (`Config.numFrames`, 100 trames = 8 s ; CLI et démo 16 s) ; défaut → ASK-9 |
| T9 | Tranche de préfill, dernier logit | **partielle** : 512 codé en dur ×2, jamais balayé, logits de toutes les positions (P-06, P-22) → K-54, K-62 | N/A (pas de `lm_head`, préfill court) | dernière position **appliquée** ; encodeur en un seul graphe (P-62) → K-13 | **N/A** (aucun préfill) |
| T10 | KV quantifié (lean) | **absente** (P-16) → K-55 | **absente**, non prioritaire (≤ 0,3 Go) | **absente**, faible priorité une fois la fenêtre 8 192 posée | **N/A** (aucun cache KV) |
| T11 | KV préalloué, écriture en place | **appliquée par l'amont** (`KVCacheSimple`) mais recopie complète à chaque tranche de 512 (P-10) → K-53 | **appliquée (amont)** | **appliquée (amont)**, tampons fp32 (P-60) → K-38 | **N/A** (aucun cache KV) |
| T12 | Tête quantifiée via `quantizedMatmul` | **appliquée** dans les packs (8 b ou 6 b g128) ; en bf16 la tête est recopiée en fp32 (P-05) → K-40 | **appliquée** (4 / 6 bits) | **absente et pire** : tête liée 16 bits par `matmul` brut, recopiée en fp32 à chaque pas (P-61) → K-38, K-46 | **N/A** : aucune tête de sortie, les logits sémantiques sont les paramètres appris (`:516`) |
| T13 | Quantification mixte par voie | **appliquée** via les packs 4 bits mixtes ; encodeur en 6 bits = premier levier de qualité (M-07) → K-42 | **absente** : LLM + FM uniformes (P-48) → K-80 | **partielle** : seuls 4 bits et fp16 chargeables (P-70) → K-78, K-80 | **N/A** : codec et table audio, seuls étages traversés, ne sont quantifiés dans aucun pack (FV-40) |
| T14 | Dé-quantifier une étape bornée par le calcul | **absente** (encodeur audio, P-27) → K-60 | **N/A** (codec non quantifié ; #29 réfuté) | **absente** (P-69) → K-60 | **N/A** (même raison que T13) |
| T15 | Pipelining `asyncEval` | **absente** (P-07) → K-45 | **absente** : `MLX.eval(xt)` à chaque frame (P-31) → K-44 | **absente** : deux synchronisations par pas (P-65) → K-47 | **partielle** : deux synchronisations par époque (`.item()` `:566`, `eval` `:601`), gain attendu < 1 % (A-10) → passager de K-64 |
| T16 | `eval` par couche / graphe borné | **partielle** : non nécessaire au décodage ; encodeur en un lot non borné (P-13) ; chargement sans `eval` (P-04) → K-56, K-59 | **partielle** : codec d'un seul graphe sur toute la séquence (P-32, P-33) → K-41, K-43 | **partielle** : décodage OK ; encodeur d'un seul graphe (P-62) → K-13 | **appliquée** : un `eval` par époque (`:601`) |
| T17 | Fuites de dtype fp32 | **défaut majeur** : tout le chemin en fp32 (P-01, P-02, P-05) → K-3, K-40 | **partielle** : LLM bf16 ; FM, tête sémantique et codec en fp32 (P-30, P-38) → K-39, K-58 | **présente, 3 sources** : mel, tables RoPE, `tCond` (P-60) → K-38 | **appliquée** : fp32 voulu (pertes, SLERP) puis recasté ; le codec calcule en fp32 comme en synthèse (P-38) → K-58 |
| T18 | `compile(shapeless:)` d'activation | **appliquée par l'amont** (`silu`/`gelu` de MLXNN compilés) ; rien d'autre (R1-R3) | **absente** ; candidats seulement dans le codec, risque ABBA (P-49, A-01) → K-71 | **absente** ; RoPE fusionnée sans `compile` (P-68) → K-65 | `silu` compilé traversé par le gradient : **ABBA** (A-01) → K-11 |
| T19 | Cache KV entre étapes | **N/A** | N/A (le FM ne lit que l'état caché) | **N/A** | **N/A** |
| T20 | Politique de calcul par étape | **absente** (P-27) → K-60 | **partielle** de fait, annulée par les promotions fp32 (P-30, P-38) → K-39, K-58 | **absente** (P-61, P-69) → K-46, K-60 | **N/A** (un seul étage calculé sous gradient) |
| T21 | Réduction du nombre de pas | **N/A** ; levier voisin fonctionnel `maxTokens` (P-11) → K-5 | **absente, non exposée** : `flowSteps`/`cfgAlpha` jamais lus (P-35) → K-48 | **N/A** tel quel ; candidat « pas spéculatifs de remplissage » (P-72) → K-73 | **absente** : 5 000 époques figées quel que soit le point d'entrée (A-11) ; « 3 000 en `enroll-lean` » à mesurer (profils.md §6) → K-64 |
| T22 | Lecture `pread` / `F_NOCACHE` | **absente** ; lecture au premier préfill (P-04) → K-59 | **absente**, faible applicabilité | **N/A** (priorité basse : un fichier de 3,1 Go ; d'abord P-66 → K-59) | **absente**, faible applicabilité (chargement du pack TTS, comme la colonne TTS) |
| T23 | Reprise / porte GPU iOS | **absente** ; iOS 17 déclaré, jamais exécuté (FA-09, ASK-2) | **absente** | absente ; pic 7 949 Mo incompatible avec jetsam (ASK-2) | **absente** : pas de point de contrôle (A-07) → K-26 |

### 2.0 bis Chemins secondaires (ajout de la critique de complétude du 2026-09-27)

Le tableau §2 regroupe le chat sous STT, le streaming sous TTS et l'encodeur hybride sous STT. Ici, chaque technique
dont le statut **diffère** du chemin parent ; toutes les autres ont le statut de la colonne parente, pour la raison
de code indiquée (même fonction appelée).

| Chemin | Même code que le parent | Techniques au statut propre |
|---|---|---|
| **Chat** (`VoxtralPipeline.chat`, `Pipeline/VoxtralPipeline.swift:393-473`) | génération par `generateStreamWithAudioEmbeds` / `generateStream` comme `transcribe` (`:444`, `:456` contre `:353`, `:366`) ; même `optimizeIfNeeded(tokenIndex: 0)` en sortie (`:408` contre `:332`) ⇒ T1-T5, T8-T20, T22-T23 = colonne STT | **T6, T7 absentes** : chaque question refait extraction, encodage et préfill (P-15) → K-49 ; échantillonnage : top-p approché (P-26, hors catalogue) → K-70 ; T21 N/A ; **baseline** : ligne chat de K-34 (ajoutée le 2026-09-27) |
| **TTS en streaming** (`synthesizeStreaming`, `TTS/Pipeline/VoxtralTTSPipeline.swift:472-671` ; production `VoxtralTTSModeling.swift:569-685`) | mêmes LLM, FM et codec que le batch ⇒ T1-T5, T9-T14, T17-T23 = colonne TTS | **T6 partielle** : préfixe mis en cache seulement avec `voiceKey` (P-40) → K-50 ; **T15 absente** dans la même boucle que le batch (P-31) → K-44 ; **T16 pire que le batch** : le codec re-décode tout l'accumulé à chaque chunk (P-33, `VoxtralTTSPipeline.swift:583`) → K-43 ; production synchrone dans la closure (S-08, MLX-019) → K-12 ; bouton `chunkSize` (10 trames) figé par le profil (profils.md §5) |
| **Encodeur hybride Core ML** (`.hybrid` / `.auto`, `CoreML/VoxtralHybridEncoder.swift:63`) | décodeur LM identique au `.mlx` ⇒ T6-T11, T15, T18-T19, T21 = colonne STT | **T4 appliquée de fait** : tour audio MLX jamais matérialisée (P-23, P-24) ; **T12/T13 sans effet sur l'encodeur** : les bits d'encodeur du pack sont ignorés, Core ML fp16 dense (M-07) → K-42 ; **T14 N/A** (Core ML) ; **T17 présente** : sortie Core ML Float32 non castée (P-01) → K-40 ; **T20 à mesurer** : unités `.cpuAndGPU`, ANE jamais mesuré (P-14, A-12) → K-42 ; **T22 analogue** : compilation Core ML à froid 1 min 09 à 2 min 25 (issues #16, #22, en session) → K-42, K-25 |

### 2.1 Rejets du catalogue confrontés au dépôt

| Rejet | Verdict sur Voxtral |
|---|---|
| R1-R3 (compiler le pas AR / FM) | Ne pas re-proposer ; P-49 se limite à des chaînes élémentaires du codec, derrière un coupe-circuit et hors gradient. |
| R12 (`MLX_MAX_OPS_PER_BUFFER`), R16 (`iogpu.wired_limit_mb`) | Non re-proposés. |
| R13 (limites mobiles fixes) | **Rencontré** : les préréglages `aggressive`/`ultra` vident le cache tous les 2-4 jetons (P-08) → K-51. |
| R14 (Core ML > 2× MLX ⇒ pas d'asset) | Sert de **règle de décision** pour l'hybride Core ML (P-14, A-12) : Core ML reste le défaut seulement s'il est ≥ 1,2× plus rapide et WER ≤ +0,3 pt → K-42. |
| R15 (CFG batché rejeté chez Y, CFG inactif) | **Contre-exemple mesuré ici** : CFG en lot de 2 retenu (`0be05af`, TTFT 414 → ≈ 280 ms, 27,7 → 31,5 fps, en session). À capitaliser comme nuance de R15. |

### 2.2 Techniques et rejets propres à Voxtral (capitalisation)

Le détail est dans `faits-et-actions.md` §5 (V-T1…V-T16 retenues, V-R1…V-R12 rejetées, V-P1…V-P16 pièges) et
dans les sections « capitalisation » des trois rapports détaillés. Les plus transposables :
- **V-T1** une synchronisation par frame dans une boucle AR à sous-pas (CFG en lot, un `eval` par intégration) ;
- **V-T2** cache KV du préfixe de conditionnement cloné, jamais prêté ;
- **V-T9** graine sur tous les chemins stochastiques : porte de parité TTS = bit-identique à graine fixée ;
- **piège** tête liée par `matmul(h, emb.weight.T)` en activations fp32 (P-61) ;
- **piège** fenêtre glissante déclarée par la config mais ignorée (P-62, P-63) ;
- **piège** masque causal maison + `RotatingKVCache` ⇒ arrêt du processus (P-03) ;
- **piège d'instrument** phases imbriquées et « Device Utilization % » instantané (P-73) ;
- **rejet** « codec 4 bits 3,4× plus lent » (#29, première mesure froide).

## 3. Constats P-xx (renvoi aux rapports détaillés)

Sévérités **après** vérification croisée. Statut : V = VÉRIFIÉ (mécanisme lu), M = À MESURER (effet).

### 3.1 STT et chat — [`audit-performance-stt.md`](audit-performance-stt.md)

| Id | Sév. | Constat (une ligne) | Statut | Fiche |
|---|---|---|---|---|
| P-01 | haute | Tout le chemin calcule en fp32 : features mel non castées, sortie Core ML Float32, fusion `where` qui promeut ; cache KV fp32 (240 Kio/jeton Mini au lieu de 120) | V ; gain M | K-40 |
| P-02 | haute | Masque additif fp32 `[T, offset+T]` construit sur CPU à chaque tranche : verrouille le fp32 (le SDPA lève si q bf16) et ne suit pas les caches amont | V | K-3 |
| P-03 | haute | `RotatingKVCache` par défaut + préfill tranché + masque maison ⇒ **arrêt du processus** au-delà de la fenêtre (audio > 2 min 30 sous 16 Go, > 5 min de 16 à 31 Go, > 8 min, > 10 min 30 à partir de 64 Go et dans l'app) ; = S-02 | V (simulation exacte) ; reproduction M | K-2 |
| P-04 | moyenne | Poids jamais matérialisés au chargement : le 1er préfill paie la lecture disque ; mesures #13/#17/#19/#21 contaminées | V | K-59 |
| P-05 | haute | Modèles bf16 : chaque `Linear` convertit son poids en fp32 à chaque appel (≈ 5× le trafic, copie de 1,61 Go du `lm_head` Mini) ; `dtype:` ignoré ; « fp16 » = bf16 | V ; gain M | K-40 |
| P-06 | moyenne | Le préfill calcule et force les logits de toutes les positions (≈ 11 % du calcul du Mini) | V ; gain M | K-54 |
| P-07 | moyenne | Aucun `asyncEval` : `eval` bloquant par tranche, `.item()` par pas | V ; gain M | K-45 |
| P-08 | moyenne | `clearCache` tous les 2-4 jetons sur ≤ 31 Go ; `resetPeakMemory` dans la boucle ; configuration globale | V ; gain M | K-51 |
| P-09 | moyenne | `Memory.cacheLimit` jamais posé ; la seule pose (`Int.max`) est fausse et morte | V ; gain M | K-52 |
| P-10 | moyenne | Cache KV réalloué et recopié à chaque tranche de préfill (longueur finale pourtant connue) | V ; gain M (pic à 30 min) | K-53 |
| P-11 | moyenne | `maxTokens = 500` : transcriptions > ≈ 3 min coupées sans signal ; la trace de référence était tronquée | V | K-5 |
| P-12 | moyenne | Pénalité de répétition 1,2 en greedy de transcription (≈ 120 nœuds par pas ; risque WER) | V ; WER M | K-61 |
| P-13 | moyenne | Encodeur MLX : toutes les fenêtres en un seul lot (pic ∝ durée) | V ; gain M | K-56 |
| P-14 | moyenne | Hybride Core ML par défaut, `.cpuAndGPU` (pas d'ANE), jamais comparé à un encodeur MLX bf16 ; compilation à froid 1 min 09 à 2 min 25 | V ; gains M | K-42 |
| P-15 | moyenne | Chat : chaque question refait extraction, encodage et préfill (3 à 6 s par question suivante) | V ; gain M | K-49 |
| P-16 | moyenne | Pas de cache KV quantifié (lean, Small, audio long) | V ; gain M | K-55 |
| P-17 | moyenne | Décodeur hérité public : masque `[T, T]` ⇒ arrêt dès la 2ᵉ tranche | V ; arrêt M | K-3 |
| P-18 | moyenne | Deux boucles de génération maison au lieu du `TokenIterator` amont ; `prepare` non conforme | V | K-75 |
| P-19 | moyenne | Aucune baseline exploitable (un passage froid, backend implicite, chiffres contradictoires) | V | K-32, K-34 |
| P-20 | moyenne | Aucun profil STT de référence | V | K-76, K-77, K-81, K-82 |
| P-21 | basse | Mel calculé deux fois par fenêtre, une synchronisation par fenêtre | V | K-69 |
| P-22 | basse | Tranche de préfill figée à 512, dupliquée, jamais balayée | V | K-62 |
| P-23 | basse | Tour audio résidente pendant tout le décodage en `.mlx` (0,52 à 1,27 Go) | V ; gain M | K-63 |
| P-24 | basse | Modules factices aléatoires (≈ 2,6 Go chacun s'ils sont évalués) : piège pour toute passe globale | V | K-59 |
| P-25 | — | **Écarté** à la vérification croisée (masque concaténé jamais atteint ; détail repris dans P-07) | — | — |
| P-26 | basse | Top-p approché : un filtre top-k = 1 000, jamais un nucleus (chat à T > 0 seulement) | V | K-70 |
| P-27 | basse | Encodeur audio en `quantizedMatmul` alors que borné par le calcul (T14/T20) | M | K-60 |
| P-28 | basse | Étapes strictement séquentielles (encodage complet avant le 1er jeton) | M | K-72 |
| P-29 | basse | Fichier audio décodé en entier au format natif (≈ 691 Mo à 30 min stéréo 48 kHz) | V | K-69 |

### 3.2 TTS — [`audit-performance-tts.md`](audit-performance-tts.md)

| Id | Sév. | Constat (une ligne) | Statut | Fiche |
|---|---|---|---|---|
| P-30 | haute | FM et tête sémantique en fp32 : copie fp32 de chaque poids bf16 à chaque appel (≈ 26 Go/frame) ; ≈ 340 `astype` par frame en 4/6 bits | V ; gain M | K-39 |
| P-31 | haute | Boucle AR synchrone (`MLX.eval(xt)` par frame ; streaming `.item()` + `eval` par frame), aucun `asyncEval` | V ; gain M | K-44 |
| P-32 | haute | Attention du codec (fenêtre ≤ 16) calculée en T×T fp32 : ≈ 2 × 10,5 Go de transitoire à 2 266 frames | V ; pic M | K-41 |
| P-33 | moyenne | Le streaming re-décode tout l'accumulé à chaque chunk (≈ 33× le travail du codec) | V | K-43 |
| P-34 | basse | Défaut bf16 (le plus lent) dans le registre, la CLI, `profile` ; le manager ne permet aucun choix | V | K-76, K-79 |
| P-35 | moyenne | T21 non exposé : `flowSteps`, `cfgAlpha`, `temperature` jamais lus | V | K-48 (doc : K-18) |
| P-36 | moyenne | Invariants du FM recalculés à chaque pas ; masque sémantique construit sur CPU par frame | V | K-57 |
| P-37 | moyenne | Attention du FM non fusionnée (≈ 105 dispatches par frame) | V | K-57 |
| P-38 | basse | Codec entièrement en fp32 (padding fp32 et `MLXArray(scale)` re-promeuvent) | V ; gain M | K-58 (doc : K-18) |
| P-39 | basse | Weight norm, centroïdes, masques recalculés à chaque décodage | V | K-58 |
| P-40 | moyenne | Pas de cache de préfixe pour les voix clonées, ZeroVoice, mélanges (chemin LipDub) | V | K-50 |
| P-41 | moyenne | `maxFrames` fixe 2 500 : un EOA manqué coûte jusqu'à 200 s d'audio (16 min en bf16, mesuré) | V | K-14 |
| P-42 | moyenne | Aucune politique mémoire TTS ; `unload()` sans `clearCache` (FluxForge contourne) | V ; effet M | K-52 |
| P-43 | basse | Poids paresseux : la 1re synthèse paie la lecture disque (coût déplacé, pas supprimé) | V ; coût M | K-59 |
| P-44 | basse | Deux passes LLM avant le 1er frame | V | K-66 |
| P-45 | moyenne | Instrument TTS incomplet (prédéfini batch seulement, sans graine, pas de TTFA) | V | K-32, K-35 |
| P-46 | basse | Un `.item()` par frame dans les coupes (≈ 100 synchronisations) | V | K-67 |
| P-47 | basse | Warm-up : frames du porteur générées puis jetées ; le streaming attend 43 frames | V ; coût M | K-68 |
| P-48 | basse | Mode de quantification ignoré (`.affine` dans les deux branches) ; packs uniformes | V | K-8 (mode), K-80 (packs) |
| P-49 | basse | Compile : candidats seulement dans le codec, interdits sous gradient (ABBA) | M | K-71 |

### 3.3 Realtime et instruments — [`audit-performance-realtime-instruments.md`](audit-performance-realtime-instruments.md)

| Id | Sév. | Constat (une ligne) | Statut | Fiche |
|---|---|---|---|---|
| P-60 | haute | Tout le chemin Realtime calcule en fp32 (mel, tables RoPE, `tCond`) ; KV fp32 | V ; gain M | K-38 |
| P-61 | haute | Tête liée : `matmul(h, W.T)` recopie la table 131 072 × 3 072 en fp32 à chaque pas (1,5 Gio) ; tête jamais quantifiée | V ; gain M | K-38, K-46 |
| P-62 | haute | Encodeur : fenêtre glissante 750 ignorée ⇒ divergence au-delà de 15 s, coût ×1,8 à 5 min, ×12 à 1 h (calcul, attendu) | V ; WER M | K-13 |
| P-63 | moyenne | Décodeur : fenêtre 8 192 ignorée, cache KV sans borne | V | K-13 |
| P-64 | moyenne | `maxTokens` compte des trames : troncature silencieuse à ≈ 5 min 27 s (≈ 39 s sous `profile`) | V | K-5 |
| P-65 | moyenne | Deux synchronisations par pas, aucun `asyncEval` | V ; gain M | K-47 |
| P-66 | moyenne | Poids paresseux ; pas de chargement « encodeur seul » | V | K-59 |
| P-67 | moyenne | Pas de `cacheLimit`, vidage aveugle tous les 256 pas, rien au `unload()` | V ; effet M | K-52 |
| P-68 | basse | RoPE entrelacée manuelle au lieu de `RoPE(traditional: true)` | V ; gain M | K-65 |
| P-69 | basse | Même quantification pour l'encodeur (borné calcul) et le décodeur | M | K-60 |
| P-70 | moyenne | Packs 6 et 8 bits publiés (format voxmlx) non chargeables, chargés faux en silence ; aucun profil | V | K-78, K-80 |
| P-71 | moyenne | « Realtime » sans API de streaming | V | hors plan (ASK-3) |
| P-72 | basse | R&D : pas spéculatifs « remplissage » | M | K-73 |
| P-73 | haute | Diagnostics #23-#25 fondés sur des artefacts d'instrument ; la fermeture de #23 a masqué P-61 | V | K-17 (doc), K-36 (mesure) |
| P-74 | moyenne | Version du profiler non épinglée ni enregistrée (1.4 contre 1.5 : sémantique différente) | V | K-22 |
| P-75 | moyenne | `profile --pipeline realtime` non validable en A/A | V | K-32 |
| P-76 | basse | RTF de la campagne q6/bf16 non références (Debug probable, froid, porteur compté) | V | K-18 (doc), K-32 |
| P-77 | basse | La bibliothèque remet à zéro le pic MLX (tous les 2-16 jetons en STT) | V | K-32 |
| P-78 | basse | Le Realtime sert de juge ASR sans validation | V | K-33 |
| P-79 | haute | Aucun instrument de baseline : proposition `VoxtralCLI bench` | V | K-32 |

### 3.4 Constats perf portés par d'autres rapports

| Id | Rapport | Constat | Fiche |
|---|---|---|---|
| A-06 | annexes | Enrôlement : ≈ 3,8 G paramètres possiblement résidents, aucune politique mémoire (résidence À MESURER) | K-37, K-64 |
| A-10 | annexes | Deux synchronisations par époque (gain attendu < 1 % : note passagère) | K-64 |
| A-12 | annexes | Gain ANE annoncé non mesuré, unités de calcul contradictoires | K-42 |
| M-07 | modèles | La précision de l'encodeur est le premier levier ASR, sans effet en backend `.auto` | K-42, K-77 |
| MLX-002 | patterns | 3 occurrences réelles (masques), 22 voulues, 5 faux positifs ; les vraies fuites ne sont pas des littéraux | K-3, K-39 |
| MLX-018/019/020 | patterns | Masque maison, stream à production synchrone, absence de `withError` | K-3, K-12, K-1 |

## 4. Protocole commun

Défini une fois dans [`PLAN.md`](PLAN.md) §0 (règles) et §5 (corpus, commandes) : binaire Release, instrument K-32
validé en A/A ≤ 3 %, `machine-check.sh --cooldown 120`, A/B/B/A, seuil 5 %, un levier par comparaison, une ligne
JSON par mesure avec la révision résolue de mlx-swift-lm, parité sur checkpoint réel (greedy, WER normalisé, ou pour
le TTS bit-identique à graine fixée / parité forcée par l'enseignant). Corpus du dépôt : C-court
(`fluxforge_short_{en,fr}_6bit.wav`), C-moyen (`fluxforge_long_{en,fr}_6bit.wav`), C-long (≈ 11 min 22 s,
concaténation), C-xlong (≈ 17 min) ; parole synthétique : biais à noter ; corpus réel long : ASK-14.
