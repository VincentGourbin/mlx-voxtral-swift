# Verdicts mlx-swift-patterns — Voxtral @ `9392ed1` (2026-09-27)

> **Identifiants définitifs (2026-09-28, claude-skills mlx-swift 0.4.0).** Les candidats du §5 ont été intégrés
> au catalogue : « MLX-016 » (`cacheLimit = Int.max`) → **MLX-010** ; « MLX-017 » → **MLX-018** ; « MLX-018 » →
> **MLX-019** ; « MLX-019 » (production synchrone) → variante de **MLX-003** ; « MLX-020 » → **MLX-021**
> (décalage d'un rang : MLX-016 du catalogue = experts MoE non quantifiés, publié par mlx-swift 0.3.0). Dans le
> catalogue final, MLX-017 = jetons d'arrêt codés en dur, MLX-020 = fenêtre du cache ≠ architecture, MLX-022…025 =
> tokenizer de secours, `consolidated.safetensors`, ABBA compile × vjp, mode de quantification. Ce rapport garde
> les ids provisoires (instantané du 2026-09-27) ; PLAN.md et les fiches citent les ids définitifs.

> **Vérification croisée : 13 constats relus, 12 gardés, 1 écarté, 8 amendés.** Relecture adverse du 2026-09-27 :
> chaque `fichier:ligne` relu à `9392ed1` ; règles relues dans mlx `1f8e74e`, mlx-swift `9019419` (et tag `0.31.6`),
> mlx-swift-lm `ee673d6`, stdlib Swift `release/6.0` (`AsyncStreamBuffer.swift`, `AsyncThrowingStream.swift`),
> référence mlx-audio `main` (Realtime) et listings HF (connecteur, 2026-09-27) ; dry-runs MLX-002 et MLX-003 rejoués,
> garde syntaxique du hunk MLX-003 rejouée (0 / 0). Les 29 verdicts d'occurrence MLX-002 et les verdicts §3-4 ont
> aussi été relus : 1 verdict reclassé (`VoxtralVoiceSLERP.swift:73`, comparaison). Détail : annexe finale.

> Test du skill `mlx-swift-patterns` (catalogue MLX-001 à MLX-015, `scripts/apply.py`, `scripts/syntax_guard.py`)
> sur `mlx-voxtral-swift`, **en lecture seule** : aucun fichier source n'a été modifié. Les dry-runs ont été lancés
> sans `--write`, et les simulations de réécriture ont tourné sur des copies placées dans le scratchpad.
>
> Révisions lues : Voxtral `9392ed1` ; mlx-swift `9019419` (sous-module mlx `1f8e74e`) ; mlx-swift-lm `main` @ `ee673d6`.
> Les numéros de ligne MLX C++ cités ici sont ceux de `1f8e74e`. Les audits sœurs citent `ce45c52` : les numéros
> diffèrent, les règles sont identiques.
>
> Environnement : Linux, sans Swift, sans GPU. Aucun build, aucun test, aucune mesure. Tout gain est **attendu**.
> Toute porte qui exige un build ou une mesure cible `macos-gpu`.
>
> Renvois : `audit-performance-stt.md` (P-xx STT), `audit-performance-tts.md` (P-3x TTS), `audit-stabilite.md`
> (S-xx), `audit-annexes-serveur.md` (A-xx). Ce rapport ne les répète pas : il juge les **patterns** et
> l'**outil**, et ajoute ce qu'ils n'avaient pas (réécritures, faux négatifs des détecteurs). Le Realtime en fp32,
> d'abord présenté ici comme nouveau, était déjà couvert, et plus complètement, par P-60
> (`audit-performance-realtime-instruments.md`) : voir §4-bis et l'annexe.

## 0. Synthèse

| Pattern | Occurrences du scan | RÉEL | VOULU | FAUX POSITIF | Faux négatifs trouvés à la main |
|---|---|---|---|---|---|
| MLX-002 (fp32) | 28 Sources + 1 Tests | 3 (`VoxtralStandardLoader.swift:477` vivant ; `VoxtralLlama.swift:530`, `:536` hérité) | 21 (TTS, enrôlement, SLERP, WAV) | 4 Sources + 1 Tests (constantes qui ne servent qu'à une **comparaison**) | `VoxtralStandardLoader.swift:476`, `VoxtralFlowMatching.swift:64` (déjà dans P-30), Realtime fp32 (= **P-60** de l'audit Realtime : mel, tables RoPE de l'encodeur, `tCond`) |
| MLX-003 (stream) | 1 | 1 (`VoxtralTTSPipeline.swift:557`) | — | — | `VoxtralTTSModeling.swift:580` (production **synchrone** dans la closure de construction, invisible au détecteur) |
| MLX-015 (lien de dossier) | 0 Sources + 3 Tests | — | 3 (tests de non-régression du correctif `1570294`) | — | aucun sur les chemins vivants |
| MLX-004, 006, 010, 012, 014 (report, 0 au scan) | 0 | 004 : 3 sites ; 006 : 2 ; 010 : 1 ; 012 : 3 | 014 : 2 (affichage) | — | tous trouvés à la main |
| MLX-001, 005, 007, 008, 009, 011, 013 | 0 | — | — | — | conformes ou sans objet (§4) |

Dry-runs :
- **MLX-002 : 0 fichier modifié.** Les 28 occurrences sont toutes détectées par la 2ᵉ regex (`MLXArray(Float(…))`),
  qu'aucune règle `replace` ne couvre. L'outil annonce « 0 fichier(s) à modifier » **sans** lister les 28 restes : il ne
  signale que les `report_only`, et MLX-002 n'en a pas. Ce défaut d'outil est remonté dans les retours skill.
- **MLX-003 : 1 hunk, sûr et syntaxiquement valide** (tree-sitter : 0 erreur avant, 0 après). Il est efficace pour
  arrêter la génération, mais **incomplet** : il ne rend pas le flux réellement progressif (S-08).

Cinq patterns génériques nouveaux sont proposés (§5) : MLX-016 à MLX-020.

---

## 1. MLX-002 — verdict par occurrence

Règles MLX utilisées (mlx `1f8e74e`, mlx-swift `9019419`) :
- Un `MLXArray(Float(x))` est un scalaire fp32 **fort** : `a * MLXArray(Float(x))` promeut `a` bf16 en fp32.
- Un scalaire Swift passé via `ScalarOrArray` est **faible** : il prend le dtype de l'autre opérande
  (`toArrays`, `Source/MLX/DType.swift:416-428`).
- Une comparaison (`.<=`, `.>`, `.<`) rend un **booléen** : la constante ne propage aucun dtype.
- SDPA : `final_type = result_type(q, k, v)` (`mlx/fast.cpp:806`). Un masque **tableau non booléen** doit se
  promouvoir vers `final_type`, sinon `invalid_argument` « Mask type must promote to output type »
  (`fast.cpp:896-903`). Avec q/k/v bf16, un masque fp32 fait donc **lever**, il ne promeut pas.
- Sans `withError`/`withErrorHandler`, une erreur MLX termine le processus : `ErrorHandler.dispatch` appelle
  `fatalError(message)` (`Source/MLX/ErrorHandler.swift:337-347`, gestionnaire installé par `MLXArray.swift:18` ;
  le commentaire `:4` dit « print … then exit »). Voxtral n'appelle jamais `withError` (grep : 0).

### 1.1 Tableau (28 Sources + 1 Tests)

| # | Occurrence | Tenseurs voisins, dtype réel | Effet | Verdict |
|---|---|---|---|---|
| 1 | `Models/VoxtralLlama.swift:527` `offsetFloat` | grilles d'indices fp32 (`:522-523`), comparaison `:528` | booléen, aucune propagation | **FAUX POSITIF** (comparaison). À traiter avec #2 |
| 2 | `Models/VoxtralLlama.swift:530` `where(…, Float(0), Float(-1e9))` | masque **additif fp32** rendu par `createCausalMask` (public), consommé par `LlamaModel` hérité (`:336` via `MLXLMBridge.swift:81`) puis SDPA `.array` (`:136-154`) | verrouille le fp32 : lève si q/k/v ne sont pas fp32. Forme `[T, T]` fausse dès que `offset > 0` (P-17) | **RÉEL** (chemin hérité public) |
| 3 | `Models/VoxtralLlama.swift:534` `windowSizeFloat` | comparaison `:535` | booléen | **FAUX POSITIF** (comparaison) |
| 4 | `Models/VoxtralLlama.swift:536` masque de fenêtre | idem #2 | idem #2 | **RÉEL** (même correctif que #2) |
| 5 | `TTS/VoiceCloning/VoxtralEnrollmentLosses.swift:124` | perte STFT : `rfft` → complexe → `abs` fp32 (`:151-156`) | aucun, le calcul est déjà fp32 | **VOULU** (perte d'enrôlement fp32) |
| 6 | `…/VoxtralEnrollmentLosses.swift:125` accumulateur | idem | idem | **VOULU** |
| 7 | `TTS/VoxtralCodecDecoder.swift:219` masque causal | scores déjà fp32 (`:210` `* MLXArray(scale)`, `:215` ALiBi fp32), softmax fp32 explicite puis recast (`:231`) | identique à la référence : `mx.full((T,T), -1e9)` est fp32 par défaut (`audio_tokenizer.py:289-300`) | **VOULU** (parité référence). Le coût vient du T×T (P-32) |
| 8 | `…/VoxtralCodecDecoder.swift:224` masque de fenêtre | idem | idem (`mx.where(dist < -w, -1e9, 0.0)` fp32) | **VOULU** |
| 9 | `…/VoxtralCodecDecoder.swift:339` epsilon du codebook | `embedding_sum.asType(.float32) / maximum(…)` | codebook fp32 comme la référence (`audio_tokenizer.py:390-392`) | **VOULU** (conséquence : P-38) |
| 10 | `…/VoxtralCodecDecoder.swift:360` décodage FSQ | `indices.asType(.float32)` explicite | identique à la référence (`:411`) | **VOULU** |
| 11 | `TTS/VoxtralCodecEncoder.swift:189` | centroïdes VQ fp32 | classe **jamais instanciée** (0 construction ; poids absents du checkpoint, en-tête `:13-15`) | **VOULU** + code mort |
| 12 | `TTS/VoxtralFlowMatching.swift:135` `invFreq` | table sinusoïdale fp32 | identique à la référence (`acoustic_head.py:119-121`) | **VOULU** |
| 13 | `…/VoxtralFlowMatching.swift:324` clip ±1 | `xt` : état d'Euler fp32 par conception (`:295`, référence `x_t` fp32) | aucune fuite : tenseur (B, 36) puis `int32` | **VOULU** |
| 14 | `…/VoxtralFlowMatching.swift:326` clip [0, n−1] | idem | idem | **VOULU** |
| 15 | `…/VoxtralFlowMatching.swift:341` `quantizeToFSQ` | seul appelant : `VoxtralCodecEncoder.swift:209` (code mort) | — | **VOULU** (hors chemin) |
| 16 | `…/VoxtralFlowMatching.swift:342` | idem | — | **VOULU** |
| 17 | `…/VoxtralFlowMatching.swift:348` `dequantizeFSQ` | 0 appelant | — | **VOULU** (code mort) |
| 18 | `TTS/VoxtralTTSProcessor.swift:362` clip WAV | `samples = waveform.asType(.float32)` (`:357`) | I/O | **VOULU** |
| 19-26 | `TTS/VoxtralVoiceSLERP.swift:53`, `:54`, `:62`, `:67`, `:73`, `:79`, `:90`, `:92` | SLERP explicitement fp32 (`:49-50`, `:89`), ≤ 218 lignes | recasté au dtype des embeddings avant le LLM (`VoxtralTTSModeling.swift:369`) : **la règle « le bloc fp32 recaste sa sortie » est respectée**. `:73` est une comparaison (`sinOmega .< MLXArray(Float(1e-6))` → booléen) | **VOULU** (7) ; `:73` **FAUX POSITIF** (comparaison ; reclassé à la vérification croisée, même règle que #1, #3, #28) |
| 27 | `Utils/VoxtralStandardLoader.swift:477` `zero` (avec `:476` `MLXArray(-Float.infinity)`, **non détecté**) | masque **additif fp32** `[T, offset+T]` du décodeur STT **vivant**, passé en tableau au SDPA (`:692-698`) | verrouille le fp32 : impossible de corriger P-01 sans lui (sinon le SDPA lève). Forme fausse avec `RotatingKVCache` (P-03) | **RÉEL** (haute) |
| 28 | `VoxtralModeling.swift:1537` `which(…, kthProb, Float(1e-9))` | `cutoff` (B, 1) ; `probs .>= cutoff` → booléen ; `which(bool, logits, -Float.infinity)` garde le dtype des logits (scalaire faible) | aucune propagation | **FAUX POSITIF** (le défaut réel est la logique top-p, voir l'audit STT) |
| T1 | `Tests/…/PerformanceOptimizationTests.swift:123` | tenseurs du test tous fp32 | — | **FAUX POSITIF**. Le test reproduit `which` de MLX, pas `sample()` de Voxtral (tautologique, S-27) |

Bilan : 3 RÉEL, 21 VOULU, 4 FAUX POSITIF en Sources, plus 1 FAUX POSITIF en Tests. Le verdict « aucune constante
ne promeut à elle seule un tenseur bf16 » de l'audit TTS §3 est **confirmé** pour les 22 occurrences TTS (21 VOULU +
1 comparaison). Sur le
chemin STT, la seule constante qui compte est celle du **masque** (#27).

### 1.2 Correctifs exacts des occurrences RÉEL

**#27 — `Utils/VoxtralStandardLoader.swift:450-481` (vivant).** Il faut un masque **booléen**, construit par le cache.
Un booléen se promeut vers tout dtype flottant, donc le SDPA l'accepte quel que soit le dtype de q/k/v
(`fast.cpp:896-905`). Sa forme vient du cache lui-même, `RotatingKVCache` enroulé compris (mlx-swift-lm
`KVCache.swift:89` exigence du protocole, `:253-267` défaut, `:910-929` rotatif). La fonction est privée : l'API ne
change pas.

```swift
private func createCausalAttentionMask(hiddenStates: MLXArray, cache: [any KVCache]?) -> MLXArray? {
    let T = hiddenStates.shape[1]
    guard T > 1 else { return nil }
    if let first = cache?.first {
        // Booléen, forme alignée sur ce que le cache rendra (Simple : [T, offset+T] ; Rotating : fenêtre).
        return first.makeMask(n: T, windowSize: nil, returnArray: true).mask
    }
    return MLXLMCommon.createCausalMask(n: T, offset: 0)   // booléen [T, T] (amont, KVCache.swift:270-292)
}
```
- Parité : pour `KVCacheSimple`, les positions autorisées sont identiques (`col ≤ row`). Le noyau convertit le
  booléen en additif 0/−inf s'il ne le gère pas nativement (`fast.cpp:941-949`, conversion booléen → 0/−inf), sinon exclut les mêmes positions.
  On attend donc la même transcription greedy. Pour `RotatingKVCache`, la forme devient correcte (P-03) : la sortie
  change par construction et doit être mesurée.
- Seul, ce correctif ne change **pas** le dtype aujourd'hui : l'état caché est fp32 à cause de P-01. C'est le
  **prérequis** de P-01.
- Variante plus rapide (`.causal`, aucun masque matérialisé) : `LlamaStandardAttention.callAsFunction`
  (`:648-698`) traduirait `attentionMask == nil && qLen > 1` en `.causal`, comme `VoxtralLlama.LlamaAttention`
  (`Models/VoxtralLlama.swift:136-146`). Cela change la sémantique d'une méthode publique (nil = causal) : risque
  **additif/comportemental**, à valider par ASK.

**#2 et #4 — `Models/VoxtralLlama.swift:333-337` (appelant) et `:515-541`.** Correctif minimal sans cassure :
`LlamaModel.callAsFunction` ne construit plus de masque, et `LlamaAttention` applique déjà `.causal` quand le masque
vaut nil et `qLen > 1` (`:136-146`). `.causal` est aligné en bas à droite (`offset = kL − qL`, repli `fast.cpp:838-846`),
donc correct pour le préfill tranché. Le TTS l'utilise déjà en production avec un cache de préfixe
(`TTS/VoxtralTTSModeling.swift:303-306`, suffixe préfillé à `offset > 0`).
```swift
        var attentionMask = mask   // nil ⇒ LlamaAttention utilise .causal pour T > 1 (fp32-agnostique, offset géré)
```
(supprimer `:333-337`). Laisser la fonction publique `createCausalMask(N:offset:windowSize:lengths:)` telle quelle,
éventuellement `@available(*, deprecated)` (additif). La réécrire en booléen `[N, offset+N]` changerait son contrat :
**cassant → ASK**. Le constat recoupe P-17 (forme `[T, T]`, arrêt dès la 2ᵉ tranche).

---

## 2. MLX-003 — `TTS/Pipeline/VoxtralTTSPipeline.swift:556-671`

**Verdict : RÉEL.** `Task { … }` sans `continuation.onTermination`.

Dry-run (`apply.py apply --pattern MLX-003 /home/user/mlx-voxtral-swift`, **sans** `--write`) :
```diff
--- a/Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift
+++ b/Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift
@@ -554,7 +554,7 @@
         let ctx = StreamContext(model: model, tokenizer: tokenizer, voiceEmb: voiceEmb, pipeline: self, prefixCache: prefix?.cache, prefixLen: prefix?.len ?? 0, genText: genText, seed: seed, hasWarmUp: hasWarmUp, warmUpLeadInFrames: warmUpLeadInFrames)
 
         return AsyncThrowingStream { continuation in
-            Task {
+            let task = Task {
                 defer { beacon?.end() }
                 // Sample offset (into the full decoded waveform) where the real
                 // content starts. Without warm-up that's 0; with warm-up it's the
@@ -668,6 +668,7 @@
 
                 ctx.pipeline.state = .ready
             }
+            continuation.onTermination = { _ in task.cancel() }
         }
     }
 
MLX-003 : 1 fichier(s) à modifier (dry-run, relancer avec --write).
```

Jugement du hunk :
- **Swift valide.** La garde syntaxique donne 0 erreur avant et 0 après (rewrite appliqué à une copie, fonction
  `errors()` de `syntax_guard.py`). En lecture des types : `task` est un `Task<Void, Never>` (le `do/catch` interne
  avale tout), `Sendable`, donc capturable par le `@Sendable` de `onTermination`. La portée est correcte :
  `onTermination` est posé dans la closure du stream, après la fermeture de la Task (`:670`). Il n'y a ni
  redéclaration ni capture de `self`.
- **Sûr, et efficace pour arrêter la génération, mais seulement pendant la génération** (amendé à la vérification
  croisée). `generateStreaming` exécute toute sa boucle **dans la closure de construction** de son propre
  `AsyncThrowingStream`, qui est synchrone (`TTS/VoxtralTTSModeling.swift:580-684` ; stdlib : `build` appelé dans
  l'`init`, `AsyncThrowingStream.swift:332` @ `release/6.0`). Cette boucle tourne donc dans cette Task-ci, et son test
  `Task.isCancelled` (`:622`) voit l'annulation propagée par le hunk : **si le consommateur s'arrête avant le premier
  chunk** (seule fenêtre possible, puisque rien n'est cédé avant la fin de la génération), la génération s'arrête à la
  frame suivante et `state = .ready` (`:669`) est atteint vite.
  **Après le premier chunk**, la génération est déjà finie et le hunk ne raccourcit rien : la boucle `for try await`
  de `:581` ne teste pas l'annulation, et un flux dont le consommateur est annulé **délivre encore ses éléments
  tamponnés** (stdlib `AsyncStreamBuffer.swift:342-353` : `cancel()` → `finish()` ; `:488-497` : `next()` rend
  `pending` avant `nil`). Chaque chunk restant repasse donc par `decodeToWaveform` de toute la séquence (`:583`) avant
  `.ready`.
- **Incomplet.** (1) Le flux ne streame pas : tout est généré avant le premier `for try await` (`:569` → `:581`).
  C'est S-08, et c'est le pattern candidat **MLX-019**. (2) Il manque `try Task.checkCancellation()` en tête du corps
  de la boucle `:581`. (3) L'état reste non protégé (S-10). (4) Le test de la fiche (« retour à l'état prêt en
  < 3 s ») n'existe pas dans le dépôt.
- Recommandation : appliquer le hunk (un pattern, un commit : `fix(MLX-003): …`), puis traiter S-08 dans le même
  lot. Deux portes `macos-gpu` distinctes :
  - **hunk seul** : annulation du consommateur pendant la génération (avant le 1er chunk, texte ≈ 350 mots, sans
    warm-up) → pipeline `.ready` en < 1 s, génération arrêtée en ≤ 1 frame après l'annulation ;
  - **lot hunk + MLX-019 + `checkCancellation` à `:581`** : annulation après 5 chunks → `.ready` en < 1 s, et frames
    générées ≤ frames au moment de l'annulation + 1. Cette porte est **inatteignable avec le hunk seul**.

---

## 3. MLX-015 — 3 occurrences en Tests

| Occurrence | Verdict |
|---|---|
| `Tests/VoxtralCoreTests/Loading/ModelLoadingSymlinkedDirectoryTests.swift:31` | **VOULU** : test de non-régression du correctif `1570294`. `loadWeights(modelPath:)` doit suivre un **dossier** lien ; le chargeur utilise `contentsOfDirectory(atPath:)` (`Utils/VoxtralModelLoading.swift:96-99`, `Utils/VoxtralStandardLoader.swift:991-993`) |
| `Tests/VoxtralCoreTests/Utils/ModelDownloaderSizeTests.swift:50` | **VOULU** : taille d'un poids relogé (lien de fichier) |
| `…/ModelDownloaderSizeTests.swift:71` | **VOULU** : lien cassé (disque non monté) → 0 |

Sur les chemins vivants :
- STT : `atPath:`.
- TTS et Realtime : noms construits puis `fileExists`, qui suit les liens (`TTS/VoxtralTTSModelLoading.swift:118-145`,
  `Realtime/VoxtralRealtimeModelLoading.swift:81-111`).

Seul `Utils/VoxtralMLXLMLoader.swift:37` passe par `MLXLMCommon.loadWeights`. C'est du code mort public (S-13).
L'affirmation du dépôt, « `contentsOfDirectory(at:)` ne voit rien derrière un lien de dossier », n'est pas
re-vérifiable sous Linux. Elle est tenue par son test.

---

## 4. Faux négatifs cherchés à la main (patterns `report` et sans occurrence)

| Pattern | Verdict | Preuve (fichier:ligne @ `9392ed1`) |
|---|---|---|
| MLX-001 | **Conforme** | `GPU.resetPeakMemory()` (`VoxtralModeling.swift:1257`, `:1434`, `Utils/VoxtralMemoryManager.swift:48`, `:92`, app) n'est **pas** déprécié dans mlx-swift `9019419` (`GPU+Metal.swift:217-222`, sans `@available`). Aucun autre `GPU.*` |
| MLX-004 | **RÉEL, 3 sites** (recoupe S-12) | Règle : « It is not safe to create `c` in one thread and consume/evaluate it in another » (mlx-swift `9019419`, `Source/MLX/Documentation.docc/MLXArray.md:26-40`). (a) `TTS/Pipeline/VoxtralTTSPipeline.swift:239` puis `:247` et `:380` puis `:389` : `applyTrims` rend une **tranche paresseuse** (`TTS/VoxtralTTSProcessor.swift:134`) dans `TTSSynthesisResult: @unchecked Sendable` (`:13-14`), qui traverse l'`async`. (b) Streaming : `fullWaveform[…]` non évalué, cédé dans `TTSStreamingChunk: @unchecked Sendable` (`VoxtralTTSPipeline.swift:629`, `:641-643`, `:648` ; type `VoxtralTTSProcessor.swift:323-325`) et évalué sur le MainActor (`StreamingDemoViewModel.swift:584-585`). (c) `GenerationChunk.accumulatedCodes = MLX.stacked(…)` non évalué (`TTS/VoxtralTTSModeling.swift:636`, `:654`, `:675` ; type `:555-557`), consommé après une suspension (changement de thread possible). **Latent** : le callback public `onFrame` reçoit `codes` non évalué pour `i > 0` (`VoxtralTTSModeling.swift:500`), sans consommateur dans le dépôt |
| MLX-004 (non-constats) | Conforme | `StreamingDemoViewModel.swift:411` `Task.detached` : seules des valeurs (`Int`, `Float`) traversent (`:419-420` ; `Progress` = `Int`/`Float`, `VoxtralVoiceEnrollment.swift:81-85`). `extractAudioEmbeddings` évalue avant de rendre (`Realtime/Pipeline/VoxtralRealtimePipeline.swift:173`). Préfixe voix évalué (`VoxtralTTSModeling.swift:415`). `_melFiltersCache` (`VoxtralFeatureExtractor.swift:250`) contient des tableaux adossés à des données Swift, pas des graphes (la course sur le dictionnaire relève de S-11) |
| MLX-005 | Sans objet | 0 `loadModelContainer`, 0 `ModelFactory` ; Voxtral ne lie pas MLXVLM |
| MLX-006 | **RÉEL, 2 sites, chemin mort** (recoupe S-17) | `VoxtralModeling.swift:1596-1647` : un seul forward, `prefill.stepSize` ignoré, logits de toutes les positions ; en plus `:1629` passe les **embeddings** comme ids. `Utils/VoxtralStandardLoader.swift:252-257` rend `.tokens(input.text)` entier. L'amont tranche (`MLXLLM/LLMModel.swift:24-62` : `resolvedStepSize`, `forEachChunk`, `asyncEval(cache)`). Aucun appelant dans le dépôt (0 `TokenIterator`/`generate` amont). **Le détecteur est aveugle** : il attend la signature `windowSize: Int?`, alors que `main` a `state:prefill:` |
| MLX-007 | Conforme | 0 `state[0|1]`. `cloneKVCaches` protège par `s.count == 2` (`VoxtralTTSModeling.swift:697-703`), et les caches sont toujours `KVCacheSimple` (`:311-313`) |
| MLX-008 | Sans objet (pas de rollback) | `RotatingKVCache(maxSize:keep:)` sans `trim` (`VoxtralModeling.swift:1139`). Le défaut réel est la forme du masque (P-03), corrigée par #27 |
| MLX-009 | Conforme | 0 `asType(….weight.dtype)`. Les `embeddings.dtype` du TTS (`VoxtralTTSModeling.swift:365`, `:369`) viennent d'une sortie d'`Embedding`/`QuantizedEmbedding` (dtype des `scales`), pas d'un `weight` packé. À respecter dans le correctif de P-01 et de §4-bis (lire le dtype sur `norm.weight`) |
| MLX-010 | **RÉEL** (recoupe P-09/A-17) | 0 pose dans `VoxtralCore` : aucun des 3 chargements (`Pipeline/VoxtralPipeline.swift:266`, `TTS/Pipeline/VoxtralTTSPipeline.swift:180`, `Realtime/Pipeline/VoxtralRealtimePipeline.swift:108`) ni aucune génération ne pose `Memory.cacheLimit`. **Le détecteur est muet** à cause de `VoxtralApp/TranscriptionManager.swift:293` (`= 0`) et `:295` (`= Int.max`), dans une fonction jamais appelée |
| MLX-011 | Sans objet | Aucun serveur (A-23) |
| MLX-012 | **RÉEL, 3 variantes** (recoupe S-03/A-02) | (a) `Utils/ModelDownloader.swift:316-321` : sans `model.safetensors.index.json`, `verifyShardedModel` rend `(true, [])`. Or l'index se classe **après** les shards (`-` 0x2D < `.` 0x2E). Le Hub liste `VincentGOURBIN/voxtral-small-8bit` ainsi (fichiers `*.json`/`*.safetensors` retenus par les globs) : `config.json`, `generation_config.json`, `model-00001…00005-of-00005.safetensors`, `model.safetensors.index.json`, `params.json`, `preprocessor_config.json`, `tekken.json` (connecteur HF, relu le 2026-09-27 ; `tekken.json` vient en dernier : une coupure juste avant lui mène aussi au repli S-05). Une coupure après le shard k < 5 laisse `config.json` + k shards sans index → « complet » → chargement partiel sans erreur (MLX-017). L'ordre exact de l'API `tree` utilisé par `downloadRepoDirect` (`:91-111`) est **À VÉRIFIER** (proxy bloqué ici). (b) TTS : `params.json` suffit (`:537-581`). (c) Realtime : `config.json`, classé **avant** `model.safetensors`, suffit (`:644-692`) ; après une coupure, « déjà téléchargé » → aucune reprise → `fileNotFound` à chaque chargement. En plus, `download()` se contente d'**imprimer** un avertissement si l'index révèle un manque (`:368-372`) |
| MLX-013 | Sans objet | 0 `ModelTypeRegistry` |
| MLX-014 | Conforme sur les chemins vivants ; **VOULU** (affichage) ailleurs | STT, Realtime et TTS décodent la liste complète (`Pipeline/VoxtralPipeline.swift:380`, `:469`, `Realtime/Pipeline/VoxtralRealtimePipeline.swift:145`). `TekkenTokenizer.decode` accumule les **octets** puis décode une seule fois (`VoxtralComponents.swift:528-558`) : correct. Décodage jeton par jeton **pour affichage console** seulement dans l'hérité `VoxtralGenerator.swift:221` (le résultat rendu vient du décodage complet `:242`) et `Scripts/VoxtralGenerate.swift:196` (mort). `VoxtralGeneratorBridge.swift:142` est dans un bloc commenté. La regex (`decode(tokenIds: [`) ne voit pas `decode([tokenId], …)` |

### 4-bis. Realtime en fp32 : faux négatif du **détecteur** MLX-002, constat = P-60 (écarté comme constat propre)

Le défaut est réel, mais il est déjà le constat **P-60** de `audit-performance-realtime-instruments.md` (§ P-60),
plus complet que la version initiale de ce paragraphe, qui a été retirée (raison dans l'annexe). Ce rapport n'en garde
que l'enseignement sur l'outil : aucune des trois sources fp32 n'est une constante littérale `MLXArray(Float(…))`,
donc la regex MLX-002 ne pouvait rien voir.
- Trois sources indépendantes (lecture à `9392ed1`) : mel fp32 jamais castée
  (`Realtime/Pipeline/VoxtralRealtimePipeline.swift:215-218`) ; tables `cos`/`sin` RoPE de l'encodeur construites en
  fp32 (`Realtime/VoxtralRealtimeEncoder.swift:76-83`, `positions.asType(.float32)` à `:81`), qui promeuvent q/k puis
  la sortie du SDPA ; `tCond` → `adaScale` fp32 (`Realtime/VoxtralRealtimeDecoder.swift:24-31`, `:54-56`,
  `:166-173`) multiplié dans chaque couche (`:124-127`).
- Corriger seulement `adaScale` et la mel **ne suffit pas** : la sortie de l'encodeur resterait fp32 à cause des
  tables RoPE, donc l'entrée `audio + texte` du décodeur (`VoxtralRealtimeModel.swift:93`, `:97`) et son cache KV
  aussi. La correction et la porte de référence sont celles de P-60.
- Preuve complémentaire, à verser à P-60 : la référence mlx-audio (`main`, lue le 2026-09-27) caste `t_cond` vers le
  dtype de `ada_down.weight` « so the matmul doesn't silently upcast » (`mlx_audio/stt/models/voxtral_realtime/voxtral_realtime.py:129-134`,
  `:596-598`) ; le portage Swift a omis ce cast.

---

## 5. Nouveaux patterns génériques proposés (candidats MLX-016 à MLX-020)

Chacun a une source vérifiable, un correctif et un test. Tous sont en mode `report` : aucune réécriture n'est sûre.

| Id | Titre | Source (vérifiable) | Correctif | Test de non-régression | Détecteur proposé |
|---|---|---|---|---|---|
| **MLX-016** | `Memory.cacheLimit = Int.max` présenté comme « restaurer le défaut » | `VoxtralApp/TranscriptionManager.swift:291-296`. Défaut réel : `max_pool_size_ = block_limit_ = min(1,5 × max_recommended_working_set, 0,95 × RAM)` (mlx `1f8e74e` `backend/metal/allocator.cpp:63-65`). Le getter mlx-swift lit la valeur réelle (`Memory.swift:251-266`) | `let previous = Memory.cacheLimit` … `Memory.cacheLimit = previous`. `Memory.clearCache()` suffit à vider ; `= 0` est inutile | après la fonction, `Memory.cacheLimit == valeur lue avant` | `cacheLimit\s*=\s*Int\.max` ; le détecteur MLX-010 doit **ignorer** ces poses |
| **MLX-017** | Poids appliqués sans vérification : `update(parameters:)` non levant (`verify: .none`) | `Utils/VoxtralStandardLoader.swift:1323`, `:1338` ; `TTS/VoxtralTTSModelLoading.swift:71` ; `Realtime/VoxtralRealtimeModelLoading.swift:55`. mlx-swift `Source/MLXNN/Module.swift:401-408` (`try! … verify: .none`). L'amont vérifie `[.all]` (mlx-swift-lm `Load.swift:404` @ `ee673d6`). Aggravé par MLX-012 et par `TTS/VoxtralTTSModelLoading.swift:125` (2ᵉ shard facultatif) | `try model.update(parameters:, verify: [.allModelKeysSet, .shapeMismatch])`, puis `.noUnusedKeys` une fois les clés assainies | un dossier sans un shard (ou une clé renommée) fait **lever** le chargement ; sorties inchangées sur les modèles du registre | `\.update\(parameters:\s*[^,()]+\)\s*$` et `verify:\s*\.none` dans les fonctions `load*` |
| **MLX-018** | Masque d'attention maison passé au SDPA : additif **fp32** (verrou de dtype, lève en bf16/fp16) ou de **forme** indépendante du cache | `Utils/VoxtralStandardLoader.swift:450-481` (+ `:692-698`) ; `Models/VoxtralLlama.swift:515-541` + `MLXLMBridge.swift:48-87` ; mlx `fast.cpp:896-903` (dtype), `:907-909` `broadcast_to` (forme), `ErrorHandler.swift:3-5` (exit) | masque **booléen** via `cache.makeMask(n:windowSize:returnArray:)` / `createAttentionMask(h:cache:…)` amont (`KVCache.swift:376-397`), ou `.causal` | préfill en 2 tranches + `RotatingKVCache` enroulé + q/k/v bf16 : aucun arrêt, masque `.bool` de forme attendue, sortie greedy identique au masque actuel en `KVCacheSimple` | `func create\w*Causal\w*Mask`, `where\([^)]*(-Float\.infinity\|-1e9)` dont le résultat alimente `mask:` |
| **MLX-019** | `AsyncThrowingStream { continuation in … }` qui **produit de façon synchrone** dans la closure de construction (aucune Task) : ni streaming ni annulation | `TTS/VoxtralTTSModeling.swift:580-684` (boucle `for i in 0..<maxTokens` + `yield`) ; stdlib : `build` est appelé immédiatement et n'échappe pas. Complément de MLX-003, que son détecteur ne voit pas (0 Task) | `let (stream, continuation) = AsyncThrowingStream.makeStream(of:)` + `let task = Task { … }` + `continuation.onTermination = { _ in task.cancel() }` + `try Task.checkCancellation()` par pas | `generateStreaming(…)` rend en < 50 ms quel que soit `maxTokens` ; premier élément disponible avant la fin de la génération | corps de closure `continuation in` contenant `continuation.yield` dans un `for`/`while`, **sans** `Task` |
| **MLX-020** | Bibliothèque MLX sans `withError` aux points d'entrée publics : une erreur MLX (forme, dtype, masque) **termine le processus hôte** | mlx-swift `9019419` (identique au tag `0.31.6`) : `MLXArray.swift:18` installe le gestionnaire ; sans gestionnaire de tâche ni global, `ErrorHandler.dispatch` appelle `fatalError(message)` (`ErrorHandler.swift:337-347`) ; Voxtral : `grep withError` = 0 ; déclencheurs avérés en lecture : P-03, P-17, MLX-018. VoxtralCore est embarqué par FluxForge (App Store) | `try withError { error in … }` (variante async) autour de `transcribe`/`chat`/`synthesize*`/`generate*` publics, converti en `VoxtralError` typé ; **`try error.check()` après chaque `eval` des boucles de génération** : une erreur capturée n'interrompt pas le bloc, les tableaux suivants sont vides et un `.item()` ou un indice pourrait piéger côté Swift avant la sortie du bloc | une entrée qui provoque une erreur MLX (ex. prompt > `maxKVCacheSize` sur le masque actuel) fait **lever** l'API ; le processus reste vivant ; temps dans le bruit (±5 %, A/B/B/A) | bibliothèque (`.library`) avec `import MLX` et 0 `withError`/`withErrorHandler` |

Amendement proposé à **MLX-002** (pas un nouveau pattern) :
- La source n°1 d'une fuite fp32 est souvent une **entrée jamais castée** (mel ou STFT fp32 : P-01 STT, Realtime
  P-60), une **table précalculée en fp32** (cos/sin RoPE, P-60) ou un **vecteur de conditionnement précalculé en
  fp32** (embedding temporel → AdaNorm), et non une constante littérale.
- La regex ne peut pas voir ce flux de dtype. Le détecteur de référence doit être le **test** « dtype du cache KV après
  préfill = dtype de calcul ».

---

## 6. Simulation d'une réécriture mécanique MLX-002 (hypothétique) : pourquoi aucune n'est sûre ici

Le skill n'a pas de règle pour la forme `MLXArray(Float(x))`. La règle naïve `MLXArray\(Float\(([^()]*)\)\)` → `\1`
(scalaire Swift faible) a été simulée sur des **copies**, avec la garde syntaxique :

```diff
-        guard !resolutions.isEmpty else { return MLXArray(Float(0)) }
-        var total = MLXArray(Float(0))
+        guard !resolutions.isEmpty else { return 0 }          # MLXArray n'est pas ExpressibleByIntegerLiteral → erreur de type
+        var total = 0                                           # Int ; `total = total + scLoss + …` (MLXArray) → erreur de type
-        let zero = MLXArray(Float(0))
+        let zero = 0                                            # Int
-    return (2.0 * indices.asType(.float32) / MLXArray(Float(levels - 1))) - 1.0
+    return (2.0 * indices.asType(.float32) / levels - 1) - 1.0   # PRÉCÉDENCE CHANGÉE : (2x/levels) − 1 − 1
-        let clamped = MLX.clip(xt, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))
+        let clamped = MLX.clip(xt, min: -1.0, max: 1.0)         # correct (scalaires faibles, dtype de xt)
```
Résultat de `syntax_guard` : **0 erreur nouvelle** sur les 3 fichiers. La garde ne voit ni les types ni la
précédence.

Même une règle correcte serait **dangereuse pour les epsilons**. `MLX.maximum(x, 1e-16)` en scalaire faible devient
0 si `x` est fp16 (plus petit sous-normal ≈ 6e-8). Ici, les 22 occurrences VOULU sont sur des tenseurs fp32 par
conception : la réécriture n'apporterait rien.

**Verdict dry-run MLX-002 : aucun hunk produit ; tout hunk mécanique serait dangereux ou inutile ; les 3 RÉEL se
corrigent à la main (§1.2).**

---

## 7. Constats (format du skill)

| id | sév. | fichier:ligne | constat | correction | risque API | effort | statut | fiche (porte chiffrée, cible) |
|---|---|---|---|---|---|---|---|---|
| MLX-002@Sources/VoxtralCore/Utils/VoxtralStandardLoader.swift:477 | haute | `:450-481` (+ `:476`, `:692-698`) | masque additif fp32 du décodeur STT vivant : verrou fp32 (lève en bf16), forme fausse sous `RotatingKVCache` | masque booléen `cache.makeMask(…, returnArray: true).mask` (§1.2) | aucun (privé) | S | VÉRIFIÉ ; parité À MESURER | *Masque STT booléen aligné sur le cache* — test rouge → vert (`.bool`, forme `[T, offset+T]` pour Simple et forme du cache pour Rotating enroulé) ; transcription greedy identique sur 100 % des clips de `docs/examples/` (Mini 4 et 8 bits, `.mlx` et `.auto`) ; préfill dans le bruit, ±5 % (A/B/B/A ; *amendé : ±3 % est sous le seuil de bruit de 5 % de `measurement.md`*) ; clip 167 s avec `maxKVCacheSize` 2048 : aucun arrêt. **macos-gpu** |
| MLX-002@Sources/VoxtralCore/Models/VoxtralLlama.swift:530 | moyenne | `:515-541`, appelant `:333-337` (et `:536`) | masque additif fp32 `[T, T]` du décodeur hérité public : verrou fp32 et arrêt dès la 2ᵉ tranche (P-17) | supprimer `:333-337` (nil ⇒ `.causal`) ; `createCausalMask` public inchangé ou déprécié | aucun / additif (dépréciation) ; réécrire `createCausalMask` = cassant → ASK | S | VÉRIFIÉ ; arrêt À MESURER | *Hérité : `.causal`* — invite de 600 jetons via `loadVoxtralModel(modelPath:dtype:lazy:)` sans arrêt ; transcription identique au chemin standard sur 3 clips. **macos-gpu** |
| MLX-003@Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:557 | moyenne (*amendé : haute → moyenne, aligné sur S-08 vérifié : seul consommateur = la démo, FluxForge n'appelle pas `synthesizeStreaming` ; ni sortie fausse ni plantage*) | `:556-671` | Task du stream non annulée à la terminaison ; le hunk n'agit que pendant la génération (après le 1er chunk, les chunks tamponnés sont encore décodés) | hunk du dry-run (§2) ; puis S-08 / MLX-019 + `try Task.checkCancellation()` en tête de la boucle `:581` | aucun | S | VÉRIFIÉ ; hunk : 0 erreur de syntaxe (rejoué) | *Streaming TTS annulable* — **hunk seul** : annulation pendant la génération (avant le 1er chunk, ≈ 350 mots, sans warm-up) → `.ready` en < 1 s, génération arrêtée en ≤ 1 frame ; **lot avec MLX-019** : annulation après 5 chunks → `.ready` en < 1 s, frames générées ≤ frames à l'annulation + 1 ; build Swift 6 sans nouvel avertissement. **macos-gpu** |
| MLX-002@Sources/VoxtralCore/TTS/VoxtralFlowMatching.swift:64 | basse | `:64` | `* MLXArray(scale)` : scalaire fp32 fort, non détecté ; inerte tant que P-30 garde le FM en fp32, fuite dès que P-30 est corrigé. *Amendé : sous-point de P-30, dont la correction le prévoit déjà (`audit-performance-tts.md:234`) ; pas de fiche propre* | `* scale` (scalaire faible) | aucun | S | VÉRIFIÉ | dans la fiche P-30 : sortie audio identique octet pour octet avant P-30 (graine fixée). **macos-gpu** |
| MLX-004@Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:239 | moyenne | `:239-253`, `:380-389`, `:629`, `:641-648` ; `TTS/VoxtralTTSModeling.swift:636`, `:654`, `:675` | tranches et `stacked` paresseux qui traversent l'isolation dans des types `@unchecked Sendable` ; règle documentée par mlx-swift (`MLXArray.md:26-40`, *preuve ajoutée*) | `MLX.eval(x)` avant `return`/`yield` (sans coût si déjà évalué) | aucun | S | VÉRIFIÉ ; plantage À MESURER | = K-S12 : synthèse hors MainActor, consommation sur MainActor, 20/20 sans plantage ; WAV identiques octet pour octet. *Amendé : sans plantage reproduit avant correctif, cette porte ne discrimine pas ; la fiche est préventive et ne revendique aucun plantage évité.* **macos-gpu** |
| MLX-006@Sources/VoxtralCore/VoxtralModeling.swift:1596 | basse | `:1596-1647` ; `Utils/VoxtralStandardLoader.swift:252-257` | `prepare` ignore `prefill.stepSize` (+ embeddings passés comme ids `:1629`) ; chemin mort dans le dépôt, API publique | ASK : retirer la conformance `LanguageModel` (cassant) **ou** `prepare` tranché calqué sur l'amont | cassant (retrait) / aucun (implémentation) | M | VÉRIFIÉ | *prepare tranché ou retrait* — si conservé : prompt de 2 000 jetons → `cache.offset` = 1 999, `.tokens` d'un jeton, pic ≤ pic du préfill tranché actuel. **ASK puis macos-gpu** |
| MLX-010@Sources/VoxtralCore/Pipeline/VoxtralPipeline.swift:266 | moyenne | chargements STT `:266`, TTS `TTS/Pipeline/VoxtralTTSPipeline.swift:180`, Realtime `Realtime/Pipeline/VoxtralRealtimePipeline.swift:108` | aucune `Memory.cacheLimit` sur aucun chemin d'inférence | champ additif `MemoryOptimizationConfig.cacheLimitBytes: Int?` (nil = ne pas toucher le réglage global de l'hôte), posé après chargement, restauré à `unload` | additif | S | VÉRIFIÉ ; gain À MESURER | = P-09 : à 10 min d'audio, `phys_footprint` ≤ actif + cacheLimit + Core ML (+5 %) ; temps ±5 %. **macos-gpu** |
| MLX-012@Sources/VoxtralCore/Utils/ModelDownloader.swift:316 | haute | `:316-321`, `:368-372`, `:537-581`, `:644-692` | complétude déduite de `config.json`/`params.json` ou de l'absence d'index (l'index se classe après les shards ; *amendé : listing Hub complété, `tekken.json` en dernier*) ; = S-03 | marqueur `.complete` écrit en fin de `downloadRepoDirect` ; sans index, exiger ≥ 1 `*.safetensors` hors `consolidated*` + le marqueur ; `download()` **lève** si incomplet ; TTS et Realtime passent par la même vérification | additif | S | VÉRIFIÉ (lecture + listing Hub) ; ordre de l'API `tree` À VÉRIFIER | *Complétude prouvée* — dossiers synthétiques : 1/5 shards sans index → non téléchargé ; Realtime `config.json` seul → non téléchargé et reprise effective ; `download()` incomplet → erreur. **macos-gpu** |
| MLX-017@Sources/VoxtralCore/Utils/VoxtralStandardLoader.swift:1338 | haute | `:1323`, `:1338` ; `TTS/VoxtralTTSModelLoading.swift:71`, `:125` ; `Realtime/VoxtralRealtimeModelLoading.swift:55` | poids manquants ou mal nommés laissés à l'initialisation aléatoire, sans erreur ; = S-04 | `try … verify: [.allModelKeysSet, .shapeMismatch]` | API : aucun (fonctions déjà `throws`) ; *amendé* : **comportemental**, un pack aujourd'hui chargé malgré une clé de modèle absente lèverait (FluxForge compris) | S | VÉRIFIÉ | = K-S04 : chargement de **tous** les modèles des registres STT, TTS et Realtime (*amendé : au lieu de « ≥ 1 par famille »*) sans erreur ; un shard retiré → erreur explicite (test rouge avant) ; sorties greedy identiques (graine fixée). **macos-gpu** |
| MLX-019@Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:580 | moyenne (*amendé : haute → moyenne, même raison que MLX-003 / S-08*) | `:580-684` ; consommation `TTS/Pipeline/VoxtralTTSPipeline.swift:569-581` | production synchrone dans la closure de construction : TTFA = génération complète ; = S-08 | `makeStream` + `Task` + `onTermination` + `checkCancellation` | aucun | M | VÉRIFIÉ (sémantique stdlib, `AsyncThrowingStream.swift:332`) | = K-S08 : texte de ≈ 350 mots, 4 bits, **sans warm-up** (*amendé : avec `warmUpText`, le 1er chunk attend 3 s d'audio, `VoxtralTTSPipeline.swift:595-596`, et la porte est inatteignable*) : premier chunk ≤ 1,5 × `ttft` batch ; audio concaténé identique au batch (graine fixée) ; `generateStreaming` rend en < 50 ms. **macos-gpu** |
| MLX-020@Sources/VoxtralCore/Pipeline/VoxtralPipeline.swift:316 | moyenne | points d'entrée publics STT `:316-473`, TTS, Realtime ; `grep withError` = 0 ; mlx-swift `ErrorHandler.swift:337-347` (`fatalError`) | une erreur MLX termine l'app hôte (FluxForge) au lieu de lever | `try withError { error in … }` aux points d'entrée publics, *amendé :* `try error.check()` après chaque `eval` des boucles (une erreur capturée ne stoppe pas le bloc) | additif (nouveaux cas d'erreur) | S | VÉRIFIÉ (lecture) | *Erreurs MLX converties* — une entrée qui provoque l'erreur P-03 (avant correctif) fait lever `VoxtralError`, processus vivant ; temps dans le bruit, ±5 % (A/B/B/A ; *amendé : « < 1 % » n'est pas mesurable sous le seuil de 5 %*). **macos-gpu** |
| MLX-016@Sources/VoxtralApp/TranscriptionManager.swift:295 | basse | `:285-297` (jamais appelée) | `Int.max` présenté comme « défaut » | supprimer la fonction (A-17) ou restaurer la valeur lue | aucun (app) | S | VÉRIFIÉ | nettoyage de code sans mesure : build de l'app vert. **macos-gpu** (build) |

Justifiés, sans correctif : 22 MLX-002 VOULU (§1.1) ; 3 MLX-015 (tests) ; MLX-014 affichage hérité
(`VoxtralGenerator.swift:221`, `Scripts/VoxtralGenerate.swift:196`).

---

## 8. Retours sur le skill (synthèse ; détail dans la sortie structurée)

1. **`apply.py apply` passe sous silence les occurrences détectées mais non réécrites.** MLX-002 : 28 détectées,
   0 réécrite, message « 0 fichier(s) à modifier ». MLX-003 : les formes `AsyncThrowingStream<…> {`,
   `Task.detached` et `AsyncStream` sont détectées mais non réécrites (test synthétique : 4 détectées, 1 réécrite).
   Correctif : relancer `find()` après `rewrite()` et imprimer les restes en « À TRAITER À LA MAIN ».
2. **MLX-002 mal classé `mechanical`** : la règle `replace` ne couvre pas la forme majoritaire. Le scinder en
   mécanique (`x = x * MLXArray(v, dtype: .float32)`) et `report` (`Float(…)`).
3. **Faux positifs MLX-002** : les constantes qui ne servent qu'à une comparaison (5 ici après vérification croisée :
   4 Sources, dont `VoxtralVoiceSLERP.swift:73`, + 1 Tests).
4. **Faux négatifs** :
   - MLX-002 : `-Float.infinity`, `* MLXArray(scale)` d'une variable, vecteurs précalculés, entrées non castées.
   - MLX-003 : production synchrone.
   - MLX-004 : `struct … : @unchecked Sendable` portant un `MLXArray`, tranches rendues.
   - MLX-006 : signature `prefill:` de mlx-swift-lm `main`.
   - MLX-010 : rendu muet par une pose `= 0`/`= Int.max` morte dans une cible app (la limite « une pose isolée le
     rend muet » est déjà écrite dans la fiche MLX-010 ; le cas nouveau est la pose dans une **cible app** ou du
     **code mort**, que le détecteur devrait exclure).
   - MLX-012 : complétude par `config.json`/`params.json` ou sans index.
   - MLX-014 : `decode([id], …)` sans étiquette.
5. **`syntax_guard` n'est pas une preuve de validité** : une réécriture qui change les types ou la précédence passe
   avec 0 erreur (§6).
6. **Source de MLX-015 périmée** : à `ee673d6`, `loadWeights` passe par `contentsOfDirectory(at:)`
   (`Load.swift:345-353`), une sélection par index (`:313-331`) et `verify: [.all]` (`:404`).
7. **`scan` tronque à 15 par défaut et ne liste pas les occurrences `Tests/`**. `patterns-scan.md` s'arrête sur
   « … 13 de plus » ; `--max 100` liste les 28 (précision ajoutée à la vérification croisée : l'option existe, c'est
   le défaut et l'absence d'avertissement qui posent problème).
8. **Il manque un gabarit « verdicts »** (occurrence · dtype réel · effet · verdict · correctif · renvoi aux audits)
   et une liste de **zones fp32 voulues** (mel/STFT, pertes, codebooks, état d'Euler, SLERP, I/O) avec la vérification
   du recast de sortie.
9. **Cinq patterns candidats** : MLX-016 à MLX-020 (§5).

---

## Annexe — Constats écartés à la vérification croisée

Relecture adverse (2026-09-27) des 13 constats du §7 contre le code à `9392ed1`, l'amont (mlx `1f8e74e`, mlx-swift
`9019419` et tag `0.31.6`, mlx-swift-lm `ee673d6`), la stdlib Swift `release/6.0`, la référence mlx-audio (`main`) et
les listings HF. Règle appliquée : en cas de doute sérieux, écarter ou rétrograder.

| Id | Raison |
|---|---|
| MLX-002@Sources/VoxtralCore/Realtime/VoxtralRealtimeDecoder.swift:168 | **Doublon incomplet de P-60** (`audit-performance-realtime-instruments.md`, § P-60). (1) « Non couvert par les audits sœurs » est faux : P-60 décrit le même défaut, avec les mêmes lignes. (2) La correction proposée (caster `adaScales` et la mel) **ne suffit pas** : les tables `cos`/`sin` RoPE de l'encodeur sont construites en fp32 (`Realtime/VoxtralRealtimeEncoder.swift:76-83`, dont `positions.asType(.float32)` à `:81`, que le rapport citait sans y voir une source fp32). Elles promeuvent q/k, donc la sortie du SDPA et celle de l'encodeur. L'entrée `audio + texte` du décodeur (`VoxtralRealtimeModel.swift:93`) resterait fp32, et la porte « dtype du cache KV couche 1 = fp16 » échouerait avec la correction proposée. (3) « Risque API : aucun » omet que `extractAudioEmbeddings` (public) change de dtype (noté par P-60). Le défaut reste réel et porté par P-60 ; le §4-bis ne garde que l'enseignement sur le détecteur et la preuve mlx-audio (`voxtral_realtime.py:129-134`, cast de `t_cond`), à verser à P-60. |

### Amendements (constats gardés)

| Id | Amendement | Raison |
|---|---|---|
| MLX-002@…/VoxtralStandardLoader.swift:477 | Porte « préfill ±3 % » → « ±5 % » | `references/measurement.md` : un écart < 5 % est du bruit, donc ±3 % n'est pas vérifiable en A/B/B/A. |
| MLX-003@…/VoxtralTTSPipeline.swift:557 | Sévérité haute → **moyenne** ; porte scindée (hunk seul / lot MLX-019) ; correction complétée (`checkCancellation` à `:581`) | S-08, vérifié par l'audit stabilité, est moyen : seul consommateur = démo, ni sortie fausse ni plantage. Après le 1er chunk, le hunk n'accélère rien : un flux annulé délivre encore son tampon (`AsyncStreamBuffer.swift:342-353`, `:488-497`) et la boucle `:581` ne teste pas l'annulation. La porte « après 5 chunks » est donc inatteignable avec le hunk seul. |
| MLX-002@…/VoxtralFlowMatching.swift:64 | Marqué sous-point de P-30, sans fiche propre | La correction de P-30 le prévoit déjà (`audit-performance-tts.md:234`). |
| MLX-004@…/VoxtralTTSPipeline.swift:239 | Preuve ajoutée ; porte déclarée non discriminante | La règle vient de mlx-swift (`MLXArray.md:26-40`). Sans plantage reproduit avant correctif, « 20/20 sans plantage » passerait aussi sans la correction. |
| MLX-012@…/ModelDownloader.swift:316 | Listing Hub complété | Le listing cité omettait `generation_config.json`, `preprocessor_config.json` et `tekken.json`. L'ordre shards < index est confirmé. |
| MLX-017@…/VoxtralStandardLoader.swift:1338 | Risque « aucun » → **comportemental** ; porte « ≥ 1 modèle par famille » → **tous les modèles des registres** | `.allModelKeysSet` fait lever tout pack qui se chargeait jusqu'ici malgré une clé absente. Un modèle par famille ne couvre pas les variantes (bf16, 4, 6, 8 bits ; format Mistral du Realtime). |
| MLX-019@…/VoxtralTTSModeling.swift:580 | Sévérité haute → **moyenne** ; porte « sans warm-up » | Même raison que S-08 et MLX-003. Avec `warmUpText`, le 1er chunk attend 3 s d'audio (`VoxtralTTSPipeline.swift:595-596`). |
| MLX-020@…/VoxtralPipeline.swift:316 | Citation corrigée ; correction complétée ; porte « surcoût < 1 % » → « ±5 % » | Le défaut réel est `fatalError(message)` dans `ErrorHandler.dispatch` (`:337-347`), pas le commentaire `:3-5`. Une erreur capturée par `withError` n'interrompt pas le bloc, d'où `error.check()` après chaque `eval`. Enfin, 1 % est sous le seuil de bruit. |

Gardés sans amendement : MLX-002@…/VoxtralLlama.swift:530 (= P-17 ; `.causal` ignore la fenêtre d'un
`RotatingKVCache` mais ne plante plus, ce que la porte couvre), MLX-006@…/VoxtralModeling.swift:1596 (`:1629` et
l'amont `LLMModel.swift:25-62` confirmés), MLX-010@…/VoxtralPipeline.swift:266 (0 pose dans `VoxtralCore` ; détecteur
muet confirmé, `apply.py:153-157`), MLX-016@…/TranscriptionManager.swift:295 (`aggressiveMemoryCleanup` : 0 appel ;
`allocator.cpp:63-65` confirmé).

Verdicts d'occurrence : `VoxtralVoiceSLERP.swift:73` reclassé de VOULU en FAUX POSITIF (comparaison). Les 28 + 1 autres
verdicts, le hunk MLX-003 (garde syntaxique 0 / 0, rejouée sur copie) et le dry-run MLX-002 (« 0 fichier(s) »,
rejoué) sont confirmés.
