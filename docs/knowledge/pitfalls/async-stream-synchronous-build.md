# Piège — `AsyncThrowingStream` dont la closure de construction produit tout (V-P5)

> Date : 2026-09-27 (audit à `9392ed1`). Statut : VÉRIFIÉ en lecture (sémantique de la stdlib) ; correctif à faire
> (fiche K-12). Sources : faits-et-actions.md §5.3 V-P5, §2.1 ; audit-stabilite.md S-08 ; pattern MLX-003.

**Symptôme** : le « streaming » TTS ne streame pas. Le premier chunk n'arrive qu'après la génération complète du
texte ; le « TTFT » affiché par la démo (`chunk.elapsed` du premier chunk) vaut donc à peu près la génération
complète. L'optimisation « premier chunk de 3 frames » (`0be05af`) est sans effet côté consommateur. Après `stop()`
dans la démo, la génération continue jusqu'à la fin de l'audio (EOA), au pire jusqu'à `maxFrames` = 2 500 frames
(200 s d'audio), ou ce sont les décodages des chunks restants qui continuent ; la pipeline reste `.synthesizing` et
refuse tout nouvel appel. Aucune durée n'est mesurée : l'effet
est établi par lecture du code (S-08).

**Cause** : `AsyncThrowingStream.init(_:bufferingPolicy:_:)` exécute sa closure de construction immédiatement et de
façon synchrone. Toute la boucle de génération est écrite dans cette closure
(`Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:569-685`). La `Task` de la pipeline n'a pas
d'`onTermination` (`Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:556-671`, pattern MLX-003) : annuler
le consommateur n'annule pas le producteur, et `Task.isCancelled` (`VoxtralTTSModeling.swift:622`) lit toujours
`false`.

**Correctif** (à faire, K-12) : produire dans une `Task` (ou `AsyncThrowingStream.makeStream()` + `Task`),
`continuation.onTermination = { _ in task.cancel() }`, `try Task.checkCancellation()` dans la boucle, état remis à
`.ready` sur annulation. Porte de K-12 : `generateStreaming` rend en < 50 ms ; premier chunk ≤ 1,5 × `ttft` du
batch ; annulation après 5 chunks → pipeline `.ready` en < 1 s ; audio concaténé identique au batch à graine fixée.

**Règle** : un stream ne produit jamais dans sa closure de construction ; sa `Task` est annulée par sa terminaison,
et sa latence se mesure en **TTFA** côté consommateur, pas en TTFT-frame (glossaire :
[`docs/Benchmarks.md`](../../Benchmarks.md) §4).
