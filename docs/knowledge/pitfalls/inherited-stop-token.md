# Piège — jeton d'arrêt hérité d'un autre tokenizer : `32000` = « ␣Capital » en Tekken

> Date : 2026-09-27 (audit à `9392ed1`). Statut : mécanisme VÉRIFIÉ en lecture ; effet sur un vrai audio À MESURER
> (K-4). Sources : audit-stabilite.md S-01 ; PLAN.md §1.

**Symptôme** (attendu, non observé sur un audio réel) : toute transcription ou réponse de chat STT qui contient
« Capital » précédé d'une espace s'arrête après ce mot, **sans erreur**, sur les deux backends (MLX et hybride).

**Cause** :
- La génération STT s'arrête sur `[2, 4, 32000]` (`Sources/VoxtralCore/VoxtralModeling.swift:1124`, `break`
  `:1262` ; même liste au chemin hybride `:1313`, `:1438`). `32000` est l'EOS de l'ancien tokenizer Llama/Mistral v1.
- En Tekken, id = rang + 1 000 (ids spéciaux) : l'id 32000 est le rang 31000 de `tekken.json`, soit le jeton de
  texte « ␣Capital » (`Sources/VoxtralCore/VoxtralComponents.swift:155-176`).
- Le Realtime n'a pas le défaut : il lit `config.eosTokenId`
  (`Sources/VoxtralCore/Realtime/VoxtralRealtimeModel.swift:120`).

**Correctif** (à faire, K-4) : dériver les jetons d'arrêt du tokenizer et de `generation_config.json`, supprimer
`32000`. Porte : test `StopTokenTests` (chaque jeton d'arrêt est un id spécial, < 1 000), rouge avec la liste
actuelle ; le clip « Capital Gains and Capital One are two different things. The capital of France is Paris. »
transcrit au-delà du premier « ␣Capital » ; greedy identique avant/après sur C-court EN, C-court FR et C-moyen EN.

**Règle** : les jetons d'arrêt se lisent dans le tokenizer et la configuration du modèle chargé ; un test exige que
chacun soit un id spécial.
