# Piège — la clé `"mode"` de `quantization` : refusée par le STT, ignorée par le TTS et le Realtime

> Date : 2026-09-27 (audit à `9392ed1`). Statut : VÉRIFIÉ en lecture (code et `config.json` du Hub) ; chargement
> effectif des packs tiers À MESURER (K-8). Sources : modeles-2026-09.md M-02 ; PLAN.md §1.

**Symptôme** :
- STT : les packs publiés en 2026 ne se chargent pas. Leur `config.json` porte `"mode": "affine"`
  (`aufklarer/Voxtral-Mini-3B-2507-MLX-8bit`, `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder`,
  `MarkusKaemmerer/Voxtral-Small-24B-2507-{4bit,8bit}-dense-encoder`) : le décodage de toute la configuration lève
  `typeMismatch`. Seuls les packs de 2025 (mzbac, VincentGOURBIN, sans `mode`) passent.
- TTS et Realtime : un pack d'un autre mode (`mxfp4`, `nvfp4`) serait chargé comme de l'affine, avec
  `verify: .none` : sortie fausse **sans erreur** (conséquence déduite du code par M-02, non observée).

**Cause** :
- STT : `QuantizationValue` n'accepte que `Bool`, `Int` ou `{group_size, bits}` ; la chaîne `"affine"` n'entre dans
  aucun cas (`Sources/VoxtralCore/Utils/VoxtralStandardLoader.swift:87-114`, décodage `:1293-1295`) ; même décodé,
  le mode serait écrasé par `.affine` (`:1245`).
- TTS : `quantConfig.mode == "affine" ? .affine : .affine`
  (`Sources/VoxtralCore/TTS/VoxtralTTSModelLoading.swift:58`).
- Realtime : `RealtimeQuantizationConfig` n'a pas de champ `mode`
  (`Sources/VoxtralCore/Realtime/VoxtralRealtimeConfiguration.swift:101-109`).

**Correctif** (à faire, K-8) : décoder `quantization` avec `MLXLMCommon.BaseConfiguration` (mode, per-layer,
`false`, `quant_method`), passer `(groupSize, bits, mode)` à `quantize(model:filter:)`, refuser explicitement tout
mode autre qu'`affine` ; TTS : repli sur `quantization_config`. Porte : 5/5 fixtures `config.json` décodées (dont
aufklarer `mode`, Markus per-layer + `mode`, `mxfp4` synthétique → erreur explicite).

**Règle** : un chargeur lit la configuration de quantification comme l'amont (mode compris) et refuse ce qu'il ne
sait pas calculer, au lieu de retomber en silence sur un mode par défaut.
