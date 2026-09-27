# Piège — tête liée calculée par `matmul` brut avec une entrée fp32 : la table est recopiée en fp32 à chaque pas

> Date : 2026-09-27 (audit à `9392ed1`). Statut : mécanisme VÉRIFIÉ (code MLX) ; gain À MESURER (K-38, K-46).
> Sources : audit-performance-realtime-instruments.md P-60, P-61, §3 bis ; PLAN.md §1.

**Symptôme** (issues #23 et #25, 2026-04-11, `realtime-4b-4bit`, « en session », machine non notée) : 33,7 ms par
pas de décodage, jugés « raisonnables » à la fermeture de #23 ; +3 980 Mio pendant un préfill de 8 jetons ; pic
process 7 949 Mo. Le budget calculé (couches 1 623 + embeddings 768 + copie fp32 1 536 + Ada ≈ 10 = 3 937 Mio) est
compatible à 1,1 % près avec les +3 980 Mio : indice, pas preuve (instantané `activeMemory` pris juste après
`eval`).

**Cause** :
- Les logits sont `MLX.matmul(h, tokEmbeddings.weight.transposed())`
  (`Sources/VoxtralCore/Realtime/VoxtralRealtimeDecoder.swift:187-189`). `h` est fp32 parce que tout le chemin
  Realtime calcule en fp32 (P-60) ; `matmul` promeut alors la table 131 072 × 3 072 par `astype(W, float32)`
  (MLX `ops.cpp:3069-3082` @`ce45c52`) : ≈ 1,5 Gio écrits puis relus **à chaque pas** (calcul depuis la forme).
- La table n'est jamais quantifiée : le prédicat de quantification saute `tok_embeddings`
  (`Sources/VoxtralCore/Realtime/VoxtralRealtimeModelLoading.swift:39`). Même sans la fuite fp32, elle pèse 32 %
  du trafic de poids du pack « 4 bits » (calcul, P-61).
- Trafic par pas estimé : ≈ 5,8 Go, dont 4,03 Go pour la tête (calcul, P-61).

**Correctif** (à faire) :
- K-38 : calcul dans le dtype du modèle (mel, tables RoPE, `tCond`/`adaScale`), ce qui supprime la copie ; porte :
  `step_ms_p50` ≥ −30 % et pic `phys_footprint` −≥ 1 Go sur le pack 4 bits (A/B/B/A), audit de dtype
  (`VOXTRAL_DTYPE_AUDIT=1`) : logits dans le dtype du modèle.
- K-46 : tête quantifiée via `QuantizedEmbedding.asLinear` (8 bits en `fast`, 4 bits en `lean`), comme la référence
  mlx-audio (`decoder.py:266`, prédicat `voxtral_realtime.py:572`).

**Règle** : une tête liée se calcule par `asLinear` dans le dtype des poids ; toute entrée fp32 d'un `matmul` sur une
table 16 bits est une copie fp32 de la table, à détecter par un audit de dtype des logits.
