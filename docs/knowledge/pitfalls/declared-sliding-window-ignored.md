# Piège — fenêtre glissante déclarée dans la configuration mais jamais appliquée (Realtime)

> Date : 2026-09-27 (audit à `9392ed1`). Statut : divergence VÉRIFIÉE en lecture (code et référence mlx-audio) ;
> effet sur le WER, le temps et la mémoire À MESURER (K-13). Sources : audit-performance-realtime-instruments.md
> P-62, P-63 ; PLAN.md §1.

**Symptôme** : aucun sur un clip court. Au-delà de la fenêtre, la sortie diverge de la référence sans erreur :
- encodeur (fenêtre 750 positions = 15 s d'audio) : chaque position voit des distances jamais vues à
  l'entraînement ; coût de l'attention pleine ×1,8 à 5 min, ×12 à 1 h par rapport au coût fenêtré (calcul, P-62) ;
- décodeur (fenêtre 8 192 pas ≈ 10 min 55 s) : cache KV sans borne, 4,46 Gio en bf16 à 1 h (45 000 pas) relus à
  chaque pas, contre 832 Mio une fois la fenêtre posée (calcul, P-63). Aujourd'hui masqué par la troncature à
  4 096 pas (P-64).
- Le Realtime sert de juge ASR à `docs/zerovoice_benchmark.md` sur des clips de 7,6 à 20,8 s : sa validité au-delà
  de 15 s n'est pas établie (P-62, P-78).

**Cause** :
- Encodeur : `encodeFull` applique `.causal` à toute la séquence ; `slidingWindow` n'est lu que dans l'`init` et
  l'attention bascule en `.none` dès qu'un cache est fourni sans masque
  (`Sources/VoxtralCore/Realtime/VoxtralRealtimeEncoder.swift:151-159`, `:320-342` ; commentaire « for audio
  within sliding window », `:327`).
- Décodeur : `createCache()` crée des `KVCacheSimple`
  (`Sources/VoxtralCore/Realtime/VoxtralRealtimeDecoder.swift:191-194`).
- Le port Swift n'a gardé que la branche courte de la référence : mlx-audio encode par tranches de 750 avec
  `RotatingKVCache(max_size=750)` (`encoder.py:188-219`) et pose `RotatingKVCache(max_size=sliding_window)` au
  décodeur (`decoder.py:226-229`).

**Correctif** (à faire, K-13) : encodeur par tranches causales avec `RotatingKVCache(maxSize: 750)` par couche,
RoPE à la position absolue du début de tranche et masque explicite ; décodeur `RotatingKVCache(8192)`. Porte :
embeddings identiques (L2 relative < 1e-3) à `encodeFull` sur un clip ≤ 15 s ; C-xlong : pic stable après 8 192 pas
(±5 %) et ms/pas au-delà de 8 192 = ms/pas à 8 000 (±5 %).

**Règle** : toute fenêtre déclarée par la configuration (`sliding_window`) est appliquée ou refusée explicitement ;
un test compare à la référence **au-delà** de la fenêtre, pas seulement en deçà.
