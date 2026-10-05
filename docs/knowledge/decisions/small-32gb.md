# Décision — Small 4 bits sur un Mac 32 Go : supporté avec un plafond de cache, non supporté sans

**Contexte** (K-34, porte : « small-4bit, 8 min d'audio : pic `phys_footprint` mesuré ≤ 24 Go (marge d'un Mac 32 Go),
sinon verdict « non supporté sur 32 Go » ») : les packs Small (≈ 14 Go de poids en 4 bits) étaient annoncés pour les
Mac 32 Go sans mesure.

**Mesure** (2026-10-05, M3 Max 96 Go, `3a0cf13b`, arbre propre ; `BENCHMARKS.md` §« 2026-10-05 — K-34 », bloc
SMALL32) : `bench stt --model small-4bit --backend mlx --input c_8min.wav` (8 min = `ffmpeg -t 480` de C-long exact),
une passe sans amorçage :
- sans plafond de cache : `SMALL32 peak_footprint_mb=37312.6` (> 24 576) → **non supporté sur 32 Go** ;
- avec `--cache-limit-mb 2048` (K-52) : `peak_footprint_mb=19012.9` (≤ 24 576) → **supporté**.

**Décision** :
- Small 4 bits sur 32 Go n'est supporté qu'avec un plafond du cache MLX : `VoxtralPipeline` avec la limite de cache de
  K-52 (2 048 Mo mesuré). Sans plafond, le cache tampon de MLX fait monter le pic à 37,3 Go sur 8 min d'audio.
- Le défaut public n'est pas changé par cette décision (réservé à Vincent) : appliquer le plafond par défaut pour
  Small sur les machines ≤ 32 Go est une question ouverte (ASK à poser au replanning #590).

**Non retenu** : un pic mesuré sur 96 Go de mémoire vaut une borne haute de ce que le processus demande ; la pression
mémoire réelle d'un Mac 32 Go (compression, swap) n'est pas mesurée ici.

Source : K-34 (#587) ; `docs/knowledge/benchmarks/m3max-stt-baseline-2026-10.md`.
