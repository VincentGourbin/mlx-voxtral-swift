# Référence Realtime figée (mlx-audio)

Sorties de mlx-audio capturées **une fois** pour K-13 (ASK-12 = A sous condition : référence seulement, aucune
dépendance Python dans le code, les tests ou le build).

- mlx-audio 0.5.7 (PyPI, tag `v0.5.7` = `94c7716212b2228f178d2f9c7619a591fd1b0b78`), mlx 0.32.3, venv temporaire
  hors du dépôt, 2026-10-01.
- Modèle : `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` (dossier local), `Model.generate(wav, max_tokens=8192)`,
  greedy, délai par défaut.
- Clips : C-moyen `docs/examples/fluxforge_long_{en,fr}_6bit.wav`.

| Fichier | SHA-256 |
|---|---|
| `mlx-audio_c-moyen_en.txt` | `1a3692f948acb739539aaed28bbfb4b6a6cb7bcdd94d6231b5a31506157c7f27` |
| `mlx-audio_c-moyen_fr.txt` | `7b8c4235046b77ff3a22ef07975440df1e6e5008acb9f8e31d719084f1725240` |

WER (K-13, pour K-33) : normalisation NFKD → ASCII, minuscules, ponctuation → espace, octets NUL retirés, puis
`jiwer.wer(ref, hyp)` (jiwer 3.0.4).
