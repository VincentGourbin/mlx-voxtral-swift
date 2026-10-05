---
okf_version: "0.1"
---

# Base de connaissances — mlx-voxtral-swift

Conclusions durables du projet (bundle OKF, structure du skill `mlx-swift-audit`, `references/knowledge-structure.md`
§5.1). [`log.md`](log.md) porte l'historique horodaté et fait foi pour les chiffres cités ailleurs. Consignes de
build, de test et de mesure : [`CLAUDE.md`](../../CLAUDE.md). Plan d'action en cours :
[`PLAN.md`](../audit/2026-09-27/PLAN.md) de l'audit du 2026-09-27.

## Benchmarks
- [`BENCHMARKS.md`](../../BENCHMARKS.md) — lignes brutes `BENCH`/`EVAL`, jamais éditées ; aucune ligne à ce jour
  (baseline À MESURER, fiches K-32 à K-37).
- [`docs/Benchmarks.md`](../Benchmarks.md) — protocole (A/B/B/A, refroidissement 120 s, seuil 5 %), corpus
  C-court à C-30min, glossaire des métriques (RTF = génération ÷ audio ; TTFT-frame ≠ TTFA).

## Decisions
- [Small 4 bits sur 32 Go](decisions/small-32gb.md) — supporté avec `--cache-limit-mb 2048` (19,0 Go sur 8 min),
  non supporté sans (37,3 Go) ; K-34.
- [Baseline STT M3 Max, 2026-10](benchmarks/m3max-stt-baseline-2026-10.md) — Mini 8/4 bits, bf16 et Small 4 bits ×
  `.mlx`/`.auto` × 4 clips, WER, occupation GPU du préfill, chat ; K-34.
- [Conclusions #23-#25 caduques](decisions/realtime-diagnostics-23-25.md) — « 0 % GPU », « 49 % systémique » et
  « 21 tok/s raisonnable » sont des artefacts d'instrument (phases imbriquées, lectures instantanées) ; re-mesure
  par K-36.

## Pitfalls
- [Stream à production synchrone](pitfalls/async-stream-synchronous-build.md) (V-P5) — le « streaming » TTS génère
  tout avant le premier chunk ; producteur jamais annulé (K-12).
- [Même % GPU partout](pitfalls/same-gpu-percent-instrument-artifact.md) (V-P8) — 48-49 % sur trois opérations
  différentes et 0 % sur les 23,89 s de la phase « Realtime Generation » (qui englobe encodage et préfill) :
  signature d'instrument (K-34, K-36).
- [Tête liée recopiée en fp32](pitfalls/tied-head-fp32-copy.md) — Realtime : ≈ 1,5 Gio de copie fp32 de la table
  131 072 × 3 072 à chaque pas (calcul ; K-38, K-46).
- [Fenêtre glissante ignorée](pitfalls/declared-sliding-window-ignored.md) — Realtime : fenêtres 750 (encodeur) et
  8 192 (décodeur) déclarées, jamais appliquées (K-13).
- [Clé `mode` de quantification](pitfalls/quantization-mode-key.md) — les packs STT de 2026 (`"mode": "affine"`)
  sont refusés ; TTS et Realtime ignorent le mode (K-8).
- [Jeton d'arrêt hérité](pitfalls/inherited-stop-token.md) — `32000` = « ␣Capital » en Tekken : la transcription
  s'arrête sur ce mot (K-4).

Les seize pièges V-P1 à V-P16 de l'audit sont listés dans
[`faits-et-actions.md`](../audit/2026-09-27/faits-et-actions.md) §5.3 ; seuls les six ci-dessus ont une fiche ici.

## Investigations
Rien pour l'instant. Rapports d'audit du 2026-09-27 : [`README.md`](../audit/2026-09-27/README.md).
