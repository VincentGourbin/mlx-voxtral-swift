# BENCHMARKS — lignes de mesure brutes

**Règle : ce fichier n'est jamais édité.** Une mesure = une nouvelle ligne ajoutée à la fin, recopiée telle quelle
depuis la sortie de `VoxtralCLI bench` (`BENCH {…}`) ou `VoxtralCLI eval` (`EVAL {…}`). Aucune ligne existante n'est
modifiée ni supprimée : une correction est une nouvelle ligne, et son explication va dans
[`docs/knowledge/log.md`](docs/knowledge/log.md).

Protocole, corpus et définition de chaque métrique : [`docs/Benchmarks.md`](docs/Benchmarks.md) (glossaire, §4).
Règles de mesure : [`PLAN.md`](docs/audit/2026-09-27/PLAN.md) §0.

## Colonnes de la ligne `BENCH` (instrument K-32)

Spécification : audit-performance-realtime-instruments.md P-79, lignes 823-837 ; schéma JSON `docs/bench.schema.json`
livré par la fiche K-32 (pas encore présent).

| Groupe | Champs |
|---|---|
| Communs | `date`, `commit`, `dirty`, `build` (Release/Debug), `mlx_swift`, `mlx_swift_lm`, `mlx_profiler` (révisions **résolues**), `chip`, `ram_gb`, `macos`, `power`, `top_process`, `pipeline`, `model`, `pack_sha256`, `profile`, `input`, `input_s`, `seed`, `pass`, `tag`, `warm` |
| Par phase | `phases_ms` (chargement, matérialisation, audio/mel, encodage, préfill, décodage, codec, post), `peak_mlx_mb`, `peak_footprint_mb` |
| Débit | `steps`, `step_ms_p50`, `step_ms_p90`, `ttft_ms`, `rtf`, `weights_bw_gbps` |
| Parité | `out_sha256`, `wer` (si une référence existe) |
| Realtime | `pad_fraction`, `encode_ms_per_audio_s`, `first_text_token_ms`, `truncated` |
| TTS | `frames`, `ttfa_ms` |
| Enrôlement | `epochs`, `epoch_ms_p50`, `final_loss` |

Une ligne sans révision résolue de `mlx_swift_lm` n'est pas une référence : la dépendance suit la branche `main`
(`Package.swift:52`).

## Lignes

Aucune ligne : la baseline est À MESURER (PLAN.md §2 ; fiches K-34 à K-37).
