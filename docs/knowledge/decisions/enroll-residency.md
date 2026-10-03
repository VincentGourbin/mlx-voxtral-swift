# Enrollment: does keeping the TTS LLM resident cost memory? (K-37, 2026-10-03)

**Question** (K-37 → K-64, `enroll-lean`): when a voice is enrolled after a synthesis in the same pipeline, the LLM
(26 layers) is materialized and stays resident, while `voxtral enroll` alone never materializes it (lazy loading).
Is the difference worth code (freeing the LLM before enrolling)?

**Measure** (M3 Max 96 GB, Release `fc8af1f7`, clean tree; `bench enroll --epochs 200 --seed 7 --passes 2`,
reference `docs/examples/clone_fr.wav`; lines in `BENCHMARKS.md`, 2026-10-03):

| Pack | (i) enroll alone: peak footprint | (ii) after a synthesis | (ii) − (i) | `epoch_ms_p50` (i) / (ii) |
|---|---|---|---|---|
| `tts-4b-6bit` | 2 670 MB | 6 147 MB | **+130 %** | 62.3 / 63.7 ms |
| `tts-4b-mlx` (bf16) | 2 674 MB | 11 593 MB | **+334 %** | 64.5 (redo) / 61.6 ms |

A/A on `epoch_ms_p50` ≤ 3 % for the four configurations (0.98 / 1.04 / 2.27 / 0.63 %). The enrolled voice is identical
across packs and scenarios (`out_sha256` `f18162fc…`, final loss 1.6647): enrollment optimizes codes through the
codec decoder, which is bf16 in every pack, and never runs the LLM.

**Decision**: (ii) − (i) ≥ 5 % → the residency part of K-64 is **kept**. Freeing (or never materializing) the LLM
before enrolling saves 3.5 GB (6-bit) to 8.9 GB (bf16) with no effect on the result, since the LLM does not take part
in enrollment. Speed is the same in both scenarios (≈ 62–65 ms per epoch).
