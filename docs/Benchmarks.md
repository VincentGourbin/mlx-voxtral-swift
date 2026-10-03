# Benchmarks — protocole, corpus et glossaire des métriques

Guide de lecture de [`BENCHMARKS.md`](../BENCHMARKS.md) (lignes brutes, jamais éditées). Créé le 2026-09-27 par
la fiche K-17 à partir du plan d'audit [`PLAN.md`](audit/2026-09-27/PLAN.md) (§0 règles, §2 baseline, §5 commandes).

**État** : aucune mesure de référence n'existe. Toute la baseline est À MESURER (PLAN.md §2) : instrument
`VoxtralCLI bench` (fiche K-32), qualité `VoxtralCLI eval` (K-33), puis baselines STT et chat (K-34), TTS (K-35),
Realtime (K-36) et enrôlement (K-37). Les chiffres publiés avant ce plan sont « en session » (§6).

## 1. Ce qui est mesuré

| Chemin | Métriques (voir §4) | Fiche |
|---|---|---|
| STT Mini / Small, backends `.mlx` et `.auto` | préfill tok/s, décodage tok/s, TTFT, pic phys_footprint, WER | K-34 |
| Chat (`VoxtralPipeline.chat`) | TTFT par question, décodage tok/s | K-34 |
| Realtime | ms par pas (p50, p90), TTFT, pic phys_footprint, WER | K-36 |
| TTS (voix prédéfinie ou clonée, batch ou streaming) | fps, TTFT-frame, TTFA, RTF, pic phys_footprint, couverture ASR | K-35 |
| Enrôlement | durée par époque, perte finale, pic phys_footprint | K-37 |

Source : PLAN.md §2 (tableaux de baseline).

## 2. Protocole (PLAN.md §0)

- **Binaire Release** (`xcodebuild -scheme VoxtralCLI -configuration Release`, commande complète dans
  [`CLAUDE.md`](../CLAUDE.md)). Tests en Debug, mesures en Release : une campagne XCTest chronométrée n'est pas une
  référence, sauf build Release avec `ENABLE_TESTABILITY=YES`.
- **Instrument dans le chemin de la bibliothèque** (piège 33) : `VoxtralCLI bench` (K-32) passe par les pipelines
  publics. L'ancien `VoxtralBenchmark` (conversions Float16 sur données aléatoires, hors du chemin
  des consommateurs) est retiré (ASK-26 = B, K-32) : ses chiffres n'étaient pas des mesures de Voxtral.
- **Machine prête** : `machine-check.sh` sans ligne `KO`, refroidissement de 120 s avant chaque point, aucune autre
  charge GPU.
- **Un levier par comparaison**, ordre **A/B/B/A** (deux passes par variante, une requête d'amorçage exclue).
- Une différence n'est lue que si elle dépasse la **dispersion A/A** ; l'instrument lui-même se valide par une A/A
  ≤ 3 % (porte de K-32). **Gain < 5 % = bruit : le levier est retiré.**
- **Une ligne JSON par mesure** (`BENCH {…}`), recopiée telle quelle dans `BENCHMARKS.md`, avec les révisions
  **résolues** de mlx-swift, mlx-swift-lm (dépendance sur `main`) et swift-mlx-profiler.
- **Parité sur le checkpoint réel**, par chemin : STT et Realtime = transcription greedy identique ou WER dans la
  tolérance de la fiche (ASK-11) ; TTS = bit-identique à graine fixée pour les leviers exacts, parité forcée par
  l'enseignant et couverture ASR pour les leviers numériques ; enrôlement = codes identiques à graine fixée.
- **Une seule convention de RTF** : génération ÷ audio (§4). **TTFT-frame ≠ TTFA** (§4).

## 3. Corpus (PLAN.md §5)

| Clip | Fichier | Durée | Source |
|---|---|---|---|
| C-court EN / FR | `docs/examples/fluxforge_short_{en,fr}_6bit.wav` | 5,0 / 4,8 s | PLAN.md:313 |
| C-moyen EN / FR | `docs/examples/fluxforge_long_{en,fr}_6bit.wav` | 167,0 / 173,8 s | PLAN.md:313 |
| C-moyen exact EN / FR (qualité, K-33) | `docs/eval/clips/c_moyen_{en,fr}.wav`, texte `docs/eval/refs/` | 146,1 / 130,9 s | `docs/eval/README.md` |
| 20 s EN / FR, ES × 3 (K-33) | `docs/eval/clips/` | 20,6 / 17,2 s ; 8–10 s | `docs/eval/corpus.json` |
| C-long exact (K-33) | 2 × (C-moyen exact EN + FR), 16 kHz mono, hors dépôt | 9 min 14 s | `docs/eval/README.md` |
| C-long | 2 × (C-moyen EN + FR), 16 kHz mono | ≈ 11 min 22 s | PLAN.md:318 |
| C-xlong | 3 × (C-moyen EN + FR) | ≈ 17 min | PLAN.md:319 |
| C-30min | C-long bouclé, tronqué | 30 min | PLAN.md:320 |

- Les anciens C-moyen ont une référence condensée (163 / 202 mots pour 413 / 451 dits) : ils restent les témoins de
  performance ; la qualité (WER) se mesure sur les clips à texte exact (K-33).
- Parole synthétique (sorties TTS du dépôt) : biais à noter (audit-performance-realtime-instruments.md P-79).
- Textes de référence : [`tts_benchmark.md`](tts_benchmark.md), section « Full test texts ». Ils font 163 mots EN et
  202 mots FR pour 167 et 174 s d'audio (« ~350 words » annoncés) : peut-être abrégés, à contrôler par K-33
  (PLAN.md §1).
- Dernières phrases (portes « rien n'est coupé ») : EN « No data sent to the cloud. » · FR « Aucune donnee envoyee
  dans le cloud. » (PLAN.md:323).
- SHA-256 des 4 clips : à noter ici par la fiche K-20.

```bash
mkdir -p .local-runs/corpus .local-runs/bench.noindex
for n in 2 3; do
  for i in $(seq $n); do echo "file '$PWD/docs/examples/fluxforge_long_en_6bit.wav'"; echo "file '$PWD/docs/examples/fluxforge_long_fr_6bit.wav'"; done \
    > .local-runs/corpus/list_$n.txt
done
ffmpeg -y -f concat -safe 0 -i .local-runs/corpus/list_2.txt -ar 16000 -ac 1 .local-runs/corpus/c_long.wav    # ≈ 11 min 22 s
ffmpeg -y -f concat -safe 0 -i .local-runs/corpus/list_3.txt -ar 16000 -ac 1 .local-runs/corpus/c_xlong.wav   # ≈ 17 min
ffmpeg -y -stream_loop 5 -i .local-runs/corpus/c_long.wav -t 1800 .local-runs/corpus/c_30min.wav               # 30 min
awk 'f && /^> /{sub(/^> /,""); print >> (".local-runs/corpus/long_" f ".txt")} /^### Long FR/{f="fr"} /^### Long EN/{f="en"}' docs/tts_benchmark.md
```
Les sorties vont sous `.local-runs/bench.noindex/`, hors index Spotlight (piège 19, PLAN.md §4).

## 4. Glossaire des métriques

Une définition par métrique. Les autres documents du dépôt renvoient ici. Champs JSON : spécification de
l'instrument dans audit-performance-realtime-instruments.md P-79, lignes 823-837 (schéma livré par K-32).

| Métrique | Champ JSON | Définition | Source |
|---|---|---|---|
| **RTF** | `rtf` | Durée de génération ÷ durée de l'audio. **< 1 = plus rapide que le temps réel.** Le « RT factor » affiché par `VoxtralCLI profile` est l'inverse (audio ÷ génération) : il n'est pas utilisé ici. | `Sources/VoxtralCore/TTS/VoxtralTTSProcessor.swift:30-33` ; `Sources/VoxtralTranscriptionTest/ProfileCommand.swift:274` ; faits-et-actions.md §2.1 |
| **TTFT-frame** (TTS) | `ttft_ms` des lignes TTS | Délai entre l'entrée dans `generate` (tokenisation et préfill compris) et le premier frame de codes évalué (`eval(codes)` au pas 0). Exclut le préfill des trames de voix quand il vient du cache par voix (voix prédéfinies ; streaming avec `voiceKey`), l'inclut sinon (voix clonée en batch ; toute mesure antérieure au cache, `f4fd21c`, 2026-07-10) : noter le cas dans la ligne. Ce n'est pas un délai perçu par l'utilisateur. | `Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:441`, `:496-499` ; `Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:210`, `:333-340`, `:512` ; faits-et-actions.md §2.1 |
| **TTFA** (TTS) | `ttfa_ms` | Délai entre l'appel de l'API et le premier échantillon audio **reçu par le consommateur** (premier chunk en streaming). | PLAN.md §0 ; faits-et-actions.md FA-04, V-P5 |
| **TTFT** (STT, chat, Realtime) | `ttft_ms`, `first_text_token_ms` (Realtime) | Délai entre l'appel et le premier jeton de texte émis. Le chargement du modèle est une phase à part de `phases_ms`. | PLAN.md §2 ; P-79 |
| **fps** (TTS) | dérivé de `frames` | Frames de codes générés ÷ durée de génération (s). | faits-et-actions.md FV-30 (« up to 19 fps » du README = 2 266 / 120,61 s) |
| **ms par pas** | `step_ms_p50`, `step_ms_p90` | Durée d'un pas de décodage autorégressif (un forward : un jeton STT, un frame TTS, une trame Realtime), en médiane et 90ᵉ centile. Les pseudo-pas sans forward sont exclus. En Realtime, 1 pas = 80 ms d'audio : le budget temps réel est de 80 ms par pas. | P-79 ; P-75 ; audit-performance-realtime-instruments.md §3 (T21) et §6 |
| **Préfill tok/s** | dérivé de `phases_ms` | Jetons de l'invite ÷ durée de la phase de préfill. | PLAN.md §2 |
| **Décodage tok/s** | dérivé de `steps`, `phases_ms` | Jetons générés ÷ durée de la phase de décodage (encodage et préfill exclus). Le « tok/s » du README divise par le temps total : il n'est pas comparable. | faits-et-actions.md §2.1 |
| **Pic phys_footprint** | `peak_footprint_mb` | Maximum de `task_info(TASK_VM_INFO).phys_footprint`, échantillonné toutes les 5 ms. Avant K-32 : ligne « peak memory footprint » de `/usr/bin/time -l`. | P-79 ; PLAN.md §5 |
| **Pic MLX par phase** | `peak_mlx_mb` | `Memory.peakMemory`, remis à zéro par l'instrument au début de chaque phase. Aujourd'hui la bibliothèque le remet à zéro en cours de run : un pic MLX lu avant K-32 ne voit pas le pic du run. | P-79 ; P-77 |
| **Bande passante des poids** | `weights_bw_gbps` | Octets de poids lus par pas × pas par seconde. | P-79 |
| **Occupation GPU** | — (trace à part) | Fraction de temps GPU occupé sur un intervalle, par Metal System Trace ou swift-mlx-profiler ≥ 1.5 en `.ioReportResidency`. Une lecture instantanée de « Device Utilization % » n'est pas une mesure. | P-73 ; P-74 ; [piège](knowledge/pitfalls/same-gpu-percent-instrument-artifact.md) |
| **WER** | `wer` | Taux d'erreur de mots normalisé (casse, ponctuation, accents ; la référence FR est sans accents). Outil : K-33. | PLAN.md §0 |
| **Couverture ASR** (TTS) | ligne `EVAL` (K-33) | Part des mots du texte source retrouvés dans la transcription d'un juge ASR commun (aller-retour TTS → STT). | faits-et-actions.md V-T15 |
| **Durée par époque** (enrôlement) | `epoch_ms_p50` | Durée médiane d'une époque d'optimisation des codes. | P-79 |
| **Perte finale** (enrôlement) | `final_loss` | Perte d'enrôlement rapportée en fin d'optimisation, à graine fixée. | P-79 |

Rapports cités : [`faits-et-actions.md`](audit/2026-09-27/faits-et-actions.md),
[`audit-performance-realtime-instruments.md`](audit/2026-09-27/audit-performance-realtime-instruments.md) (P-73 à
P-79).

## 5. Reproduire une mesure et contribuer une ligne

Commandes de référence (PLAN.md §5, disponibles après K-32 et K-33) :

```bash
$CLI bench stt --model mini-3b-8bit --backend mlx --input docs/examples/fluxforge_long_en_6bit.wav --language en \
  --passes 2 --warmup 1 --cooldown 120 --tag A
$CLI eval stt --model mini-3b-8bit --backend mlx --corpus docs/eval/corpus.json
VOXTRAL_DTYPE_AUDIT=1 $CLI bench realtime --model realtime-4b-4bit --input docs/examples/fluxforge_short_en_6bit.wav --passes 1
```

1. Build Release et `machine-check.sh` (voir [`CLAUDE.md`](../CLAUDE.md)).
2. Lancer la mesure ; recopier **telle quelle** chaque ligne `BENCH`/`EVAL` à la fin de `BENCHMARKS.md`. Ne jamais
   modifier une ligne existante : une correction est une nouvelle ligne.
3. Ajouter l'entrée datée dans [`docs/knowledge/log.md`](knowledge/log.md) (et dans `PLAN.md` §7 pour une fiche).

## 6. Chiffres publiés avant le 2026-09-27

Tous « en session » : révisions anciennes, passages froids, pas d'A/B/B/A, deux conventions de RTF, TTFT ≠ TTFA.
Aucun n'est une référence. Inventaire avec leur source : faits-et-actions.md §2.3 à §2.6 (FV-10 à FV-57) ; ce qui
doit être re-mesuré : §2.8.
