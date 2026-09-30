# CLAUDE.md — mlx-voxtral-swift

Consignes d'agent, tirées du plan d'audit du 2026-09-27 ([`PLAN.md`](docs/audit/2026-09-27/PLAN.md) §0, §5).
Mémoire : [`docs/knowledge/index.md`](docs/knowledge/index.md) ; toute conclusion durable va dans [`log.md`](docs/knowledge/log.md).

## Rôles (décidés par Vincent le 2026-09-28)
- La session Voxtral du Mac est la seule à committer ici ; tout autre agent passe par une tâche action-plans.
- Planification et vérification (applied → verified) : une session cloud, dans action-plans uniquement.
- Réponses aux ASK et fusions : Vincent.

## Build : `xcodebuild`, jamais `swift build`
`swift build` ne compile ni n'embarque complètement les shaders Metal de MLX (commentaire du mainteneur,
issue #11). Toutes les mesures se font sur ce binaire Release :

```bash
xcodebuild -scheme VoxtralCLI -configuration Release -derivedDataPath .build/xcode -destination 'platform=macOS' build
```
`$CLI` = `.build/xcode/Build/Products/Release/VoxtralCLI`. Autres schémas (même commande, `-scheme <nom>`) :
VoxtralApp, VoxtralBenchmark, VoxtralTTSStreamingDemo.

## Tests : Debug, sans parallélisme
```bash
xcodebuild test -scheme MLXVoxtralSwift-Package -destination 'platform=macOS' \
  -derivedDataPath .build/xcode-test -parallel-testing-enabled NO
```
- Jamais de tests en parallèle : un gradient (enrôlement) concurrent d'une inférence peut bloquer le processus
  (deadlock compile × vjp, piège 20 ; mlx-swift 0.31.6 n'a pas le correctif `df9ae26` — PLAN.md §1).
- Tests gardés par variable : préfixe `TEST_RUNNER_` (ex. `TEST_RUNNER_VOXTRAL_LONG_AUDIO=1 xcodebuild test …`) ;
  cibler une classe avec `-only-testing:VoxtralCoreTests/<Classe>` ; TSan : `-enableThreadSanitizer YES`.
- Une porte de test se prouve par `xcodebuild test` (Debug, les tests font `@testable import VoxtralCore`) ;
  une porte de mesure par `$CLI bench` (Release). Un test chronométré n'est pas une mesure de référence.

## Mesures : Release, machine prête, A/B/B/A
```bash
~/.claude/skills/mlx-swift-audit/scripts/machine-check.sh $CLI --cooldown 120 --procs 'Voxtral.*|FluxForge.*'
```
- Aucune ligne `KO` avant de mesurer ; une autre charge GPU (autre app MLX) invalide la mesure (la noter).
- Refroidissement 120 s, un levier par comparaison, A/B/B/A avec amorçage exclu ; une différence ne compte que
  si elle dépasse la dispersion A/A ; gain < 5 % = bruit, le levier est retiré (PLAN.md §0).
- Une ligne `BENCH {…}` par mesure, recopiée telle quelle dans [`BENCHMARKS.md`](BENCHMARKS.md), jamais éditée.
  Protocole, corpus et glossaire (RTF = génération ÷ audio ; TTFT-frame ≠ TTFA) :
  [`docs/Benchmarks.md`](docs/Benchmarks.md).
- Toute valeur publiée avant le 2026-09-27 est « en session » : jamais une référence (PLAN.md §0).

## Dépendances : `mlx-swift-lm` sur `main`
- `Package.swift:52` suit `branch: "main"` (raison : `Package.swift:46-51`) et `Package.resolved` n'est pas suivi
  (`.gitignore:27`) : noter la révision **résolue** de mlx-swift, mlx-swift-lm et swift-mlx-profiler dans chaque
  mesure (piège 21). À l'audit : mlx-swift 0.31.6 (`0bb916c`), mlx-swift-lm `main@ee673d6` (PLAN.md §1).
- Changer une exigence de version passe par ASK-28 (épinglage, synchronisé avec FluxForge).

## API publique
- Consommateurs connus : FluxForge Studio (App Store) et SongAnalysisDb. Symboles consommés :
  `VoxtralPipeline(.mini3b4bit)`, `ModelRegistry`, `ModelDownloader.customModelsDirectory`,
  `RuntimeBeacon.isEnabled`, `VoxtralTTSPipeline` (audit-stabilite.md §0).
- Tout retrait ou renommage public est cassant : dépréciation d'abord (K-30), décision de Vincent (ASK-23) ;
  sinon, changements additifs seulement.

## Commits
- `type(scope): summary` en anglais (`fix`, `feat`, `docs`, `perf`, `test`, `chore` : `git log`) ; une fiche
  d'audit = un commit `type(K-n): …`, avec son entrée de journal dans `PLAN.md` §7.
