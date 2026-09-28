# Audit mlx-voxtral-swift — 2026-09-27

> Premier essai en conditions réelles du skill **`mlx-swift-audit`** (bêta, dépôt `claude-skills`). Il s'appuie sur
> `mlx-swift-patterns` (motifs MLX-0xx), `task-dispatch` et `task-runner` (tracker `VincentGourbin/action-plans`).
> Objectif : des modèles Voxtral optimisés au standard de septembre 2026 (profils `<bits>bit-fast|lean`), sans
> perdre aucune des actions déjà ouvertes sur Voxtral.
>
> **Révision auditée** : `9392ed1` (= tag `v2.2.2`) · **branche** : `claude/action-plan-skills-beta-wifgmu` ·
> **amont résolu** : mlx-swift 0.31.6 (`0bb916c`, MLX C++ `ce45c52`), mlx-swift-lm `main@ee673d6`.
>
> **État au 2026-09-27 (fin de la phase 4)** : phases 0 à 4 terminées (cadrage, scan, audit vérifié, profils,
> plan), sans build, test ni mesure (session cloud Linux, sans Mac ni GPU), sans commit ni écriture sur GitHub.
>
> **État au 2026-09-28** : le dossier est suivi par git depuis `22a117f`. Phase 5, côté cloud : **K-17, K-81, K-18,
> K-19 et K-21 sont faites** (portes des quatre premières rejouées par la revue adverse du 2026-09-28), **K-20 est
> partielle** (`.serena` retiré de l'index ; les 22 WAV attendent ASK-30). K-21 a écrit dans le tracker (ASK-31
> tranchée par la demande de Vincent, pas encore reportée dans la section « Réponses » d'[`ASK.md`](ASK.md)) : #71,
> #307 et #349 fermés `verified`, #556 et #557 créés. Les 76 fiches `macos-gpu` ne sont pas dispatchées (aucune tâche
> `kind:task` Voxtral dans `action-plans`, relu le 2026-09-28) ; 22 fiches restent ⛔ (21 `macos-gpu`, qui
> naîtront `blocked`, et K-20). Le détail vit dans la colonne État et le journal (§7) de [`PLAN.md`](PLAN.md).

## Par où commencer

1. [`PLAN.md`](PLAN.md) : règles de mesure, faits vérifiés, 82 fiches en 6 lots avec porte chiffrée, pièges,
   commandes, hors plan, journal, et en annexe A la disposition de chaque action existante.
2. [`ASK.md`](ASK.md) : 31 décisions fermées ; 23 fiches (⛔) attendaient une réponse à la fin de la phase 4, 22
   depuis K-21 (ASK-31). La section « Réponses » est à remplir.
3. [`profils.md`](profils.md) : matrice `<bits>bit-fast|lean` par modèle (STT Mini/Small, Realtime, TTS,
   enrôlement), poids recommandés au 2026-09-27, esquisse Swift `VoxtralReferenceProfile`, brouillon de
   `References.md`.
4. [`fiches/`](fiches/) : une fiche autonome par ligne du plan (`K-1.md` … `K-82.md`), exécutable par un agent Mac
   sans autre contexte.
5. [`tasks.yaml`](tasks.yaml) : les 76 fiches `macos-gpu` au format `task-dispatch`.

## Critique de complétude (2026-09-27, après la phase 4)

Relecture contre le `SKILL.md` de `mlx-swift-audit` (phases 0-4 et règles). Corrigé dans ce dossier (rien dans
`Sources/` ni `Tests/`) :
- **Ordre imposé** : K-65 n'atteignait aucune baseline (+ K-36) ; K-74 comparait à une baseline absente (+ K-34) ;
  K-82 exigeait les profils d'enrôlement sans K-64 (+ K-64) ; les baselines TTS et Realtime (K-35, K-36) ne
  dépendaient pas de l'outil d'éval K-33 alors que 7 fiches citent un WER ou une couverture ASR « de référence »
  (+ K-33, portes étendues à la qualité).
- **Chemin chat sans baseline** alors que K-49 et K-70 le mesurent : mode `bench chat` ajouté à K-32, lignes chat à
  K-34 et PLAN §2.
- **K-9 et K-13** (lot 1) citaient un WER avant l'outil K-33 : méthode `jiwer` épinglée écrite dans les fiches.
- **Catalogue T1…T23** : 14 cellules « — » de la colonne Enrôlement tranchées, statuts rendus explicites, chemins
  secondaires (chat, streaming TTS, encodeur hybride) ajoutés (`audit-performance.md` §2.0 bis).
- **Phase 0** : matrice point d'entrée × stabilité / annexes / perf / profil / fiches (`faits-et-actions.md` §1.2 bis).
- **Profils** : matrice complète 4 modèles × largeurs × `fast|lean` avec cases vides justifiées ; matrice Small
  explicite ; boutons existants oubliés (`chunkSize`, `warmUpLeadInFrames`, hyperparamètres d'enrôlement) et bouton
  à créer (compile du codec, K-71).
- **Annexe A du plan** : correspondance élément par élément pour Q-M1…Q-M8, K-M01…K-M09, PK-1…PK-4 ; tracker relu.
- `tasks.yaml` revalidé à blanc : « 76 tâche(s) valides ».

## Fichiers

| Fichier | Phase | Contenu |
|---|---|---|
| [`scan.md`](scan.md) | 1 | sortie de `scan.py` : volumes, dépendances, motifs, absence de `BENCHMARKS.md` |
| [`patterns-scan.md`](patterns-scan.md) | 1 | sortie de `apply.py scan` (catalogue MLX-001…MLX-015) |
| [`patterns-verdicts.md`](patterns-verdicts.md) | 2 | verdicts par occurrence ; 13 constats relus, 12 gardés, 1 écarté ; MLX-001, 005, 007, 008, 009, 011, 013, 014, 015 conformes ou sans objet ; fiches MLX-016…MLX-020 proposées au skill (ids provisoires : correspondance avec le catalogue mlx-swift 0.4.0 en tête du rapport) |
| [`faits-et-actions.md`](faits-et-actions.md) | 0 | cadrage, consommateurs de l'API, 44 faits `FV-xx`, 52 actions `ACT-xx` (issues, PR, plans, TODO, docs), 9 constats `FA-xx`, capitalisation |
| [`audit-stabilite.md`](audit-stabilite.md) | 2 | 29 constats `S-xx` (12 gardés tels quels, 17 amendés, 0 écarté) |
| [`audit-annexes-serveur.md`](audit-annexes-serveur.md) | 2 | 23 constats `A-xx` : enrôlement, Core ML, app, scripts, serveur (13 amendés) |
| [`audit-performance-stt.md`](audit-performance-stt.md) | 2 | P-01…P-29 (28 gardés, P-25 écarté), simulation du `RotatingKVCache` |
| [`audit-performance-tts.md`](audit-performance-tts.md) | 2 | P-30…P-49 (20 gardés) |
| [`audit-performance-realtime-instruments.md`](audit-performance-realtime-instruments.md) | 2 | P-60…P-79 (20 gardés), spécification de l'instrument `bench` |
| [`audit-performance.md`](audit-performance.md) | 2 | rapport consolidé : catalogue T1…T23 × STT / TTS / Realtime / enrôlement, 68 constats P-xx avec leur fiche |
| [`modeles-2026-09.md`](modeles-2026-09.md) | 3 | état de l'art des poids au 2026-09-27 (Hub HF), M-01…M-07, packs PK-1…PK-4 à publier |
| [`profils.md`](profils.md) | 3 | matrice des profils, boutons existants ou à créer, valeurs « mesurée (source) » ou « à mesurer » |
| [`PLAN.md`](PLAN.md) | 4 | plan exécutable (gabarit `plan.md.tmpl`) |
| [`ASK.md`](ASK.md) | 4 | décisions à prendre |
| [`fiches/K-n.md`](fiches/) | 4 | 82 fiches (gabarit `fiche.md.tmpl`) |
| [`tasks.yaml`](tasks.yaml) | 4 | tâches Mac pour `task-dispatch` |

Chaque rapport de phase 2 a subi une **vérification croisée adverse** : relecture du code à `9392ed1`, de l'amont
aux révisions résolues et des listings HF. Le bilan figure en tête de chaque rapport ; les sous-affirmations
écartées sont en annexe. Règle appliquée partout : un indice n'est pas un constat, tout gain est « attendu », et
toute valeur non mesurée dans le dépôt est « À MESURER ».

## Top 10 des constats

| # | Constat | Effet | Fiche |
|---|---|---|---|
| 1 | **S-02 = P-03** : tous les préréglages mémoire posent `maxKVCacheSize` ⇒ `RotatingKVCache` + masque maison de mauvaise forme | `fatalError` au préfill au-delà de 6, 11, 17 ou 22 fenêtres de 30 s ; l'app (8 192) s'arrête vers 11 min d'audio | K-1, K-2, K-3 |
| 2 | **S-01** : jeton d'arrêt `32000` hérité de Llama ; avec Tekken c'est « ␣Capital » | toute transcription s'arrête au premier « Capital », sans erreur | K-4 |
| 3 | **P-11 / P-64** : `maxTokens` = 500 en STT (≈ 3 min de parole), trames en Realtime (≈ 5 min 27 s) | troncature silencieuse des audios longs | K-5 |
| 4 | **A-01** : `enrollVoice` ne marque jamais la pipeline occupée ; le vjp traverse un `silu` compilé | deadlock ABBA possible entre enrôlement et synthèse ; le correctif amont (`df9ae26`) n'est dans aucun tag | K-11 |
| 5 | **S-03, S-04, S-05, A-02** : « téléchargé » = un `.json` présent, sans SHA-256 ; poids appliqués avec `verify: .none` ; tokenizer de repli « démo » | modèle incomplet ou faux chargé **en silence** | K-6, K-7 |
| 6 | **M-01, M-02** : l'original `realtime-4b` ne se charge plus (dérive du dépôt Mistral) ; la clé `"mode"` des packs 2026 fait refuser toute la config STT | les packs recommandés en 2026 sont inutilisables | K-8, K-9 |
| 7 | **P-60 / P-61** : tête liée Realtime `matmul(h, emb.T)` avec `h` fp32 | copie fp32 de 1,5 Gio de la table par pas ; `tok_embeddings` jamais quantifié | K-38, K-46 |
| 8 | **P-01 / P-05** (STT) et **P-30** (TTS) : mel fp32 jamais casté, FM casté en fp32 | tout le calcul est en fp32 (cache KV Mini 240 Kio/jeton) ; chaque `Linear` recopie son poids | K-40, K-39 |
| 9 | **P-32** : attention du codec TTS en T×T fp32 pour une fenêtre ≤ 16 | ≈ 10,5 Go par tenseur à 2 266 frames (calcul) | K-41 |
| 10 | **P-79 / P-73** : aucun instrument de référence ; diagnostics Realtime #23-#25 fondés sur des artefacts du profileur | aucune mesure publiée n'est A/B/B/A ; tous les chiffres sont « en session » | K-32, K-33, K-34…K-37 |

À noter aussi : le « streaming » TTS génère tout avant le premier chunk et ne s'annule pas (S-08, K-12) ; les
fenêtres glissantes Realtime sont ignorées (P-62, P-63, K-13) ; `flowSteps`, `cfgAlpha` et `temperature` du TTS ne
sont lus nulle part (P-35, K-48).

## Le plan en chiffres

| Lot | Fiches | Contenu |
|---|---|---|
| 1 — stabilité bloquante | K-1…K-16 (16) | erreurs MLX, cache KV, masques, arrêt, troncature, téléchargement et chargement vérifiés, exclusion enrôlement/inférence, streaming TTS, annulation |
| 2 — hygiène sans risque | K-17…K-31 (15) | mémoire du projet, docs, annexes Python, git, tracker, dépendances, registres, tests et CI, dépréciations |
| 3 — baseline mesurée | K-32…K-37 (6) | instrument `VoxtralCLI bench` (A/A ≤ 3 %), `eval` (WER, juge ASR), baselines STT, TTS, Realtime, enrôlement |
| 4 — leviers perf | K-38…K-75 (38) | par gain attendu décroissant : dtypes, codec, asyncEval, tête quantifiée, mémoire, KV, etc. |
| 5 — type de profils + CLI | K-76 (1) | `VoxtralSTTReferenceProfile`…, `VoxtralCLI references`, `--reference` |
| 6 — mesures de la matrice et docs | K-77…K-82 (6) | matrice STT, Realtime, TTS ; packs ; `References.md`, `Weights.md` |

- **Cible** : 6 fiches `cloud` (K-17…K-21, K-81 : docs, scripts Python, tracker) et 76 fiches `macos-gpu`
  (build, tests, mesure ou écoute).
- **Gravité** : 19 hautes, 37 moyennes, 26 basses.
- **Décisions** : 23 fiches ⛔ attendent une réponse d'`ASK.md` ; K-2 peut en poser une après sa mesure (ASK-7).
- **Actions existantes** : les 52 actions `ACT-xx`, les questions des rapports et les fiches proposées par chaque
  rapport ont chacune une disposition (fiche, ASK, fermeture ou hors plan) dans `PLAN.md` annexe A. Aucune n'est
  perdue.

## Exécution

**Prérequis** : ce dossier doit être **sur la branche distante** `claude/action-plan-skills-beta-wifgmu`, car chaque
tâche Mac commence par `git pull` puis lit sa fiche. Il est commité depuis `22a117f` : pousser les derniers commits
avant de dispatcher.

Depuis claude-skills 0.6.0 (agent-tracker 0.2.0, contrat de tableau partagé avec `mlx-swift-audit`), `tasks.yaml` et
`PLAN.md` donnent le même lot (rejoué le 2026-09-28) :

```bash
D=~/.claude/skills/task-dispatch/scripts/dispatch.py

# Essai à blanc depuis tasks.yaml (valide, résume, ne crée rien)
python3 $D docs/audit/2026-09-27/tasks.yaml
#   → « Vagues : 12 » · « Naissent blocked (⛔) : 21 » · « Erreurs : aucune »
#   → « 76 tâche(s) valides — rien créé (--create pour créer ; sans gh : --emit-json --issue-map map.json ; … »

# Même essai depuis PLAN.md (colonnes Cible, Prérequis, État)
python3 $D docs/audit/2026-09-27/PLAN.md --runs-on macos-gpu --requires mlx,xcode,gh,git \
  --repo VincentGourbin/mlx-voxtral-swift --branch claude/action-plan-skills-beta-wifgmu --project mlx-voxtral-swift
#   → « Ignorées : 5 — K-17 (fait), K-18 (fait), K-19 (fait), K-21 (fait), K-81 (fait) »
#   → « Exclues : 1 — K-20 (cible cloud) » · mêmes vagues, mêmes 21 blocked · « 76 tâche(s) valides — rien créé … »

# Création des issues de tâche dans VincentGourbin/action-plans (vagues successives selon depends_on)
python3 $D docs/audit/2026-09-27/tasks.yaml --create

# Sans gh : vague de créations en JSON, puis carte id → numéro d'issue (skill task-dispatch, « Sans gh »)
python3 $D docs/audit/2026-09-27/tasks.yaml --emit-json --issue-map docs/audit/2026-09-27/map.json
```

- **Première vague** (aucune dépendance) : K-1, K-4, K-6, K-7, K-10, K-11, K-14, K-16, K-22. K-10 et K-22 sont ⛔ :
  elles naissent `blocked` (`needs_decision` : ASK-15, ASK-28) et ne sont ni proposées ni réclamables par
  `task-runner` tant que la réponse datée n'est pas inscrite dans `ASK.md` et la tâche rouverte.
- **`tasks.yaml` ou `PLAN.md`** : mêmes 76 tâches, mêmes portes, cibles, capacités, dépôt, branche, `depends_on`,
  vagues et décisions ⛔ (comparaison des sorties `--emit-json` et `--verbose`, 2026-09-28). Trois écarts restent :
  (1) gravité : `tasks.yaml` porte celle de chaque fiche (19 `high`, 34 `medium`, 23 `low`), le mode Markdown pose
  `--severity` (défaut `medium`) sur tout le lot ; (2) titres : `[voxtral K-n]` et objectifs reformulés dans
  `tasks.yaml`, `[mlx-voxtral-swift K-n]` et l'Objet du plan en Markdown ; (3) instructions : `tasks.yaml` demande en
  plus l'entrée de journal (`PLAN.md` §7, `docs/knowledge/log.md`) et cite K-81 comme prérequis cloud de K-82. Créer
  depuis `tasks.yaml` ; le mode Markdown sert de contre-vérification.
- **Historique** : l'essai à blanc du mode Markdown du 2026-09-27 (`dispatch.py PLAN.md --runs-on macos-gpu
  --project mlx-voxtral-swift`, agent-tracker 0.1.0) donnait « 82 tâche(s) valides », mais le lot était faux : (1) la
  colonne `Cible` est ignorée, les 6 fiches cloud deviendraient `macos-gpu` ; (2) le découpage des cellules ne tient
  pas compte de `\|` (échappement GFM) : les portes de **K-18, K-20, K-23, K-41, K-43, K-64 et K-78** sont tronquées
  au premier `\|` et la suite glisse dans `Effort` (K-20 : `gate: '`git ls-files -ci --exclude-standard \'`) ; (3)
  aucun `depends_on` (l'ordre imposé disparaît) ; (4) la colonne `État` (⛔ ASK) est perdue ; (5) `repo` et `branch`
  restent `null` sans `--repo/--branch`, et les instructions citent « `PLAN.md` » sans chemin ni fiche
  `fiches/K-n.md` ; (6) le titre garde la syntaxe de lien Markdown, coupée à 110 caractères. Défauts remontés au
  skill `task-dispatch` et corrigés dans agent-tracker 0.2.0 ; `PLAN.md` a reçu la colonne `Prérequis` (`809f45d`).
- **Fiches cloud** : K-17, K-81, K-18 et K-19 (2026-09-27) et K-21 (2026-09-28, après ASK-31) sont faites ; K-20 est
  partielle (`git ls-files -ci --exclude-standard` : 24 → 22, les 22 WAV attendent ASK-30). Le dispatch ignore les
  fiches `fait` et n'envoie jamais une fiche `cloud` au Mac.
- **Côté Mac** (skill `task-runner`) : une fiche = un commit. La porte observée est recopiée telle quelle dans le
  rapport de tâche, et chaque exécution ajoute une entrée au journal de `PLAN.md` §7. Porte non atteinte ⇒ rien
  n'est livré. Gain < 5 % ⇒ le levier est retiré.
