# Plan — mlx-voxtral-swift au niveau de livraison de YuE2 et qwen38

> Décidé par Vincent le 2026-11-07. Remplace le plan d'audit du 2026-09-27 (97 fiches), gelé en archive dans
> [`docs/audit/2026-09-27/`](docs/audit/2026-09-27/README.md) : on le cite, on ne le met plus à jour.

## But

Un framework d'inférence complet et intégrable, au même niveau de livraison que `yue2-mlx-swift` et `qwen38-mlx-swift` :
profils de référence lean / fast, gestion des modèles et de leur téléchargement, mémoire optimisée, et exécution sur
iPhone pour le TTS, la transcription et la compréhension audio. Pas de serveur d'inférence (ASK-1 = A).

## Acquis sur la branche

Stabilité (erreurs MLX levées, téléchargements vérifiés, annulation, chargeurs stricts) et vitesse : STT mini-3b de 135,7
à 28,7 ms par pas, Realtime 4 bits de 26,9 à 11,7 ms, pic du codec TTS de 22 à 3,6 Go, streaming TTS sans re-décodage.
Détail : [`docs/knowledge/log.md`](docs/knowledge/log.md). Rien de cela n'est encore sur `main`.

## Lots

État : lot 1 livré ([PR #51](https://github.com/VincentGourbin/mlx-voxtral-swift/pull/51)) ; lot 2 en revue
([PR #52](https://github.com/VincentGourbin/mlx-voxtral-swift/pull/52)) ; prochain : lot 3.

Reste à mesurer pour le lot 2, Lexar monté, A/B/B/A contre `86bb24b2` : `lean` sur audio long (mini-3b-8bit, small-4bit) et en
TTS, K-51 (`step_ms_p50` sous 32 Go), K-55 (small-4bit, WER), K-59 (1er préfill = préfill à chaud), K-62 (256 à 2 048),
K-63 (mini-3b-8bit, pic de décodage). Un levier sous sa porte sort du profil `lean`.

| Lot | Contenu | Fini quand | Reprend |
|---|---|---|---|
| 1. Catalogue et téléchargement | Un catalogue pour STT, Realtime et TTS : id, dépôt, révision, fichiers, taille, licence. | `download`, `verify`, `list` et `delete` marchent à l'identique pour les trois familles, en API et en CLI ; ajout sans retrait d'API. | — |
| 2. Mémoire et socle iOS | Un gestionnaire remplace les préréglages `light` à `ultra` : plafond par étape, limites `lean` calées sur la mémoire disponible (`os_proc_available_memory` sur iOS), rapport mémoire, résidence des poids, KV 8 bits, tranche de préfill. Plafond TTS de 2 Go par défaut, désactivable (ASK-37 = B). Porte GPU en arrière-plan et build iOS en CI. | Un `lean` borne son pic sur un audio long ; le plafond TTS est actif par défaut ; le build iOS passe en CI. | K-51, K-52, K-55, K-59, K-62, K-63, K-95 |
| 3. Vitesse | `asyncEval` en TTS et STT, pas de flow matching réglables, dernier logit seul au préfill, session de chat audio réutilisable. | Chaque levier gagne au moins 5 % en A/B/B/A, sinon il est retiré. | K-44, K-45, K-48, K-49, K-54 |
| 4. Profils | Un type par pipeline, `all`, `named`, `applyGlobalPolicy`, `VoxtralCLI references`, `--reference`, chargement par profil. Toute la matrice 4 / 6 / 8 / 16 bits × fast / lean (ASK-4 = A). La tête Realtime quantifiée passe de `VOXTRAL_REALTIME_HEAD_BITS` à un champ de profil (K-46) ; le défaut public ne change pas. | Un profil fixe tous les réglages ; chaque champ est un bouton existant. | K-76 |
| 5. Mesure et docs | Un script à la `bench-references.sh`, une passe par profil. `docs/References.md` sans « à mesurer », `docs/Weights.md` avec SHA-256 et révisions, `docs/API.md`, `docs/CLI.md` ; `docs/Profiles.md` retiré. | La table de référence est remplie et citée par le README. | K-77, K-78, K-79, K-82 |
| 3.0.0 | Corriger K-85 et les tests gardés `PARITY` et `TTS_STREAM` ; publier ou retirer `tts-4b-8bit`. Fusion sur `main` et tag, par Vincent. | Suite et tests gardés verts ; tag publié. | K-85 |
| 6. iPhone | Essai sur appareil du TTS, de la transcription et de la compréhension audio ; `docs/iOS.md`. Hors périmètre : Realtime et enrôlement. | Les trois fonctions tournent avec le profil `lean`, sans jetsam ; pic visé ≤ 4 Go (ASK-2 = A), voix clonée : question 6 ci-dessous. | — |

Les lots 1 à 5 se font dans l'ordre : un profil fige des réglages qui doivent exister, et la table mesure des profils
finis. Après la 3.0.0, l'API n'évolue plus que par ajout : tout ce que l'iPhone exige de public arrive donc avant le tag
(lot 2), et seul l'essai sur appareil attend le lot 6.

## Règles

- Un lot = une branche = une PR vers `claude/action-plan-skills-beta-wifgmu`, relue par un seul vérificateur.
- Porte d'un lot : build, tests, et un A/B si le lot touche la vitesse. L'A/B/B/A sert aux leviers de vitesse ; la table
  de référence se mesure en une passe après refroidissement, comme YuE2.
- Un écart sous le bruit se tranche par l'agent et se note dans `log.md`.
- Reste à Vincent : API cassante, défauts publics, dépendances, publication, fusion sur `main`, tags.
- Tout ce qui touche le son passe par une écoute à l'aveugle (ASK-13 = A).
- Le suivi vit ici et dans les PR. Le tracker `action-plans` ne garde que le blocage amont #556 et l'attente de K-90.

## Défauts connus

- TTS, texte long : la voix perd 13 à 25 dB entre le premier et le dernier quart (constat du 2026-10-12).
- TTS, porteur de warm-up : coupe trop tôt sur 2 prises sur 10 et mots perdus sur une ; défaut antérieur à K-68 (ASK-39).
- TTS, 4 bits avec voix clonée : qualité dégradée, limite documentée dans le README (K-97).
- Catalogue : `verify` ne compare pas la révision du manifeste à celle du catalogue ; le snapshot Hub retenu est le dernier
  par nom, pas le plus récent ; `delete` de l'API efface toutes les copies, dossier Hub partagé compris, sans option de
  portée (`deletionTargets` permet de demander confirmation avant).
- iPhone : en TTS en flux, la porte GPU bloque deux fils du pool coopératif tant que l'app est en arrière-plan. Realtime et
  enrôlement ne sont pas couverts par la porte (hors périmètre).
- Realtime : le vidage du cache tous les 256 pas reste, sans A/B (reste de K-52).
- Avant la 3.0.0 : `tts-4b-8bit` au registre sans pack publié (K-90), Small `.auto` bimodal avec dernier segment français
  traduit (K-85), tests gardés `PARITY` (référence d'avant K-40) et `TTS_STREAM` (course de `.ready`) en échec depuis K-89.

## En attente de Vincent

Proposition de l'agent en gras.

1. Dépendances (ASK-28 = « premier tag ensuite », ASK-29) : passer à mlx-swift-lm 3.32.3, qui tire mlx-swift 0.32.x et le
   correctif du deadlock, avant le lot 5 pour ne pas mesurer deux fois ? **Oui.** La conformance `LanguageModel` (K-75,
   ASK-24 = A) part avec ce passage.
2. Flux Realtime micro → texte (ASK-3) : A) dans le plan ; **B) Realtime reste une transcription de fichier.**
3. Enregistrement réel de 12 min pour le WER (ASK-14) : B) tu en fournis un, hors dépôt ; **C) corpus synthétique seul, biais
   noté.**
4. Volume qui baisse sur les textes longs : **A) mesure et découpage du texte en option** ; B) limite documentée.
5. K-90 : écoute des 8 paires (`.local-runs/listen/k90-vs-bf16/`, `k90-vs-6bit/`) et relecture de la carte
   (`.local-runs/k90/card/README.md`) avant publication.
6. Voix clonée sur iPhone : le 4 bits la dégrade (K-97) et le pack 6 bits pèse 3,47 Go de poids, peu compatible avec
   4 Go de pic. **A) 6 bits, pic visé ≤ 5 Go avec Increased Memory Limit** ; B) voix prédéfinies seulement sur iPhone.
7. STT sous 32 Go de RAM : `.aggressive` et `.ultra` vidaient le cache dans la boucle de décodage ; sans ce vidage (K-51),
   rien ne bornait plus le cache sur ces machines. **A) plafond de 2 Go par défaut, désactivable, comme Small (posé par le
   lot 2)** ; B) aucun plafond par défaut, et un pic plus haut sur audio long.
