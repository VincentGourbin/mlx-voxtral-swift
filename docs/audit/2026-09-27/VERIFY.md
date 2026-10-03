# Vérification et autonomie de la session Voxtral du Mac (décision de Vincent du 2026-10-03)

À partir du 2026-10-03, la session Voxtral du Mac planifie, dispatche, exécute, vérifie (`applied` → `verified`) et
replanifie seule ce dépôt. La session cloud ne vérifie plus rien, sauf audit demandé par Vincent. Ce document remplace,
pour ce dépôt, la répartition du 2026-09-28 (`ASK.md` §Dérogations, entrée « Rôles »).

## 1. Vérifier une tâche `applied` (avant de réclamer une tâche qui en dépend)

L'agent ne juge pas sa propre porte. Deux sous-agents en contexte neuf (Agent ou Workflow), en **lecture seule**
(aucun commit, push, checkout ni écriture dans le tracker) :

- **Vérificateur** : lit le corps de l'issue (porte et tous les blocs « Amendement… », qui priment sur la fiche), tous
  les commentaires, la fiche, les commits et la CI. Pour chaque clause : la preuve observée et un contrôle
  indépendant. Il ne conclut `verified` que si chaque clause est prouvée.
- **Contradicteur** : reçoit le verdict et cherche à le réfuter à partir des sources.
- `verified` seulement si les deux concluent `verified`. En cas de désaccord, un troisième sous-agent tranche la seule
  question disputée.

### Règles de preuve (vérifications du 2026-10-01 et du 2026-10-03)

- **Rouge ET vert** recopiés tels quels : « Executed N tests, with M failures » et le code de sortie. Si le test appelle
  du code créé par le correctif, défaire le comportement en gardant les signatures, dans un worktree jetable. Avant
  `e104b9d`, ce worktree demande `mkdir -p Sources/VoxtralApp/Resources/VoxtralEncoderFull.mlmodelc`.
- **Test discriminant** : une mutation de la branche visée doit le faire échouer (les tests initiaux de K-3 et K-29 ne
  l'étaient pas).
- **Backend public par défaut** (`.auto`) couvert, sauf si la porte se restreint explicitement à un autre (K-15).
- **Mesures** : Release, arbre propre, `dirty:false`, commit poussé, un tag distinct par série. Avant chaque série, la
  sortie de `machine-check --procs 'Voxtral.*|FluxForge.*'` sans `KO`, recopiée (K-37). Lignes `BENCH` recopiées dans
  `BENCHMARKS.md`, A/A ≤ 3 % recalculé.
- **Recalculer** tout ce qui est dans le dépôt : dispersions, WER, ratios, SHA-256, comptes (en excluant les blocs
  `/* */`).
- **La CI** ne compile pas les exécutables et saute les tests gardés : elle ne prouve pas ces clauses.

### Écriture

- Porte tenue : commentaire « Porte vérifiée (auto-vérification, <date>) » avec une ligne de preuve par clause et les
  réserves, puis `status: verified`, label `status:verified` et fermeture « completed ».
- Sinon : commentaire listant les manques ; l'agent les fournit lui-même et relance la vérification.
- Une entrée « Vérification du <date> » dans `PLAN.md` §7.

## 2. Décisions

L'agent décide seul :
- reformuler une clause impossible à tenir telle qu'écrite ;
- remplacer une mesure par une mesure équivalente ou plus stricte ;
- déclarer sans objet une clause qui porte sur du code retiré ;
- fixer l'ordre des tâches et créer des fiches de suite.

Chaque décision s'inscrit dans `ASK.md` §Dérogations (« décision de l'agent », date, clause, raison, preuve) avant le
passage en `verified`.

Restent à Vincent :
- retrait ou renommage d'API publique ;
- changement d'un défaut public visible d'un consommateur ;
- toute dérogation qui affaiblit une garantie mesurée (seuil de qualité, parité, troncature, sécurité) ;
- dépendances et licences ;
- fusions sur `main` et tags ;
- FluxForge.

Pour celles-là : une ASK avec une question fermée et l'option recommandée, la tâche passe en `blocked`, et l'agent
continue sur une autre. Au plus un message par jour, qui regroupe les ASK ouvertes.

Inchangé : jamais de fusion ni de tag ; une fiche = un commit `type(K-n)`, avec son entrée de journal.

## 3. Ordre de travail à partir du 2026-10-03

1. K-37 (#586) : refaire les quatre séries comme demandé dans le dernier commentaire de l'issue.
2. K-34, K-35, K-36 (#587 à #589) : lire leurs amendements du 2026-10-03 et le commentaire sur K-33 ; télécharger
   `small-4bit` avant K-34.
3. Inscrire dans `ASK.md` la décision K-14 du 2026-10-03 (porte reformulée).
4. Les quatre baselines vérifiées : #590 (replanification des lots 4 à 6 : plan recalé commité directement, puis
   `dispatch.py --create` après un essai à blanc ; la procédure « patch + sha256 pour le Mac » ne s'applique plus),
   puis les tâches dispatchées, dans l'ordre des dépendances.
5. claude-skills : #600 et #606 attendent la fusion de claude-skills#1 (par Vincent) ; #606 reçoit un point de plus :
   capitaliser ce protocole de vérification dans agent-tracker.
