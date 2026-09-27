# Piège — le même % GPU sur des opérations différentes est une signature d'instrument (V-P8)

> Date : 2026-09-27 (audit à `9392ed1`). Statut : VÉRIFIÉ en lecture (swift-mlx-profiler, tags `1.1.1` à `1.4.0`) ;
> occupation réelle À MESURER (K-34, K-36). Sources : faits-et-actions.md §5.3 V-P8 ;
> audit-performance-realtime-instruments.md F-R1, F-R4, §3 bis, P-73, P-74.

**Symptôme** (issues d'avril 2026, profils « en session », machine non notée) : 48-49 % de GPU sur des opérations
sans rapport (encodage Core ML, préfill STT, encodeur Realtime : #13, #14, #24) ; « 0 % GPU » pendant les 23,89 s de
« Realtime Generation » à « 21 tok/s » (#23, `realtime-4b-4bit`, 2026-04-11). Les issues ont été fermées sur une
cause « allocation Metal systémique » jamais prouvée ; la fermeture de #23 a masqué la copie fp32 de la tête (P-61).

**Cause** : l'instrument, pas le GPU.
- **Lecture instantanée** : « Device Utilization % » est une valeur instantanée. Lue au bord d'une phase ou juste
  après une synchronisation, elle ne mesure pas une occupation ; le profiler le documente sur une phase de 5 ms
  (« 41-49 % where it was ~82 % », `GPUUtilization.swift:11-16`, profiler 1.5.x).
- **Moyenne de bords** (profiler ≤ 1.4, `ProfilingSession.swift:200-213`, `:229`) : une phase sans pas n'a que deux
  lectures ; (≈ 0 au repos) + (≈ 98 après un `eval`) donne ≈ 49 en moyenne entière. Hypothèse compatible avec le
  « 49 % », non établie.
- **Phases imbriquées** : la phase englobante perd ses pas au profit des sous-phases ; son GPU % retombe à la
  moyenne de ses deux bords, d'où « 0 % » (#23). Ses « 71 % du temps » additionnent des phases imbriquées (72 %
  recalculé, §3 bis).

**Correctif** (à faire) : occupation mesurée par Metal System Trace et par swift-mlx-profiler 1.5.x `.fineGrained`
(`.ioReportResidency`, moyenne d'intervalle) : écart entre les deux ≤ 10 pts pour le décodage Realtime (K-36),
écart noté pour le préfill STT (K-34) ; version du profiler épinglée et enregistrée dans chaque mesure (P-74, K-22). Décision associée :
[conclusions #23-#25 caduques](../decisions/realtime-diagnostics-23-25.md).

**Règle** : un pourcentage GPU identique sur des opérations différentes, ou nul pendant un calcul, se recoupe par
Metal System Trace ou `ioreg` avant toute conclusion ; une lecture instantanée n'est jamais une occupation.
