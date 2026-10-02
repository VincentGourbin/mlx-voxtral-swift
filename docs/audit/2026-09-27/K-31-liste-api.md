# K-31 — Liste proposée de la surface publique 3.0 (à relire par Vincent, ASK-25 = A)

Proposée le 2026-10-02 par la session Mac, à partir de `5dca7ee`+K-28 (`e104b9d`), **validée telle quelle par Vincent
le 2026-10-02** et appliquée (K-31). Décisions et amendements : §E.

- État : `grep -c '^\s*public' -r Sources/VoxtralCore` = **1 110** lignes ; 183 déclarations de premier niveau.
- Cible de la porte : ≤ 300. Les fichiers des façades seules en portent 373 : il faut aussi élaguer **dans** les
  façades (membres internes rendus `internal`).
- Consommateurs inventoriés (grep des fichiers qui font `import VoxtralCore`) :
  - **FluxForge Studio** (21 fichiers) : `VoxtralPipeline` (`.Model`, `.Configuration`, `loadModel`, `transcribe`,
    `unloadModel`), `VoxtralTranscriptionManager.isModelDownloaded`, `ModelRegistry.model(withId:)`, `.repoId`,
    `ModelDownloader` (`download`, `downloadTTSModel`, `findModelPath`, `findTTSModelPath`, `isModelDownloaded`,
    `isTTSModelDownloaded`, `deleteModel`, `customModelsDirectory`, `reconfigureHubApi` déprécié),
    `DownloadProgressCallback`, `VoxtralTTSPipeline` (`.Configuration`, `synthesize`, `synthesizeToFile`,
    `enrollVoice`, `recommendedWarmUpVocalise`), `VoxtralTTSRegistry` (`model(withId:)`, `defaultModel`),
    `VoxtralTTSModelInfo`, `VoxtralVoice`, `VoxtralVoiceEnrollment.Config`, `WAVWriter.write`,
    `VoxtralCoreMLEncoder.downloadFromHuggingFace`, `VoxtralCoreMLVariant`, `RuntimeBeacon.isEnabled`.
    Contournements : `private typealias FluxModelRegistry = ModelRegistry` (4 fichiers) et `VoxtralCore.ModelRegistry` /
    `VoxtralCore.ModelDownloader` qualifiés (collision avec les `ModelRegistry`/`ModelDownloader` de FluxForge).
  - **SongAnalysisDb** (1 fichier) : `VoxtralPipeline` (`.Model`, `.Configuration`, `ProgressCallback`, `loadModel`).

## A. Conservé public (façades)

| Domaine | Symboles | Membres publics conservés |
|---|---|---|
| STT | `VoxtralPipeline`, `VoxtralPipelineError` | `Model` (+ `repoId`, `displayName`), `Backend`, `Configuration`, `ProgressCallback`, `init`, `loadModel`, `unload`, `transcribe`, `chat`, `quickTranscribe`, `isReady`, `state`, `lastTokenCount`, `memorySummary`, `recommendedModel`, `automaticMaxTokens` |
| STT | `VoxtralTranscriptionManager`, `TranscriptionResult`, `VoxtralTranscriptionError` | inchangés (hors `chat(systemPrompt:)` déprécié → retiré) |
| STT | `MemoryOptimizationConfig` | utilisé par `VoxtralPipeline.Configuration` |
| TTS | `VoxtralTTSPipeline`, `VoxtralTTSSynthesisManager`, `VoxtralTTSError` | `Configuration`, `loadModel`, `unload`, `synthesize`, `synthesizeStreaming`, `synthesizeToFile`, `enrollVoice`, `recommendedWarmUpVocalise`, `lastSynthesisTruncated`, `frameCap(forText:)` ; **plus** `ttsModel`/`tokenizer`/`voiceEmbeddings` (aujourd'hui `private(set)` publics) → `internal` |
| TTS | `VoxtralVoice`, `TTSSynthesisResult`, `TTSStreamingChunk`, `WAVWriter` | inchangés |
| TTS | `VoxtralVoiceEnrollment` | `Config`, `Progress`, les deux erreurs ; l'instance (`init(model:)` prend un `VoxtralTTSModel` interne, `prepareReference`, `optimize`, `codesToVoiceEmbedding`) → `internal`, l'enrôlement passe par `VoxtralTTSPipeline.enrollVoice` ; `EnrollmentLossComputer` → `internal` |
| Realtime | `VoxtralRealtimePipeline`, `VoxtralRealtimeManager`, `VoxtralRealtimeError` | `Configuration`, `loadModel(modelId:)`, `transcribe(audio:)`, `unload`, `state`, `isReady`, `lastTranscriptionTruncated` ; `extractAudioEmbeddings` (rend un `MLXArray`) → `internal` |
| Registres | `ModelRegistry` → **`VoxtralModelRegistry`** ; `VoxtralModelInfo` ; `VoxtralTTSRegistry`, `VoxtralTTSModelInfo` ; `VoxtralRealtimeRegistry`, `VoxtralRealtimeModelInfo` | `models`, `defaultModel`, `model(withId:)` ; `printAvailableModels` → `internal` |
| Téléchargement | `ModelDownloader` → **`VoxtralModelDownloader`** ; `ModelDownloaderError` → **`VoxtralModelDownloaderError`** ; `DownloadProgressCallback` → **`VoxtralDownloadProgressCallback`** | les 9 appels de FluxForge ci-dessus + `downloadRealtimeModel`, `findRealtimeModelPath`, `downloadByRepoId` |
| Core ML | `VoxtralCoreMLEncoder` (statique seulement), `VoxtralCoreMLVariant`, `VoxtralCoreMLError` | `downloadFromHuggingFace`, `downloadForMLXModel`, `variant(forConfigAt:)`, `fromMLXModelRepoId` ; l'instance et ses fabriques (`init`, `encode`, `fromHuggingFace`, `forMLXModel`, MLMultiArray) → `internal` |
| Infra | `RuntimeBeacon` → **`VoxtralRuntimeBeacon`** ; `VoxtralError` ; `VoxtralDebug` ; `VoxtralCoreVersion` | `isEnabled`, `begin`, `Session` ; `VoxtralDebug.isEnabled` seulement |

**Renommages** : l'ancien nom reste un `@available(*, deprecated, renamed:)` typealias pendant la 3.x (FluxForge
compile alors sans ses `typealias FluxModelRegistry`), retrait en 4.0.

## B. Rendu `internal` (détails d'implémentation, aucun consommateur connu)

- **Modèles STT** : `VoxtralStandardModel`, `VoxtralStandardConfiguration`, `VoxtralStandardEncoder(Layer)`,
  `VoxtralStandardProjector`, `AudioAttention`, `LanguageModelContainer`, `LlamaStandard*`, `LlamaStandardConfig`,
  `loadVoxtralStandardModel` (×2), `VoxtralForConditionalGeneration`, `VoxtralModelOutput`, `VoxtralModel` (typealias).
- **Entrées STT** : `VoxtralProcessor` (la faute `applyTranscritionRequest` disparaît avec lui ; sinon alias déprécié
  `applyTranscriptionRequest`), `VoxtralFeatureExtractor`, `loadAudio`, `logMelSpectrogram`, `N_FFT`, `HOP_LENGTH`,
  `N_MELS`, `TekkenTokenizer`, `ChatTemplateProcessor` (`applyChatTemplate` non typé : interne, plus d'API à typer),
  `AudioEncoder`.
- **Core ML hybride** : `VoxtralHybridEncoder`, `VoxtralEncoderBackend`, `VoxtralEncoderStatus`, `VoxtralCoreMLConfig`,
  `MLXCoreMLBridge`, `MLXCoreMLBridgeError`.
- **TTS** : `VoxtralTTSModel`, `VoxtralTTSConfiguration` (+ 6 extensions), `MMAudioEmbeddings`,
  `AudioCodebookEmbeddingsContainer`, `cloneKVCaches`, `loadVoxtralTTSModel`, codec (`VoxtralCodecDecoder`,
  `VoxtralCodecEncoder`, `MistralAudioCodebook`, `SemanticCodebook`, `AcousticCodebook`, `Codec*`, `ConvBlock`,
  `WeightNorm*`, `WeightParametrizations`), flow matching (`FlowMatchingAudioTransformer`, `AcousticTransformerBlock`,
  `BidirectionalAttention`, `FMFeedForward`, `TimeEmbedding`, `quantizeToFSQ`, `dequantizeFSQ`), `trimLeadingCarrier`,
  `trimLeadingCarrierAdaptive`, `trimLeadInSilence` (×2), `trimTrailingSilence`, `VoxtralVoicePresetManager`.
- **Realtime** : `VoxtralRealtimeModel`, `VoxtralRealtimeEncoder`, `VoxtralRealtimeDecoder`, `Realtime*Config`,
  `VoxtralRealtimeConfiguration`, `RealtimeEncoder*`, `RealtimeCausalConv1d`, `RealtimeDecoderLayer`, `AdaRMSNorm`,
  `computeTimeEmbedding`, `loadVoxtralRealtimeModel`.
- **Utilitaires** : `VoxtralMemoryManager`, `VoxtralMLXProfiler` (typealias).

## C. Retiré (déprécié en 2.3 par K-30/K-27, ASK-23 : retrait en 3.0)

La liste « Deprecated » du `CHANGELOG.md` (K-30, K-27, K-28) : famille portage Python (`VoxtralGenerator`,
`VoxtralGenerationParameters`, `ProcessedInputs`, la classe `VoxtralCLI`, `VoxtralConfig`, `VoxtralTextConfig`,
`VoxtralEncoderConfig`, `PythonVoxtralConfig`, `VoxtralAttention`, `VoxtralEncoder(Layer)`,
`VoxtralMultiModalProjector`, `LlamaModel`/`LlamaAttention`/`LlamaMLP`/`LlamaDecoderLayer`/`LlamaConfig`, `MLXLMRope`,
`createCausalMask`, `initializeRope`, `LlamaModelWrapper`), chargeurs (`loadVoxtralModel` ×2, `loadVoxtralModelWithMLXLM`,
`downloadModel`, `loadConfig`, `loadWeights`), quantification hors ligne (`VoxtralQuantization.swift` : 10 fonctions ;
`MLXLMBridge.swift` : 7 fonctions + extension), `writeDebugToDump`, `ModelDownloader.hubApi`/`reconfigureHubApi`
(FluxForge doit retirer `Fluxforge_StudioApp.swift:104`), `VoxtralCoreMLEncoder.resourceBundle`,
`toMLMultiArrayNoCopy`, `chat(systemPrompt:userMessage:)`, `EnrollmentLossComputer(validating:)` public.

## D. Questions ouvertes (à trancher dans la relecture)

1. **Branche et calendrier** : la 3.0 casse l'API. L'appliquer sur `claude/action-plan-skills-beta-wifgmu` ferait
   fusionner la 2.3 (dépréciations, ASK-23) et la 3.0 d'un coup chez FluxForge. Proposé : fusionner/étiqueter la 2.3
   d'abord, puis K-31 sur une branche `api-3.0`.
2. **Voix composées** (`VoxtralZeroVoice`, `VoiceRecipe`, `VoiceFamily`, `blendVoices`, `slerpVoices`,
   `calibrateVoiceNorms`, `alignVoiceLengths`, `resampleVoiceEmbedding`) : documentées dans le README (ZeroVoice) mais
   sans consommateur connu. Proposé : publiques (A), sinon `internal`.
3. **Préfixe** de `ModelDownloader` et `RuntimeBeacon` (collisions FluxForge/LTX) : proposé oui (`VoxtralModelDownloader`,
   `VoxtralRuntimeBeacon`), avec typealias dépréciés.
4. **Porte FluxForge** : « compile sans typealias de contournement » exige une branche de test **dans FluxForge**
   (hors de ce dépôt) ; proposé : branche locale non poussée, constat recopié ici.

## E. Décisions de Vincent (2026-10-02) et amendements d'application

- D1 : appliquer **sur cette branche** (3.0 directe, sans 2.3 publiée). D2 : ZeroVoice **public**. D3 : les **trois**
  préfixes (`VoxtralModelRegistry`, `VoxtralModelDownloader`, `VoxtralRuntimeBeacon`, plus `VoxtralModelDownloaderError`
  et `VoxtralDownloadProgressCallback`), typealias dépréciés. D4 : FluxForge vérifié par Vincent à la transmission.
  D5 : **338 lignes `public` acceptées** (porte ≤ 300 non tenue : la liste validée prime).
- Restés publics parce que les exécutables du dépôt les utilisent : `VoxtralPipeline.encoderStatus` (VoxtralApp),
  `VoxtralTTSPipeline.textTokenCount` (`bench`), `VoxtralRealtimePipeline.extractAudioEmbeddings` (CLI),
  `printAvailableModels` des trois registres (CLI), `VoxtralModelDownloader.listDownloadedModels`, `resolveModel`,
  `modelSize`, `formatSize`, `manifestFileName` (CLI) ; `Configuration.defaultFramesPerTextToken`/`defaultFramesCapBase`
  (valeurs par défaut d'un `init` public).
- Rendus internes en plus de §B (non listés en §A) : `VoxtralPipeline.availableModels`/`automaticTokensPerSecond`,
  `VoxtralModelRegistry.model(withRepoId:)`/`officialModels`/`miniModels`/`smallModels`, les helpers « modèle par
  défaut » et `localPath`/`modelsDirectory`/`resolveTTSModel`/`isRealtimeModelDownloaded` du téléchargeur,
  `downloadRepoDirect`/`verifyShardedModel`/`hubCachePath`, `VoxtralVoice.embeddingFileName`/`safetensorsFileName`,
  `VoxtralDebug.log`/`console`/`always`/`logGeneration`.
- Supprimés : 9 fichiers hérités (`VoxtralGenerator`, `VoxtralGeneratorBridge`, `VoxtralQuantization`,
  `VoxtralModelLoading`, `VoxtralPythonCompatLoader`, `VoxtralMLXLMLoader`, `LlamaModelWrapper`, `CustomLoadWeights`,
  `QuantizedLinearWeightLoader`), `MLXLMBridge.swift` réduit au typealias interne, et les membres dépréciés restants.
  La faute `applyTranscritionRequest` est corrigée sans alias (le processeur est interne).
- Restent internes, non supprimés : la famille de configuration et de décodeur hérités (`VoxtralConfig`,
  `LlamaModel`, `VoxtralAttention`…) que `VoxtralForConditionalGeneration(config:)` référence encore ; leur retrait
  demande de séparer ce modèle du chemin hérité (suite proposée, hors K-31).
