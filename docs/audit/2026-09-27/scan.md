# Scan mlx-swift-audit — `/home/user/mlx-voxtral-swift`

Révision : `9392ed1 fix(build): restore LanguageModel conformance for mlx-swift-lm's prefill change (#50)` · 166 fichiers suivis · 116 fichiers Swift

## 1. Volume Swift par module

| Module | Lignes |
|---|---|
| `Sources/VoxtralCore` | 20489 |
| `Tests/VoxtralCoreTests` | 7250 |
| `Sources/VoxtralApp` | 1490 |
| `Sources/VoxtralTTSStreamingDemo` | 1325 |
| `Sources/VoxtralTranscriptionTest` | 1077 |
| `Sources/VoxtralBenchmark` | 241 |
| `Examples` | 129 |
| `Package.swift` | 114 |

## 2. Dépendances (Package.swift / Package.resolved)

- `.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),`
- `.package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),`
- `.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),`
- `.package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),`
- `.package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0")`

## 3. Hygiène du dépôt

Plus gros fichiers suivis :

| Taille | Fichier |
|---|---|
| 9375 Ko | `docs/examples/fluxforge_long_fr_bf16.wav` |
| 9375 Ko | `docs/examples/fluxforge_long_fr_4bit.wav` |
| 8674 Ko | `docs/examples/fluxforge_long_en_bf16.wav` |
| 8498 Ko | `docs/examples/fluxforge_long_en_4bit.wav` |
| 8145 Ko | `docs/examples/fluxforge_long_fr_6bit.wav` |
| 7826 Ko | `docs/examples/fluxforge_long_en_6bit.wav` |
| 3694 Ko | `samples/fr_voxtral_tts_demo.wav` |
| 3668 Ko | `samples/en_voxtral_tts_demo.wav` |
| 3428 Ko | `docs/examples/tts_bench_long_6bit.wav` |
| 3229 Ko | `docs/examples/tts_bench_long_bf16.wav` |
| 3004 Ko | `docs/examples/tts_bench_long_4bit.wav` |
| 1493 Ko | `.serena/cache/swift/document_symbols.pkl` |
| 536 Ko | `screenshots/voxtral-transcribe.png` |
| 494 Ko | `screenshots/voxtral-chat.png` |
| 401 Ko | `docs/examples/clone_fr.wav` |

Fichiers suspects suivis (artefacts, poids, traces) :

- `docs/examples/clone_en.wav`
- `docs/examples/clone_fr.wav`
- `docs/examples/fluxforge_long_en_4bit.wav`
- `docs/examples/fluxforge_long_en_6bit.wav`
- `docs/examples/fluxforge_long_en_bf16.wav`
- `docs/examples/fluxforge_long_fr_4bit.wav`
- `docs/examples/fluxforge_long_fr_6bit.wav`
- `docs/examples/fluxforge_long_fr_bf16.wav`
- `docs/examples/fluxforge_short_en_4bit.wav`
- `docs/examples/fluxforge_short_en_6bit.wav`
- `docs/examples/fluxforge_short_en_bf16.wav`
- `docs/examples/fluxforge_short_fr_4bit.wav`
- `docs/examples/fluxforge_short_fr_6bit.wav`
- `docs/examples/fluxforge_short_fr_bf16.wav`
- `docs/examples/tts_bench_long_4bit.wav`
- `docs/examples/tts_bench_long_6bit.wav`
- `docs/examples/tts_bench_long_bf16.wav`
- `docs/examples/tts_bench_short_4bit.wav`
- `docs/examples/tts_bench_short_6bit.wav`
- `docs/examples/tts_bench_short_bf16.wav`
- `samples/en_voxtral_tts_demo.wav`
- `samples/fr_voxtral_tts_demo.wav`

Fichiers non-code à la racine :

- `create_app_bundle.sh`
- `llms.txt`

## 4. Indices perf / mémoire

| Motif | Occ. (Sources / Tests) | Sens | Catalogue |
|---|---|---|---|
| `cache-limit` | 2 / 0 | Pose d'une limite de cache MLX | T1/T2, piège 7 |
| `memory-limit` | 0 / 0 | Seuil GC MLX (pas un plafond dur) | T2, piège 17 |
| `clear-cache` | 14 / 2 | Libération du cache MLX | T3 |
| `async-eval` | 0 / 0 | Pipelining du décodage | T15, piège 6 |
| `item-call` | 49 / 50 | Synchronisation GPU→CPU (coûteuse dans une boucle) | T15 |
| `as-array` | 9 / 6 | Copie GPU→CPU | T15 |
| `concat` | 26 / 1 | Concat (cache KV recopié à chaque pas ?) | T11, piège 3 |
| `fp32-cast` | 43 / 14 | Passage en fp32 (fuite de dtype ?) | T17 |
| `compile` | 0 / 0 | compile() MLX (utile ? deadlock ABBA avec vjp) | T18, R1-R3, piège 20 |
| `dequantize` | 0 / 0 | Dé-quantification (jamais le head entier) | T12, T14 |
| `quantized-kv` | 0 / 0 | KV cache quantifié | T10 |
| `prefill-step` | 8 / 0 | Tranche de préfill | T9 |
| `kv-trim` | 0 / 0 | Réutilisation de préfixe KV | T6 |
| `eval-params` | 0 / 0 | eval global des paramètres (pic au chargement) | piège 4 |
| `resize-image` | 0 / 0 | Redimensionnement d'image implicite ? | T8, piège 15 |

<details><summary><code>cache-limit</code> — 2 hors tests</summary>

- `Sources/VoxtralApp/TranscriptionManager.swift:293` — `Memory.cacheLimit = 0  // Temporarily disable caching`
- `Sources/VoxtralApp/TranscriptionManager.swift:295` — `Memory.cacheLimit = Int.max  // Restore default (unlimited)`

</details>

<details><summary><code>clear-cache</code> — 14 hors tests</summary>

- `Sources/VoxtralApp/TranscriptionManager.swift:273` — `Memory.clearCache()`
- `Sources/VoxtralApp/TranscriptionManager.swift:281` — `Memory.clearCache()`
- `Sources/VoxtralApp/TranscriptionManager.swift:287` — `Memory.clearCache()`
- `Sources/VoxtralApp/TranscriptionManager.swift:294` — `Memory.clearCache()`
- `Sources/VoxtralApp/TranscriptionManager.swift:396` — `Memory.clearCache()`
- `Sources/VoxtralApp/TranscriptionManager.swift:467` — `Memory.clearCache()`
- `Sources/VoxtralCore/Realtime/VoxtralRealtimeModel.swift:139` — `Memory.clearCache()`
- `Sources/VoxtralCore/Utils/VoxtralMemoryManager.swift:40` — `Memory.clearCache()`
- `Sources/VoxtralCore/Utils/VoxtralMemoryManager.swift:47` — `Memory.clearCache()`
- `Sources/VoxtralCore/Utils/VoxtralMemoryManager.swift:87` — `Memory.clearCache()`
- `Sources/VoxtralCore/VoxtralModeling.swift:1253` — `Memory.clearCache()`
- `Sources/VoxtralCore/VoxtralModeling.swift:1279` — `Memory.clearCache()`
- … 2 de plus

</details>

<details><summary><code>item-call</code> — 49 hors tests</summary>

- `Sources/VoxtralCore/MLXLMBridge.swift:333` — `let val = array[i].item(UInt32.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:336` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:339` — `let val = array[i].item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:342` — `let val = array[i].item(Int32.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:345` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:360` — `let val = array[i].item(UInt32.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:363` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:366` — `let val = array[i].item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:369` — `let val = array[i].item(Int32.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:372` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:380` — `let minVal = array.min().asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:381` — `let maxVal = array.max().asType(.float32).item(Float.self)`
- … 37 de plus

</details>

<details><summary><code>as-array</code> — 9 hors tests</summary>

- `Sources/VoxtralCore/CoreML/MLXCoreMLBridge.swift:120` — `let floatArray = mlxArray.asArray(Float.self)`
- `Sources/VoxtralCore/CoreML/MLXCoreMLBridge.swift:131` — `let floatArray = mlxArray.asType(.float32).asArray(Float.self)`
- `Sources/VoxtralCore/CoreML/MLXCoreMLBridge.swift:141` — `let intArray = mlxArray.asArray(Int32.self)`
- `Sources/VoxtralCore/TTS/VoxtralTTSProcessor.swift:390` — `scaled.asArray(Int16.self).withUnsafeBufferPointer { buffer in`
- `Sources/VoxtralCore/VoxtralGeneratorBridge.swift:200` — `let tokenIds = generatedTokens.asArray(Int.self)`
- `Sources/VoxtralCore/VoxtralProcessor.swift:227` — `var idsList = batchIds.asArray(Int.self)`
- `Sources/VoxtralCore/VoxtralProcessor.swift:312` — `mlxTokenIds[batchIdx].asArray(Int.self)`
- `Sources/VoxtralCore/VoxtralProcessor.swift:334` — `processedTokenIds = mlxTokenIds.asArray(Int.self)`
- `Sources/VoxtralTTSStreamingDemo/StreamingDemoViewModel.swift:586` — `let floatArray = samples.asArray(Float.self)`

</details>

<details><summary><code>concat</code> — 26 hors tests</summary>

- `Sources/VoxtralCore/CoreML/VoxtralHybridEncoder.swift:249` — `let concatenated = concatenated(allEmbeddings, axis: 0)  // [numChunks * 375, 3072]`
- `Sources/VoxtralCore/Realtime/Pipeline/VoxtralRealtimePipeline.swift:208` — `let padded = MLX.concatenated([`
- `Sources/VoxtralCore/Realtime/VoxtralRealtimeDecoder.swift:30` — `return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)])  // [dim]`
- `Sources/VoxtralCore/Realtime/VoxtralRealtimeEncoder.swift:44` — `padded = MLX.concatenated([MLX.zeros([B, padding, C]), x], axis: 1)`
- `Sources/VoxtralCore/Realtime/VoxtralRealtimeModel.swift:99` — `prefixEmbeds = MLX.concatenated([combinedPart, textOnlyPart], axis: 0)`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralEnrollmentLosses.swift:164` — `return MLX.concatenated(`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:445` — `let fullEmb = MLX.concatenated([semanticEmb, acousticEmb], axis: -1)  // (T, 292)`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:646` — `let codes = MLX.concatenated([sem2d, acoustic], axis: -1)    // (T, 1+nAcoustic)`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:681` — `let allRows = MLX.concatenated([codeRows, MLXArray(endRows)], axis: 0)             // (T*cb + cb)`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:687` — `let result = MLX.concatenated([voiceEmb, endFrame], axis: 0)             // (T+1, dim)`
- `Sources/VoxtralCore/TTS/VoxtralCodecDecoder.swift:83` — `padded = MLX.concatenated([MLX.zeros([B, K - 1, C]), x], axis: 1)`
- `Sources/VoxtralCore/TTS/VoxtralCodecDecoder.swift:390` — `return MLX.concatenated([semanticEmb, acousticEmb], axis: -1)  // (B, T, 292)`
- … 14 de plus

</details>

<details><summary><code>fp32-cast</code> — 43 hors tests</summary>

- `Sources/VoxtralCore/CoreML/MLXCoreMLBridge.swift:108` — `let converted = mlxArray.asType(.float32)`
- `Sources/VoxtralCore/CoreML/MLXCoreMLBridge.swift:131` — `let floatArray = mlxArray.asType(.float32).asArray(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:146` — `let indices = MLXArray(0..<(headDim/2)).asType(.float32)`
- `Sources/VoxtralCore/MLXLMBridge.swift:150` — `let positions = MLXArray(0..<maxPositionEmbeddings).asType(.float32)`
- `Sources/VoxtralCore/MLXLMBridge.swift:336` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:345` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:363` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:372` — `let val = array[i].asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:380` — `let minVal = array.min().asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:381` — `let maxVal = array.max().asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:382` — `let meanVal = array.mean().asType(.float32).item(Float.self)`
- `Sources/VoxtralCore/MLXLMBridge.swift:520` — `let val = flattened[i].asType(.float32).item(Float.self)`
- … 31 de plus

</details>

<details><summary><code>prefill-step</code> — 8 hors tests</summary>

- `Sources/VoxtralCore/VoxtralModeling.swift:1169` — `let prefillChunkSize = 512`
- `Sources/VoxtralCore/VoxtralModeling.swift:1172` — `if totalSeqLen > prefillChunkSize {`
- `Sources/VoxtralCore/VoxtralModeling.swift:1173` — `for chunkStart in stride(from: 0, to: totalSeqLen, by: prefillChunkSize) {`
- `Sources/VoxtralCore/VoxtralModeling.swift:1174` — `let chunkEnd = min(chunkStart + prefillChunkSize, totalSeqLen)`
- `Sources/VoxtralCore/VoxtralModeling.swift:1352` — `let prefillChunkSize = 512`
- `Sources/VoxtralCore/VoxtralModeling.swift:1355` — `if totalSeqLen > prefillChunkSize {`
- `Sources/VoxtralCore/VoxtralModeling.swift:1356` — `for chunkStart in stride(from: 0, to: totalSeqLen, by: prefillChunkSize) {`
- `Sources/VoxtralCore/VoxtralModeling.swift:1357` — `let chunkEnd = min(chunkStart + prefillChunkSize, totalSeqLen)`

</details>

## 5. Indices stabilité

| Motif | Occ. (Sources / Tests) | Sens | Catalogue |
|---|---|---|---|
| `try-bang` | 1 / 0 | try! : crash sur erreur |  |
| `fatal` | 15 / 0 | Arrêt dur (entrée utilisateur ?) |  |
| `try-q` | 45 / 22 | Erreur avalée silencieusement ? |  |
| `unchecked-sendable` | 16 / 0 | Sendable non vérifié | feedback MLXArray/thread |
| `nonisolated-unsafe` | 8 / 5 | État global non isolé |  |
| `detached` | 1 / 1 | Task.detached (task-locals perdus, eval avant transfert) |  |
| `todo` | 2 / 0 | Dette marquée |  |
| `value-and-grad` | 1 / 0 | Gradient : ne pas mélanger avec l'inférence (ABBA) | piège 20 |
| `free-load` | 0 / 0 | Chargement via ModelFactoryRegistry (VLM d'abord) |  |
| `print` | 366 / 56 | print() dans une bibliothèque (profiler plutôt) | measurement |

<details><summary><code>try-bang</code> — 1 hors tests</summary>

- `Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:171` — `let bulletPattern = try! NSRegularExpression(pattern: "^\\s*(?:[-*+•]|\\d+[.)]) +")`

</details>

<details><summary><code>fatal</code> — 15 hors tests</summary>

- `Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:54` — `fatalError("Unsupported tok_embeddings type: \(type(of: tokEmbeddings))")`
- `Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:73` — `fatalError("Unsupported embeddings type: \(type(of: embeddings))")`
- `Sources/VoxtralCore/Utils/CustomLoadWeights.swift:43` — `fatalError("Received \(extras.count) parameters not in model: \n\(extrasList)")`
- `Sources/VoxtralCore/Utils/CustomLoadWeights.swift:50` — `fatalError("Missing \(missing.count) parameters: \n\(missingList)")`
- `Sources/VoxtralCore/Utils/CustomLoadWeights.swift:57` — `fatalError("Expected shape \(currentValue.shape) but received shape \(newValue.shape) for parameter \(key)")`
- `Sources/VoxtralCore/Utils/VoxtralModelLoading.swift:206` — `fatalError("Failed to cast quantized model back to VoxtralForConditionalGeneration")`
- `Sources/VoxtralCore/VoxtralModeling.swift:514` — `fatalError("Unsupported language_model type: \(type(of: language_model))")`
- `Sources/VoxtralCore/VoxtralModeling.swift:643` — `fatalError("Unsupported language_model type: \(type(of: language_model))")`
- `Sources/VoxtralCore/VoxtralModeling.swift:916` — `fatalError("Either input_ids or inputs_embeds must be provided")`
- `Sources/VoxtralCore/VoxtralModeling.swift:930` — `fatalError("Unsupported language_model type: \(type(of: language_model))")`
- `Sources/VoxtralCore/VoxtralModeling.swift:1018` — `fatalError("Unsupported lm_head type: \(type(of: lm_head))")`
- `Sources/VoxtralCore/VoxtralModeling.swift:1478` — `fatalError("Unsupported language_model type: \(type(of: language_model))")`
- … 3 de plus

</details>

<details><summary><code>try-q</code> — 45 hors tests</summary>

- `Sources/VoxtralApp/ContentView.swift:798` — `try? await manager.deleteModel(model.id)`
- `Sources/VoxtralBenchmark/BenchmarkCLI.swift:167` — `multiArray = try? mlxArray.toMLMultiArray()`
- `Sources/VoxtralCore/Realtime/VoxtralRealtimeConfiguration.swift:199` — `if let _ = try? JSONDecoder().decode(VoxtralRealtimeConfiguration.self, from: data) {`
- `Sources/VoxtralCore/TTS/VoxtralTTSModelLoading.swift:93` — `guard let data = try? Data(contentsOf: configURL),`
- `Sources/VoxtralCore/TTS/VoxtralTTSModelLoading.swift:94` — `let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:115` — `if let attrs = try? FileManager.default.attributesOfItem(atPath: dest.path),`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:149` — `try? await Task.sleep(nanoseconds: backoff * 1_000_000_000)`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:192` — `guard let re = try? NSRegularExpression(pattern: "^\(escaped)$") else { return false }`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:251` — `guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:317` — `let data = try? Data(contentsOf: indexPath),`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:318` — `let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:457` — `guard let attrs = try? fm.attributesOfItem(atPath: itemPath) else { continue }`
- … 33 de plus

</details>

<details><summary><code>unchecked-sendable</code> — 16 hors tests</summary>

- `Sources/VoxtralCore/CoreML/VoxtralCoreMLEncoder.swift:177` — `public class VoxtralCoreMLEncoder: @unchecked Sendable {`
- `Sources/VoxtralCore/Pipeline/VoxtralPipeline.swift:23` — `public class VoxtralPipeline: @unchecked Sendable {`
- `Sources/VoxtralCore/Pipeline/VoxtralTranscriptionManager.swift:52` — `public class VoxtralTranscriptionManager: @unchecked Sendable {`
- `Sources/VoxtralCore/Realtime/Pipeline/VoxtralRealtimeManager.swift:21` — `public class VoxtralRealtimeManager: @unchecked Sendable {`
- `Sources/VoxtralCore/Realtime/Pipeline/VoxtralRealtimePipeline.swift:19` — `public class VoxtralRealtimePipeline: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:20` — `public class VoxtralTTSPipeline: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:537` — `final class StreamContext: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSSynthesisManager.swift:20` — `public class VoxtralTTSSynthesisManager: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift:555` — `public struct GenerationChunk: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/VoxtralTTSProcessor.swift:13` — `public struct TTSSynthesisResult: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/VoxtralTTSProcessor.swift:323` — `public struct TTSStreamingChunk: @unchecked Sendable {`
- `Sources/VoxtralCore/TTS/VoxtralVoicePresets.swift:76` — `public class VoxtralVoicePresetManager: @unchecked Sendable {`
- … 4 de plus

</details>

<details><summary><code>nonisolated-unsafe</code> — 8 hors tests</summary>

- `Sources/VoxtralCore/CoreML/VoxtralCoreMLEncoder.swift:184` — `nonisolated(unsafe) public static var resourceBundle: Bundle?`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:28` — `nonisolated(unsafe) public static var customModelsDirectory: URL? = nil`
- `Sources/VoxtralCore/Utils/ModelDownloader.swift:32` — `nonisolated(unsafe) private static var _hubApi: HubApi? = nil`
- `Sources/VoxtralCore/Utils/VoxtralDebug.swift:11` — `nonisolated(unsafe) public static var enabled: Bool = false`
- `Sources/VoxtralCore/Utils/VoxtralDebug.swift:14` — `nonisolated(unsafe) public static var verboseGeneration: Bool = false`
- `Sources/VoxtralCore/VoxtralFeatureExtractor.swift:250` — `nonisolated(unsafe) var _melFiltersCache: [Int: MLXArray] = [:]`
- `Sources/VoxtralCore/VoxtralModeling.swift:19` — `nonisolated(unsafe) public var writeDebugToDump: (String) -> Void = { message in`
- `Sources/VoxtralCore/VoxtralModeling.swift:902` — `nonisolated(unsafe) private static var _mergeCallCount = 0`

</details>

<details><summary><code>detached</code> — 2 hors tests</summary>

- `Examples/ReferenceImplementation.swift:126` — `- Do NOT use Task.detached for loadModel() - it may cause threading issues`
- `Sources/VoxtralTTSStreamingDemo/StreamingDemoViewModel.swift:411` — `Task.detached { [weak self] in`

</details>

<details><summary><code>todo</code> — 2 hors tests</summary>

- `Sources/VoxtralCore/VoxtralComponents.swift:28` — `//import Transformers  // TODO: Integrate later when swift-transformers is stable`
- `Sources/VoxtralCore/VoxtralComponents.swift:566` — `// TODO: implémenter la logique spécifique transcription si nécessaire`

</details>

<details><summary><code>value-and-grad</code> — 1 hors tests</summary>

- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralVoiceEnrollment.swift:558` — `let (values, grads) = MLX.valueAndGrad(lossFn, argumentNumbers: [0, 1])(`

</details>

<details><summary><code>print</code> — 377 hors tests</summary>

- `Examples/ReferenceImplementation.swift:34` — `print("[Reference] Model already loaded")`
- `Examples/ReferenceImplementation.swift:38` — `print("[Reference] Creating pipeline...")`
- `Examples/ReferenceImplementation.swift:46` — `print("[Reference] Loading model (this may download on first run)...")`
- `Examples/ReferenceImplementation.swift:50` — `print("[Reference] \(Int(progress * 100))% - \(status)")`
- `Examples/ReferenceImplementation.swift:56` — `print("[Reference] Model loaded successfully!")`
- `Examples/ReferenceImplementation.swift:68` — `print("[Reference] Transcribing: \(audioURL.lastPathComponent)")`
- `Examples/ReferenceImplementation.swift:72` — `print("[Reference] Transcription complete (\(result.count) chars)")`
- `Examples/ReferenceImplementation.swift:85` — `print("[Reference] Chat with audio: \(audioURL.lastPathComponent)")`
- `Examples/ReferenceImplementation.swift:86` — `print("[Reference] Prompt: \(prompt.prefix(50))...")`
- `Examples/ReferenceImplementation.swift:90` — `print("[Reference] Chat complete (\(result.count) chars)")`
- `Examples/ReferenceImplementation.swift:101` — `print("[Reference] Model unloaded")`
- `Sources/VoxtralApp/TranscriptionManager.swift:173` — `print("[VoxtralApp] loadModel: already loading, skipping")`
- … 365 de plus

</details>

## 6. Alertes synthétiques

- 🟡 `cacheLimit` posé dans très peu de fichiers : vérifier que chaque chemin d'inférence en a un.
- 🟡 Aucun `asyncEval` : décodage probablement synchrone (T15).
- 🟡 Aucune réutilisation de préfixe KV repérée (T6) : chaque tour rejoue l'historique ?
- 🔴 Aucun type de profils de référence (`<bits>bit-fast|lean`, `.all`, `named`) : standard absent.
- 🟡 Module `VoxtralApp` : aucun test ne l'importe ni ne porte son nom.
- 🟡 Module `VoxtralBenchmark` : aucun test ne l'importe ni ne porte son nom.
- 🟡 Module `VoxtralTTSStreamingDemo` : aucun test ne l'importe ni ne porte son nom.
- 🟡 Module `VoxtralTranscriptionTest` : aucun test ne l'importe ni ne porte son nom.
- 🟡 22 fichier(s) suivis ressemblent à des artefacts (section 3).

## 7. Standard documentaire (references/knowledge-structure.md)

| Élément | Présent | Rôle |
|---|---|---|
| `BENCHMARKS.md` | ❌ | Lignes de mesure brutes, jamais éditées |
| `docs/References.md` | ❌ | Table des profils de référence mesurés |
| `docs/Weights.md` | ❌ | Poids / packs recommandés par profil |
| `docs/Benchmarks.md` | ❌ | Protocole de mesure |
| `docs/knowledge/index.md` | ❌ | Index de la base de connaissance |
| `docs/knowledge/log.md` | ❌ | Journal horodaté |
| `docs/knowledge/decisions` | ❌ | Décisions |
| `docs/knowledge/pitfalls` | ❌ | Pièges |
| `CHANGELOG.md` | ❌ | Journal des versions |
| `PLAN.md` | ❌ | Plan / chantiers |

