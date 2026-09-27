# Scan mlx-swift-patterns — `/home/user/mlx-voxtral-swift` @ `9392ed1`

| Pattern | Mode | Sévérité | Sources | Tests | Titre |
|---|---|---|---|---|---|
| MLX-001 | mechanical | basse | 0 | 0 | API mémoire GPU.* dépréciée → Memory.* |
| MLX-002 | mechanical | haute | 28 | 1 | Constante scalaire fp32 qui promeut tout le graphe |
| MLX-003 | mechanical | haute | 1 | 0 | AsyncThrowingStream dont la Task n'est pas annulée à la terminaison |
| MLX-004 | report | haute | 0 | 0 | MLXArray paresseux qui traverse une frontière de thread/acteur |
| MLX-005 | report | haute | 0 | 0 | loadModelContainer libre pour un modèle aussi publié par MLXVLM |
| MLX-006 | report | moyenne | 0 | 0 | prepare surchargé qui ignore prefillStepSize |
| MLX-007 | report | haute | 0 | 0 | QuantizedKVCache lu comme (K, V) par des couches à KV partagé |
| MLX-008 | report | haute | 0 | 0 | Rollback spéculatif sur RotatingKVCache au-delà de la fenêtre |
| MLX-009 | report | haute | 0 | 0 | asType(layer.weight.dtype) sur un module potentiellement quantifié |
| MLX-010 | report | moyenne | 0 | 0 | Aucune Memory.cacheLimit sur les chemins d'inférence |
| MLX-011 | report | haute | 0 | 0 | Serveur d'inférence exposé (0.0.0.0, /metrics non authentifié, file://) |
| MLX-012 | report | haute | 0 | 0 | Modèle « téléchargé » dès un seul fichier safetensors |
| MLX-013 | report | haute | 0 | 0 | Dictionnaire de closures @Sendable passé à ModelTypeRegistry(creators:) |
| MLX-014 | report | moyenne | 0 | 0 | Détokenisation incrémentale par graphèmes ou token par token |
| MLX-015 | report | basse | 0 | 3 | Dossier de modèle exposé par un lien symbolique sur le dossier racine |

## MLX-002 — Constante scalaire fp32 qui promeut tout le graphe (28)

- `Sources/VoxtralCore/Models/VoxtralLlama.swift:527` — `let offsetFloat = MLXArray(Float(offset))`
- `Sources/VoxtralCore/Models/VoxtralLlama.swift:530` — `var mask = MLX.where(causalCondition, MLXArray(Float(0)), MLXArray(Float(-1e9)))`
- `Sources/VoxtralCore/Models/VoxtralLlama.swift:534` — `let windowSizeFloat = MLXArray(Float(windowSize))`
- `Sources/VoxtralCore/Models/VoxtralLlama.swift:536` — `let windowMask = MLX.where(windowCondition, MLXArray(Float(0)), MLXArray(Float(-1e9)))`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralEnrollmentLosses.swift:124` — `guard !resolutions.isEmpty else { return MLXArray(Float(0)) }`
- `Sources/VoxtralCore/TTS/VoiceCloning/VoxtralEnrollmentLosses.swift:125` — `var total = MLXArray(Float(0))`
- `Sources/VoxtralCore/TTS/VoxtralCodecDecoder.swift:219` — `let causalMaskValues = MLX.where(dist .> MLXArray(Int32(0)), MLXArray(Float(-1e9)), MLXArray(Float(0)))`
- `Sources/VoxtralCore/TTS/VoxtralCodecDecoder.swift:224` — `let windowMask = MLX.where(dist .< MLXArray(Int32(-windowSize)), MLXArray(Float(-1e9)), MLXArray(Float(0)))`
- `Sources/VoxtralCore/TTS/VoxtralCodecDecoder.swift:339` — `MLXArray(Float(1e-8))`
- `Sources/VoxtralCore/TTS/VoxtralCodecDecoder.swift:360` — `return (2.0 * indices.asType(.float32) / MLXArray(Float(codebookSize - 1))) - 1.0`
- `Sources/VoxtralCore/TTS/VoxtralCodecEncoder.swift:189` — `MLXArray(Float(1e-8))`
- `Sources/VoxtralCore/TTS/VoxtralFlowMatching.swift:135` — `MLXArray(-log(theta)) * MLXArray(0..<half).asType(.float32) / MLXArray(Float(half))`
- `Sources/VoxtralCore/TTS/VoxtralFlowMatching.swift:324` — `let clamped = MLX.clip(xt, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))`
- `Sources/VoxtralCore/TTS/VoxtralFlowMatching.swift:326` — `let acousticCodes = MLX.clip(scaled, min: MLXArray(Float(0)), max: MLXArray(Float(acousticCodebookSize - 1)))`
- `Sources/VoxtralCore/TTS/VoxtralFlowMatching.swift:341` — `let clamped = MLX.clip(x, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))`
- … 13 de plus

## MLX-003 — AsyncThrowingStream dont la Task n'est pas annulée à la terminaison (1)

- `Sources/VoxtralCore/TTS/Pipeline/VoxtralTTSPipeline.swift:556` — `return AsyncThrowingStream { continuation in`
