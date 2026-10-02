// swift-tools-version: 6.2
// MLX Voxtral Swift - Speech-to-Text with Apple Silicon acceleration
// Based on the Python implementation: https://github.com/mzbac/mlx.voxtral
// Aligned with flux-2-swift-mlx for API compatibility

import PackageDescription

let package = Package(
    name: "MLXVoxtralSwift",
    platforms: [
        .macOS(.v15),
        .iOS(.v17)
    ],
    products: [
        // Core library for integration into other projects
        .library(
            name: "VoxtralCore",
            targets: ["VoxtralCore"]
        ),
        // Standalone macOS app with SwiftUI interface
        .executable(
            name: "VoxtralApp",
            targets: ["VoxtralApp"]
        ),
        // Command-line transcription tool
        .executable(
            name: "VoxtralCLI",
            targets: ["VoxtralTranscriptionTest"]
        ),
        // TTS Streaming demo app
        .executable(
            name: "VoxtralTTSStreamingDemo",
            targets: ["VoxtralTTSStreamingDemo"]
        ),
    ],
    dependencies: [
        // Aligned with flux-2-swift-mlx dependency versions
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // Pinned to branch (not a version) because the app that embeds this framework
        // also depends on mlx-swift-lm's `main` (needed by a sibling framework), and
        // SwiftPM refuses to resolve a version requirement against a branch requirement
        // on the same package. `main` currently sits past the `prepare(...)` protocol
        // change (added `prefill: PrefillParameters`) that this package's LanguageModel
        // conformances were updated for. Revisit once ml-explore cuts a tag beyond 3.31.4.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.5.1")
    ],
    targets: [
        // Core library containing all Voxtral model implementations
        .target(
            name: "VoxtralCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Only the Hub module of swift-transformers is imported (ModelDownloader)
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler")
            ]
        ),
        // SwiftUI macOS application
        .executableTarget(
            name: "VoxtralApp",
            dependencies: [
                "VoxtralCore"
            ],
            // The Core ML encoder is downloaded at run time (VoxtralHybridEncoder), never bundled (ASK-27, K-28)
            exclude: ["Resources/Info.plist"]
        ),
        // CLI transcription tool
        .executableTarget(
            name: "VoxtralTranscriptionTest",
            dependencies: [
                "VoxtralCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler")
            ]
        ),
        // TTS Streaming demo app
        .executableTarget(
            name: "VoxtralTTSStreamingDemo",
            dependencies: [
                "VoxtralCore"
            ]
        ),
        // Unit tests
        .testTarget(
            name: "VoxtralCoreTests",
            dependencies: [
                "VoxtralCore"
            ]
        ),
    ]
)
