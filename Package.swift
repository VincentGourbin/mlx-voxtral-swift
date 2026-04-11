// swift-tools-version: 6.2
// MLX Voxtral Swift - Speech-to-Text with Apple Silicon acceleration
// Based on the Python implementation: https://github.com/mzbac/mlx.voxtral
// Aligned with flux-2-swift-mlx for API compatibility

import PackageDescription

let package = Package(
    name: "MLXVoxtralSwift",
    platforms: [
        .macOS(.v15)
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
        // Performance benchmark tool
        .executable(
            name: "VoxtralBenchmark",
            targets: ["VoxtralBenchmark"]
        ),
        // TTS Streaming demo app
        .executable(
            name: "VoxtralTTSStreamingDemo",
            targets: ["VoxtralTTSStreamingDemo"]
        ),
    ],
    dependencies: [
        // Aligned with flux-2-swift-mlx dependency versions
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.30.6"),
        .package(path: "../swift-mlx-profiler")
    ],
    targets: [
        // Core library containing all Voxtral model implementations
        .target(
            name: "VoxtralCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler")
            ]
        ),
        // SwiftUI macOS application
        .executableTarget(
            name: "VoxtralApp",
            dependencies: [
                "VoxtralCore"
            ],
            exclude: ["Resources/Info.plist"],
            resources: [
                .copy("Resources/VoxtralEncoderFull.mlmodelc")
            ]
        ),
        // CLI transcription tool
        .executableTarget(
            name: "VoxtralTranscriptionTest",
            dependencies: [
                "VoxtralCore",
                .product(name: "MLXProfiler", package: "swift-mlx-profiler")
            ]
        ),
        // Performance benchmark
        .executableTarget(
            name: "VoxtralBenchmark",
            dependencies: [
                "VoxtralCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
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
