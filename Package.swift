// swift-tools-version: 6.0
// MLX Voxtral Swift - Speech-to-Text with Apple Silicon acceleration
// Based on the Python implementation: https://github.com/mzbac/mlx.voxtral

import PackageDescription

let package = Package(
    name: "MLXVoxtralSwift",
    platforms: [
        .macOS(.v14)
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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.29.0")
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
                .product(name: "MLXLLM", package: "mlx-swift-lm")
            ]
        ),
        // SwiftUI macOS application
        .executableTarget(
            name: "VoxtralApp",
            dependencies: [
                "VoxtralCore"
            ],
            exclude: ["Resources"]
        ),
        // CLI transcription tool
        .executableTarget(
            name: "VoxtralTranscriptionTest",
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
