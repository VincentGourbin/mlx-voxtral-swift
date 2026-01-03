/**
 * ModelDownloader - Downloads Voxtral models from HuggingFace Hub
 *
 * Uses the Hub module from swift-transformers for downloads.
 * Provides progress tracking and local caching.
 */

import Foundation
import Hub

/// Progress callback for download updates
public typealias DownloadProgressCallback = (Double, String) -> Void

/// Model downloader with HuggingFace Hub integration
public class ModelDownloader {

    /// Default Hub API instance (uses system cache directory)
    private static var hubApi: HubApi = {
        HubApi(downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first)
    }()

    /// Default models directory (in user's home)
    public static var modelsDirectory: URL {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(".voxtral").appendingPathComponent("models")
    }

    /// Check if a model is already downloaded
    public static func isModelDownloaded(_ model: VoxtralModelInfo, in directory: URL? = nil) -> Bool {
        let modelPath = localPath(for: model, in: directory)
        let configPath = modelPath.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: configPath.path)
    }

    /// Get local path for a model
    public static func localPath(for model: VoxtralModelInfo, in directory: URL? = nil) -> URL {
        let baseDir = directory ?? modelsDirectory
        // Use repo ID as folder name, replacing "/" with "--"
        let folderName = model.repoId.replacingOccurrences(of: "/", with: "--")
        return baseDir.appendingPathComponent(folderName)
    }

    /// List all downloaded models
    public static func listDownloadedModels(in directory: URL? = nil) -> [VoxtralModelInfo] {
        return ModelRegistry.models.filter { model in
            findModelPath(for: model) != nil
        }
    }

    /// Get the HuggingFace Hub cache path for a model
    public static func hubCachePath(for model: VoxtralModelInfo) -> URL? {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        // Find the latest snapshot
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            return modelPath
        }

        return nil
    }

    /// Find a model path (checks Hub cache first, then local directory)
    public static func findModelPath(for model: VoxtralModelInfo) -> URL? {
        // Check Hub cache first
        if let hubPath = hubCachePath(for: model) {
            return hubPath
        }

        // Check local models directory
        let localDir = localPath(for: model)
        if FileManager.default.fileExists(atPath: localDir.appendingPathComponent("config.json").path) {
            return localDir
        }

        return nil
    }

    /// Download a model using Hub API
    public static func download(
        _ model: VoxtralModelInfo,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = findModelPath(for: model) {
            progress?(1.0, "Model already downloaded")
            return existingPath
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        progress?(0.1, "Downloading model files...")

        // Use Hub API to download the snapshot
        let modelUrl = try await hubApi.snapshot(
            from: model.repoId,
            matching: ["*.json", "*.safetensors"]
        )

        progress?(1.0, "Download complete!")
        print("\nDownload complete: \(modelUrl.path)")

        return modelUrl
    }

    /// Download a model by repo ID directly
    public static func downloadByRepoId(
        _ repoId: String,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        progress?(0.0, "Starting download...")
        print("\nDownloading from HuggingFace: \(repoId)")

        let modelUrl = try await hubApi.snapshot(
            from: repoId,
            matching: ["*.json", "*.safetensors"]
        )

        progress?(1.0, "Download complete!")
        print("Model available at: \(modelUrl.path)")

        return modelUrl
    }

    /// Resolve a model identifier to a local path, downloading if necessary
    public static func resolveModel(
        _ identifier: String,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Try to find by ID first
        if let model = ModelRegistry.model(withId: identifier) {
            if let existingPath = findModelPath(for: model) {
                return existingPath
            }
            return try await download(model, progress: progress)
        }

        // Try to find by repo ID
        if let model = ModelRegistry.model(withRepoId: identifier) {
            if let existingPath = findModelPath(for: model) {
                return existingPath
            }
            return try await download(model, progress: progress)
        }

        // Check if it's a local path
        let localURL = URL(fileURLWithPath: identifier)
        if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("config.json").path) {
            return localURL
        }

        // Try as a direct HuggingFace repo ID
        return try await downloadByRepoId(identifier, progress: progress)
    }
}
