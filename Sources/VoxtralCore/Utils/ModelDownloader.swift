/**
 * ModelDownloader - Downloads Voxtral models from HuggingFace Hub
 *
 * Uses the Hub module from swift-transformers for downloads.
 * Provides progress tracking and local caching.
 */

import Foundation
import Hub

/// Progress callback for download updates
/// Swift 6: @Sendable for safe cross-isolation usage
public typealias DownloadProgressCallback = @Sendable (Double, String) -> Void

/// Model downloader with HuggingFace Hub integration
public class ModelDownloader {

    /// Default Hub API instance (uses system cache directory, forces online mode)
    // Swift 6: nonisolated(unsafe) for lazy-initialized singleton
    nonisolated(unsafe) private static var hubApi: HubApi = {
        // Disable network monitor that can incorrectly trigger offline mode
        // This happens when connection is detected as "constrained" or "expensive"
        setenv("CI_DISABLE_NETWORK_MONITOR", "1", 1)

        return HubApi(
            downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first,
            useOfflineMode: false
        )
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
    /// Checks both the new Library/Caches location and the legacy ~/.cache/huggingface location
    public static func hubCachePath(for model: VoxtralModelInfo) -> URL? {
        // First check the new location: ~/Library/Caches/models/{org}/{repo}
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let newPath = cacheDir
                .appendingPathComponent("models")
                .appendingPathComponent(model.repoId)

            if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
                return newPath
            }
        }

        // Then check the legacy location: ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/...
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
    /// Only returns paths for complete downloads (all sharded files present)
    public static func findModelPath(for model: VoxtralModelInfo) -> URL? {
        // Check Hub cache first
        if let hubPath = hubCachePath(for: model) {
            let verification = verifyShardedModel(at: hubPath)
            if verification.complete {
                return hubPath
            }
        }

        // Check local models directory
        let localDir = localPath(for: model)
        if FileManager.default.fileExists(atPath: localDir.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: localDir)
            if verification.complete {
                return localDir
            }
        }

        // Check project voxtral_models directory
        let projectModelsDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("voxtral_models")
            .appendingPathComponent(model.repoId.split(separator: "/").last.map(String.init) ?? model.id)
        if FileManager.default.fileExists(atPath: projectModelsDir.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: projectModelsDir)
            if verification.complete {
                return projectModelsDir
            }
        }

        return nil
    }

    /// Verify that a sharded model has all required safetensors files
    public static func verifyShardedModel(at path: URL) -> (complete: Bool, missing: [String]) {
        let indexPath = path.appendingPathComponent("model.safetensors.index.json")

        // If no index file, it's either a single-file model or not sharded
        guard FileManager.default.fileExists(atPath: indexPath.path),
              let data = try? Data(contentsOf: indexPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = json["weight_map"] as? [String: String] else {
            return (true, [])
        }

        // Get unique safetensors files from the weight map
        let requiredFiles = Set(weightMap.values)
        var missingFiles: [String] = []

        for filename in requiredFiles {
            let filePath = path.appendingPathComponent(filename)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                missingFiles.append(filename)
            }
        }

        return (missingFiles.isEmpty, missingFiles)
    }

    /// Download a model using Hub API
    public static func download(
        _ model: VoxtralModelInfo,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded and complete
        if let existingPath = findModelPath(for: model) {
            let verification = verifyShardedModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Model already downloaded")
                return existingPath
            } else {
                print("Warning: Incomplete download detected. Missing files: \(verification.missing)")
                print("Re-downloading...")
            }
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

        // Verify the download is complete
        let verification = verifyShardedModel(at: modelUrl)
        if !verification.complete {
            print("\nWarning: Download may be incomplete. Missing files: \(verification.missing)")
            print("You may need to manually download these files or re-run the download.")
        }

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

    /// Get the size of a downloaded model in bytes
    public static func modelSize(for model: VoxtralModelInfo) -> Int64? {
        guard let path = findModelPath(for: model) else { return nil }
        return directorySize(at: path)
    }

    /// Calculate directory size recursively
    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
            return 0
        }

        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(fileSize)
            }
        }
        return totalSize
    }

    /// Format bytes as human-readable string
    public static func formatSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    /// Delete a downloaded model
    public static func deleteModel(_ model: VoxtralModelInfo) throws {
        guard let path = findModelPath(for: model) else {
            throw ModelDownloaderError.modelNotFound
        }

        // Determine if it's in Hub cache (need to delete parent folder) or local directory
        let pathString = path.path

        if pathString.contains("/.cache/huggingface/hub/") {
            // Legacy Hub cache: delete the models--org--repo folder
            // path is .../snapshots/hash, so go up 2 levels
            let modelFolder = path.deletingLastPathComponent().deletingLastPathComponent()
            try FileManager.default.removeItem(at: modelFolder)
        } else if pathString.contains("/Library/Caches/models/") {
            // New Hub cache: delete the repo folder
            try FileManager.default.removeItem(at: path)
        } else {
            // Local directory
            try FileManager.default.removeItem(at: path)
        }
    }
}

/// Errors for model downloading
public enum ModelDownloaderError: LocalizedError {
    case modelNotFound
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model not found locally"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        }
    }
}
