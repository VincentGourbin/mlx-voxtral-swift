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

/// Cross-platform home directory (macOS: ~/, iOS: app container Documents)
private func platformHomeDirectory() -> URL {
    #if os(iOS) || os(tvOS) || os(visionOS)
    return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    #else
    return FileManager.default.homeDirectoryForCurrentUser
    #endif
}

/// Model downloader with HuggingFace Hub integration
public class ModelDownloader {

    /// Override the default models directory. Set before first download.
    nonisolated(unsafe) public static var customModelsDirectory: URL? = nil

    /// Hub API instance
    // Swift 6: nonisolated(unsafe) for lazy-initialized singleton
    nonisolated(unsafe) private static var _hubApi: HubApi? = nil

    public static var hubApi: HubApi {
        if let existing = _hubApi { return existing }
        let api = createHubApi()
        _hubApi = api
        return api
    }

    private static func createHubApi() -> HubApi {
        // Disable network monitor that can incorrectly trigger offline mode
        // This happens when connection is detected as "constrained" or "expensive"
        setenv("CI_DISABLE_NETWORK_MONITOR", "1", 1)

        let base: URL?
        if let custom = customModelsDirectory {
            // HubApi appends "models/" to downloadBase, so pass the parent
            base = custom.deletingLastPathComponent()
        } else {
            base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        }

        // cache: nil disables swift-transformers' content-addressed blob cache
        // (~/.cache/huggingface/hub), so downloads land directly under
        // downloadBase — i.e. everything lives under ~/Library/Caches/models.
        return HubApi(
            downloadBase: base,
            cache: nil,
            useOfflineMode: false
        )
    }

    /// Recreate the HubApi to pick up a new customModelsDirectory.
    /// Call after setting customModelsDirectory.
    public static func reconfigureHubApi() {
        _hubApi = createHubApi()
    }

    // MARK: - Direct downloader (URLSession)

    /// swift-huggingface's downloader stalls on HuggingFace's cross-host LFS
    /// redirect for large weight files (it fetches metadata, then hangs at 0
    /// bytes on the CDN GET). A plain URLSession download follows that redirect
    /// and streams the file fine, so we do the byte transfer ourselves: list
    /// the repo tree via the Hub API, then download each matching file with
    /// URLSession into ~/Library/Caches/models/{org}/{repo}/{path}.
    /// Portable (no Python/curl), resumable-by-skip (completed files are kept).
    public static func downloadRepoDirect(
        repoId: String,
        revision: String = "main",
        matching globs: [String],
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        struct TreeEntry: Decodable { let type: String; let path: String; let size: Int? }

        let destDir = modelsDirectory.appendingPathComponent(repoId)
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        // 1. List the repo files.
        let treeURL = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/\(revision)?recursive=true")!
        var treeReq = URLRequest(url: treeURL)
        if case let .fixed(token) = tokenProviderValue, !token.isEmpty {
            treeReq.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (treeData, treeResp) = try await URLSession.shared.data(for: treeReq)
        guard (treeResp as? HTTPURLResponse)?.statusCode == 200 else {
            throw VoxtralError.loadingFailed("Cannot list files for \(repoId) (HTTP \((treeResp as? HTTPURLResponse)?.statusCode ?? -1))")
        }
        let entries = try JSONDecoder().decode([TreeEntry].self, from: treeData)
        let files = entries.filter { entry in
            entry.type == "file" && globs.contains { matchesGlob(entry.path, $0) }
        }
        guard !files.isEmpty else {
            throw VoxtralError.loadingFailed("No matching files for \(repoId)")
        }
        let totalBytes = files.reduce(0) { $0 + ($1.size ?? 0) }

        // 2. Download each file (skip ones already complete).
        var doneBytes = 0
        for file in files {
            let dest = destDir.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

            if let attrs = try? FileManager.default.attributesOfItem(atPath: dest.path),
               let size = attrs[.size] as? Int, size == (file.size ?? -1) {
                doneBytes += file.size ?? 0
                progress?(fraction(doneBytes, totalBytes), "Skipped \(file.path)")
                continue
            }

            progress?(fraction(doneBytes, totalBytes), "Downloading \(file.path)…")
            let fileURL = URL(string: "https://huggingface.co/\(repoId)/resolve/\(revision)/\(file.path)")!
            var req = URLRequest(url: fileURL)
            if case let .fixed(token) = tokenProviderValue, !token.isEmpty {
                req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            // Retry transient network failures (connection lost / offline) with
            // backoff — a brief drop shouldn't fail a multi-GB model download.
            let maxAttempts = 5
            var attempt = 0
            while true {
                attempt += 1
                do {
                    let (tmp, resp) = try await URLSession.shared.download(for: req)
                    guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                        throw VoxtralError.loadingFailed("Download failed for \(file.path) (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1))")
                    }
                    if FileManager.default.fileExists(atPath: dest.path) {
                        try FileManager.default.removeItem(at: dest)
                    }
                    try FileManager.default.moveItem(at: tmp, to: dest)
                    break
                } catch let error as URLError where Self.isTransient(error) && attempt < maxAttempts {
                    let backoff = UInt64(pow(2.0, Double(attempt))) // 2,4,8,16 s
                    progress?(fraction(doneBytes, totalBytes),
                              "Network dropped on \(file.path), retry \(attempt)/\(maxAttempts - 1) in \(backoff)s…")
                    try? await Task.sleep(nanoseconds: backoff * 1_000_000_000)
                }
            }
            doneBytes += file.size ?? 0
            progress?(fraction(doneBytes, totalBytes), "Downloaded \(file.path)")
        }

        progress?(1.0, "Download complete")
        return destDir
    }

    private static func fraction(_ done: Int, _ total: Int) -> Double {
        total > 0 ? min(1.0, Double(done) / Double(total)) : 1.0
    }

    /// Transient network errors worth retrying (connection lost, offline,
    /// timeout, DNS/host lookup).
    private static func isTransient(_ error: URLError) -> Bool {
        switch error.code {
        case .networkConnectionLost, .notConnectedToInternet, .timedOut,
             .cannotConnectToHost, .cannotFindHost, .dnsLookupFailed,
             .resourceUnavailable, .dataNotAllowed:
            return true
        default:
            return false
        }
    }

    /// Optional HF token for gated/private repos (from HF_TOKEN env).
    private static var tokenProviderValue: TokenProvider {
        if let t = ProcessInfo.processInfo.environment["HF_TOKEN"], !t.isEmpty {
            return .fixed(t)
        }
        return .none
    }

    private enum TokenProvider { case none, fixed(String) }

    /// fnmatch-style glob match with FNM_PATHNAME semantics (`*` does not cross `/`).
    static func matchesGlob(_ path: String, _ pattern: String) -> Bool {
        let escaped = NSRegularExpression.escapedPattern(for: pattern)
            .replacingOccurrences(of: "\\*", with: "[^/]*")
            .replacingOccurrences(of: "\\?", with: "[^/]")
        guard let re = try? NSRegularExpression(pattern: "^\(escaped)$") else { return false }
        return re.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)) != nil
    }

    /// Models directory. Canonical location for all downloaded models:
    /// ~/Library/Caches/models (the same base HubApi downloads to), unless
    /// overridden via customModelsDirectory.
    public static var modelsDirectory: URL {
        if let custom = customModelsDirectory { return custom }
        let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cachesDir.appendingPathComponent("models")
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
        // {org}/{repo} subdirectories — the layout HubApi resolves models into.
        return baseDir.appendingPathComponent(model.repoId)
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
        let homeDir = platformHomeDirectory()
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

    /// Find a model path (checks custom directory first, then Hub cache, then local directory)
    /// Only returns paths for complete downloads (all sharded files present)
    public static func findModelPath(for model: VoxtralModelInfo) -> URL? {
        // Check custom models directory first (HubApi downloads to customDir/{org}/{repo})
        if let customDir = customModelsDirectory {
            let customModelPath = customDir.appendingPathComponent(model.repoId)
            if FileManager.default.fileExists(atPath: customModelPath.appendingPathComponent("config.json").path) {
                let verification = verifyShardedModel(at: customModelPath)
                if verification.complete {
                    return customModelPath
                }
            }
        }

        // Check Hub cache
        if let hubPath = hubCachePath(for: model) {
            let verification = verifyShardedModel(at: hubPath)
            if verification.complete {
                return hubPath
            }
        }

        // Check local models directory (~/Library/Caches/models)
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

        let modelUrl = try await downloadRepoDirect(
            repoId: model.repoId,
            matching: ["*.json", "*.safetensors"],
            progress: progress
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

        let modelUrl = try await downloadRepoDirect(
            repoId: repoId,
            matching: ["*.json", "*.safetensors"],
            progress: progress
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

    /// Calculate directory size recursively.
    ///
    /// Walks with `atPath:` APIs (not the `URL`-based family) because a relocated
    /// model's large weight files are replaced with file symlinks to an external
    /// disk: `resourceValues(forKeys: [.fileSizeKey])` on a symlink reports the
    /// link's own size (a few bytes), not its target's. Each symlinked entry is
    /// resolved via `destinationOfSymbolicLink(atPath:)` (a raw `readlink`) rather
    /// than `resolvingSymlinksInPath()`, which silently no-ops and leaks the
    /// symlink's own near-zero size when the target is missing (e.g. an unmounted
    /// external disk); a broken symlink contributes 0 instead.
    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: url.path) else {
            return 0
        }

        var totalSize: Int64 = 0
        for case let relativePath as String in enumerator {
            let itemPath = url.appendingPathComponent(relativePath).path
            if (itemPath as NSString).lastPathComponent.hasPrefix(".") { continue }
            guard let attrs = try? fm.attributesOfItem(atPath: itemPath) else { continue }

            if (attrs[.type] as? FileAttributeType) == .typeSymbolicLink {
                guard let rawTarget = try? fm.destinationOfSymbolicLink(atPath: itemPath) else { continue }
                let targetPath = rawTarget.hasPrefix("/")
                    ? rawTarget
                    : URL(fileURLWithPath: itemPath).deletingLastPathComponent().appendingPathComponent(rawTarget).path
                guard let targetAttrs = try? fm.attributesOfItem(atPath: targetPath) else { continue }
                totalSize += (targetAttrs[.size] as? Int64) ?? 0
            } else {
                totalSize += (attrs[.size] as? Int64) ?? 0
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

    // MARK: - Convenience Methods for Default Model

    /// Check if the default/recommended model is downloaded
    public static func isDefaultModelDownloaded() -> Bool {
        findModelPath(for: ModelRegistry.defaultModel) != nil
    }

    /// Download the default/recommended model
    public static func downloadDefaultModel(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        try await download(ModelRegistry.defaultModel, progress: progress)
    }

    /// Delete the default/recommended model
    public static func deleteDefaultModel() throws {
        try deleteModel(ModelRegistry.defaultModel)
    }

    /// Get the default model info
    public static var defaultModel: VoxtralModelInfo {
        ModelRegistry.defaultModel
    }

    // MARK: - TTS Model Support

    /// Check if a TTS model is downloaded (checks for params.json instead of config.json)
    public static func isTTSModelDownloaded(_ model: VoxtralTTSModelInfo) -> Bool {
        findTTSModelPath(for: model) != nil
    }

    /// Find a TTS model path (checks Hub cache, then local directories)
    /// TTS models use params.json instead of config.json
    public static func findTTSModelPath(for model: VoxtralTTSModelInfo) -> URL? {
        let configFile = "params.json"

        // Check custom models directory
        if let customDir = customModelsDirectory {
            let customModelPath = customDir.appendingPathComponent(model.repoId)
            if FileManager.default.fileExists(atPath: customModelPath.appendingPathComponent(configFile).path) {
                return customModelPath
            }
        }

        // Check Hub cache (new location)
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let newPath = cacheDir
                .appendingPathComponent("models")
                .appendingPathComponent(model.repoId)
            if FileManager.default.fileExists(atPath: newPath.appendingPathComponent(configFile).path) {
                return newPath
            }
        }

        // Check legacy Hub cache
        let homeDir = platformHomeDirectory()
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
           let latestSnapshot = contents.sorted().last {
            let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
            if FileManager.default.fileExists(atPath: modelPath.appendingPathComponent(configFile).path) {
                return modelPath
            }
        }

        // Check local models directory
        let localDir = modelsDirectory.appendingPathComponent(model.repoId)
        if FileManager.default.fileExists(atPath: localDir.appendingPathComponent(configFile).path) {
            return localDir
        }

        return nil
    }

    /// Download a TTS model (includes voice embeddings)
    public static func downloadTTSModel(
        _ model: VoxtralTTSModelInfo,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = findTTSModelPath(for: model) {
            progress?(1.0, "TTS model already downloaded")
            return existingPath
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        progress?(0.1, "Downloading model files...")

        // Download safetensors, json, and voice embeddings (.pt files)
        let modelUrl = try await downloadRepoDirect(
            repoId: model.repoId,
            matching: ["*.json", "*.safetensors", "voice_embedding/*.pt", "voice_embedding/*.safetensors", "tekken.json"],
            progress: progress
        )

        progress?(1.0, "Download complete!")
        print("TTS model available at: \(modelUrl.path)")

        return modelUrl
    }

    /// Resolve a TTS model identifier, downloading if necessary
    public static func resolveTTSModel(
        _ identifier: String,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Try by ID in TTS registry
        if let model = VoxtralTTSRegistry.model(withId: identifier) {
            if let existingPath = findTTSModelPath(for: model) {
                return existingPath
            }
            return try await downloadTTSModel(model, progress: progress)
        }

        // Check if it's a local path with params.json
        let localURL = URL(fileURLWithPath: identifier)
        if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("params.json").path) {
            return localURL
        }

        // Try as a direct HuggingFace repo ID
        return try await downloadByRepoId(identifier, progress: progress)
    }

    // MARK: - Realtime Model Methods

    public static func isRealtimeModelDownloaded(_ model: VoxtralRealtimeModelInfo) -> Bool {
        findRealtimeModelPath(for: model) != nil
    }

    /// Find a Realtime model path (checks Hub cache, then local directories)
    public static func findRealtimeModelPath(for model: VoxtralRealtimeModelInfo) -> URL? {
        // Realtime models may have config.json (mlx-community) or params.json (Mistral)
        let configFiles = ["config.json", "params.json"]

        // Check custom models directory
        if let customDir = customModelsDirectory {
            let customModelPath = customDir.appendingPathComponent(model.repoId)
            for configFile in configFiles {
                if FileManager.default.fileExists(atPath: customModelPath.appendingPathComponent(configFile).path) {
                    return customModelPath
                }
            }
        }

        // Check Hub cache (new location)
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let newPath = cacheDir.appendingPathComponent("models").appendingPathComponent(model.repoId)
            for configFile in configFiles {
                if FileManager.default.fileExists(atPath: newPath.appendingPathComponent(configFile).path) {
                    return newPath
                }
            }
        }

        // Check legacy Hub cache
        let homeDir = platformHomeDirectory()
        let hubCache = homeDir.appendingPathComponent(".cache/huggingface/hub")
        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
           let latestSnapshot = contents.sorted().last {
            let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
            for configFile in configFiles {
                if FileManager.default.fileExists(atPath: modelPath.appendingPathComponent(configFile).path) {
                    return modelPath
                }
            }
        }

        // Check local models directory
        let localDir = modelsDirectory.appendingPathComponent(model.repoId)
        for configFile in configFiles {
            if FileManager.default.fileExists(atPath: localDir.appendingPathComponent(configFile).path) {
                return localDir
            }
        }

        return nil
    }

    /// Download a Realtime model
    public static func downloadRealtimeModel(
        _ model: VoxtralRealtimeModelInfo,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        if let existingPath = findRealtimeModelPath(for: model) {
            progress?(1.0, "Realtime model already downloaded")
            return existingPath
        }

        progress?(0.0, "Starting download of \(model.name)...")
        progress?(0.1, "Downloading model files...")

        let modelUrl = try await downloadRepoDirect(
            repoId: model.repoId,
            matching: ["*.json", "*.safetensors", "tekken.json"],
            progress: progress
        )

        progress?(1.0, "Download complete!")
        return modelUrl
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
