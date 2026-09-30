/**
 * ModelDownloader - Downloads Voxtral models from HuggingFace Hub
 *
 * Uses the Hub module from swift-transformers for downloads.
 * Provides progress tracking and local caching.
 */

import CryptoKit
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
    public static var customModelsDirectory: URL? {
        get { _customModelsDirectory.get() }
        set { _customModelsDirectory.set(newValue) }
    }
    private static let _customModelsDirectory = Locked<URL?>(nil)

    /// Hub API instance (lazily created once, under a lock)
    private static let _hubApi = Locked<HubApi?>(nil)

    public static var hubApi: HubApi {
        _hubApi.withLock { api in
            if let existing = api { return existing }
            let created = createHubApi()
            api = created
            return created
        }
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
        _hubApi.set(createHubApi())
    }

    // MARK: - Direct downloader (URLSession)

    /// swift-huggingface's downloader stalls on HuggingFace's cross-host LFS
    /// redirect for large weight files (it fetches metadata, then hangs at 0
    /// bytes on the CDN GET). A plain URLSession download follows that redirect
    /// and streams the file fine, so we do the byte transfer ourselves: list
    /// the repo tree via the Hub API, then download each matching file with
    /// URLSession into ~/Library/Caches/models/{org}/{repo}/{path}.
    /// Portable (no Python/curl), resumable-by-skip (completed files are kept).
    ///
    /// Each LFS file is checked against the tree's SHA-256 (`lfs.oid`) before it
    /// replaces its destination; the completeness manifest (`manifestFileName`)
    /// is removed first and written last, so an interrupted download is never
    /// taken for a complete one (MLX-012).
    public static func downloadRepoDirect(
        repoId: String,
        revision: String = "main",
        matching globs: [String],
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        struct LFS: Decodable { let oid: String; let size: Int? }
        struct TreeEntry: Decodable { let type: String; let path: String; let size: Int?; let lfs: LFS? }

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

        // SHA-256 already proven for kept files, then no manifest until the end.
        let manifestURL = destDir.appendingPathComponent(manifestFileName)
        let previous = readManifest(at: destDir)
        try? FileManager.default.removeItem(at: manifestURL)

        // 2. Download each file (skip ones already complete).
        var doneBytes = 0
        var written: [DownloadManifest.Entry] = []
        for file in files {
            let dest = destDir.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            let expectedSHA = file.lfs?.oid.lowercased()

            let previousSHA = previous?.files.first(where: { $0.path == file.path })?.sha256
            if let size = fileSize(at: dest), size == (file.size ?? -1),
               expectedSHA == nil || previousSHA == nil || previousSHA == expectedSHA {
                doneBytes += file.size ?? 0
                written.append(.init(path: file.path, size: size, sha256: expectedSHA ?? previousSHA))
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
                        try? FileManager.default.removeItem(at: tmp)
                        throw VoxtralError.loadingFailed("Download failed for \(file.path) (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1))")
                    }
                    try placeVerifiedFile(tmp, at: dest, name: file.path,
                                          expectedSize: file.size, expectedSHA256: expectedSHA)
                    break
                } catch let error as URLError where Self.isTransient(error) && attempt < maxAttempts {
                    let backoff = UInt64(pow(2.0, Double(attempt))) // 2,4,8,16 s
                    progress?(fraction(doneBytes, totalBytes),
                              "Network dropped on \(file.path), retry \(attempt)/\(maxAttempts - 1) in \(backoff)s…")
                    try? await Task.sleep(nanoseconds: backoff * 1_000_000_000)
                }
            }
            doneBytes += file.size ?? 0
            written.append(.init(path: file.path, size: fileSize(at: dest) ?? file.size ?? 0, sha256: expectedSHA))
            progress?(fraction(doneBytes, totalBytes), "Downloaded \(file.path)")
        }

        // 3. Every file is in place: the manifest is the last thing written.
        try writeManifest(DownloadManifest(repoId: repoId, revision: revision, files: written), to: destDir)

        progress?(1.0, "Download complete")
        return destDir
    }

    /// Moves a downloaded temporary file to `dest` only if its size and SHA-256
    /// match the Hub tree; otherwise deletes it and throws (nothing is placed).
    static func placeVerifiedFile(
        _ tmp: URL, at dest: URL, name: String, expectedSize: Int?, expectedSHA256: String?
    ) throws {
        let fm = FileManager.default
        if let expectedSize, let size = fileSize(at: tmp), size != expectedSize {
            try? fm.removeItem(at: tmp)
            throw VoxtralError.loadingFailed("Size mismatch for \(name): \(size) bytes, expected \(expectedSize)")
        }
        if let expectedSHA256 {
            let actual = try sha256Hex(of: tmp)
            guard actual == expectedSHA256.lowercased() else {
                try? fm.removeItem(at: tmp)
                throw VoxtralError.loadingFailed("SHA-256 mismatch for \(name): \(actual), expected \(expectedSHA256)")
            }
        }
        if fm.fileExists(atPath: dest.path) || (try? fm.destinationOfSymbolicLink(atPath: dest.path)) != nil {
            try fm.removeItem(at: dest)
        }
        try fm.moveItem(at: tmp, to: dest)
    }

    /// SHA-256 of a file, read in 8 MiB chunks (weights never sit whole in memory).
    static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk: Data? = try autoreleasepool { try handle.read(upToCount: 8 << 20) }
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Size of the file at `url`, following symlinks (relocated weights); nil if absent.
    static func fileSize(at url: URL) -> Int? {
        var info = stat()
        guard stat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { return nil }
        return Int(info.st_size)
    }

    /// Throws when a folder just downloaded does not pass `isComplete(folder:)`.
    private static func requireComplete(_ folder: URL, repoId: String, requiresVoices: Bool = false) throws {
        guard isComplete(folder: folder, requiresVoices: requiresVoices) else {
            throw VoxtralError.loadingFailed("Download of \(repoId) is incomplete (\(folder.path)); run it again to resume")
        }
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
        isComplete(folder: localPath(for: model, in: directory))
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
    /// Only returns paths for complete downloads (`isComplete(folder:)`)
    public static func findModelPath(for model: VoxtralModelInfo) -> URL? {
        candidateFolders(for: model).first { isComplete(folder: $0) }
    }

    /// The folder holding a model on disk, complete or not (size, deletion of a partial download).
    static func locateModelFolder(for model: VoxtralModelInfo) -> URL? {
        candidateFolders(for: model).first
    }

    /// Folders with a `config.json`, in lookup order: custom models directory (HubApi
    /// downloads to customDir/{org}/{repo}), Hub cache, ~/Library/Caches/models, then the
    /// project's voxtral_models directory.
    private static func candidateFolders(for model: VoxtralModelInfo) -> [URL] {
        var folders: [URL] = []
        if let customDir = customModelsDirectory {
            folders.append(customDir.appendingPathComponent(model.repoId))
        }
        if let hubPath = hubCachePath(for: model) {
            folders.append(hubPath)
        }
        folders.append(localPath(for: model))
        folders.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("voxtral_models")
            .appendingPathComponent(model.repoId.split(separator: "/").last.map(String.init) ?? model.id))
        return folders.filter {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path)
        }
    }

    /// Verify that a sharded model has all required safetensors files.
    /// Without an index, only a single-file `model.safetensors` counts as complete:
    /// the Hub lists `model.safetensors.index.json` after the shards, so "no index"
    /// is the normal state of an interrupted download (MLX-012).
    public static func verifyShardedModel(at path: URL) -> (complete: Bool, missing: [String]) {
        let indexName = "model.safetensors.index.json"
        let indexPath = path.appendingPathComponent(indexName)

        guard FileManager.default.fileExists(atPath: indexPath.path) else {
            let single = fileSize(at: path.appendingPathComponent("model.safetensors")) != nil
            return single ? (true, []) : (false, [indexName])
        }
        guard let data = try? Data(contentsOf: indexPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = json["weight_map"] as? [String: String], !weightMap.isEmpty else {
            return (false, [indexName])
        }

        // Get unique safetensors files from the weight map
        let requiredFiles = Set(weightMap.values)
        var missingFiles: [String] = []

        for filename in requiredFiles.sorted() {
            if fileSize(at: path.appendingPathComponent(filename)) == nil {
                missingFiles.append(filename)
            }
        }

        return (missingFiles.isEmpty, missingFiles)
    }

    // MARK: - Completeness manifest

    /// Written last by `downloadRepoDirect`: its presence proves the download finished.
    public static let manifestFileName = ".voxtral-complete.json"

    struct DownloadManifest: Codable {
        struct Entry: Codable { let path: String; let size: Int; let sha256: String? }
        var version = 1
        let repoId: String
        let revision: String?
        let files: [Entry]
    }

    static func readManifest(at folder: URL) -> DownloadManifest? {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent(manifestFileName)) else { return nil }
        return try? JSONDecoder().decode(DownloadManifest.self, from: data)
    }

    static func writeManifest(_ manifest: DownloadManifest, to folder: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: folder.appendingPathComponent(manifestFileName), options: .atomic)
    }

    /// True when `folder` holds a readable manifest and every file it lists is present with its size.
    static func hasCompleteManifest(_ folder: URL) -> Bool {
        guard let manifest = readManifest(at: folder), !manifest.files.isEmpty else { return false }
        return manifest.files.allSatisfy { fileSize(at: folder.appendingPathComponent($0.path)) == $0.size }
    }

    /// Single completeness rule for STT, TTS and Realtime folders.
    ///
    /// With a manifest: every listed file is present with its recorded size. Without one
    /// (folders downloaded before the manifest existed): the safetensors index, all its
    /// shards, `tekken.json` and, if `requiresVoices`, at least one voice embedding must be
    /// present; the manifest is then written so the next check is a plain read.
    static func isComplete(folder: URL, requiresVoices: Bool = false) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: folder.appendingPathComponent(manifestFileName).path) {
            return hasCompleteManifest(folder)
        }

        guard fm.fileExists(atPath: folder.appendingPathComponent("model.safetensors.index.json").path),
              verifyShardedModel(at: folder).complete,
              fileSize(at: folder.appendingPathComponent("tekken.json")) != nil else { return false }
        let voices = (try? fm.contentsOfDirectory(atPath: folder.appendingPathComponent("voice_embedding").path))?
            .filter { $0.hasSuffix(".safetensors") || $0.hasSuffix(".pt") } ?? []
        if requiresVoices && voices.isEmpty { return false }

        let top = ((try? fm.contentsOfDirectory(atPath: folder.path)) ?? [])
            .filter { !$0.hasPrefix(".") && ($0.hasSuffix(".json") || $0.hasSuffix(".safetensors")) }
        let paths = top + voices.map { "voice_embedding/\($0)" }
        let entries = paths.sorted().compactMap { path in
            fileSize(at: folder.appendingPathComponent(path)).map { DownloadManifest.Entry(path: path, size: $0, sha256: nil) }
        }
        try? writeManifest(DownloadManifest(repoId: folder.lastPathComponent, revision: nil, files: entries), to: folder)
        return true
    }

    /// Download a model using Hub API
    public static func download(
        _ model: VoxtralModelInfo,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        // findModelPath only returns complete folders (`isComplete(folder:)`)
        if let existingPath = findModelPath(for: model) {
            progress?(1.0, "Model already downloaded")
            return existingPath
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        progress?(0.1, "Downloading model files...")

        let modelUrl = try await downloadRepoDirect(
            repoId: model.repoId,
            revision: model.revision ?? "main",
            matching: ["*.json", "*.safetensors"],
            progress: progress
        )
        try requireComplete(modelUrl, repoId: model.repoId)

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
        guard let path = locateModelFolder(for: model) else { return nil }
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
        guard let path = locateModelFolder(for: model) else {
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
    /// TTS models use params.json instead of config.json; only complete folders
    /// (`isComplete(folder:requiresVoices:)`) are returned
    public static func findTTSModelPath(for model: VoxtralTTSModelInfo) -> URL? {
        let configFile = "params.json"
        func complete(_ folder: URL) -> Bool {
            FileManager.default.fileExists(atPath: folder.appendingPathComponent(configFile).path)
                && isComplete(folder: folder, requiresVoices: true)
        }

        // Check custom models directory
        if let customDir = customModelsDirectory {
            let customModelPath = customDir.appendingPathComponent(model.repoId)
            if complete(customModelPath) {
                return customModelPath
            }
        }

        // Check Hub cache (new location)
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let newPath = cacheDir
                .appendingPathComponent("models")
                .appendingPathComponent(model.repoId)
            if complete(newPath) {
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
            if complete(modelPath) {
                return modelPath
            }
        }

        // Check local models directory
        let localDir = modelsDirectory.appendingPathComponent(model.repoId)
        if complete(localDir) {
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
            revision: model.revision ?? "main",
            matching: ["*.json", "*.safetensors", "voice_embedding/*.pt", "voice_embedding/*.safetensors", "tekken.json"],
            progress: progress
        )
        try requireComplete(modelUrl, repoId: model.repoId, requiresVoices: true)

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
                if FileManager.default.fileExists(atPath: customModelPath.appendingPathComponent(configFile).path), isComplete(folder: customModelPath) {
                    return customModelPath
                }
            }
        }

        // Check Hub cache (new location)
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let newPath = cacheDir.appendingPathComponent("models").appendingPathComponent(model.repoId)
            for configFile in configFiles {
                if FileManager.default.fileExists(atPath: newPath.appendingPathComponent(configFile).path), isComplete(folder: newPath) {
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
                if FileManager.default.fileExists(atPath: modelPath.appendingPathComponent(configFile).path), isComplete(folder: modelPath) {
                    return modelPath
                }
            }
        }

        // Check local models directory
        let localDir = modelsDirectory.appendingPathComponent(model.repoId)
        for configFile in configFiles {
            if FileManager.default.fileExists(atPath: localDir.appendingPathComponent(configFile).path), isComplete(folder: localDir) {
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
            revision: model.revision ?? "main",
            matching: ["*.json", "*.safetensors", "tekken.json"],
            progress: progress
        )
        try requireComplete(modelUrl, repoId: model.repoId)

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
