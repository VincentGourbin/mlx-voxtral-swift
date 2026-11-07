/**
 * VoxtralModelCatalog - One catalog for the STT, Realtime and TTS models
 *
 * Built from the three registries, which stay the only source of model data; the catalog adds each repository's
 * licence and the files the downloader fetches. `VoxtralModelDownloader` then offers the same `download`, `verify`,
 * `listDownloaded` and `delete` for every family.
 */

import Foundation

/// The three model families
public enum VoxtralModelFamily: String, CaseIterable, Sendable {
    case stt, realtime, tts

    public var displayName: String {
        switch self {
        case .stt: return "Speech-to-text"
        case .realtime: return "Realtime"
        case .tts: return "Text-to-speech"
        }
    }

    /// Configuration files of a complete folder, any one of which is enough: the rule of `path(for:)`
    var configFiles: [String] {
        switch self {
        case .stt: return ["config.json"]
        case .realtime: return ["config.json", "params.json"]
        case .tts: return ["params.json"]
        }
    }
}

/// Licence of a model repository: its Hub tag or, for a pack without one, the licence of its base model
public struct VoxtralModelLicense: Sendable, Equatable {
    /// Licence id (`apache-2.0`, `cc-by-nc-4.0`), or `unknown`
    public let id: String
    public let name: String
    public let url: String?
    /// nil when the licence is unknown
    public let allowsCommercialUse: Bool?
    public let summary: String

    public static let apache2 = VoxtralModelLicense(
        id: "apache-2.0",
        name: "Apache License 2.0",
        url: "https://www.apache.org/licenses/LICENSE-2.0",
        allowsCommercialUse: true,
        summary: "Free to use, modify and redistribute, commercially too, with attribution."
    )

    public static let ccByNc4 = VoxtralModelLicense(
        id: "cc-by-nc-4.0",
        name: "Creative Commons Attribution-NonCommercial 4.0",
        url: "https://creativecommons.org/licenses/by-nc/4.0/",
        allowsCommercialUse: false,
        summary: "Free to use, share and adapt with attribution; commercial use is not permitted."
    )

    /// Neither the repository nor a known base model gives a licence
    public static let unknown = VoxtralModelLicense(
        id: "unknown",
        name: "Unknown",
        url: nil,
        allowsCommercialUse: nil,
        summary: "The repository declares no licence; check it before any use."
    )
}

/// A downloadable model of any family. The fields come from its registry entry.
public struct VoxtralModelEntry: Identifiable, Sendable, Equatable {
    enum Source: Sendable {
        case stt(VoxtralModelInfo)
        case realtime(VoxtralRealtimeModelInfo)
        case tts(VoxtralTTSModelInfo)
    }

    let source: Source

    public let family: VoxtralModelFamily
    public let id: String
    public let repoId: String
    public let name: String
    public let description: String
    public let size: String
    public let quantization: String
    public let parameters: String
    public let recommended: Bool
    /// Hub revision (commit or tag) to download; nil = `main`
    public let revision: String?
    /// Exact bytes of the weight files downloaded
    public let approximateBytes: Int64?
    /// Repository files the downloader fetches: globs, or exact paths when the entry lists its own
    public let files: [String]
    /// Repository files left out of `files`
    public let excludedFiles: [String]
    public let license: VoxtralModelLicense

    init(_ model: VoxtralModelInfo) {
        source = .stt(model)
        family = .stt
        id = model.id; repoId = model.repoId; name = model.name; description = model.description
        size = model.size; quantization = model.quantization; parameters = model.parameters
        recommended = model.recommended; revision = model.revision; approximateBytes = model.approximateBytes
        files = VoxtralModelDownloader.sttDownloadGlobs
        excludedFiles = VoxtralModelDownloader.unusedConsolidatedWeights
        license = VoxtralModelCatalog.license(forId: model.id)
    }

    init(_ model: VoxtralRealtimeModelInfo) {
        source = .realtime(model)
        family = .realtime
        id = model.id; repoId = model.repoId; name = model.name; description = model.description
        size = model.size; quantization = model.quantization; parameters = model.parameters
        recommended = model.recommended; revision = model.revision; approximateBytes = model.approximateBytes
        files = model.files ?? VoxtralModelDownloader.realtimeDownloadGlobs
        excludedFiles = model.files == nil ? VoxtralModelDownloader.unusedConsolidatedWeights : []
        license = VoxtralModelCatalog.license(forId: model.id)
    }

    init(_ model: VoxtralTTSModelInfo) {
        source = .tts(model)
        family = .tts
        id = model.id; repoId = model.repoId; name = model.name; description = model.description
        size = model.size; quantization = model.quantization; parameters = model.parameters
        recommended = model.recommended; revision = model.revision; approximateBytes = model.approximateBytes
        files = VoxtralModelDownloader.ttsDownloadGlobs
        excludedFiles = []
        license = VoxtralModelCatalog.license(forId: model.id)
    }

    public static func == (a: VoxtralModelEntry, b: VoxtralModelEntry) -> Bool {
        a.family == b.family && a.id == b.id
    }
}

/// Every model the framework can download, whatever its family
public enum VoxtralModelCatalog {

    /// All models: STT, then Realtime, then TTS, in registry order
    public static let all: [VoxtralModelEntry] =
        VoxtralModelRegistry.models.map(VoxtralModelEntry.init)
        + VoxtralRealtimeRegistry.models.map(VoxtralModelEntry.init)
        + VoxtralTTSRegistry.models.map(VoxtralModelEntry.init)

    /// The models of one family
    public static func entries(in family: VoxtralModelFamily) -> [VoxtralModelEntry] {
        all.filter { $0.family == family }
    }

    /// Find a model by id (ids are unique across families)
    public static func entry(withId id: String) -> VoxtralModelEntry? {
        all.first { $0.id == id }
    }

    /// Find a model by Hugging Face repository
    public static func entry(withRepoId repoId: String) -> VoxtralModelEntry? {
        all.first { $0.repoId == repoId }
    }

    /// Hub `license:` tag of each repository (docs/Weights.md). `mini-3b-8bit` and `mini-3b-4bit` have no tag: they
    /// carry the licence of their base model, `mistralai/Voxtral-Mini-3B-2507`. `tts-4b-8bit` is our own derivative,
    /// under the licence of the TTS checkpoint.
    static let licenses: [String: VoxtralModelLicense] = [
        "mini-3b": .apache2,
        "small-24b": .apache2,
        "mini-3b-8bit": .apache2,
        "mini-3b-4bit": .apache2,
        "small-24b-8bit": .apache2,
        "small-4bit": .apache2,
        "realtime-4b-4bit": .apache2,
        "realtime-4b-6bit": .apache2,
        "realtime-4b-8bit": .apache2,
        "realtime-4b-fp16": .apache2,
        "realtime-4b": .apache2,
        "tts-4b-mlx": .ccByNc4,
        "tts-4b": .ccByNc4,
        "tts-4b-4bit": .ccByNc4,
        "tts-4b-6bit": .ccByNc4,
        "tts-4b-8bit": .ccByNc4,
    ]

    static func license(forId id: String) -> VoxtralModelLicense {
        licenses[id] ?? .unknown
    }
}

/// What `VoxtralModelDownloader.verify` found in a model folder
public struct VoxtralModelVerification: Sendable, Equatable {
    public enum Problem: Sendable, Equatable, CustomStringConvertible {
        /// No readable manifest: the download did not finish
        case noManifest
        case missing(String)
        case sizeMismatch(path: String, actual: Int, expected: Int)
        case checksumMismatch(path: String, actual: String, expected: String)
        /// Present with its size, but reading it failed: a failing disk, or no permission
        case unreadable(String)

        /// The file at fault; nil when the problem is the folder's
        var path: String? {
            switch self {
            case .noManifest: return nil
            case .missing(let path), .unreadable(let path): return path
            case .sizeMismatch(let path, _, _), .checksumMismatch(let path, _, _): return path
            }
        }

        public var description: String {
            switch self {
            case .noManifest:
                return "no readable \(VoxtralModelDownloader.manifestFileName): the download did not finish"
            case .missing(let path):
                return "\(path): missing"
            case .sizeMismatch(let path, let actual, let expected):
                return "\(path): \(actual) bytes, expected \(expected)"
            case .checksumMismatch(let path, let actual, let expected):
                return "\(path): SHA-256 \(actual), expected \(expected)"
            case .unreadable(let path):
                return "\(path): cannot be read"
            }
        }
    }

    /// The folder checked
    public let folder: URL
    /// Files whose SHA-256 was recomputed and matches the manifest
    public let checksummedFiles: [String]
    /// Files checked by size only: small Hub files have no SHA-256. In a folder from before the manifest, every file
    /// is here, against sizes read from the folder itself: that proves nothing about its content
    public let sizeOnlyFiles: [String]
    public let problems: [Problem]

    public var isValid: Bool { problems.isEmpty }

    /// True when every weight file in the manifest was hashed and matches, and there was at least one. False: `isValid`
    /// says nothing about the weights
    public var weightsChecksummed: Bool {
        func isWeight(_ path: String) -> Bool { path.hasSuffix(".safetensors") && !path.hasPrefix("voice_embedding/") }
        return checksummedFiles.contains(where: isWeight) && !sizeOnlyFiles.contains(where: isWeight)
            && !problems.contains { $0.path.map(isWeight) ?? false }
    }
}

/// Hashing reads whole models: it gets its own queue, away from the cooperative pool and from MLX work
private let modelVerificationQueue = DispatchQueue(label: "voxtral.model-verification", qos: .utility)

// MARK: - Same operations for every family

extension VoxtralModelDownloader {

    /// The folder of a complete download; nil if the model is absent or partial
    public static func path(for entry: VoxtralModelEntry) -> URL? {
        switch entry.source {
        case .stt(let model): return findModelPath(for: model)
        case .realtime(let model): return findRealtimeModelPath(for: model)
        case .tts(let model): return findTTSModelPath(for: model)
        }
    }

    /// True when `entry` is completely downloaded
    public static func isDownloaded(_ entry: VoxtralModelEntry) -> Bool {
        path(for: entry) != nil
    }

    /// The folder holding `entry` on disk: the one a pipeline loads, else the first partial copy; nil if there is none
    public static func localFolder(for entry: VoxtralModelEntry) -> URL? {
        path(for: entry) ?? localFolders(for: entry).first
    }

    /// Every folder holding `entry` on disk, complete or not, in lookup order
    static func localFolders(for entry: VoxtralModelEntry) -> [URL] {
        repoFolders(entry.repoId).filter { folder in
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(atPath: folder.path, isDirectory: &isDirectory) && isDirectory.boolValue
        }
    }

    /// What `delete(_:)` removes: every copy of `entry` on disk. A legacy Hub snapshot counts for its whole
    /// `models--org--repo` folder, with all its revisions. Paths as looked up: `delete(_:)` says where a link leads.
    public static func deletionTargets(for entry: VoxtralModelEntry) -> [URL] {
        var targets: [URL] = []
        for folder in localFolders(for: entry) {
            let target = deletionTarget(of: folder, repoId: entry.repoId)
            if !targets.contains(where: { $0.path == target.path }) { targets.append(target) }
        }
        return targets
    }

    /// Models completely downloaded, of one family or of all
    public static func listDownloaded(_ family: VoxtralModelFamily? = nil) -> [VoxtralModelEntry] {
        (family.map(VoxtralModelCatalog.entries(in:)) ?? VoxtralModelCatalog.all).filter(isDownloaded)
    }

    /// Download a model of any family; returns its folder at once if it is already complete
    public static func download(
        _ entry: VoxtralModelEntry,
        progress: VoxtralDownloadProgressCallback? = nil
    ) async throws -> URL {
        switch entry.source {
        case .stt(let model): return try await download(model, progress: progress)
        case .realtime(let model): return try await downloadRealtimeModel(model, progress: progress)
        case .tts(let model): return try await downloadTTSModel(model, progress: progress)
        }
    }

    /// Checks the folder `localFolder(for:)` returns against its manifest: every file present with its size, SHA-256
    /// recomputed where the manifest holds one, and the files any model needs (configuration, weights, `tekken.json`,
    /// voices for TTS). No network. A model looked up before its manifest existed gets one with sizes only, so read
    /// `weightsChecksummed` before trusting its weights. The manifest's revision is not compared with the entry's: a copy
    /// of another revision passes if it matches its own manifest.
    /// Throws `VoxtralModelDownloaderError.modelNotFound` if no folder holds the model, `CancellationError` if the
    /// calling Task is cancelled.
    public static func verify(
        _ entry: VoxtralModelEntry,
        progress: VoxtralDownloadProgressCallback? = nil
    ) async throws -> VoxtralModelVerification {
        guard let folder = localFolder(for: entry) else {
            throw VoxtralModelDownloaderError.modelNotFound
        }
        return try await runOffCooperativePool(on: modelVerificationQueue) {
            try verifyFolder(folder, family: entry.family, progress: progress)
        }
    }

    static func verifyFolder(
        _ folder: URL,
        family: VoxtralModelFamily,
        progress: VoxtralDownloadProgressCallback? = nil
    ) throws -> VoxtralModelVerification {
        var checksummed: [String] = []
        var sizeOnly: [String] = []
        var problems: [VoxtralModelVerification.Problem] = []

        if let manifest = readManifest(at: folder), !manifest.files.isEmpty {
            let totalBytes = manifest.files.reduce(0) { $0 + $1.size }
            var doneBytes = 0
            for file in manifest.files {
                try VoxtralCancellation.check()
                progress?(totalBytes > 0 ? Double(doneBytes) / Double(totalBytes) : 1.0, "Checking \(file.path)")
                doneBytes += file.size

                let url = folder.appendingPathComponent(file.path)
                guard let size = fileSize(at: url) else {
                    problems.append(.missing(file.path))
                    continue
                }
                guard size == file.size else {
                    problems.append(.sizeMismatch(path: file.path, actual: size, expected: file.size))
                    continue
                }
                guard let expected = file.sha256?.lowercased() else {
                    sizeOnly.append(file.path)
                    continue
                }
                do {
                    let actual = try sha256Hex(of: url)
                    if actual == expected {
                        checksummed.append(file.path)
                    } else {
                        problems.append(.checksumMismatch(path: file.path, actual: actual, expected: expected))
                    }
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    // A failing disk is what `verify` is there to report: the other files still get checked
                    problems.append(.unreadable(file.path))
                }
            }
        } else {
            problems.append(.noManifest)
        }

        // A manifest proves what was received, not what the repository had to hold
        var needed = missingModelFiles(folder, requiresVoices: family == .tts)
        if !family.configFiles.contains(where: { fileSize(at: folder.appendingPathComponent($0)) != nil }) {
            needed.append(family.configFiles[0])
        }
        for path in needed where !problems.contains(.missing(path)) {
            problems.append(.missing(path))
        }

        progress?(1.0, "Check complete")
        return VoxtralModelVerification(folder: folder, checksummedFiles: checksummed, sizeOnlyFiles: sizeOnly, problems: problems)
    }

    /// Deletes every copy of a model of any family, complete or partial: `deletionTargets(for:)`. A folder that is itself
    /// a symbolic link loses the link, not what it points to. A link above it is followed: under an organisation folder
    /// moved to another disk, the real folder goes. Throws `VoxtralModelDownloaderError.modelNotFound` if there is none.
    public static func delete(_ entry: VoxtralModelEntry) throws {
        let targets = deletionTargets(for: entry)
        guard !targets.isEmpty else {
            throw VoxtralModelDownloaderError.modelNotFound
        }
        // Two paths can reach the same folder through a link: the first removal takes the folder and breaks the link,
        // which `fileExists` no longer sees. Left there, it would stop the next download.
        for target in targets where (try? FileManager.default.attributesOfItem(atPath: target.path)) != nil {
            try FileManager.default.removeItem(at: target)
        }
    }
}
