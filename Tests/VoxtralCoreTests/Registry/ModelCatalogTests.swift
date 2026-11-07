/**
 * ModelCatalogTests - one catalog over the three registries, same operations for every family
 *
 * No network, no GPU. Disk operations run in a temporary `customModelsDirectory`, on entries with made-up repo ids:
 * a real model in the caches can neither be matched nor deleted. Downloads go through `HubStubProtocol`.
 */

import XCTest
@testable import VoxtralCore

final class ModelCatalogTests: XCTestCase {

    private var sandbox: URL!
    private var previousCustomDir: URL?
    private let fm = FileManager.default

    override func setUp() {
        super.setUp()
        sandbox = fm.temporaryDirectory.appendingPathComponent("voxtral-catalog-\(UUID().uuidString)")
        try? fm.createDirectory(at: sandbox, withIntermediateDirectories: true)
        previousCustomDir = VoxtralModelDownloader.customModelsDirectory
        VoxtralModelDownloader.customModelsDirectory = sandbox
    }

    override func tearDown() {
        VoxtralModelDownloader.customModelsDirectory = previousCustomDir
        try? fm.removeItem(at: sandbox)
        super.tearDown()
    }

    // MARK: - Fixtures

    /// One made-up entry per family, on a repo id no real cache can hold
    private func entries() -> [VoxtralModelEntry] {
        let tag = UUID().uuidString
        return [
            VoxtralModelEntry(VoxtralModelInfo(
                id: "test-stt", repoId: "voxtral-tests/stt-\(tag)", name: "stt", description: "", size: "",
                quantization: "", parameters: "")),
            VoxtralModelEntry(VoxtralRealtimeModelInfo(
                id: "test-realtime", repoId: "voxtral-tests/realtime-\(tag)", name: "realtime", description: "", size: "",
                quantization: "", parameters: "")),
            VoxtralModelEntry(VoxtralTTSModelInfo(
                id: "test-tts", repoId: "voxtral-tests/tts-\(tag)", name: "tts", description: "", size: "",
                quantization: "", parameters: "")),
        ]
    }

    private func folder(of entry: VoxtralModelEntry) -> URL { sandbox.appendingPathComponent(entry.repoId) }

    /// Files of a whole model of this family
    private func modelFiles(_ entry: VoxtralModelEntry) -> [String: Data] {
        var files: [String: Data] = [
            "model.safetensors": Data(repeating: 7, count: 4_096),
            "tekken.json": Data("{}".utf8),
            entry.family == .tts ? "params.json" : "config.json": Data("{}".utf8),
        ]
        if entry.family == .tts { files["voice_embedding/neutral.safetensors"] = Data(repeating: 9, count: 512) }
        return files
    }

    private func write(_ files: [String: Data], in folder: URL) throws {
        for (name, data) in files {
            let url = folder.appendingPathComponent(name)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
    }

    /// Writes a whole model and its manifest, with a SHA-256 for the weight file, as `downloadRepoDirect` does
    private func install(_ entry: VoxtralModelEntry) throws {
        let files = modelFiles(entry)
        try write(files, in: folder(of: entry))
        let listed = files.keys.sorted().map { path in
            VoxtralModelDownloader.DownloadManifest.Entry(
                path: path, size: files[path]!.count,
                sha256: path == "model.safetensors" ? HubStubProtocol.sha256(files[path]!) : nil)
        }
        try VoxtralModelDownloader.writeManifest(
            .init(repoId: entry.repoId, revision: "main", files: listed), to: folder(of: entry))
    }

    private func verify(_ entry: VoxtralModelEntry) async throws -> VoxtralModelVerification {
        try await VoxtralModelDownloader.verify(entry)
    }

    // MARK: - The catalog

    func testCatalogHoldsEveryRegistryEntryOnce() {
        let expected = VoxtralModelRegistry.models.map(\.id) + VoxtralRealtimeRegistry.models.map(\.id)
            + VoxtralTTSRegistry.models.map(\.id)
        XCTAssertEqual(VoxtralModelCatalog.all.map(\.id), expected)
        XCTAssertEqual(Set(expected).count, expected.count, "an id must name one model across the families")
        XCTAssertEqual(Set(VoxtralModelCatalog.all.map(\.repoId)).count, expected.count)
        XCTAssertEqual(VoxtralModelCatalog.entries(in: .stt).map(\.id), VoxtralModelRegistry.models.map(\.id))
        XCTAssertEqual(VoxtralModelCatalog.entries(in: .realtime).map(\.id), VoxtralRealtimeRegistry.models.map(\.id))
        XCTAssertEqual(VoxtralModelCatalog.entries(in: .tts).map(\.id), VoxtralTTSRegistry.models.map(\.id))
    }

    func testEntryCarriesItsRegistryFields() throws {
        let info = try XCTUnwrap(VoxtralRealtimeRegistry.model(withId: "realtime-4b-6bit"))
        let entry = try XCTUnwrap(VoxtralModelCatalog.entry(withId: "realtime-4b-6bit"))
        XCTAssertEqual(entry.family, .realtime)
        XCTAssertEqual(entry.repoId, info.repoId)
        XCTAssertEqual(entry.revision, info.revision)
        XCTAssertNotNil(entry.revision)
        XCTAssertEqual(entry.size, info.size)
        XCTAssertEqual(entry.approximateBytes, info.approximateBytes)
        XCTAssertEqual(entry.quantization, info.quantization)
        XCTAssertEqual(entry.recommended, info.recommended)
    }

    func testLookup() {
        XCTAssertEqual(VoxtralModelCatalog.entry(withId: "tts-4b-6bit")?.family, .tts)
        XCTAssertEqual(VoxtralModelCatalog.entry(withRepoId: "mzbac/voxtral-mini-3b-8bit")?.id, "mini-3b-8bit")
        XCTAssertNil(VoxtralModelCatalog.entry(withId: "nope"))
        XCTAssertNil(VoxtralModelCatalog.entry(withRepoId: "nope/nope"))
    }

    func testEveryEntryHasARecordedLicence() {
        XCTAssertEqual(Set(VoxtralModelCatalog.licenses.keys), Set(VoxtralModelCatalog.all.map(\.id)))
    }

    func testLicences() throws {
        for entry in VoxtralModelCatalog.entries(in: .tts) {
            XCTAssertEqual(entry.license, .ccByNc4, entry.id)
            XCTAssertEqual(entry.license.allowsCommercialUse, false, entry.id)
        }
        for entry in VoxtralModelCatalog.entries(in: .realtime) {
            XCTAssertEqual(entry.license, .apache2, entry.id)
        }
        // These two repositories have no Hub tag: they carry the licence of mistralai/Voxtral-Mini-3B-2507
        for id in ["mini-3b-8bit", "mini-3b-4bit"] {
            let entry = try XCTUnwrap(VoxtralModelCatalog.entry(withId: id))
            XCTAssertEqual(entry.license, .apache2, id)
        }
    }

    func testFilesFollowWhatTheDownloaderFetches() throws {
        let stt = try XCTUnwrap(VoxtralModelCatalog.entry(withId: "mini-3b"))
        XCTAssertEqual(stt.files, VoxtralModelDownloader.sttDownloadGlobs)
        XCTAssertEqual(stt.excludedFiles, VoxtralModelDownloader.unusedConsolidatedWeights)

        let tts = try XCTUnwrap(VoxtralModelCatalog.entry(withId: "tts-4b"))
        XCTAssertEqual(tts.files, VoxtralModelDownloader.ttsDownloadGlobs)
        XCTAssertEqual(tts.excludedFiles, [], "consolidated.safetensors is the only weight file of the TTS checkpoint")

        let original = try XCTUnwrap(VoxtralModelCatalog.entry(withId: "realtime-4b"))
        XCTAssertEqual(original.files, ["consolidated.safetensors", "params.json", "tekken.json"])
        XCTAssertEqual(original.excludedFiles, [])

        let realtime = try XCTUnwrap(VoxtralModelCatalog.entry(withId: "realtime-4b-4bit"))
        XCTAssertEqual(realtime.files, VoxtralModelDownloader.realtimeDownloadGlobs)
        XCTAssertEqual(realtime.excludedFiles, VoxtralModelDownloader.unusedConsolidatedWeights)
    }

    // MARK: - Path, list

    func testAbsentModelHasNoPathNoFolder() {
        for entry in entries() {
            XCTAssertNil(VoxtralModelDownloader.path(for: entry), entry.id)
            XCTAssertNil(VoxtralModelDownloader.localFolder(for: entry), entry.id)
            XCTAssertFalse(VoxtralModelDownloader.isDownloaded(entry), entry.id)
        }
    }

    func testInstalledModelIsFoundInEveryFamily() throws {
        for entry in entries() {
            try install(entry)
            XCTAssertEqual(VoxtralModelDownloader.path(for: entry)?.path, folder(of: entry).path, entry.id)
            XCTAssertTrue(VoxtralModelDownloader.isDownloaded(entry), entry.id)
        }
    }

    func testPartialFolderHasALocalFolderButNoPath() throws {
        for entry in entries() {
            try write(["model-00001-of-00002.safetensors": Data(repeating: 1, count: 64)], in: folder(of: entry))
            XCTAssertNil(VoxtralModelDownloader.path(for: entry), entry.id)
            XCTAssertEqual(VoxtralModelDownloader.localFolder(for: entry)?.path, folder(of: entry).path, entry.id)
        }
    }

    func testListDownloadedSeesARealIdInTheCustomDirectory() throws {
        // Read only: a real id, written in the sandbox, is listed for its family alone
        let entry = try XCTUnwrap(VoxtralModelCatalog.entry(withId: "realtime-4b-fp16"))
        try install(entry)
        XCTAssertTrue(VoxtralModelDownloader.listDownloaded().contains(entry))
        XCTAssertTrue(VoxtralModelDownloader.listDownloaded(.realtime).contains(entry))
        XCTAssertFalse(VoxtralModelDownloader.listDownloaded(.tts).contains(entry))
    }

    // MARK: - Verify

    func testVerifyPassesInEveryFamily() async throws {
        for entry in entries() {
            try install(entry)
            let result = try await verify(entry)
            XCTAssertTrue(result.isValid, "\(entry.id): \(result.problems)")
            XCTAssertEqual(result.folder.path, folder(of: entry).path)
            XCTAssertEqual(result.checksummedFiles, ["model.safetensors"], entry.id)
            XCTAssertEqual(result.sizeOnlyFiles.count, modelFiles(entry).count - 1, entry.id)
        }
    }

    func testVerifyCatchesACorruptedWeightOfTheSameSize() async throws {
        for entry in entries() {
            try install(entry)
            let weight = folder(of: entry).appendingPathComponent("model.safetensors")
            try Data(repeating: 8, count: 4_096).write(to: weight)

            XCTAssertTrue(VoxtralModelDownloader.isDownloaded(entry), "size alone cannot see this: that is what verify is for")
            let result = try await verify(entry)
            XCTAssertFalse(result.isValid, entry.id)
            guard case .checksumMismatch(let path, _, _)? = result.problems.first else {
                return XCTFail("\(entry.id): \(result.problems)")
            }
            XCTAssertEqual(path, "model.safetensors")
            XCTAssertEqual(result.checksummedFiles, [])
        }
    }

    func testVerifyCatchesATruncatedFile() async throws {
        let entry = entries()[0]
        try install(entry)
        try Data(repeating: 7, count: 100).write(to: folder(of: entry).appendingPathComponent("model.safetensors"))

        let result = try await verify(entry)
        XCTAssertEqual(result.problems, [.sizeMismatch(path: "model.safetensors", actual: 100, expected: 4_096)])
    }

    func testVerifyReportsAMissingFileOnce() async throws {
        let entry = entries()[1]
        try install(entry)
        try fm.removeItem(at: folder(of: entry).appendingPathComponent("tekken.json"))

        let result = try await verify(entry)
        XCTAssertEqual(result.problems, [.missing("tekken.json")])
    }

    func testVerifySeesWhatTheManifestDoesNotList() async throws {
        // A manifest proves what was received: a TTS folder without voices fails even with a clean manifest
        let entry = entries()[2]
        var files = modelFiles(entry)
        files["voice_embedding/neutral.safetensors"] = nil
        try write(files, in: folder(of: entry))
        let listed = files.keys.sorted().map {
            VoxtralModelDownloader.DownloadManifest.Entry(path: $0, size: files[$0]!.count, sha256: nil)
        }
        try VoxtralModelDownloader.writeManifest(.init(repoId: entry.repoId, revision: nil, files: listed), to: folder(of: entry))

        let result = try await verify(entry)
        XCTAssertEqual(result.problems, [.missing("voice_embedding/*")])
    }

    func testVerifyOfAPartialDownload() async throws {
        let entry = entries()[0]
        try write([
            "config.json": Data("{}".utf8),
            "model-00001-of-00002.safetensors": Data(repeating: 1, count: 64),
        ], in: folder(of: entry))

        let result = try await verify(entry)
        XCTAssertFalse(result.isValid)
        XCTAssertEqual(result.problems.first, .noManifest)
        XCTAssertTrue(result.problems.contains(.missing("tekken.json")), "\(result.problems)")
        XCTAssertFalse(fm.fileExists(atPath: folder(of: entry).appendingPathComponent(VoxtralModelDownloader.manifestFileName).path))
    }

    func testVerifyOfAnUnreadableManifest() async throws {
        let entry = entries()[0]
        try install(entry)
        try Data("not json".utf8).write(to: folder(of: entry).appendingPathComponent(VoxtralModelDownloader.manifestFileName))

        let result = try await verify(entry)
        XCTAssertEqual(result.problems, [.noManifest])
    }

    func testVerifyWithoutAnySHA256IsSaidSo() async throws {
        // Folder from before the manifest: the lookup writes one with sizes only, nothing is hashed
        let entry = entries()[0]
        let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        let map = Dictionary(uniqueKeysWithValues: shards.enumerated().map { ("layer.\($0.offset).weight", $0.element) })
        try write([
            "config.json": Data("{}".utf8),
            "tekken.json": Data("{}".utf8),
            "model.safetensors.index.json": try JSONSerialization.data(withJSONObject: ["weight_map": map]),
            shards[0]: Data(repeating: 1, count: 64),
            shards[1]: Data(repeating: 2, count: 64),
        ], in: folder(of: entry))

        let result = try await verify(entry)
        XCTAssertTrue(result.isValid, "\(result.problems)")
        XCTAssertEqual(result.checksummedFiles, [])
        XCTAssertEqual(result.sizeOnlyFiles.count, 5)
    }

    func testVerifyOfAnAbsentModelThrows() async throws {
        for entry in entries() {
            do {
                _ = try await verify(entry)
                XCTFail("\(entry.id) must throw")
            } catch VoxtralModelDownloaderError.modelNotFound {
                // expected
            }
        }
    }

    func testVerifyReportsProgressToTheEnd() async throws {
        let entry = entries()[0]
        try install(entry)
        let fractions = Locked<[Double]>([])
        _ = try await VoxtralModelDownloader.verify(entry) { fraction, _ in
            fractions.withLock { $0.append(fraction) }
        }
        let seen = fractions.get()
        XCTAssertEqual(seen.last, 1.0)
        XCTAssertEqual(seen, seen.sorted())
    }

    func testCancelledVerifyStops() async throws {
        let entry = entries()[0]
        try install(entry)
        let task = Task<VoxtralModelVerification, Error> {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await VoxtralModelDownloader.verify(entry)
        }
        do {
            _ = try await task.value
            XCTFail("a cancelled verification must throw")
        } catch is CancellationError {
            // expected
        }
    }

    // MARK: - Delete

    func testDeleteRemovesAWholeModelInEveryFamily() throws {
        for entry in entries() {
            try install(entry)
            try VoxtralModelDownloader.delete(entry)
            XCTAssertFalse(fm.fileExists(atPath: folder(of: entry).path), entry.id)
            XCTAssertFalse(VoxtralModelDownloader.isDownloaded(entry), entry.id)
            XCTAssertThrowsError(try VoxtralModelDownloader.delete(entry), entry.id) {
                guard case VoxtralModelDownloaderError.modelNotFound = $0 else { return XCTFail("\($0)") }
            }
        }
    }

    func testDeleteRemovesAPartialDownloadWithoutConfig() throws {
        // `tts-4b` receives consolidated.safetensors before params.json: a partial folder may hold no config at all
        for entry in entries() {
            try write(["consolidated.safetensors": Data(repeating: 1, count: 64)], in: folder(of: entry))
            try VoxtralModelDownloader.delete(entry)
            XCTAssertFalse(fm.fileExists(atPath: folder(of: entry).path), entry.id)
        }
    }

    func testDeleteLeavesOtherModelsAlone() throws {
        let all = entries()
        for entry in all { try install(entry) }
        try VoxtralModelDownloader.delete(all[1])
        XCTAssertTrue(VoxtralModelDownloader.isDownloaded(all[0]))
        XCTAssertFalse(VoxtralModelDownloader.isDownloaded(all[1]))
        XCTAssertTrue(VoxtralModelDownloader.isDownloaded(all[2]))
    }

    func testRemovingALegacyHubSnapshotTakesItsRepoFolder() throws {
        let repoFolder = sandbox.appendingPathComponent(".cache/huggingface/hub/models--voxtral-tests--legacy")
        let snapshot = repoFolder.appendingPathComponent("snapshots/abc123")
        try write(["config.json": Data("{}".utf8)], in: snapshot)

        try VoxtralModelDownloader.removeModelFolder(snapshot)
        XCTAssertFalse(fm.fileExists(atPath: repoFolder.path))
        XCTAssertTrue(fm.fileExists(atPath: repoFolder.deletingLastPathComponent().path), "the Hub cache itself stays")
    }

    // MARK: - Download

    func testDownloadGoesThroughTheFamilyAndVerifies() async throws {
        URLProtocol.registerClass(HubStubProtocol.self)
        defer { URLProtocol.unregisterClass(HubStubProtocol.self) }

        for entry in entries() {
            HubStubProtocol.repoId = entry.repoId
            HubStubProtocol.files = modelFiles(entry)
            HubStubProtocol.corruptSHA = []
            HubStubProtocol.requested = []

            let path = try await VoxtralModelDownloader.download(entry)
            XCTAssertEqual(path.path, folder(of: entry).path, entry.id)
            XCTAssertTrue(VoxtralModelDownloader.isDownloaded(entry), entry.id)

            let result = try await verify(entry)
            XCTAssertTrue(result.isValid, "\(entry.id): \(result.problems)")
            XCTAssertTrue(result.checksummedFiles.contains("model.safetensors"), entry.id)

            // A second call finds the folder and fetches nothing
            HubStubProtocol.requested = []
            _ = try await VoxtralModelDownloader.download(entry)
            XCTAssertEqual(HubStubProtocol.requested, [], entry.id)
        }
    }
}
