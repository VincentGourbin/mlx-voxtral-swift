/**
 * DownloadCallSiteTests - K-88 (suite of K-6, K-9, K-24)
 *
 * Each public download entry point asks the Hub for the right files at the right revision: STT `download` (registry
 * revision, `consolidated*` excluded, completeness checked), `resolveModel` with a bare repo id (`downloadByRepoId`,
 * `consolidated*` excluded, completeness checked), `downloadTTSModel` (registry revision, `consolidated*` and voices kept, voices
 * required), `downloadRealtimeModel` (the entry's own file list, K-9). Stubbed Hub (`HubStubProtocol`): no network.
 */

import XCTest
@testable import VoxtralCore

final class DownloadCallSiteTests: XCTestCase {

    private var sandbox: URL!
    private var previousCustomDir: URL?
    private let fm = FileManager.default

    override func setUp() {
        super.setUp()
        sandbox = fm.temporaryDirectory.appendingPathComponent("voxtral-callsites-\(UUID().uuidString)")
        try? fm.createDirectory(at: sandbox, withIntermediateDirectories: true)
        previousCustomDir = VoxtralModelDownloader.customModelsDirectory
        VoxtralModelDownloader.customModelsDirectory = sandbox
        HubStubProtocol.files = [:]
        HubStubProtocol.corruptSHA = []
        HubStubProtocol.requested = []
        URLProtocol.registerClass(HubStubProtocol.self)
    }

    override func tearDown() {
        URLProtocol.unregisterClass(HubStubProtocol.self)
        VoxtralModelDownloader.customModelsDirectory = previousCustomDir
        try? fm.removeItem(at: sandbox)
        super.tearDown()
    }

    private func repo(_ kind: String) -> String { "voxtral-tests/\(kind)-\(UUID().uuidString)" }
    private func blob(_ byte: UInt8, _ count: Int = 1_024) -> Data { Data(repeating: byte, count: count) }
    private func index(_ shard: String) -> Data {
        try! JSONSerialization.data(withJSONObject: ["metadata": [:], "weight_map": ["layer.0.weight": shard]])
    }

    /// Files fetched (`resolve`) and revisions used, from the stub's log
    private func fetched(_ repoId: String) -> (files: Set<String>, revisions: Set<String>) {
        var files = Set<String>(), revisions = Set<String>()
        for path in HubStubProtocol.requested {
            if let r = path.range(of: "/\(repoId)/resolve/") {
                let parts = path[r.upperBound...].split(separator: "/", maxSplits: 1).map(String.init)
                revisions.insert(parts[0]); if parts.count == 2 { files.insert(parts[1]) }
            } else if let r = path.range(of: "/api/models/\(repoId)/tree/") {
                revisions.insert(String(path[r.upperBound...]))
            }
        }
        return (files, revisions)
    }

    private func sttRemote() -> [String: Data] {
        ["config.json": Data("{}".utf8), "model.safetensors": blob(1), "model.safetensors.index.json": index("model.safetensors"),
         "tekken.json": Data("{\"vocab\": []}".utf8), "consolidated.safetensors": blob(2), "README.md": Data("x".utf8)]
    }

    func testSTTDownloadPinsTheRevisionAndSkipsConsolidated() async throws {
        let repoId = repo("stt")
        let model = VoxtralModelInfo(id: "t", repoId: repoId, name: "t", description: "", size: "", quantization: "",
                                     parameters: "", revision: "0123abcd")
        HubStubProtocol.repoId = repoId; HubStubProtocol.files = sttRemote()
        let url = try await VoxtralModelDownloader.download(model)
        let got = fetched(repoId)
        XCTAssertEqual(got.revisions, ["0123abcd"])
        XCTAssertEqual(got.files, ["config.json", "model.safetensors", "model.safetensors.index.json", "tekken.json"])
        XCTAssertTrue(fm.fileExists(atPath: url.appendingPathComponent(".voxtral-complete.json").path))
    }

    func testSTTDownloadRefusesAnIncompleteRepo() async throws {
        let repoId = repo("stt-incomplete")
        let model = VoxtralModelInfo(id: "t", repoId: repoId, name: "t", description: "", size: "", quantization: "",
                                     parameters: "")
        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = ["config.json": Data("{}".utf8), "model.safetensors.index.json": index("model-00001.safetensors"),
                                 "tekken.json": Data("{\"vocab\": []}".utf8)]
        do {
            _ = try await VoxtralModelDownloader.download(model)
            XCTFail("a repo without its weights must not be returned as downloaded")
        } catch {
            XCTAssertTrue("\(error)".contains("model-00001.safetensors"), "\(error)")
        }
    }

    func testResolveByRepoIdSkipsConsolidated() async throws {
        let repoId = repo("bare")
        HubStubProtocol.repoId = repoId; HubStubProtocol.files = sttRemote()
        _ = try await VoxtralModelDownloader.resolveModel(repoId)
        let got = fetched(repoId)
        XCTAssertEqual(got.revisions, ["main"])
        XCTAssertEqual(got.files, ["config.json", "model.safetensors", "model.safetensors.index.json", "tekken.json"])
    }

    func testTTSDownloadKeepsConsolidatedAndVoicesAtTheRevision() async throws {
        let repoId = repo("tts")
        let model = VoxtralTTSModelInfo(id: "t", repoId: repoId, name: "t", description: "", size: "", quantization: "",
                                        parameters: "", revision: "feedbeef")
        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = ["params.json": Data("{\"dim\": 8}".utf8), "consolidated.safetensors": blob(3),
                                 "tekken.json": Data("{\"vocab\": []}".utf8),
                                 "voice_embedding/neutral_female.pt": blob(4, 256), "README.md": Data("x".utf8)]
        _ = try await VoxtralModelDownloader.downloadTTSModel(model)
        let got = fetched(repoId)
        XCTAssertEqual(got.revisions, ["feedbeef"])
        XCTAssertEqual(got.files, ["params.json", "consolidated.safetensors", "tekken.json", "voice_embedding/neutral_female.pt"])
    }

    func testTTSDownloadWithoutVoicesIsRefused() async throws {
        let repoId = repo("tts-novoice")
        let model = VoxtralTTSModelInfo(id: "t", repoId: repoId, name: "t", description: "", size: "", quantization: "",
                                        parameters: "")
        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = ["params.json": Data("{\"dim\": 8}".utf8), "consolidated.safetensors": blob(3),
                                 "tekken.json": Data("{\"vocab\": []}".utf8)]
        do {
            _ = try await VoxtralModelDownloader.downloadTTSModel(model)
            XCTFail("a TTS repo without voices must not be returned as downloaded")
        } catch {
            XCTAssertTrue("\(error)".contains("voice_embedding"), "\(error)")
        }
    }

    func testRealtimeDownloadFollowsTheEntryFileList() async throws {
        let repoId = repo("realtime")
        let model = VoxtralRealtimeModelInfo(id: "t", repoId: repoId, name: "t", description: "", size: "",
                                             quantization: "", parameters: "", revision: "c0ffee00",
                                             files: ["params.json", "consolidated.safetensors", "tekken.json"])
        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = ["params.json": Data("{\"dim\": 8}".utf8), "config.json": Data("{}".utf8),
                                 "consolidated.safetensors": blob(5), "model.safetensors": blob(6),
                                 "tekken.json": Data("{\"vocab\": []}".utf8)]
        _ = try await VoxtralModelDownloader.downloadRealtimeModel(model)
        let got = fetched(repoId)
        XCTAssertEqual(got.revisions, ["c0ffee00"])
        XCTAssertEqual(got.files, ["params.json", "consolidated.safetensors", "tekken.json"])
    }
}
