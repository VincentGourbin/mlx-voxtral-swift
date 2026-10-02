/**
 * DownloadCompletenessTests - K-6 (S-03, A-02, A-05, MLX-012)
 *
 * A model folder is "downloaded" only when its completeness manifest is present and
 * verified, for STT, TTS and Realtime alike; each weight is checked against the Hub's
 * SHA-256 before it is kept; the legacy `downloadModel(modelId:)` stub throws instead of
 * creating an empty folder. Everything runs on temporary folders with a stubbed Hub
 * (URLProtocol on `URLSession.shared`): no network, no GPU.
 *
 * Only APIs that existed before K-6 are used, so the suite compiles (and fails) with
 * the fix stashed (piège 38).
 */

import CryptoKit
import XCTest
@testable import VoxtralCore

/// Serves a fake Hugging Face repo: `GET /api/models/<repo>/tree/<rev>` and `GET /<repo>/resolve/<rev>/<path>`.
final class HubStubProtocol: URLProtocol {
    nonisolated(unsafe) static var repoId = ""
    nonisolated(unsafe) static var files: [String: Data] = [:]
    /// Files whose tree entry advertises a wrong SHA-256 (`lfs.oid`)
    nonisolated(unsafe) static var corruptSHA: Set<String> = []

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "huggingface.co" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let path = request.url?.path ?? ""
        let body: Data?
        if path.hasPrefix("/api/models/\(Self.repoId)/tree/") {
            let entries: [[String: Any]] = Self.files.keys.sorted().map { name in
                let data = Self.files[name]!
                var entry: [String: Any] = ["type": "file", "path": name, "size": data.count]
                if name.hasSuffix(".safetensors") || name == "tekken.json" {
                    let oid = Self.corruptSHA.contains(name) ? Self.sha256(Data("corrupt".utf8)) : Self.sha256(data)
                    entry["lfs"] = ["oid": oid, "size": data.count]
                }
                return entry
            }
            body = try? JSONSerialization.data(withJSONObject: entries)
        } else if let range = path.range(of: "/\(Self.repoId)/resolve/") {
            let rest = path[range.upperBound...]
            let file = rest.split(separator: "/", maxSplits: 1).last.map(String.init) ?? ""
            body = Self.files[file]
        } else {
            body = nil
        }
        let status = body == nil ? 404 : 200
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body ?? Data())
        client?.urlProtocolDidFinishLoading(self)
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

final class DownloadCompletenessTests: XCTestCase {

    private var sandbox: URL!
    private var previousCustomDir: URL?
    private let fm = FileManager.default

    override func setUp() {
        super.setUp()
        sandbox = fm.temporaryDirectory.appendingPathComponent("voxtral-completeness-\(UUID().uuidString)")
        try? fm.createDirectory(at: sandbox, withIntermediateDirectories: true)
        previousCustomDir = VoxtralModelDownloader.customModelsDirectory
        VoxtralModelDownloader.customModelsDirectory = sandbox
        HubStubProtocol.files = [:]
        HubStubProtocol.corruptSHA = []
        URLProtocol.registerClass(HubStubProtocol.self)
    }

    override func tearDown() {
        URLProtocol.unregisterClass(HubStubProtocol.self)
        VoxtralModelDownloader.customModelsDirectory = previousCustomDir
        try? fm.removeItem(at: sandbox)
        super.tearDown()
    }

    // MARK: - Fixtures

    /// A repo id no real cache can hold, so only the sandbox is ever matched.
    private func uniqueRepo(_ kind: String) -> String { "voxtral-tests/\(kind)-\(UUID().uuidString)" }

    private func write(_ files: [String: Data], in folder: URL) throws {
        for (name, data) in files {
            let url = folder.appendingPathComponent(name)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
    }

    private func blob(_ byte: UInt8, _ count: Int = 4_096) -> Data { Data(repeating: byte, count: count) }

    private func index(_ shards: [String]) -> Data {
        let map = Dictionary(uniqueKeysWithValues: shards.enumerated().map { ("layer.\($0.offset).weight", $0.element) })
        return try! JSONSerialization.data(withJSONObject: ["metadata": [:], "weight_map": map])
    }

    private func sttModel(_ repoId: String) -> VoxtralModelInfo {
        VoxtralModelInfo(id: "test-stt", repoId: repoId, name: "test", description: "", size: "", quantization: "", parameters: "")
    }

    // MARK: - 1. 1 shard of 5 without index

    func testOneShardOfFiveWithoutIndexIsNotDownloaded() throws {
        let model = sttModel(uniqueRepo("stt"))
        let folder = sandbox.appendingPathComponent(model.repoId)
        try write([
            "config.json": Data("{}".utf8),
            "model-00001-of-00005.safetensors": blob(1),
            "tekken.json": Data("{}".utf8),
        ], in: folder)

        XCTAssertNil(VoxtralModelDownloader.findModelPath(for: model), "1/5 shards without index must not count as downloaded")
        XCTAssertFalse(VoxtralModelDownloader.isModelDownloaded(model))
    }

    // MARK: - 2. Corrupted index

    func testCorruptedIndexIsNotDownloaded() throws {
        let model = sttModel(uniqueRepo("stt"))
        let folder = sandbox.appendingPathComponent(model.repoId)
        var files: [String: Data] = [
            "config.json": Data("{}".utf8),
            "model.safetensors.index.json": Data("{\"weight_map\": {".utf8),
            "tekken.json": Data("{}".utf8),
        ]
        for i in 1...5 { files["model-0000\(i)-of-00005.safetensors"] = blob(UInt8(i)) }
        try write(files, in: folder)

        XCTAssertNil(VoxtralModelDownloader.findModelPath(for: model), "an unreadable index must not count as downloaded")
    }

    // MARK: - 3. TTS params.json only, then effective resume

    func testTTSParamsOnlyIsNotDownloadedThenResumes() async throws {
        let repoId = uniqueRepo("tts")
        let model = VoxtralTTSModelInfo(id: "test-tts", repoId: repoId, name: "test", description: "",
                                        size: "", quantization: "", parameters: "")
        let folder = sandbox.appendingPathComponent(repoId)
        let remote: [String: Data] = [
            "params.json": Data("{\"dim\": 8}".utf8),
            "config.json": Data("{}".utf8),
            "model.safetensors": blob(7),
            "model.safetensors.index.json": index(["model.safetensors"]),
            "tekken.json": Data("{\"vocab\": []}".utf8),
            "voice_embedding/neutral_female.safetensors": blob(9, 512),
        ]
        try write(["params.json": remote["params.json"]!], in: folder)

        XCTAssertNil(VoxtralModelDownloader.findTTSModelPath(for: model), "params.json alone must not count as downloaded")

        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = remote
        let url = try await VoxtralModelDownloader.downloadTTSModel(model)

        XCTAssertEqual(url.standardizedFileURL, folder.standardizedFileURL)
        for (name, data) in remote {
            XCTAssertEqual(try? Data(contentsOf: folder.appendingPathComponent(name)), data, "\(name) not resumed")
        }
        XCTAssertTrue(fm.fileExists(atPath: folder.appendingPathComponent(".voxtral-complete.json").path))
        XCTAssertEqual(VoxtralModelDownloader.findTTSModelPath(for: model)?.standardizedFileURL, folder.standardizedFileURL)
    }

    // MARK: - 4. Realtime config.json only, then effective resume

    func testRealtimeConfigOnlyIsNotDownloadedThenResumes() async throws {
        let repoId = uniqueRepo("realtime")
        let model = VoxtralRealtimeModelInfo(id: "test-realtime", repoId: repoId, name: "test", description: "",
                                             size: "", quantization: "", parameters: "")
        let folder = sandbox.appendingPathComponent(repoId)
        let remote: [String: Data] = [
            "config.json": Data("{\"dim\": 8}".utf8),
            "model.safetensors": blob(5),
            "model.safetensors.index.json": index(["model.safetensors"]),
            "tekken.json": Data("{\"vocab\": []}".utf8),
        ]
        try write(["config.json": remote["config.json"]!], in: folder)

        XCTAssertNil(VoxtralModelDownloader.findRealtimeModelPath(for: model), "config.json alone must not count as downloaded")

        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = remote
        _ = try await VoxtralModelDownloader.downloadRealtimeModel(model)

        for (name, data) in remote {
            XCTAssertEqual(try? Data(contentsOf: folder.appendingPathComponent(name)), data, "\(name) not resumed")
        }
        XCTAssertEqual(VoxtralModelDownloader.findRealtimeModelPath(for: model)?.standardizedFileURL, folder.standardizedFileURL)
    }

    // MARK: - 5. Wrong SHA-256 is rejected

    func testWrongSHA256IsRejected() async throws {
        let repoId = uniqueRepo("stt")
        let folder = sandbox.appendingPathComponent(repoId)
        HubStubProtocol.repoId = repoId
        HubStubProtocol.files = [
            "config.json": Data("{}".utf8),
            "model.safetensors": blob(3),
        ]
        HubStubProtocol.corruptSHA = ["model.safetensors"]

        do {
            _ = try await VoxtralModelDownloader.downloadRepoDirect(repoId: repoId, matching: ["*.json", "*.safetensors"])
            XCTFail("a weight whose SHA-256 differs from the Hub's must be rejected")
        } catch {}
        XCTAssertFalse(fm.fileExists(atPath: folder.appendingPathComponent("model.safetensors").path),
                       "the rejected weight must not be kept")
        XCTAssertFalse(fm.fileExists(atPath: folder.appendingPathComponent(".voxtral-complete.json").path))
    }

}
