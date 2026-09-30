/**
 * TekkenStrictLoadTests - K-7 (S-05, MLX-022)
 *
 * A missing or invalid tekken.json raises a typed error in the STT, TTS and Realtime loading
 * paths instead of a silent byte-level demo tokenizer; on a real tekken.json the strict loader
 * produces the same ids as the legacy (deprecated) initializer.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class TekkenStrictLoadTests: XCTestCase {

    private let fm = FileManager.default
    private var sandbox: URL!
    private var savedCustomDir: URL?

    override func setUp() {
        super.setUp()
        sandbox = fm.temporaryDirectory.appendingPathComponent("voxtral-tekken-\(UUID().uuidString)")
        try? fm.createDirectory(at: sandbox, withIntermediateDirectories: true)
        savedCustomDir = ModelDownloader.customModelsDirectory
        ModelDownloader.customModelsDirectory = sandbox
    }

    override func tearDown() {
        ModelDownloader.customModelsDirectory = savedCustomDir
        try? fm.removeItem(at: sandbox)
        super.tearDown()
    }

    private func assertFileNotFound(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
        guard case VoxtralError.fileNotFound(let path) = error else {
            return XCTFail("expected VoxtralError.fileNotFound, got \(error)", file: file, line: line)
        }
        XCTAssertTrue(path.hasSuffix("tekken.json"), path, file: file, line: line)
    }

    /// A model folder the downloader accepts as complete (manifest), without tekken.json.
    private func completeFolderWithoutTekken(repoId: String, configFile: String) throws -> URL {
        let folder = sandbox.appendingPathComponent(repoId)
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        let config = Data("{}".utf8)
        try config.write(to: folder.appendingPathComponent(configFile))
        let manifest = """
        {"version": 1, "repoId": "\(repoId)", "files": [{"path": "\(configFile)", "size": \(config.count)}]}
        """
        try Data(manifest.utf8).write(to: folder.appendingPathComponent(ModelDownloader.manifestFileName))
        return folder
    }

    func testLoadWithoutTekkenJSONThrows() {
        XCTAssertThrowsError(try TekkenTokenizer.load(modelPath: sandbox.path)) { assertFileNotFound($0) }
    }

    func testLoadWithInvalidTekkenJSONThrows() throws {
        try Data("{ not json".utf8).write(to: sandbox.appendingPathComponent("tekken.json"))
        XCTAssertThrowsError(try TekkenTokenizer.load(modelPath: sandbox.path)) { error in
            guard case VoxtralError.invalidTokenizer = error else {
                return XCTFail("expected VoxtralError.invalidTokenizer, got \(error)")
            }
        }
    }

    // STT: VoxtralPipeline loads its tokenizer through VoxtralProcessor.fromPretrained
    func testSTTProcessorWithoutTekkenThrows() {
        XCTAssertThrowsError(try VoxtralProcessor.fromPretrained(sandbox.path)) { assertFileNotFound($0) }
    }

    func testTTSPipelineWithoutTekkenThrows() async throws {
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit"))
        _ = try completeFolderWithoutTekken(repoId: info.repoId, configFile: "params.json")
        let pipeline = VoxtralTTSPipeline()
        do {
            try await pipeline.loadModel(modelInfo: info)
            XCTFail("loadModel must fail without tekken.json")
        } catch {
            assertFileNotFound(error)
        }
    }

    func testRealtimePipelineWithoutTekkenThrows() async throws {
        let info = try XCTUnwrap(VoxtralRealtimeRegistry.model(withId: "realtime-4b-4bit"))
        _ = try completeFolderWithoutTekken(repoId: info.repoId, configFile: "config.json")
        let pipeline = VoxtralRealtimePipeline()
        do {
            try await pipeline.loadModel(modelId: info.id)
            XCTFail("loadModel must fail without tekken.json")
        } catch {
            assertFileNotFound(error)
        }
    }

    @available(*, deprecated, message: "compares with the deprecated legacy initializer on purpose")
    func testIdsIdenticalWithRealTekken() throws {
        ModelDownloader.customModelsDirectory = savedCustomDir
        guard let info = ModelRegistry.model(withId: "mini-3b-8bit"),
              let folder = ModelDownloader.findModelPath(for: info) else {
            throw XCTSkip("mini-3b-8bit is not downloaded")
        }
        let strict = try TekkenTokenizer.load(modelPath: folder.path)
        let legacy = TekkenTokenizer(modelPath: folder.path)
        let sentences = [
            "Capital Gains and Capital One are two different things.",
            "The capital of France is Paris.",
            "Flux Forge Studio turns your Mac into a complete AI creative studio.",
            "Generate high-quality images and videos from text.",
            "Train your own custom models, all locally on your Mac.",
            "No data is sent to the cloud.",
            "Hello, world! How are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "It costs $12.50, or roughly €11.",
            "Transcribe this audio, please: 3, 2, 1, go.",
            "Bonjour, ceci est un test de transcription.",
            "Le capital de la société a augmenté de 10 %.",
            "Où est la gare ? À gauche, après le café.",
            "Fluxforge Studio transforme votre Mac en studio de création IA complet.",
            "Aucune donnée envoyée dans le cloud.",
            "L'été dernier, nous sommes allés à Montréal.",
            "Il était une fois un garçon qui aimait les mathématiques.",
            "Voxtral transcrit et comprend l'audio.",
            "Émile a reçu 42 messages — c'est beaucoup !",
            "Ça marche très bien, merci beaucoup.",
        ]
        var identical = 0
        for sentence in sentences {
            let ids = strict.encode(sentence)
            XCTAssertEqual(ids, legacy.encode(sentence), sentence)
            XCTAssertTrue(ids.allSatisfy { $0 >= 1_000 }, "text ids are rank + 1000: \(sentence)")
            if ids == legacy.encode(sentence) { identical += 1 }
        }
        print("[tekken-strict] ids identical on \(identical)/\(sentences.count) sentences")
        XCTAssertEqual(identical, sentences.count)
    }
}
