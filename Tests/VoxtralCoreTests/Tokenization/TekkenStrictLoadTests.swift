/**
 * TekkenStrictLoadTests - K-7 (S-05, MLX-022)
 *
 * A missing or invalid tekken.json raises a typed error in the STT, TTS and Realtime loading
 * paths instead of a silent byte-level demo tokenizer (the legacy fallback initializer was removed in 3.0, K-31).
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
        savedCustomDir = VoxtralModelDownloader.customModelsDirectory
        VoxtralModelDownloader.customModelsDirectory = sandbox
    }

    override func tearDown() {
        VoxtralModelDownloader.customModelsDirectory = savedCustomDir
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
        try Data(manifest.utf8).write(to: folder.appendingPathComponent(VoxtralModelDownloader.manifestFileName))
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

}
