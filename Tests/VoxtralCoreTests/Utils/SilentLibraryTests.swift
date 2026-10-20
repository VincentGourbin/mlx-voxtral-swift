/**
 * SilentLibraryTests - K-88 (contract of K-23: the library writes nothing to stdout unless `VoxtralDebug.enabled`)
 *
 * With `VoxtralDebug.enabled = false`, loading a TTS pack and synthesizing a sentence, then loading an STT pack and
 * transcribing a clip, write 0 bytes to stdout (file descriptor 1 is redirected to a file meanwhile). Uses tts-4b-4bit and
 * mini-3b-4bit, skipped when either is not downloaded.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class SilentLibraryTests: XCTestCase {

    /// Runs `body` with stdout redirected to a temporary file; returns what was written
    private func capturingStdout(_ body: () async throws -> Void) async throws -> Data {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-stdout-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: file) }
        let handle = try FileHandle(forWritingTo: file)
        fflush(stdout)
        let saved = dup(STDOUT_FILENO)
        dup2(handle.fileDescriptor, STDOUT_FILENO)
        var thrown: Error?
        do { try await body() } catch { thrown = error }
        fflush(stdout)
        dup2(saved, STDOUT_FILENO)
        close(saved)
        try handle.close()
        if let thrown { throw thrown }
        return try Data(contentsOf: file)
    }

    func testNothingOnStdoutDuringLoadAndInference() async throws {
        guard let ttsInfo = VoxtralTTSRegistry.model(withId: "tts-4b-4bit"),
              VoxtralModelDownloader.findTTSModelPath(for: ttsInfo) != nil,
              let sttInfo = VoxtralModelRegistry.model(withId: "mini-3b-4bit"),
              VoxtralModelDownloader.findModelPath(for: sttInfo) != nil else {
            throw XCTSkip("tts-4b-4bit or mini-3b-4bit is not downloaded")
        }
        let clip = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("docs/eval/clips/c_20s_en.wav")
        let saved = VoxtralDebug.enabled
        VoxtralDebug.enabled = false
        defer { VoxtralDebug.enabled = saved }

        var samples = 0, text = ""
        let written = try await capturingStdout {
            let tts = VoxtralTTSPipeline()
            try await tts.loadModel(modelInfo: ttsInfo)
            samples = try await tts.synthesize(text: "Hello from a silent library.", voice: .neutralFemale, seed: 1)
                .waveform.dim(0)
            tts.unload()
            let stt = VoxtralPipeline(model: .mini3b4bit)
            try await stt.loadModel()
            text = try await stt.transcribe(audio: clip)
            stt.unload()
        }
        let preview = String(decoding: written.prefix(300), as: UTF8.self)
        print("[k88] SILENT stdout \(written.count) bytes ; synthesis \(samples) samples ; transcription \(text.count) chars")
        XCTAssertGreaterThan(samples, 0)
        XCTAssertFalse(text.isEmpty)
        XCTAssertEqual(written.count, 0, "stdout: \(preview)")
    }
}
