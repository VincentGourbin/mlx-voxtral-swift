/**
 * ModelDownloaderSizeTests - Regression tests for ModelDownloader.modelSize(for:)
 *
 * ModelDownloader.directorySize(at:) must follow a file symlink to its target's
 * real size (relocated models on an external disk), not report the symlink's own
 * near-zero size, and a broken symlink (unmounted external disk) must contribute
 * 0 rather than leak the symlink's own size. See
 * Fluxforge Studio/docs/FRAMEWORK_ASKS_STORAGE.md ask #5.
 */

import XCTest
@testable import VoxtralCore

final class ModelDownloaderSizeTests: XCTestCase {

    /// Runs `body` with ModelDownloader.customModelsDirectory sandboxed into a
    /// fresh temp directory, restoring global state afterwards.
    private func withSandbox<T>(_ body: (URL) throws -> T) rethrows -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-modelsize-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let previous = ModelDownloader.customModelsDirectory
        ModelDownloader.customModelsDirectory = dir
        defer {
            ModelDownloader.customModelsDirectory = previous
            try? FileManager.default.removeItem(at: dir)
        }
        return try body(dir)
    }

    func testModelSizeFollowsSymlinkedWeightToItsRealSize() throws {
        try withSandbox { customDir in
            let model = ModelRegistry.defaultModel
            let fm = FileManager.default
            let modelDir = customDir.appendingPathComponent(model.repoId)
            let externalDir = customDir.appendingPathComponent("external")
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
            try fm.createDirectory(at: externalDir, withIntermediateDirectories: true)

            // A regular, non-relocated file.
            let configData = Data(repeating: 0x41, count: 500)
            try configData.write(to: modelDir.appendingPathComponent("config.json"))

            // A "relocated" weight: real bytes live on the external target, the
            // model directory only holds an absolute file symlink to it.
            let targetData = Data(repeating: 0x42, count: 50_000)
            let targetURL = externalDir.appendingPathComponent("model.safetensors")
            try targetData.write(to: targetURL)
            let symlinkURL = modelDir.appendingPathComponent("model.safetensors")
            try fm.createSymbolicLink(at: symlinkURL, withDestinationURL: targetURL)

            let size = ModelDownloader.modelSize(for: model)

            XCTAssertEqual(size, Int64(configData.count + targetData.count))
        }
    }

    func testModelSizeBrokenSymlinkContributesZero() throws {
        try withSandbox { customDir in
            let model = ModelRegistry.defaultModel
            let fm = FileManager.default
            let modelDir = customDir.appendingPathComponent(model.repoId)
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)

            let configData = Data(repeating: 0x41, count: 1_234)
            try configData.write(to: modelDir.appendingPathComponent("config.json"))

            // Simulates an unmounted external disk: the symlink target doesn't exist.
            let missingTarget = customDir.appendingPathComponent("not-mounted/model.safetensors")
            let symlinkURL = modelDir.appendingPathComponent("model.safetensors")
            try fm.createSymbolicLink(at: symlinkURL, withDestinationURL: missingTarget)

            let size = ModelDownloader.modelSize(for: model)

            XCTAssertEqual(size, Int64(configData.count))
        }
    }
}
