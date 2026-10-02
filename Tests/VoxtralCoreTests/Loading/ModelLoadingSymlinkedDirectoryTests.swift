/**
 * ModelLoadingSymlinkedDirectoryTests - Regression test for loadWeights(from:) (the live STT loader, K-29; the
 * legacy loadWeights(modelPath:) was removed in 3.0, K-31)
 *
 * `contentsOfDirectory(at:)` (the `URL`-based API) silently returns nothing for
 * files one level inside a *symlinked* directory; `contentsOfDirectory(atPath:)`
 * follows the symlink transparently. This is the v2 prerequisite for symlinking
 * a whole model directory (today only individual weight files are symlinked).
 * See Fluxforge Studio/docs/FRAMEWORK_ASKS_STORAGE.md ask #6.
 */

import XCTest
import MLX
@testable import VoxtralCore

final class ModelLoadingSymlinkedDirectoryTests: XCTestCase {

    /// The live STT loader (`loadVoxtralStandardModel` → `loadWeights(from:)`) follows a symlinked model directory
    /// and skips `consolidated.safetensors` (K-29)
    func testLiveLoaderFollowsSymlinkedModelDirectory() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("voxtral-symlinkdir-live-\(UUID().uuidString)")
        let realModelDir = root.appendingPathComponent("real-model")
        try fm.createDirectory(at: realModelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let expected = MLXArray([Float(4), 5, 6])
        try MLX.save(arrays: ["test.weight": expected], url: realModelDir.appendingPathComponent("model.safetensors"))
        try MLX.save(arrays: ["unused.weight": MLXArray([Float(0)])],
                     url: realModelDir.appendingPathComponent("consolidated.safetensors"))
        let symlinkedModelDir = root.appendingPathComponent("model")
        try fm.createSymbolicLink(at: symlinkedModelDir, withDestinationURL: realModelDir)

        let weights = try loadWeights(from: symlinkedModelDir)

        XCTAssertEqual(weights["test.weight"]?.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertNil(weights["unused.weight"], "consolidated.safetensors is not read")
    }
}
