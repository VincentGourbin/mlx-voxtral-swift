/**
 * ModelLoadingSymlinkedDirectoryTests - Regression test for loadWeights(modelPath:)
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

    func testLoadWeightsFollowsSymlinkedModelDirectory() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("voxtral-symlinkdir-\(UUID().uuidString)")
        let realModelDir = root.appendingPathComponent("real-model")
        let parentDir = root.appendingPathComponent("parent")
        try fm.createDirectory(at: realModelDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: parentDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let expected = MLXArray([Float(1), 2, 3])
        try MLX.save(arrays: ["test.weight": expected], url: realModelDir.appendingPathComponent("weights.safetensors"))

        // The model directory itself is a symlink (simulates a v2 whole-directory relocation).
        let symlinkedModelDir = parentDir.appendingPathComponent("model")
        try fm.createSymbolicLink(at: symlinkedModelDir, withDestinationURL: realModelDir)

        let weights = try loadWeights(modelPath: symlinkedModelDir)

        XCTAssertEqual(weights["test.weight"]?.asArray(Float.self), expected.asArray(Float.self))
    }
}
