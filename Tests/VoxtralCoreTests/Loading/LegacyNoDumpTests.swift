/**
 * LegacyNoDumpTests - K-23 (S-14)
 *
 * The legacy loading path writes nothing to /tmp: `writeDebugToDump` used to append every message of the legacy
 * loaders to /tmp/swift_debug_generation.txt (unbounded growth), even without debug enabled. The test runs the
 * legacy loader `loadVoxtralModel(modelPath:dtype:lazy:)` on the bf16 Mini 3B folder (it emits its messages, then
 * fails on that folder with keyNotFound, a separate legacy defect). A full `VoxtralGenerator` generation is not
 * possible: that legacy entry point stops the process at load (`needModuleInfo` for `standardModel`, K-30).
 * Skipped when the folder is absent.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class LegacyNoDumpTests: XCTestCase {

    private let dumpPath = "/tmp/swift_debug_generation.txt"

    func testLegacyLoaderWritesNoDumpFile() throws {
        let info = try XCTUnwrap(ModelRegistry.model(withId: "mini-3b"))
        guard let folder = ModelDownloader.findModelPath(for: info) else { throw XCTSkip("mini-3b is not downloaded") }
        try? FileManager.default.removeItem(atPath: dumpPath)
        let wasEnabled = VoxtralDebug.enabled
        VoxtralDebug.enabled = false
        defer { VoxtralDebug.enabled = wasEnabled }

        _ = try? loadVoxtralModel(modelPath: folder.path, dtype: .float16, lazy: true)
        let exists = FileManager.default.fileExists(atPath: dumpPath)
        print("[nodump] legacy loader ran ; \(dumpPath) exists = \(exists)")
        XCTAssertFalse(exists, "NODUMP \(dumpPath) must stay absent")
    }
}
