/**
 * RuntimeBeaconRaceTests - K-28 (A-20)
 *
 * `update` is "safe from any thread": 1 000 concurrent `update` calls racing one `end` must never leave a manifest
 * behind (before the fix, a write released the lock before writing and could recreate the file after its removal).
 * Repeated over several rounds to make the race likely; the session stays referenced so `deinit` cannot clean up.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class RuntimeBeaconRaceTests: XCTestCase {

    func testConcurrentUpdatesRacingEndLeaveNoManifest() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("beacon-race-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        RuntimeBeacon.directoryOverride = dir
        RuntimeBeacon.isEnabled = true
        defer {
            RuntimeBeacon.isEnabled = false
            RuntimeBeacon.directoryOverride = nil
            try? FileManager.default.removeItem(at: dir)
        }

        let rounds = 50, updates = 1_000
        var residualRounds = 0
        for _ in 0 ..< rounds {
            let session = try XCTUnwrap(RuntimeBeacon.begin(task: "race"))
            DispatchQueue.concurrentPerform(iterations: updates + 1) { i in
                if i == updates / 2 {
                    session.end()
                } else {
                    session.update(phase: "p\(i)", step: i, totalSteps: updates)
                }
            }
            let residual = (try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))
                .filter { $0.pathExtension == "json" }
            if !residual.isEmpty {
                residualRounds += 1
                residual.forEach { try? FileManager.default.removeItem(at: $0) }
            }
            withExtendedLifetime(session) {}
        }
        print("[K-28] RuntimeBeaconRace: \(residualRounds) manifeste(s) résiduel(s) sur \(rounds) tours de \(updates) update")
        XCTAssertEqual(residualRounds, 0)
    }
}
