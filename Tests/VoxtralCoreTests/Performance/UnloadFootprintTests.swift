/**
 * UnloadFootprintTests - K-52
 *
 * With the opt-in MLX cache limit, unloading a TTS or Realtime pipeline gives the memory back:
 * the process footprint returns within 200 MB of what it was before loading.
 *
 * Heavy (real 4-bit models): skipped unless VOXTRAL_UNLOAD_FOOTPRINT=1
 * (`TEST_RUNNER_VOXTRAL_UNLOAD_FOOTPRINT=1 xcodebuild test …`).
 */

import Darwin
import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class UnloadFootprintTests: XCTestCase {

    private func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    /// The GPU driver takes back freed Metal buffers asynchronously (≈ 1 s): wait up to 10 s.
    private func settledFootprint(below limit: Double) async -> Double {
        var value = footprintMB()
        for _ in 0 ..< 20 where value > limit {
            try? await Task.sleep(nanoseconds: 500_000_000)
            value = footprintMB()
        }
        return value
    }

    private func report(_ label: String, reference: Double, loaded: Double) async {
        var line = "[unload-footprint] \(label) ref \(Int(reference)) MB · loaded \(Int(loaded)) MB"
        for delay in [0.0, 1.0, 5.0] {
            if delay > 0 { try? await Task.sleep(nanoseconds: UInt64(delay * 1e9)) }
            line += " · +\(Int(delay))s \(Int(footprintMB() - reference)) MB"
        }
        print(line + " · MLX active \(Memory.activeMemory / 1_048_576) MB cache \(Memory.cacheMemory / 1_048_576) MB")
        let vmmap = Process()
        vmmap.executableURL = URL(fileURLWithPath: "/usr/bin/vmmap")
        vmmap.arguments = ["--summary", "\(getpid())"]
        let pipe = Pipe(); vmmap.standardOutput = pipe; vmmap.standardError = Pipe()
        try? vmmap.run(); vmmap.waitUntilExit()
        let text = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        for row in text.split(separator: "\n") where row.contains("IOAccelerator") || row.contains("MALLOC") || row.contains("mapped file") || row.contains("TOTAL") || row.contains("Physical footprint") || row.contains("IOKit") {
            print("[unload-footprint]   \(label) vmmap: \(row.prefix(150))")
        }
    }

    private func requireHeavy() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_UNLOAD_FOOTPRINT"] == "1",
                          "Set VOXTRAL_UNLOAD_FOOTPRINT=1 to run")
    }

    func testTTSUnloadReturnsFootprint() async throws {
        try requireHeavy()
        Memory.clearCache()
        let reference = footprintMB()
        let pipeline = VoxtralTTSPipeline(configuration: .init(maxFrames: 200, cacheLimitBytes: 512 * 1_048_576))
        try await pipeline.loadModel(modelInfo: XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit")))
        _ = try await pipeline.synthesize(text: "Bonjour, ceci est un test de mémoire.", voice: .frFemale, seed: 42)
        let loaded = footprintMB()
        pipeline.unload()
        await report("TTS", reference: reference, loaded: loaded)
        let after = await settledFootprint(below: reference + 200)
        print("[unload-footprint] TTS settled Δ \(Int(after - reference)) MB")
        XCTAssertLessThanOrEqual(after, reference + 200)
    }

    func testRealtimeUnloadReturnsFootprint() async throws {
        try requireHeavy()
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        Memory.clearCache()
        let reference = footprintMB()
        let pipeline = VoxtralRealtimePipeline(configuration: .init(cacheLimitBytes: 512 * 1_048_576))
        try await pipeline.loadModel(modelId: "realtime-4b-4bit")
        _ = try await pipeline.transcribe(audio: root.appendingPathComponent("docs/examples/fluxforge_short_en_6bit.wav"))
        let loaded = footprintMB()
        pipeline.unload()
        await report("RT", reference: reference, loaded: loaded)
        let after = await settledFootprint(below: reference + 200)
        print("[unload-footprint] RT settled Δ \(Int(after - reference)) MB")
        XCTAssertLessThanOrEqual(after, reference + 200)
    }
}
