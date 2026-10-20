/**
 * BenchInstrumentTests — K-89: the pure parts of `VoxtralCLI bench` (`VoxtralBenchSupport`) on fixed inputs (interval union, exclusive
 * phases adding up to at most the pass, phase keys). The pad fractions of the Realtime pipeline are tested in
 * `VoxtralCoreTests/Realtime/PadFractionTests.swift`.
 */

import XCTest
import VoxtralBenchSupport

final class BenchInstrumentTests: XCTestCase {

    func testUnionLengthMergesOverlapsAndNestedIntervals() {
        XCTAssertEqual(BenchIntervals.unionLength([(start: 0, end: 10), (start: 5, end: 15), (start: 20, end: 25)]), 20)
        XCTAssertEqual(BenchIntervals.unionLength([(start: 0, end: 10), (start: 2, end: 3)]), 10)
        XCTAssertEqual(BenchIntervals.unionLength([(start: 30, end: 40), (start: 0, end: 10)]), 20, "unsorted input")
        XCTAssertEqual(BenchIntervals.unionLength([(start: 0, end: 10), (start: 10, end: 12)]), 12, "touching intervals")
        XCTAssertEqual(BenchIntervals.unionLength([(start: UInt64, end: UInt64)]()), 0)
    }

    /// STT `.mlx` pass as the pipeline records it (µs): features, then "Generation" around the MLX encoder, the
    /// prefill and the decode steps, then token decoding.
    private let sttPass: [(name: String, start: UInt64, end: UInt64)] = [
        ("Audio Feature Extraction", 0, 20_000),
        ("Generation", 20_000, 520_000),
        ("Audio Encoding (MLX)", 20_000, 190_000),
        ("Prefill", 190_000, 400_000),
        ("Token Decoding", 520_000, 521_000),
    ]

    func testExclusivePhasesAddUpToAtMostThePass() {
        let phases = BenchIntervals.exclusivePhaseMs(sttPass)
        XCTAssertEqual(phases["audio"], 20)
        XCTAssertEqual(phases["encode"], 170)
        XCTAssertEqual(phases["prefill"], 210)
        XCTAssertEqual(phases["decode"], 120, "Generation minus the encoder and the prefill it contains")
        XCTAssertEqual(phases["post"], 1)
        let total = Double(sttPass.map(\.end).max()! - sttPass.map(\.start).min()!) / 1000
        XCTAssertLessThanOrEqual(phases.values.reduce(0, +), total)
    }

    func testExclusivePhasesOfARealtimePass() {
        // Realtime: "Realtime Generation" around the encoder and the prefill, one after the other
        let phases = BenchIntervals.exclusivePhaseMs([
            (name: "Realtime Generation", start: 0, end: 1_000_000),
            (name: "Realtime Encoding", start: 0, end: 300_000),
            (name: "Prefill", start: 300_000, end: 400_000),
        ] as [(name: String, start: UInt64, end: UInt64)])
        XCTAssertEqual(phases["decode"], 600)
        XCTAssertEqual(phases.values.reduce(0, +), 1_000)
    }

    func testPhaseKeysOfTheProfiledPhases() {
        XCTAssertEqual(BenchIntervals.phaseKey("Audio Feature Extraction"), "audio")
        XCTAssertEqual(BenchIntervals.phaseKey("Audio Encoding (MLX)"), "encode")
        XCTAssertEqual(BenchIntervals.phaseKey("Audio Encoding (CoreML)"), "encode")
        XCTAssertEqual(BenchIntervals.phaseKey("Generation"), "decode")
        XCTAssertEqual(BenchIntervals.phaseKey("Token Decoding"), "post")
    }
}
