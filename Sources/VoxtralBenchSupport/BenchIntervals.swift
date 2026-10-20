/**
 * BenchIntervals — the pure arithmetic of `VoxtralCLI bench` (interval union, exclusive phase time, phase keys),
 * in a library target so that VoxtralCLITests can test it: Xcode 26 cannot import an executable target (K-89).
 */

import Foundation

public enum BenchIntervals {
    /// Total length of the union of [start, end] intervals, in the same unit
    public static func unionLength<T: BinaryInteger>(_ intervals: [(start: T, end: T)]) -> Double {
        var total = 0.0
        var current: (start: T, end: T)?
        for interval in intervals.sorted(by: { $0.start < $1.start }) {
            if let c = current, interval.start <= c.end {
                current = (c.start, max(c.end, interval.end))
            } else {
                if let c = current { total += Double(c.end - c.start) }
                current = interval
            }
        }
        if let c = current { total += Double(c.end - c.start) }
        return total
    }

    /// Exclusive time per phase key, in ms: a phase that contains others (STT "Generation" around the MLX encoder,
    /// the LLM prefill and decode; Realtime "Realtime Generation" around encode and prefill) counts only its own
    /// time, so the phases add up to at most total_ms (K-32b; before, decode was counted twice; K-89: tested)
    public static func exclusivePhaseMs<T: BinaryInteger>(_ phases: [(name: String, start: T, end: T)]) -> [String: Double] {
        var result: [String: Double] = [:]
        for (index, phase) in phases.enumerated() {
            let inner = phases.enumerated().filter { other in
                other.offset != index && other.element.start >= phase.start && other.element.end <= phase.end
                    && (other.element.start, other.element.end) != (phase.start, phase.end)
            }.map { (start: $0.element.start, end: $0.element.end) }
            let exclusive = Double(phase.end - phase.start) - unionLength(inner)
            result[phaseKey(phase.name), default: 0] += max(0, exclusive) / 1000
        }
        return result
    }

    public static func phaseKey(_ name: String) -> String {
        let lower = name.lowercased()
        if lower.contains("mel") || lower.contains("feature") { return "audio" }
        if lower.contains("encod") && !lower.contains("decod") { return "encode" }
        if lower.contains("prefill") { return "prefill" }
        if lower.contains("generation") || lower.contains("semantic") { return "decode" }
        if lower.contains("codec") { return "codec" }
        if lower.contains("post") || lower.contains("token decoding") { return "post" }
        return lower.replacingOccurrences(of: " ", with: "_")
    }
}
