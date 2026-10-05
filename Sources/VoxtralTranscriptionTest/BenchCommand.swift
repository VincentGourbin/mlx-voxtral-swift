/**
 * BenchCommand - `VoxtralCLI bench`: the baseline instrument (K-32, P-79)
 *
 * Goes through the library's public pipelines only (piège 33), refuses a Debug binary and a busy
 * machine (other MLX processes, live beacons of other runtimes), excludes model loading and warm-up
 * passes, and prints one `BENCH {json}` line per pass (also appended to `<out>/bench.jsonl`), then
 * an `AA …` verdict when `--passes ≥ 2`. Lines follow `docs/bench.schema.json`.
 */

import ArgumentParser
import AVFoundation
import CryptoKit
import Darwin
import Foundation
import MLX
import MLXProfiler
import VoxtralCore

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Measure a pipeline (baseline instrument): one BENCH JSON line per pass, A/A verdict",
        subcommands: [BenchSTT.self, BenchTTS.self, BenchRealtime.self, BenchEnroll.self, BenchChat.self]
    )
}

// MARK: - Common options

struct BenchCommonOptions: ParsableArguments {
    @Option(name: .long, help: "Measured passes") var passes: Int = 2
    @Option(name: .long, help: "Warm-up passes (excluded)") var warmup: Int = 1
    @Option(name: .long, help: "Cooldown in seconds before each measured pass") var cooldown: Int = 120
    @Option(name: .long, help: "Tag for A/B/B/A series (A|B)") var tag: String = "A"
    @Option(name: .long, help: "Output directory for bench.jsonl") var out: String = ".local-runs/bench.noindex"
    @Option(name: .long, help: "MLX buffer-cache limit in MB set by the pipeline (K-52); omit to leave it unset")
    var cacheLimitMb: Int?
    @Flag(name: .long, help: "Afterwards, run one separate diagnostic pass (not a BENCH line) and export its Chrome trace to --out")
    var trace: Bool = false
    @Flag(name: .long, help: "With --trace: also record a Metal System Trace of that pass and print GPU occupancy per phase (GPUPHASE lines)")
    var metalTrace: Bool = false
    @Flag(name: .long, help: "Advertise activity to external monitors (SiliconScope)") var beacon: Bool = false

    var cacheLimitBytes: Int? { cacheLimitMb.map { $0 * 1_048_576 } }
}

// MARK: - Pass measurement

/// Samples MLX active memory and the process footprint every 5 ms, on the profiling session's clock.
final class MemorySampler: @unchecked Sendable {
    struct Sample { let us: UInt64; let activeMB: Double; let footprintMB: Double }
    private let lock = NSLock()
    private var samples: [Sample] = []
    private var running = true
    private let clock: () -> UInt64

    init(clock: @escaping () -> UInt64) {
        self.clock = clock
        let thread = Thread { [weak self] in
            while let self, self.lock.withLock({ self.running }) {
                let sample = Sample(us: self.clock(), activeMB: Double(Memory.activeMemory) / 1_048_576,
                                    footprintMB: BenchSystem.footprintMB())
                self.lock.withLock { self.samples.append(sample) }
                usleep(5_000)
            }
        }
        thread.qualityOfService = .utility
        thread.start()
    }

    func stop() -> [Sample] {
        lock.withLock { running = false; return samples }
    }
}

/// One measured pass: profiling session (phases, steps), memory sampler, exact MLX peak.
struct PassMeasurement {
    var totalMs: Double = 0
    var phases: [String: Double] = [:]
    var phasePeakMLX: [String: Double] = [:]
    var peakMLXMB: Double = 0
    var peakFootprintMB: Double = 0
    var stepMs: [Double] = []
    var firstStepEndMs: Double?
    var events: [ProfilingEvent] = []

    /// Set for the `--trace` diagnostic pass: fine-grained session, Chrome trace written there (K-32b)
    @TaskLocal static var traceURL: URL?
    /// With `--trace --metal-trace`: record a Metal System Trace of the diagnostic pass (K-34, K-36)
    @TaskLocal static var metalTrace = false

    static func run(_ body: () async throws -> Void) async throws -> PassMeasurement {
        let session = traceURL == nil
            ? ProfilingSession(config: ProfilingConfig(
                trackMemory: false, trackPerStepMemory: false, exportChromeTrace: false, printSummary: false,
                enableSampling: false, trackSystemMemory: false, preventIdleSleep: true, recordPowerManagement: false))
            // .fineGrained (IOReport residency, interval mean) for the per-phase GPU occupancy (K-36)
            : ProfilingSession(config: ProfilingConfig(
                trackMemory: true, trackPerStepMemory: true, exportChromeTrace: true, printSummary: false,
                enableSampling: true, samplingIntervalMs: 16, snapshotAtPhaseBoundaries: false,
                gpuBackend: .ioReportResidency))
        let profiler = MLXProfiler.shared
        profiler.enable()
        profiler.activeSession = session
        defer { profiler.activeSession = nil; profiler.disable() }

        var recorder: MetalSystemTrace.Recorder?
        if let traceURL, metalTrace {
            let output = traceURL.deletingPathExtension().appendingPathExtension("trace")
            recorder = try session.startMetalSystemTrace(output: output)
            if recorder?.waitUntilRecording() != true { print("METAL_TRACE not recording: \(recorder?.log() ?? "")") }
        }
        GPU.resetPeakMemory()  // the instrument resets the peak, never the library (P-77)
        let sampler = MemorySampler(clock: { session.currentTimestampUsPublic() })
        let startUs = session.currentTimestampUsPublic()
        let startEpoch = Date().timeIntervalSince1970
        let start = CFAbsoluteTimeGetCurrent()
        try await body()
        var m = PassMeasurement()
        m.totalMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let samples = sampler.stop()
        session.finish()

        m.peakMLXMB = Double(Memory.peakMemory) / 1_048_576
        m.peakFootprintMB = samples.map(\.footprintMB).max() ?? BenchSystem.footprintMB()
        var kernels: [GPUKernelInterval] = []
        if let recorder {
            let bundle = try recorder.stop()
            let summary = try session.mergeMetalSystemTrace(bundle)
            kernels = session.mergedGPUKernelIntervals
            print("METAL_TRACE \(bundle.path) intervals=\(summary.intervalCount) busy=\(String(format: "%.1f", summary.busyPercent))%")
        }
        if let traceURL {
            try ChromeTraceExporter.export(session: session).write(to: traceURL)
        }
        let phases = session.phaseSummaries()
        if traceURL != nil {
            // GPU occupancy per phase by each instrument; epoch bounds let an external `ioreg` sampler be aligned
            for phase in phases {
                var record: [String: Any] = [
                    "phase": phase.name, "duration_ms": BenchJSON.round(phase.durationMs) ?? 0,
                    "start_epoch": startEpoch + Double(phase.startUs - startUs) / 1e6,
                    "end_epoch": startEpoch + Double(phase.endUs - startUs) / 1e6,
                ]
                if let gpu = phase.gpu { record["profiler_gpu_mean"] = BenchJSON.round(gpu.mean) ?? 0 }
                if !kernels.isEmpty, phase.endUs > phase.startUs {
                    let clipped = kernels.compactMap { k -> (start: UInt64, end: UInt64)? in
                        let a = max(k.startUs, phase.startUs), b = min(k.endUs, phase.endUs)
                        return a < b ? (a, b) : nil
                    }
                    record["xctrace_busy_pct"] = BenchJSON.round(
                        BenchJSON.unionLength(clipped) / Double(phase.endUs - phase.startUs) * 100) ?? 0
                }
                let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
                print("GPUPHASE \(String(decoding: data, as: UTF8.self))")
            }
        }
        let intervals = phases.map { (start: $0.startUs, end: $0.endUs) }
        for (index, phase) in phases.enumerated() {
            let key = BenchJSON.phaseKey(phase.name)
            // Exclusive time: a phase that contains others (STT "Generation" around the LLM prefill and decode,
            // Realtime "Realtime Generation" around encode and prefill) counts only its own time, so the
            // phases add up to at most total_ms (K-32b; before, decode was counted twice)
            let inner = intervals.enumerated().filter { other in
                other.offset != index && other.element.start >= phase.startUs && other.element.end <= phase.endUs
                    && (other.element.start, other.element.end) != (phase.startUs, phase.endUs)
            }.map(\.element)
            let exclusiveUs = Double(phase.endUs - phase.startUs) - BenchJSON.unionLength(inner)
            m.phases[key, default: 0] += max(0, exclusiveUs) / 1000
            let inPhase = samples.filter { $0.us >= phase.startUs && $0.us <= phase.endUs }.map(\.activeMB)
            if let peak = inPhase.max() { m.phasePeakMLX[key] = max(m.phasePeakMLX[key] ?? 0, peak) }
        }
        m.events = session.getEvents()
        let steps = m.events.filter {
            $0.category == .generationStep || $0.category == .semanticCodeGen
        }.filter { $0.durationUs != nil }
        // The first recorded step of the STT/TTS loops includes the prefill: a pseudo-step, excluded
        m.stepMs = steps.filter { ($0.stepIndex ?? 0) > 1 }.map { Double($0.durationUs!) / 1000 }
        if let first = steps.first(where: { ($0.stepIndex ?? 0) == 1 }) ?? steps.first {
            m.firstStepEndMs = Double(first.timestampUs + (first.durationUs ?? 0) - startUs) / 1000
        }
        return m
    }
}

// MARK: - Machine, hygiene, JSON

enum BenchSystem {
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }

    static func shell(_ args: [String]) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = args
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return nil }
        // Read before waiting: a large output (ps) would fill the pipe and block the child forever
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Revisions resolved in Package.resolved (the build uses -onlyUsePackageVersionsFromResolvedFile)
    static func resolvedRevisions() -> [String: String] {
        guard let data = FileManager.default.contents(atPath: "Package.resolved"),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pins = json["pins"] as? [[String: Any]] else { return [:] }
        var result: [String: String] = [:]
        for pin in pins {
            guard let identity = pin["identity"] as? String, let state = pin["state"] as? [String: Any],
                  let revision = state["revision"] as? String else { continue }
            let label = (state["version"] as? String) ?? (state["branch"] as? String) ?? "rev"
            result[identity] = "\(label)@\(revision.prefix(9))"
        }
        return result
    }

    static func environment() -> [String: Any] {
        let revisions = resolvedRevisions()
        let osVersion = ProcessInfo.processInfo.operatingSystemVersion
        return [
            "commit": shell(["git", "rev-parse", "--short=9", "HEAD"]) ?? "unknown",
            "dirty": !(shell(["git", "status", "--porcelain", "--untracked-files=no"]) ?? "").isEmpty,
            "build": RunEnvironment.buildConfiguration,
            "mlx_swift": revisions["mlx-swift"] ?? "unknown",
            "mlx_swift_lm": revisions["mlx-swift-lm"] ?? "unknown",
            "mlx_profiler": revisions["swift-mlx-profiler"] ?? "unknown",
            "chip": sysctlString("machdep.cpu.brand_string") ?? "unknown",
            "ram_gb": Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824),
            "macos": "\(osVersion.majorVersion).\(osVersion.minorVersion).\(osVersion.patchVersion)",
            "power": RunEnvironment.powerSource() ?? "unknown",
        ]
    }

    static func topProcess() -> String {
        let out = shell(["ps", "-Ao", "%cpu=,comm=", "-r"]) ?? ""
        return out.split(separator: "\n").first.map {
            $0.split(separator: " ", omittingEmptySubsequences: true).map(String.init).joined(separator: " ")
        } ?? "unknown"
    }

    /// Reasons the machine is not fit to measure (empty = fine)
    static func busyReasons() -> [String] {
        var reasons: [String] = []
        let me = getpid()
        let heavy = try! NSRegularExpression(pattern: "^(gemma4-cli|qwen38|yue2|mlx_lm|mlx-lm|Voxtral.*|FluxForge.*)$|BenchUI|-cli$")
        for line in (shell(["ps", "-Ao", "pid=,comm="]) ?? "").split(separator: "\n") {
            let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
            guard parts.count == 2, let pid = Int32(parts[0]), pid != me else { continue }
            let name = URL(fileURLWithPath: String(parts[1])).lastPathComponent
            if heavy.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)) != nil {
                reasons.append("MLX process \(name) (pid \(pid))")
            }
        }
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/ai-runtime-beacons")
        for file in (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [] where file.hasSuffix(".json") {
            guard let pid = Int32(file.split(separator: "-").first ?? ""), pid != me, kill(pid, 0) == 0 else { continue }
            let runtime = (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(file))))
                .flatMap { ($0 as? [String: Any])?["runtime"] as? String } ?? "?"
            reasons.append("live beacon of \(runtime) (pid \(pid))")
        }
        return reasons
    }
}

enum BenchJSON {
    /// Total length of the union of [start, end] intervals, in the same unit
    static func unionLength<T: BinaryInteger>(_ intervals: [(start: T, end: T)]) -> Double {
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

    static func phaseKey(_ name: String) -> String {
        let lower = name.lowercased()
        if lower.contains("mel") || lower.contains("feature") { return "audio" }
        if lower.contains("encod") && !lower.contains("decod") { return "encode" }
        if lower.contains("prefill") { return "prefill" }
        if lower.contains("generation") || lower.contains("semantic") { return "decode" }
        if lower.contains("codec") { return "codec" }
        if lower.contains("post") || lower.contains("token decoding") { return "post" }
        return lower.replacingOccurrences(of: " ", with: "_")
    }

    static func percentile(_ values: [Double], _ p: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let index = min(sorted.count - 1, max(0, Int((p * Double(sorted.count - 1)).rounded())))
        return sorted[index]
    }

    static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    /// Weights identity: the K-6 completeness manifest (sizes + SHA-256 of every file)
    static func packSHA(folder: URL?) -> String? {
        guard let folder, let data = FileManager.default.contents(
            atPath: folder.appendingPathComponent(VoxtralModelDownloader.manifestFileName).path) else { return nil }
        return String(sha256(data).prefix(16))
    }

    static func round(_ value: Double?, _ digits: Int = 1) -> Double? {
        value.map { (($0 * pow(10, Double(digits))).rounded()) / pow(10, Double(digits)) }
    }

    /// One `<tag> {json}` line on stdout, appended to `<out>/<file>` (`EVAL` lines of `voxtral eval` too, K-33)
    static func emit(_ record: [String: Any], out: String, tag: String = "BENCH", file fileName: String = "bench.jsonl") {
        var full = BenchSystem.environment()
        full["date"] = ISO8601DateFormatter().string(from: Date())
        full["top_process"] = BenchSystem.topProcess()
        for (key, value) in record { full[key] = value }
        guard let data = try? JSONSerialization.data(withJSONObject: decimals(full), options: [.sortedKeys, .withoutEscapingSlashes]),
              let line = String(data: data, encoding: .utf8) else { return }
        print("\(tag) \(line)")
        let dir = URL(fileURLWithPath: out)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent(fileName)
        if !FileManager.default.fileExists(atPath: file.path) { FileManager.default.createFile(atPath: file.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: file) {
            handle.seekToEndOfFile()
            handle.write(Data((line + "\n").utf8))
            try? handle.close()
        }
    }

    /// Doubles written as exact decimals (≤ 4 digits) instead of binary expansions (116.40000000000001)
    static func decimals(_ value: Any) -> Any {
        switch value {
        case let dict as [String: Any]: return dict.mapValues { decimals($0) }
        case let array as [Any]: return array.map { decimals($0) }
        case let bool as Bool: return bool
        case let int as Int: return int
        case let double as Double: return NSDecimalNumber(string: String(format: "%.4f", double))
        default: return value
        }
    }

    /// `AA dispersion …` over the measured passes: ≤ 3 % on every metric and identical outputs.
    static func printAA(_ records: [[String: Any]], metrics: [String], label: String = "") {
        guard records.count >= 2 else { return }
        var parts: [String] = []
        var pass = true
        for metric in metrics {
            let values = records.compactMap { $0[metric] as? Double }.filter { $0 > 0 }
            guard values.count == records.count, let low = values.min(), let high = values.max() else {
                parts.append("\(metric)=n/a"); continue
            }
            let dispersion = (high - low) / low * 100
            if dispersion > 3 { pass = false }
            parts.append(String(format: "%@=%.2f%%", metric, dispersion))
        }
        let outputs = Set(records.compactMap { $0["out_sha256"] as? String })
        let identical = outputs.count == 1
        if !identical { pass = false }
        parts.append("out_sha256=\(identical ? "identical" : "DIFFERENT")")
        print("AA \(label)dispersion \(parts.joined(separator: " ")) → \(pass ? "PASS" : "FAIL") (≤ 3 %)")
    }
}

// MARK: - Runner shared by the subcommands

enum BenchRunner {
    /// Refuses a Debug binary; before each pass, cools down and refuses a busy machine.
    static func guardBuild() throws {
        if RunEnvironment.isDebugBuild {
            print("REFUSED debug build")
            throw ExitCode(2)
        }
    }

    static func prepare(pass: Int, cooldown: Int) throws {
        if cooldown > 0 {
            print("… cooldown \(cooldown) s before pass \(pass)")
            sleep(UInt32(cooldown))
        }
        let reasons = BenchSystem.busyReasons()
        if !reasons.isEmpty {
            print("REFUSED busy: \(reasons.joined(separator: "; "))")
            throw ExitCode(3)
        }
    }

    /// Warm-up passes (excluded, no cooldown before the first) then measured passes.
    static func run(
        _ common: BenchCommonOptions,
        base: [String: Any],
        metrics: [String] = ["total_ms", "step_ms_p50"],
        pass body: (_ pass: Int, _ warm: Bool) async throws -> [String: Any]
    ) async throws {
        try guardBuild()
        if common.beacon { VoxtralRuntimeBeacon.isEnabled = true }
        for w in 0 ..< common.warmup {
            try prepare(pass: -(w + 1), cooldown: 0)
            _ = try await body(-(w + 1), true)
            print("… warm-up \(w + 1)/\(common.warmup) done (excluded)")
        }
        var records: [[String: Any]] = []
        for pass in 1 ... max(1, common.passes) {
            try prepare(pass: pass, cooldown: common.cooldown)
            var record = base
            for (key, value) in try await body(pass, false) { record[key] = value }
            record["pass"] = pass
            record["tag"] = common.tag
            record["warm"] = true
            BenchJSON.emit(record, out: common.out)
            records.append(record)
        }
        BenchJSON.printAA(records, metrics: metrics)
        if common.trace {
            let dir = URL(fileURLWithPath: common.out)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
            let url = dir.appendingPathComponent("trace-\(base["pipeline"] ?? "bench")-\(stamp).json")
            try prepare(pass: 0, cooldown: common.cooldown)
            _ = try await PassMeasurement.$metalTrace.withValue(common.metalTrace) {
                try await PassMeasurement.$traceURL.withValue(url) { try await body(0, false) }
            }
            print("TRACE \(url.path) (diagnostic pass, not a BENCH line; open in https://ui.perfetto.dev/)")
        }
    }

    /// Fields every pipeline derives from a PassMeasurement
    static func fields(_ m: PassMeasurement, inputSeconds: Double?) -> [String: Any] {
        var record: [String: Any] = [
            "total_ms": BenchJSON.round(m.totalMs) ?? 0,
            "phases_ms": m.phases.mapValues { BenchJSON.round($0) ?? 0 },
            "peak_mlx_mb": BenchJSON.round(m.peakMLXMB) ?? 0,
            "peak_mlx_mb_by_phase": m.phasePeakMLX.mapValues { BenchJSON.round($0) ?? 0 },
            "peak_footprint_mb": BenchJSON.round(m.peakFootprintMB) ?? 0,
            "steps": m.stepMs.count,
        ]
        if let p50 = BenchJSON.percentile(m.stepMs, 0.5) { record["step_ms_p50"] = BenchJSON.round(p50, 2) }
        if let p90 = BenchJSON.percentile(m.stepMs, 0.9) { record["step_ms_p90"] = BenchJSON.round(p90, 2) }
        if let ttft = m.firstStepEndMs { record["ttft_ms"] = BenchJSON.round(ttft) }
        if let inputSeconds, inputSeconds > 0 {
            record["rtf"] = BenchJSON.round(m.totalMs / 1000 / inputSeconds, 4)
        }
        return record
    }

    static func audioSeconds(_ path: String) -> Double? {
        guard let file = try? AVAudioFileBox(path) else { return nil }
        return file.seconds
    }

    static func waveformSHA(_ waveform: MLXArray) -> String {
        let samples = waveform.asType(.float32).asArray(Float.self)
        var pcm = [Int16](repeating: 0, count: samples.count)
        for (i, s) in samples.enumerated() { pcm[i] = Int16(max(-1, min(1, s)) * 32767) }
        return BenchJSON.sha256(pcm.withUnsafeBufferPointer { Data(buffer: $0) })
    }
}

/// Duration of an audio file without loading it
struct AVAudioFileBox {
    let seconds: Double
    init(_ path: String) throws {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        seconds = Double(file.length) / file.processingFormat.sampleRate
    }
}

func parseSTTModelID(_ id: String) -> VoxtralPipeline.Model? { VoxtralPipeline.Model(rawValue: id) }

func parseBackend(_ value: String) throws -> VoxtralPipeline.Backend {
    switch value {
    case "mlx": return .mlx
    case "auto": return .auto
    case "hybrid": return .hybrid
    default: throw ValidationError("Unknown backend: \(value) (mlx, auto, hybrid)")
    }
}

// MARK: - stt

struct BenchSTT: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "stt", abstract: "VoxtralPipeline.transcribe")
    @Option(name: .long) var model: String = "mini-3b-8bit"
    @Option(name: .long, help: "mlx | auto") var backend: String = "mlx"
    @Option(name: .long) var input: String
    @Option(name: .long) var language: String?
    @Option(name: .long, help: "Text-token cap (default: from the audio duration)") var maxTokens: Int?
    @OptionGroup var common: BenchCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        guard let pipelineModel = parseSTTModelID(model) else { throw ValidationError("Unknown STT model: \(model)") }
        var config = VoxtralPipeline.Configuration.default
        config.maxTokens = maxTokens
        config.memoryOptimization.cacheLimitBytes = common.cacheLimitBytes
        let pipeline = VoxtralPipeline(model: pipelineModel, backend: try parseBackend(backend), configuration: config)
        let loadStart = CFAbsoluteTimeGetCurrent()
        try await pipeline.loadModel()
        let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        let url = URL(fileURLWithPath: input)
        let seconds = BenchRunner.audioSeconds(input)
        let base: [String: Any] = [
            "pipeline": "stt", "model": model, "backend": backend, "input": input,
            "input_s": BenchJSON.round(seconds, 2) ?? 0, "language": language ?? "auto",
            "profile": common.cacheLimitMb.map { "cacheLimit=\($0)MB" } ?? "default", "load_ms": BenchJSON.round(loadMs) ?? 0,
        ]
        try await BenchRunner.run(common, base: base) { _, _ in
            var text = ""
            let m = try await PassMeasurement.run { text = try await pipeline.transcribe(audio: url, language: language) }
            var record = BenchRunner.fields(m, inputSeconds: seconds)
            record["out_sha256"] = BenchJSON.sha256(Data(text.utf8))
            record["chars"] = text.count
            record["truncated"] = pipeline.lastResultTruncated
            return record
        }
        pipeline.unload()
    }
}

// MARK: - realtime

struct BenchRealtime: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "realtime", abstract: "VoxtralRealtimePipeline.transcribe")
    @Option(name: .long) var model: String = "realtime-4b-4bit"
    @Option(name: .long) var input: String
    @Option(name: .long, help: "Transcription delay (ms)") var delay: Int = 480
    @OptionGroup var common: BenchCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        let pipeline = VoxtralRealtimePipeline(configuration: .init(transcriptionDelayMs: delay, cacheLimitBytes: common.cacheLimitBytes))
        let loadStart = CFAbsoluteTimeGetCurrent()
        try await pipeline.loadModel(modelId: model)
        let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        let url = URL(fileURLWithPath: input)
        let seconds = BenchRunner.audioSeconds(input)
        let base: [String: Any] = [
            "pipeline": "realtime", "model": model, "input": input, "input_s": BenchJSON.round(seconds, 2) ?? 0,
            "delay_ms": delay, "profile": common.cacheLimitMb.map { "cacheLimit=\($0)MB" } ?? "default",
            "load_ms": BenchJSON.round(loadMs) ?? 0,
        ]
        try await BenchRunner.run(common, base: base) { _, _ in
            var text = ""
            let m = try await PassMeasurement.run { text = try await pipeline.transcribe(audio: url) }
            var record = BenchRunner.fields(m, inputSeconds: seconds)
            record["out_sha256"] = BenchJSON.sha256(Data(text.utf8))
            record["chars"] = text.count
            record["truncated"] = pipeline.lastTranscriptionTruncated
            if let pad = pipeline.lastPadFraction { record["pad_fraction"] = BenchJSON.round(pad, 4) }
            if let pad = pipeline.lastStreamingPadFraction { record["streaming_pad_fraction"] = BenchJSON.round(pad, 4) }
            if let encode = m.phases["encode"], let seconds, seconds > 0 {
                record["encode_ms_per_audio_s"] = BenchJSON.round(encode / seconds, 2)
            }
            return record
        }
        pipeline.unload()
    }
}

// MARK: - tts

struct BenchTTS: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "tts", abstract: "VoxtralTTSPipeline.synthesize / synthesizeStreaming")
    @Option(name: .long) var model: String = "tts-4b-6bit"
    @Option(name: .long, help: "Preset voice (e.g. neutral_female)") var voice: String?
    @Option(name: .long, help: "Voice embedding .safetensors (key \"embedding\")") var voiceEmbedding: String?
    @Option(name: .long) var textFile: String
    @Option(name: .long) var seed: UInt64 = 42
    @Flag(name: .long) var streaming: Bool = false
    @Flag(name: .long, help: "Use the recommended warm-up vocalise") var warmUp: Bool = false
    @Option(name: .long, help: "Maximum frames") var maxFrames: Int = 2500
    @OptionGroup var common: BenchCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        guard voice != nil || voiceEmbedding != nil else { throw ValidationError("--voice or --voice-embedding is required") }
        guard let info = VoxtralTTSRegistry.model(withId: model) else { throw ValidationError("Unknown TTS model: \(model)") }
        let text = try String(contentsOfFile: textFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        let pipeline = VoxtralTTSPipeline(configuration: .init(maxFrames: maxFrames, cacheLimitBytes: common.cacheLimitBytes))
        let loadStart = CFAbsoluteTimeGetCurrent()
        try await pipeline.loadModel(modelInfo: info)
        let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000

        // Voice as an embedding (same path for batch and streaming, and the seed is honoured)
        let embeddingURL: URL
        if let voiceEmbedding {
            embeddingURL = URL(fileURLWithPath: voiceEmbedding)
        } else {
            guard let folder = VoxtralModelDownloader.findTTSModelPath(for: info) else { throw ValidationError("TTS model folder not found") }
            embeddingURL = folder.appendingPathComponent("voice_embedding/\(voice!).safetensors")
        }
        let arrays = try MLX.loadArrays(url: embeddingURL)
        guard let embedding = arrays["embedding"] else { throw ValidationError("\(embeddingURL.lastPathComponent) has no 'embedding'") }
        let warmUpText = warmUp ? VoxtralTTSPipeline.recommendedWarmUpVocalise : nil

        let base: [String: Any] = [
            "pipeline": "tts", "model": model, "input": textFile, "voice": voice ?? voiceEmbedding ?? "",
            "seed": Int(seed), "streaming": streaming, "warm_up": warmUp,
            "pack_sha256": BenchJSON.packSHA(folder: VoxtralModelDownloader.findTTSModelPath(for: info)) ?? "",
            "profile": common.cacheLimitMb.map { "cacheLimit=\($0)MB" } ?? "default", "load_ms": BenchJSON.round(loadMs) ?? 0,
        ]
        try await BenchRunner.run(common, base: base, metrics: ["total_ms", "step_ms_p50"]) { _, _ in
            var sha = ""
            var frames = 0
            var audioSeconds = 0.0
            var ttfaMs: Double?
            let start = CFAbsoluteTimeGetCurrent()
            let m = try await PassMeasurement.run {
                if streaming {
                    var chunks: [MLXArray] = []
                    for try await chunk in pipeline.synthesizeStreaming(
                        text: text, voiceEmbedding: embedding, seed: seed, warmUpText: warmUpText) {
                        if ttfaMs == nil { ttfaMs = (CFAbsoluteTimeGetCurrent() - start) * 1000 }
                        chunks.append(chunk.waveform)
                        frames = chunk.totalFrames
                    }
                    let waveform = MLX.concatenated(chunks, axis: 0)
                    sha = BenchRunner.waveformSHA(waveform)
                    audioSeconds = Double(waveform.dim(0)) / Double(pipeline.sampleRate)
                } else {
                    let result = try await pipeline.synthesize(
                        text: text, voiceEmbedding: embedding, seed: seed, warmUpText: warmUpText)
                    sha = BenchRunner.waveformSHA(result.waveform)
                    frames = result.numFrames
                    audioSeconds = result.duration
                    ttfaMs = result.timeToFirstToken * 1000
                }
            }
            var record = BenchRunner.fields(m, inputSeconds: nil)
            record["out_sha256"] = sha
            record["frames"] = frames
            record["audio_s"] = BenchJSON.round(audioSeconds, 2)
            record["ttfa_ms"] = BenchJSON.round(ttfaMs)
            // Frame cap of the text and whether it was reached without an end of audio (K-14)
            record["text_tokens"] = pipeline.textTokenCount(text) ?? 0
            record["frame_cap"] = pipeline.frameCap(forText: text)
            if !streaming { record["truncated"] = pipeline.lastSynthesisTruncated }
            if warmUp && !streaming { record["carrier_frames"] = pipeline.lastCarrierFrames }  // K-35: carrier / total frames
            if audioSeconds > 0 { record["rtf"] = BenchJSON.round(m.totalMs / 1000 / audioSeconds, 4) }
            return record
        }
        pipeline.unload()
    }
}

// MARK: - enroll

struct BenchEnroll: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "enroll", abstract: "VoxtralTTSPipeline.enrollVoice")
    @Option(name: .long) var model: String = "tts-4b-6bit"
    @Option(name: .long, help: "Reference audio") var input: String
    @Option(name: .long) var epochs: Int = 100
    @Option(name: .long, help: "Enrollment seed (K-26): same seed and reference give the same voice")
    var seed: UInt64 = 42
    @Flag(name: .long, help: "Synthesize once before enrolling (resident LLM scenario)") var afterSynthesis: Bool = false
    @OptionGroup var common: BenchCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        guard let info = VoxtralTTSRegistry.model(withId: model) else { throw ValidationError("Unknown TTS model: \(model)") }
        let pipeline = VoxtralTTSPipeline(configuration: .init(maxFrames: 200, cacheLimitBytes: common.cacheLimitBytes))
        try await pipeline.loadModel(modelInfo: info)
        if afterSynthesis { _ = try await pipeline.synthesize(text: "Hello there.", voice: .neutralFemale, seed: seed) }
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-bench-enroll.safetensors")
        let base: [String: Any] = [
            "pipeline": "enroll", "model": model, "input": input, "epochs": epochs, "seed": Int(seed),
            "after_synthesis": afterSynthesis, "scenario": afterSynthesis ? "after_synthesis" : "cli", "profile": common.cacheLimitMb.map { "cacheLimit=\($0)MB" } ?? "default",
        ]
        try await BenchRunner.run(common, base: base, metrics: ["total_ms", "epoch_ms_p50"]) { _, _ in
            var epochTimes: [Double] = []
            var last = CFAbsoluteTimeGetCurrent()
            var finalLoss: Double?
            var embedding: MLXArray?
            var config = VoxtralVoiceEnrollment.Config()
            config.epochs = epochs
            config.logEvery = 1
            config.seed = seed  // K-37: the seed was recorded in the line but never applied
            let m = try await PassMeasurement.run {
                embedding = try pipeline.enrollVoice(
                    referenceURL: URL(fileURLWithPath: input), outputURL: output, config: config,
                    progress: { progress in
                        let now = CFAbsoluteTimeGetCurrent()
                        epochTimes.append((now - last) * 1000)
                        last = now
                        finalLoss = Double(progress.totalLoss)
                    })
            }
            var record = BenchRunner.fields(m, inputSeconds: nil)
            record["epoch_ms_p50"] = BenchJSON.round(BenchJSON.percentile(Array(epochTimes.dropFirst()), 0.5), 2)
            record["final_loss"] = BenchJSON.round(finalLoss, 4)
            record["out_sha256"] = embedding.map { BenchRunner.waveformSHA($0.flattened()) } ?? ""
            return record
        }
        pipeline.unload()
    }
}

// MARK: - chat

struct BenchChat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "chat", abstract: "VoxtralPipeline.chat, one line per question")
    @Option(name: .long) var model: String = "mini-3b-8bit"
    @Option(name: .long, help: "mlx | auto") var backend: String = "mlx"
    @Option(name: .long) var input: String
    @Option(name: .long) var questions: String = "docs/eval/chat-questions.json"
    @Option(name: .long) var temperature: Float = 0.0
    @Option(name: .long) var topP: Float = 0.95
    @Option(name: .long) var language: String?
    @OptionGroup var common: BenchCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        guard let pipelineModel = parseSTTModelID(model) else { throw ValidationError("Unknown STT model: \(model)") }
        let data = try Data(contentsOf: URL(fileURLWithPath: questions))
        guard let list = (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["questions"] as? [String] else {
            throw ValidationError("\(questions) must be {\"questions\": [String]}")
        }
        var config = VoxtralPipeline.Configuration.default
        config.temperature = temperature
        config.topP = topP
        config.memoryOptimization.cacheLimitBytes = common.cacheLimitBytes
        let pipeline = VoxtralPipeline(model: pipelineModel, backend: try parseBackend(backend), configuration: config)
        try await pipeline.loadModel()
        let url = URL(fileURLWithPath: input)
        try BenchRunner.guardBuild()
        if common.beacon { VoxtralRuntimeBeacon.isEnabled = true }
        for (index, question) in list.enumerated() {
            for w in 0 ..< common.warmup {
                try BenchRunner.prepare(pass: -(w + 1), cooldown: 0)
                _ = try await pipeline.chat(audio: url, prompt: question, language: language)
            }
            var records: [[String: Any]] = []
            for pass in 1 ... max(1, common.passes) {
                try BenchRunner.prepare(pass: pass, cooldown: common.cooldown)
                var answer = ""
                let m = try await PassMeasurement.run { answer = try await pipeline.chat(audio: url, prompt: question, language: language) }
                var record = BenchRunner.fields(m, inputSeconds: nil)
                record["pipeline"] = "chat"
                record["model"] = model
                record["backend"] = backend
                record["input"] = input
                record["question"] = index + 1
                record["temperature"] = Double(temperature)
                record["out_sha256"] = BenchJSON.sha256(Data(answer.utf8))
                if !m.stepMs.isEmpty {
                    record["tok_s"] = BenchJSON.round(Double(m.stepMs.count) / (m.stepMs.reduce(0, +) / 1000), 2)
                }
                record["pass"] = pass
                record["tag"] = common.tag
                record["warm"] = true
                BenchJSON.emit(record, out: common.out)
                records.append(record)
            }
            BenchJSON.printAA(records, metrics: ["ttft_ms", "tok_s"], label: "q\(index + 1) ")
        }
        pipeline.unload()
    }
}
