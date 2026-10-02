import Foundation

/// Minimal ffmpeg/ffprobe wrapper for building an enrollment reference from a
/// video or audio file (macOS demo only — shells out to a locally installed
/// ffmpeg, which handles formats AVFoundation can't, e.g. .webm/VP9/Opus).
enum FFmpeg {

    enum FFmpegError: LocalizedError {
        case notInstalled
        case failed(String)
        var errorDescription: String? {
            switch self {
            case .notInstalled: return "ffmpeg not found. Install it with: brew install ffmpeg"
            case let .failed(msg): return "ffmpeg failed: \(msg)"
            }
        }
    }

    private static let searchPaths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]

    private static func tool(_ name: String) -> String? {
        searchPaths.map { "\($0)/\(name)" }.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    static var ffmpegPath: String? { tool("ffmpeg") }
    static var ffprobePath: String? { tool("ffprobe") }
    static var isAvailable: Bool { ffmpegPath != nil && ffprobePath != nil }

    /// Total duration (seconds) of any media file.
    static func duration(of url: URL) async throws -> Double {
        guard let probe = ffprobePath else { throw FFmpegError.notInstalled }
        let out = try await run(probe, [
            "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", url.path,
        ])
        guard let d = Double(out.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw FFmpegError.failed("could not read duration")
        }
        return d
    }

    /// Extract `[start, end]` of `source` as a 24 kHz mono WAV at `dest`.
    static func extractSegment(from source: URL, start: Double, end: Double, to dest: URL) async throws {
        guard let ff = ffmpegPath else { throw FFmpegError.notInstalled }
        _ = try await run(ff, [
            "-nostdin", "-loglevel", "error", "-y",
            "-i", source.path,
            "-ss", String(format: "%.3f", start),
            "-to", String(format: "%.3f", end),
            "-ac", "1", "-ar", "24000", "-vn",
            dest.path,
        ])
    }

    /// Concatenate 24 kHz mono WAVs into a single WAV.
    static func concat(_ wavs: [URL], to dest: URL) async throws {
        guard let ff = ffmpegPath else { throw FFmpegError.notInstalled }
        if wavs.count == 1 {
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: wavs[0], to: dest)
            return
        }
        let listURL = dest.deletingLastPathComponent().appendingPathComponent("concat_\(UUID().uuidString).txt")
        let list = wavs.map { "file '\($0.path)'" }.joined(separator: "\n")
        try list.write(to: listURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: listURL) }
        _ = try await run(ff, [
            "-nostdin", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", listURL.path,
            "-c", "copy", dest.path,
        ])
    }

    // MARK: - Process runner

    /// Runs a tool and returns its stdout. Both pipes are drained while it runs, so a process writing more than a
    /// pipe buffer cannot block; cancelling the calling task terminates the process (A-18, K-28).
    static func run(_ path: String, _ args: [String]) async throws -> String {
        let run = ProcessRun(path: path, args: args)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { run.start($0) }
        } onCancel: {
            run.cancel()
        }
    }

    /// One process, its two drained pipes, and the cancellation flag, shared by the task and its cancel handler.
    private final class ProcessRun: @unchecked Sendable {
        private let process = Process()
        private let outPipe = Pipe(), errPipe = Pipe()
        private let lock = NSLock()
        private var cancelled = false

        init(path: String, args: [String]) {
            process.executableURL = URL(fileURLWithPath: path)
            process.arguments = args
            process.standardOutput = outPipe
            process.standardError = errPipe
        }

        func start(_ continuation: CheckedContinuation<String, Error>) {
            let out = PipeDrain(), err = PipeDrain()
            process.terminationHandler = { [self] p in
                let output = out.wait(), errors = err.wait()
                if lock.withLock({ cancelled }) {
                    continuation.resume(throwing: CancellationError())
                } else if p.terminationStatus == 0 {
                    continuation.resume(returning: output)
                } else {
                    let tail = errors.count > 2_000 ? "…" + errors.suffix(2_000) : errors
                    continuation.resume(throwing: FFmpegError.failed(tail.isEmpty ? "exit \(p.terminationStatus)" : tail))
                }
            }
            // Cancelled before launch: never start; a launch failure resumes here, not in the handler
            let launchError: Error? = lock.withLock {
                if cancelled { return CancellationError() }
                do { try process.run() } catch { return error }
                return nil
            }
            if let launchError {
                process.terminationHandler = nil
                continuation.resume(throwing: launchError)
                return
            }
            out.start(outPipe.fileHandleForReading)
            err.start(errPipe.fileHandleForReading)
        }

        func cancel() {
            lock.withLock {
                cancelled = true
                if process.isRunning { process.terminate() }
            }
        }
    }

    /// Reads a pipe to its end on its own thread while the process runs.
    private final class PipeDrain: @unchecked Sendable {
        private let done = DispatchSemaphore(value: 0)
        private var data = Data()

        func start(_ handle: FileHandle) {
            Thread.detachNewThread { [self] in
                data = handle.readDataToEndOfFile()
                done.signal()
            }
        }

        func wait() -> String {
            done.wait()
            return String(decoding: data, as: UTF8.self)
        }
    }
}
