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

    private static func run(_ path: String, _ args: [String]) async throws -> String {
        try await withCheckedThrowingContinuation { cont in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: path)
            proc.arguments = args
            let outPipe = Pipe(), errPipe = Pipe()
            proc.standardOutput = outPipe
            proc.standardError = errPipe
            proc.terminationHandler = { p in
                let out = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                let err = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                if p.terminationStatus == 0 {
                    cont.resume(returning: out)
                } else {
                    cont.resume(throwing: FFmpegError.failed(err.isEmpty ? "exit \(p.terminationStatus)" : err))
                }
            }
            do { try proc.run() } catch { cont.resume(throwing: error) }
        }
    }
}
