/**
 * OffPoolExecution - K-15 (cooperative cancellation, no blocked cooperative threads)
 *
 * Model loading and generation are long synchronous MLX work. Run inside an `async` function, they held a thread
 * of Swift's cooperative pool for minutes ("cooperative threads must never block", mlx-swift-lm Load.swift) and a
 * cancelled Task kept generating to the end. `runOffCooperativePool` runs the work on a dedicated serial queue
 * and relays the caller's cancellation as a flag; generation loops poll `VoxtralCancellation` at every step, on
 * that queue (where `Task.isCancelled` is always false) as well as inside a Task.
 */

import Foundation

enum VoxtralCancellation {
    private static let key = "VoxtralCancellationFlag"

    /// True when the caller of the current off-pool work, or the current Task, was cancelled
    static var isCancelled: Bool {
        if Task.isCancelled { return true }
        return (Thread.current.threadDictionary[key] as? Locked<Bool>)?.get() ?? false
    }

    /// Throws `CancellationError` when cancelled (a generation step boundary)
    static func check() throws {
        if isCancelled { throw CancellationError() }
    }

    fileprivate static func bind<R>(_ flag: Locked<Bool>, _ body: () throws -> R) rethrows -> R {
        let dictionary = Thread.current.threadDictionary
        let previous = dictionary[key]
        dictionary[key] = flag
        defer { dictionary[key] = previous }
        return try body()
    }
}

/// One serial queue for all MLX work of the pipelines: MLX work is serialized anyway (PipelineGate, K-11)
private let mlxWorkQueue = DispatchQueue(label: "voxtral.mlx-work", qos: .userInitiated)

/// Carries non-Sendable work and its result across the queue hop (accessed by one thread at a time)
private final class OffPoolWork<R>: @unchecked Sendable {
    let body: () throws -> R
    var result: Result<R, Error>?
    init(_ body: @escaping () throws -> R) { self.body = body }
}

/// Runs `body` on the dedicated MLX queue, off Swift's cooperative pool; a cancellation of the calling Task sets
/// the flag `VoxtralCancellation` reads, and the work stops at its next step (K-15)
func runOffCooperativePool<R>(_ body: @escaping () throws -> R) async throws -> R {
    let flag = Locked(false)
    let work = OffPoolWork(body)
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            mlxWorkQueue.async {
                work.result = Result { try VoxtralCancellation.bind(flag) { try work.body() } }
                continuation.resume()
            }
        }
        return try work.result!.get()
    } onCancel: {
        flag.set(true)
    }
}
