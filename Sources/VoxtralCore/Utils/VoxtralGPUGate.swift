/**
 * VoxtralGPUGate - stops submitting GPU work while the host app is in the background (iOS)
 *
 * iOS refuses every GPU submission from a backgrounded app, at once
 * (`kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`), and mlx-core raises the error inside a Metal
 * completion handler, where nothing can catch it: the process terminates. MLX commits command buffers inside an `eval`,
 * so the only place to stop is between two evals. The pipelines wait here before each unit of GPU work (a token, a
 * prefill slice, an encoder layer, a TTS frame, a codec layer). Same pattern as YuE2GPUGate.
 */

import Foundation

/// Closed by the host while it cannot use the GPU; open by default. Close it as soon as the scene leaves `.active`, and
/// reopen it when it comes back: an operation in progress parks between two units of work and resumes where it stopped.
///
/// ```swift
/// .onChange(of: scenePhase) { _, phase in
///     if phase == .active { VoxtralGPUGate.shared.resume() } else { VoxtralGPUGate.shared.suspend() }
/// }
/// ```
public final class VoxtralGPUGate: @unchecked Sendable {
    public static let shared = VoxtralGPUGate()

    private let condition = NSCondition()
    private var suspended = false

    init() {}

    /// Closes the gate: GPU work stops at the next boundary
    public func suspend() {
        condition.lock()
        suspended = true
        condition.unlock()
    }

    /// Opens the gate and wakes the operations waiting on it
    public func resume() {
        condition.lock()
        suspended = false
        condition.broadcast()
        condition.unlock()
    }

    public var isSuspended: Bool {
        condition.lock()
        defer { condition.unlock() }
        return suspended
    }

    /// Blocks while the gate is closed. A cancelled caller is released, polled every 100 ms: its operation then stops
    /// without touching the GPU.
    func wait() {
        condition.lock()
        defer { condition.unlock() }
        while suspended, !VoxtralCancellation.isCancelled {
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.1))
        }
    }
}

extension VoxtralCancellation {
    /// A point between two units of GPU work: waits while `VoxtralGPUGate` is closed, then throws `CancellationError` if
    /// the caller cancelled
    static func gpuCheckpoint() throws {
        VoxtralGPUGate.shared.wait()
        try check()
    }

    /// Same as `gpuCheckpoint()`, for the loops that stop by themselves: true when the caller cancelled
    static var cancelledAtGPUCheckpoint: Bool {
        VoxtralGPUGate.shared.wait()
        return isCancelled
    }
}
