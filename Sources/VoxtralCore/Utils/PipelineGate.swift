/**
 * PipelineGate - atomic pipeline state and one GPU operation at a time (K-11, S-10, A-01)
 *
 * A pipeline's `guard state … ; state = …` pairs were not atomic, and an enrollment (gradient)
 * could run while a synthesis compiled kernels: mlx-swift 0.31.6 takes its compile and vjp locks
 * in opposite orders (ABBA deadlock, fixed upstream by `df9ae26`, untagged). The gate checks and
 * sets under one lock, refuses a second operation with a `busy` error, and hands out a generation
 * token so a Task that outlives `unload()` or a reload cannot write a stale state.
 */

import os

final class PipelineGate<State: Sendable>: Sendable {
    private struct Slot: Sendable {
        var state: State
        var operation: String?
        var generation: UInt64 = 0
    }

    private enum Claim: Sendable {
        case granted(UInt64)
        case busy(String)
        case refused
    }

    private let lock: OSAllocatedUnfairLock<Slot>

    init(_ state: State) {
        lock = OSAllocatedUnfairLock(initialState: Slot(state: state))
    }

    var state: State { lock.withLock { $0.state } }

    /// The operation holding the pipeline, if any (tests and diagnostics).
    var operation: String? { lock.withLock { $0.operation } }

    /// Claims the pipeline for `operation`. Throws `busy(...)` while another operation runs and
    /// `refusal` when `accepts(state)` is false; otherwise sets `state` (when given) and returns the
    /// generation to pass to `end`. `newGeneration` (loads) invalidates every earlier token.
    func begin(
        _ operation: String,
        accepts: @Sendable (State) -> Bool,
        refusal: @autoclosure () -> Error,
        busy: (String) -> Error,
        state newState: State? = nil,
        newGeneration: Bool = false
    ) throws -> UInt64 {
        let claim = lock.withLock { slot -> Claim in
            if let running = slot.operation { return .busy(running) }
            guard accepts(slot.state) else { return .refused }
            slot.operation = operation
            if let newState { slot.state = newState }
            if newGeneration { slot.generation &+= 1 }
            return .granted(slot.generation)
        }
        switch claim {
        case .granted(let generation): return generation
        case .busy(let running): throw busy("\(operation) refused: \(running) in progress")
        case .refused: throw refusal()
        }
    }

    /// Releases the operation begun with `generation` and sets `state` (when given). Ignored when
    /// the pipeline was unloaded or reloaded since: a stale Task never overwrites the current state.
    func end(_ generation: UInt64, state newState: State? = nil) {
        lock.withLock { slot in
            guard slot.generation == generation else { return }
            slot.operation = nil
            if let newState { slot.state = newState }
        }
    }

    /// Unload: new generation, no operation, `state`.
    func reset(_ newState: State) {
        lock.withLock { slot in
            slot.generation &+= 1
            slot.operation = nil
            slot.state = newState
        }
    }
}
