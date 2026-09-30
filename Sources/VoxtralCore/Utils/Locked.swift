/**
 * Locked - a value whose every read and write goes through a lock
 *
 * Replaces the `nonisolated(unsafe)` globals of VoxtralCore (S-11): the unsafety is
 * confined here, where each access to `value` is serialized by `lock`.
 */

import Foundation

/// `@unchecked Sendable`: `value` is only reached through `withLock`, `get` and `set`,
/// all of which hold `lock`. A non-Sendable payload (MLXArray, closure) must be safe to
/// use from another thread once handed out (MLXArray: evaluated first, MLX-004).
final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value

    init(_ value: Value) { self.value = value }

    func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R {
        lock.lock()
        defer { lock.unlock() }
        return try body(&value)
    }

    func get() -> Value { withLock { $0 } }

    func set(_ newValue: Value) { withLock { $0 = newValue } }
}
