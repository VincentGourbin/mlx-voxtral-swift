/**
 * MLXErrorBoundary - MLX errors become `VoxtralError.mlx` at the public entry points (K-1, MLX-021)
 *
 * Without a handler, mlx-swift's `ErrorHandler.dispatch` calls `fatalError`, which terminates the
 * host app (FluxForge). `withError` installs a task-local handler; a caught error does not stop the
 * block, so loops call `try error.check()` after each `eval` before reading shapes or items.
 */

import MLX

/// The error box of the innermost `withMLXErrors`, for non-throwing inner loops (decoder layers)
/// that must stop before reading the shape of an empty array left by a caught error.
enum MLXErrorScope {
    @TaskLocal static var box: ErrorBox?

    /// True once an MLX error has been caught in the enclosing boundary
    static var hasError: Bool { box?.firstError != nil }
}

/// Runs `body` with an MLX error handler and rethrows a caught MLX error as `VoxtralError.mlx`.
func withMLXErrors<R>(_ body: (ErrorBox) throws -> R) throws -> R {
    do {
        return try withError { box in try MLXErrorScope.$box.withValue(box) { try body(box) } }
    } catch let MLXError.caught(message) {
        throw VoxtralError.mlx(message)
    }
}

/// Async variant: the handler is task-local, so work spawned in another `Task` needs its own boundary.
func withMLXErrors<R>(_ body: (ErrorBox) async throws -> R) async throws -> R {
    do {
        return try await withError { box in try await MLXErrorScope.$box.withValue(box) { try await body(box) } }
    } catch let MLXError.caught(message) {
        throw VoxtralError.mlx(message)
    }
}
