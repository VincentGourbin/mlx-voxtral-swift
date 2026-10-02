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

/// A configuration error on a non-throwing path (unsupported module type, missing input): recorded in the enclosing
/// MLX error boundary, which throws it as `VoxtralError.invalidConfiguration` at the entry point's next check, with an
/// empty result. Outside any boundary (a direct call to the model) the process stops as before (K-27). The throwing
/// entry points validate the module types up front, so they never reach this.
func unsupportedConfiguration(_ message: String) -> MLXArray {
    guard let box = MLXErrorScope.box else { fatalError(message) }
    box.firstError = VoxtralError.invalidConfiguration(message)
    return MLXArray.zeros([0])
}
