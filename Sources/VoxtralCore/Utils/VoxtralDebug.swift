/**
 * VoxtralDebug - Centralized logging for VoxtralCore
 *
 * The library writes nothing to stdout unless `enabled` (CLI `--debug`) or for `console` output a caller asked for
 * (model listings). Messages also go to the unified log (`os.Logger`, subsystem `com.vincentgourbin.voxtral`),
 * readable with Console or `log stream` (K-23).
 */

import Foundation
import os

public enum VoxtralDebug {
    /// Enable/disable all debug output
    public static var enabled: Bool {
        get { _enabled.get() }
        set { _enabled.set(newValue) }
    }
    private static let _enabled = Locked(false)

    /// Enable/disable verbose generation logs (token-by-token)
    public static var verboseGeneration: Bool {
        get { _verboseGeneration.get() }
        set { _verboseGeneration.set(newValue) }
    }
    private static let _verboseGeneration = Locked(false)

    private static let logger = Logger(subsystem: "com.vincentgourbin.voxtral", category: "VoxtralCore")

    /// Debug message: stdout and the unified log, only when enabled
    public static func log(_ message: String) {
        guard enabled else { return }
        logger.debug("\(message, privacy: .public)")
        print(message)
    }

    /// Token-by-token generation message (only if verboseGeneration is enabled)
    public static func logGeneration(_ message: String) {
        if verboseGeneration {
            print(message)
        }
    }

    /// Important message (fallback, error): always in the unified log, on stdout only when enabled
    public static func always(_ message: String) {
        logger.notice("\(message, privacy: .public)")
        if enabled { print(message) }
    }

    /// Output the caller asked for (e.g. `printAvailableModels`): always on stdout
    public static func console(_ message: String = "") {
        print(message)
    }
}
