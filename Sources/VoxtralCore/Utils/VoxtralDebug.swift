/**
 * VoxtralDebug - Centralized debug logging for VoxtralCore
 * Set VoxtralDebug.enabled = true to see debug output
 */

import Foundation

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

    /// Log a debug message (only if enabled)
    public static func log(_ message: String) {
        if enabled {
            print(message)
        }
    }

    /// Log a verbose generation message (only if verboseGeneration is enabled)
    public static func logGeneration(_ message: String) {
        if verboseGeneration {
            print(message)
        }
    }

    /// Always log (for important messages like errors)
    public static func always(_ message: String) {
        print(message)
    }
}
