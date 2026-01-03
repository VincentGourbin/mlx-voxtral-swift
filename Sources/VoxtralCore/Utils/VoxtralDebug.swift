/**
 * VoxtralDebug - Centralized debug logging for VoxtralCore
 * Set VoxtralDebug.enabled = true to see debug output
 */

import Foundation

public enum VoxtralDebug {
    /// Enable/disable all debug output
    public static var enabled: Bool = false

    /// Enable/disable verbose generation logs (token-by-token)
    public static var verboseGeneration: Bool = false

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
