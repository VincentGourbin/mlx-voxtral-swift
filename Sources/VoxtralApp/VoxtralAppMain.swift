/**
 * VoxtralApp - SwiftUI macOS application for Voxtral audio transcription
 */

import SwiftUI
import AppKit
import MLX

@main
struct VoxtralAppMain: App {
    init() {
        // Suppress MLX Metal shader compilation warnings
        // These are internal to the MLX framework and not actionable
        setenv("MTL_SHADER_VALIDATION", "0", 1)
    }
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 900, height: 600)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Voxtral") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            .applicationName: "Voxtral",
                            .applicationVersion: "1.0",
                            .credits: NSAttributedString(string: "Audio transcription powered by MLX Swift")
                        ]
                    )
                }
            }
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Make this a regular app that appears in the Dock
        NSApplication.shared.setActivationPolicy(.regular)

        // Bring to front
        NSApplication.shared.activate(ignoringOtherApps: true)

        // Make the first window key and front
        if let window = NSApplication.shared.windows.first {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
