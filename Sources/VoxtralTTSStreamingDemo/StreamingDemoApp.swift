import SwiftUI
import AppKit

@main
struct StreamingDemoApp: App {

    init() {
        // Required for SPM executables to show as a foreground GUI app
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup {
            StreamingDemoView()
                .frame(minWidth: 600, minHeight: 700)
        }
        .windowResizability(.contentMinSize)
    }
}
