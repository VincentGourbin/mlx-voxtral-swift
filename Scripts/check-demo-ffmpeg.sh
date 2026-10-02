#!/bin/bash
# Checks the demo's process runner (Sources/VoxtralTTSStreamingDemo/FFmpeg.swift; the demo has no test target, K-28):
#   1. a process writing 1 MB on stderr (and 1 MB on stdout) terminates instead of blocking on a full pipe;
#   2. cancelling the task kills the process in under 1 s;
#   3. voice names: `../x` and `a/b` are refused before any enrollment.
# Exit status 0 = the three checks pass.

set -euo pipefail
cd "$(dirname "$0")/.."
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/main.swift" <<'SWIFT'
import Foundation

var failures = 0
@MainActor func check(_ ok: Bool, _ line: String) { print((ok ? "OK  " : "KO  ") + line); if !ok { failures += 1 } }

// Watchdog: a blocked runner never returns
Thread.detachNewThread { Thread.sleep(forTimeInterval: 30); print("KO  blocked > 30 s"); exit(1) }

// 1. 1 MB on stderr, 1 MB on stdout, exit 3
var start = Date()
do {
    _ = try await FFmpeg.run("/bin/sh", ["-c",
        "head -c 1048576 /dev/zero | tr '\\0' x >&2; head -c 1048576 /dev/zero | tr '\\0' y; exit 3"])
    check(false, "STDERR 1 Mo : no error thrown")
} catch {
    check(true, String(format: "STDERR 1 Mo → processus terminé (%.2f s, erreur : %@)",
                       Date().timeIntervalSince(start), String(describing: type(of: error))))
}

// 2. Cancellation
let marker = "sleep 61.\(Int.random(in: 100...999))"
let task = Task { try await FFmpeg.run("/bin/sh", ["-c", "exec /bin/\(marker)"]) }
try await Task.sleep(nanoseconds: 500_000_000)
start = Date()
task.cancel()
let result = await task.result
let elapsed = Date().timeIntervalSince(start)
let alive = Process(); alive.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
alive.arguments = ["-f", marker]; alive.standardOutput = FileHandle.nullDevice
try alive.run(); alive.waitUntilExit()
var cancelled = false
if case .failure(let error) = result, error is CancellationError { cancelled = true }
check(cancelled && elapsed < 1 && alive.terminationStatus != 0,
      String(format: "annulation → processus tué en %.3f s (CancellationError: %@, processus restant: %@)",
             elapsed, cancelled ? "oui" : "non", alive.terminationStatus == 0 ? "oui" : "non"))

// 3. Voice names
for (name, valid) in [("../x", false), ("a/b", false), (".hidden", false), ("my_voice", true), ("Voix 2", true)] {
    let accepted = (try? VoiceName.validate(name)) != nil
    check(accepted == valid, "nom \"\(name)\" \(accepted ? "accepté" : "refusé")")
}
exit(failures == 0 ? 0 : 1)
SWIFT

xcrun swiftc -swift-version 6 -O -o "$WORK/check" \
    Sources/VoxtralTTSStreamingDemo/FFmpeg.swift Sources/VoxtralTTSStreamingDemo/VoiceName.swift "$WORK/main.swift"
"$WORK/check"
