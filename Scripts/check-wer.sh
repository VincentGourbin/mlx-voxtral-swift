#!/bin/bash
# Checks the WER of `voxtral eval` (Sources/VoxtralTranscriptionTest/WER.swift) on hand-computed cases (K-33).
# Exit status 0 = every case matches.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/main.swift" <<'SWIFT'
import Foundation
var failures = 0
@MainActor func expect(_ ref: String, _ hyp: String, wer: Double, s: Int, d: Int, i: Int) {
    let r = WER.score(reference: ref, hypothesis: hyp)
    let ok = abs(r.wer - wer) < 1e-9 && r.substitutions == s && r.deletions == d && r.insertions == i
    print((ok ? "OK  " : "KO  ") + "\"\(ref)\" / \"\(hyp)\" → WER \(r.wer) (S\(r.substitutions) D\(r.deletions) I\(r.insertions))")
    if !ok { failures += 1 }
}
expect("the cat sat", "the cat sat", wer: 0, s: 0, d: 0, i: 0)
expect("The cat, sat!", "the CAT sat", wer: 0, s: 0, d: 0, i: 0)                 // case and punctuation
expect("Générez des vidéos", "generez des videos", wer: 0, s: 0, d: 0, i: 0)    // accents
expect("the cat sat", "the bat sat", wer: 1.0 / 3, s: 1, d: 0, i: 0)
expect("the cat sat on the mat", "the cat on mat", wer: 2.0 / 6, s: 0, d: 2, i: 0)
expect("a b", "a x b y", wer: 1.0, s: 0, d: 0, i: 2)
expect("", "", wer: 0, s: 0, d: 0, i: 0)
expect("n'importe quel projet", "n importe quel projet", wer: 0, s: 0, d: 0, i: 0)
let okContains = WER.contains("Bla. Aucune donnée envoyée dans le cloud.", sentence: "Aucune donnee envoyee dans le cloud.")
    && !WER.contains("cloudy", sentence: "cloud")
print((okContains ? "OK  " : "KO  ") + "contains (accents, word boundaries)"); if !okContains { failures += 1 }
let last = WER.lastSentence(of: "One. Two three!\nAucune donnée envoyée dans le cloud.\n")
print((last == "Aucune donnée envoyée dans le cloud" ? "OK  " : "KO  ") + "lastSentence → \(last)")
if last != "Aucune donnée envoyée dans le cloud" { failures += 1 }
let cov = WER.coverage("bla No account is required, no data is sent to the cloud", sentence: "No account is required, and no data is sent to the cloud.")
print((abs(cov - 11.0 / 12) < 1e-9 ? "OK  " : "KO  ") + "coverage one word missing → \(cov)"); if abs(cov - 11.0 / 12) >= 1e-9 { failures += 1 }
let full = WER.coverage("x Aucune donnée envoyée dans le cloud.", sentence: "Aucune donnee envoyee dans le cloud")
print((full == 1 ? "OK  " : "KO  ") + "coverage full sentence → \(full)"); if full != 1 { failures += 1 }
exit(failures == 0 ? 0 : 1)
SWIFT
xcrun swiftc -swift-version 6 -O -o "$WORK/check" Sources/VoxtralTranscriptionTest/WER.swift "$WORK/main.swift"
"$WORK/check"
