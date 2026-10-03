/**
 * WER - normalized word error rate for `voxtral eval` (K-33). Foundation only, so `Scripts/check-wer.sh` compiles it
 * on its own.
 *
 * Normalization (same as the frozen mlx-audio reference, docs/eval/realtime-reference/README.md): NUL bytes removed,
 * Unicode NFKD with combining marks dropped (accents ignored), lowercase, every character that is not a letter or a
 * digit becomes a space, runs of spaces collapse. WER = (substitutions + deletions + insertions) / reference words,
 * from a word-level Levenshtein alignment.
 */

import Foundation

enum WER {
    struct Score: Equatable {
        let referenceWords: Int
        let hypothesisWords: Int
        let substitutions: Int
        let deletions: Int
        let insertions: Int

        var errors: Int { substitutions + deletions + insertions }
        /// Fraction (0.05 = 5 %); 0 for an empty reference and an empty hypothesis
        var wer: Double {
            referenceWords == 0 ? (hypothesisWords == 0 ? 0 : 1) : Double(errors) / Double(referenceWords)
        }
    }

    static func normalize(_ text: String) -> String {
        let decomposed = text.replacingOccurrences(of: "\u{0}", with: "").decomposedStringWithCompatibilityMapping
        var out = String.UnicodeScalarView()
        for scalar in decomposed.unicodeScalars {
            if scalar.properties.generalCategory == .nonspacingMark { continue }
            if CharacterSet.alphanumerics.contains(scalar) {
                out.append(contentsOf: String(scalar).lowercased().unicodeScalars)
            } else {
                out.append(" ")
            }
        }
        return String(out).split(separator: " ").joined(separator: " ")
    }

    static func words(_ text: String) -> [Substring] {
        normalize(text).split(separator: " ")
    }

    static func score(reference: String, hypothesis: String) -> Score {
        let ref = words(reference), hyp = words(hypothesis)
        // dp[i][j] = (cost, substitutions, deletions, insertions) aligning ref[..<i] with hyp[..<j]
        typealias Cell = (cost: Int, sub: Int, del: Int, ins: Int)
        var previous = [Cell](repeating: (0, 0, 0, 0), count: hyp.count + 1)
        for j in 0 ... hyp.count { previous[j] = (j, 0, 0, j) }
        for i in 1 ... max(ref.count, 1) where !ref.isEmpty {
            var current = [Cell](repeating: (0, 0, 0, 0), count: hyp.count + 1)
            current[0] = (i, 0, i, 0)
            for j in stride(from: 1, through: hyp.count, by: 1) {
                if ref[i - 1] == hyp[j - 1] {
                    current[j] = previous[j - 1]
                    continue
                }
                let substitution = previous[j - 1], deletion = previous[j], insertion = current[j - 1]
                let best = min(substitution.cost, deletion.cost, insertion.cost)
                if best == substitution.cost {
                    current[j] = (best + 1, substitution.sub + 1, substitution.del, substitution.ins)
                } else if best == deletion.cost {
                    current[j] = (best + 1, deletion.sub, deletion.del + 1, deletion.ins)
                } else {
                    current[j] = (best + 1, insertion.sub, insertion.del, insertion.ins + 1)
                }
            }
            previous = current
        }
        let last = previous[hyp.count]
        return Score(referenceWords: ref.count, hypothesisWords: hyp.count,
                     substitutions: last.sub, deletions: last.del, insertions: last.ins)
    }

    /// The normalized `sentence` occurs in the normalized `text` (casing, punctuation and accents ignored)
    static func contains(_ text: String, sentence: String) -> Bool {
        let needle = normalize(sentence)
        return !needle.isEmpty && (" " + normalize(text) + " ").contains(" " + needle + " ")
    }

    /// Last sentence of a reference text (split on . ! ?)
    static func lastSentence(of text: String) -> String {
        let sentences = text.split(whereSeparator: { ".!?".contains($0) })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !normalize($0).isEmpty }
        return sentences.last ?? ""
    }
}
