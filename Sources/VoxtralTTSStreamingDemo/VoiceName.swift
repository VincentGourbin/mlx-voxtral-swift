import Foundation

/// File name of an enrolled voice, checked before the enrollment starts: a name that leaves the voices folder
/// (`../x`) or names a missing sub-folder (`a/b`) used to fail only when saving, after the whole enrollment (A-19, K-28).
enum VoiceName {

    enum Invalid: LocalizedError, Equatable {
        case empty
        case tooLong
        case characters(String)

        var errorDescription: String? {
            switch self {
            case .empty: return "Enter a name for the cloned voice"
            case .tooLong: return "Voice name too long (64 characters at most)"
            case let .characters(name):
                return "Invalid voice name \u{201C}\(name)\u{201D}: use letters, digits, space, _ - . (not starting with a dot)"
            }
        }
    }

    static let maxLength = 64

    /// The trimmed name, or `Invalid`
    static func validate(_ raw: String) throws -> String {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw Invalid.empty }
        guard name.count <= maxLength else { throw Invalid.tooLong }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " _-."))
        guard name.unicodeScalars.allSatisfy(allowed.contains), !name.hasPrefix("."), !name.contains("..") else {
            throw Invalid.characters(name)
        }
        return name
    }
}
