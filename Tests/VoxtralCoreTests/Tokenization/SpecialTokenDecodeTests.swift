/**
 * SpecialTokenDecodeTests - K-13 (amended gate: 0 NUL byte in the Realtime output)
 *
 * Every id below `default_num_special_tokens` is a control token. `decode(skipSpecialTokens: true)` must skip
 * them all, like mistral-common and mlx-audio, not only BOS/EOS/PAD: the Realtime model emits [STREAMING_PAD] (32)
 * and [STREAMING_WORD] (33) between words, which mapped to rank max(0, id - 1000) = 0, the byte 0x00.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class SpecialTokenDecodeTests: XCTestCase {

    private var sandbox: URL!

    override func setUpWithError() throws {
        sandbox = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-decode-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: sandbox, withIntermediateDirectories: true)
        // Rank 0 is the byte 0x00 as in the real tekken.json; ranks 1-2 are "Hi" and " there"
        let vocab = [Data([0]), Data("Hi".utf8), Data(" there".utf8)].enumerated().map { rank, bytes in
            #"{"rank": \#(rank), "token_bytes": "\#(bytes.base64EncodedString())", "token_str": null}"#
        }
        let json = """
        {"config": {"pattern": "[^\\\\s]+|\\\\s+", "num_vocab_tokens": 3, "default_vocab_size": 1003,
                    "default_num_special_tokens": 1000, "version": "v7"},
         "vocab": [\(vocab.joined(separator: ","))]}
        """
        try Data(json.utf8).write(to: sandbox.appendingPathComponent("tekken.json"))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: sandbox)
    }

    func testControlTokensAreSkipped() throws {
        let tokenizer = try TekkenTokenizer.load(modelPath: sandbox.path)
        let bos = 1, eos = 2, streamingPad = 32, streamingWord = 33
        let decoded = tokenizer.decode([bos, streamingPad, streamingWord, 1001, streamingPad, streamingWord, 1002, eos])
        print("[decode] \(decoded.debugDescription)")
        XCTAssertFalse(decoded.contains("\u{0}"), "a control token decoded as a NUL byte")
        XCTAssertEqual(decoded, "Hi there")
    }
}
