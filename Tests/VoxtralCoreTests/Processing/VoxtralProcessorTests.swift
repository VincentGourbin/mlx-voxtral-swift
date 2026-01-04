/**
 * VoxtralProcessorTests - Unit tests for VoxtralProcessor
 */

import XCTest
import MLX
@testable import VoxtralCore

final class VoxtralProcessorTests: XCTestCase {

    // MARK: - Initialization Tests

    func testProcessorInitWithDefaultTokenizer() {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        XCTAssertNotNil(processor, "Processor should initialize with default tokenizer")
    }

    func testProcessorInitWithFeatureExtractor() {
        let tokenizer = TekkenTokenizer()
        let featureExtractor = VoxtralFeatureExtractor()
        let processor = VoxtralProcessor(
            featureExtractor: featureExtractor,
            tokenizer: tokenizer
        )

        XCTAssertNotNil(processor, "Processor should initialize with feature extractor")
    }

    // MARK: - Decode Tests

    func testDecodeEmptyTokens() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let result = try processor.decode([], skipSpecialTokens: true)
        XCTAssertEqual(result, "", "Empty tokens should decode to empty string")
    }

    func testDecodeWithSkipSpecialTokens() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        // Encode some text
        let text = "Hello world"
        let tokens = tokenizer.encode(text, addSpecialTokens: false)

        let decoded = try processor.decode(tokens, skipSpecialTokens: true)
        XCTAssertEqual(decoded, text, "Decode should preserve original text")
    }

    func testDecodeWithMLXArray() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        // Create MLXArray of token IDs
        let tokens = tokenizer.encode("Test")
        let mlxTokens = MLXArray(tokens.map { Int32($0) })

        let decoded = try processor.decode(mlxTokens, skipSpecialTokens: true)
        XCTAssertEqual(decoded, "Test", "Should decode MLXArray tokens")
    }

    // MARK: - Batch Decode Tests

    func testBatchDecodeEmpty() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let result = try processor.batchDecode([], skipSpecialTokens: true)
        XCTAssertTrue(result.isEmpty, "Empty batch should return empty array")
    }

    func testBatchDecodeMultipleSequences() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let texts = ["Hello", "World"]
        let tokenLists = texts.map { tokenizer.encode($0, addSpecialTokens: false) }

        let decoded = try processor.batchDecode(tokenLists, skipSpecialTokens: true)
        XCTAssertEqual(decoded.count, 2, "Should decode same number of sequences")
        XCTAssertEqual(decoded[0], "Hello", "First sequence should match")
        XCTAssertEqual(decoded[1], "World", "Second sequence should match")
    }

    // MARK: - CallAsFunction Tests

    func testCallAsFunctionWithTextOnly() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let result = try processor(text: "Hello world", audio: nil)

        XCTAssertNotNil(result["input_ids"], "Should return input_ids for text")
    }

    // MARK: - Edge Cases

    func testDecodeInvalidTokenFormat() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        // Try to decode an unsupported type (String instead of tokens)
        XCTAssertThrowsError(try processor.decode("not tokens", skipSpecialTokens: true)) { error in
            guard case VoxtralError.invalidTokenFormat = error else {
                XCTFail("Expected invalidTokenFormat error")
                return
            }
        }
    }

    func testDecodeVeryLongSequence() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        // Create a long sequence of token IDs
        let longTokens = Array(repeating: 100, count: 1000)
        let decoded = try processor.decode(longTokens, skipSpecialTokens: true)

        XCTAssertNotNil(decoded, "Should handle long sequences")
    }

    // MARK: - Tokenizer Integration Tests

    func testProcessorUsesTokenizer() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let text = "Hello world"
        let tokensFromTokenizer = tokenizer.encode(text)
        let decoded = try processor.decode(tokensFromTokenizer, skipSpecialTokens: true)

        XCTAssertEqual(decoded, text, "Processor should correctly use tokenizer")
    }

    func testProcessorWithUnicodeText() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let text = "Hello 世界"
        let tokens = tokenizer.encode(text)
        let decoded = try processor.decode(tokens, skipSpecialTokens: true)

        XCTAssertEqual(decoded, text, "Should handle unicode text")
    }

    func testProcessorWithEmoji() throws {
        let tokenizer = TekkenTokenizer()
        let processor = VoxtralProcessor(tokenizer: tokenizer)

        let text = "Hello 🎵"
        let tokens = tokenizer.encode(text)
        let decoded = try processor.decode(tokens, skipSpecialTokens: true)

        XCTAssertEqual(decoded, text, "Should handle emoji")
    }
}
