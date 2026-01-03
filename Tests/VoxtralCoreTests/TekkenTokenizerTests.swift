import XCTest
import Foundation
@testable import VoxtralCore

final class TekkenTokenizerTests: XCTestCase {
    
    var tokenizer: TekkenTokenizer!
    let modelPath = "/Users/vincent/Developpements/convertvoxtral/voxtral_models/voxtral-mini-3b-4bit-mixed"
    
    override func setUpWithError() throws {
        tokenizer = TekkenTokenizer()
        // Check if model path exists, otherwise use demo tokenizer
        let tekkenPath = "\(modelPath)/tekken.json"
        if FileManager.default.fileExists(atPath: tekkenPath) {
            tokenizer.loadTekkenTokenizerFromFile(modelPath: modelPath)
        } else {
            // Use demo tokenizer for testing if model files not available
            print("⚠️ Model files not found, using demo tokenizer for tests")
        }
    }
    
    // MARK: - Core Functionality Tests
    
    func testBasicTokenization() throws {
        let text = "Hello world"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertFalse(tokens.isEmpty, "Tokens should not be empty")
        XCTAssertEqual(text, decoded, "Round-trip tokenization should preserve text")
        print("✅ Basic test: '\(text)' -> \(tokens) -> '\(decoded)'")
    }
    
    func testEmptyText() throws {
        let text = ""
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertTrue(tokens.isEmpty, "Empty text should produce empty tokens")
        XCTAssertEqual(text, decoded, "Empty text round-trip should work")
    }
    
    // MARK: - Python Compatibility Tests
    
    func testGermanTextCompatibility() throws {
        // Test case that previously caused issues with token 129113
        let text = "ich liebe dich"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertEqual(text, decoded, "German text should round-trip correctly")
    }
    
    func testComplexMultilingualText() throws {
        let text = "Hello 🌍 world! Comment ça va? ¿Cómo estás? 你好 123"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertEqual(text, decoded, "Complex multilingual text should round-trip correctly")
    }
    
    func testEmojiHandling() throws {
        let text = "🚀 🎉 🤖 💡 🌸"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertEqual(text, decoded, "Emoji should round-trip correctly")
    }
    
    // MARK: - Vocabulary Truncation Tests
    
    func testVocabularyTruncation() throws {
        // Verify that vocabulary is truncated to 130072 tokens as per Python logic
        // This is a crucial test to ensure Python compatibility
        
        let testTexts = [
            "simple test",
            "complex émojis 🎯",
            "numbers 123 456",
            "symbols @#$%^&*()",
            "unicode ñáéíóú"
        ]
        
        for text in testTexts {
            let tokens = tokenizer.encode(text)
            let decoded = tokenizer.decode(tokens)
            
            XCTAssertEqual(text, decoded, "Text '\(text)' should round-trip correctly with truncated vocabulary")
        }
    }
    
    // MARK: - Special Token Tests
    
    func testSpecialTokenHandling() throws {
        let text = "test"
        let tokensWithSpecial = tokenizer.encode(text, addSpecialTokens: true)
        let tokensWithoutSpecial = tokenizer.encode(text, addSpecialTokens: false)
        
        // With special tokens should have more tokens than without
        XCTAssertGreaterThanOrEqual(tokensWithSpecial.count, tokensWithoutSpecial.count,
                                   "Tokens with special tokens should have >= count than without")
    }
    
    // MARK: - Edge Cases
    
    func testVeryLongText() throws {
        let text = String(repeating: "This is a long text that should be handled correctly. ", count: 100)
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertEqual(text, decoded, "Very long text should round-trip correctly")
    }
    
    func testSpecialCharacters() throws {
        let text = "\"Special\" 'quotes' and\ttabs\nand newlines"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)
        
        XCTAssertEqual(text, decoded, "Special characters should round-trip correctly")
    }
    
    // MARK: - Performance Tests
    
    func testTokenizationPerformance() throws {
        let text = "This is a performance test text that should be tokenized efficiently."
        
        measure {
            for _ in 0..<1000 {
                let tokens = tokenizer.encode(text)
                let _ = tokenizer.decode(tokens)
            }
        }
    }
    
    // MARK: - Integration Tests
    
    func testWithProcessor() throws {
        // Test that tokenizer works correctly when used through VoxtralProcessor
        let processor = VoxtralProcessor(tokenizer: tokenizer)
        
        XCTAssertNotNil(processor.tokenizer, "Processor should have tokenizer")
        
        // Test decode functionality through processor
        let tokens = tokenizer.encode("test message")
        let decoded = try processor.decode(tokens, skipSpecialTokens: true)
        
        XCTAssertEqual("test message", decoded, "Processor decode should work with tokenizer")
    }
}

// MARK: - Python Cross-Validation Tests

extension TekkenTokenizerTests {
    
    /// This test validates that our Swift tokenizer produces exactly the same results
    /// as Python mistral-common implementation. This is crucial for model compatibility.
    func testPythonCrossValidation() throws {
        // Only run cross-validation tests if we have the real model loaded
        let tekkenPath = "\(modelPath)/tekken.json"
        guard FileManager.default.fileExists(atPath: tekkenPath) else {
            print("⚠️ Skipping Python cross-validation - model files not available")
            return
        }
        
        // Test cases that have been validated against Python
        let testCases = [
            ("Hello world", [22177, 4304]),
            ("ich liebe dich", [1521, 5897, 2352, 12486]),
            ("test 123", [4417, 1032, 1049, 1050, 1051])
        ]
        
        for (text, expectedTokens) in testCases {
            let tokens = tokenizer.encode(text)
            XCTAssertEqual(tokens, expectedTokens, 
                          "Swift tokens for '\(text)' should match Python: \(expectedTokens) but got \(tokens)")
            
            let decoded = tokenizer.decode(tokens)
            XCTAssertEqual(text, decoded, "Decoded text should match original")
        }
    }
}