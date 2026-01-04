/**
 * MLXTestHelpers - Helper functions for testing with MLX arrays
 */

import Foundation
import MLX
import XCTest

// MARK: - MLXArray Test Extensions

extension MLXArray {
    /// Create a test array filled with a constant value
    /// - Parameters:
    ///   - shape: The shape of the array
    ///   - fill: The value to fill with (default 0.0)
    /// - Returns: A new MLXArray
    static func testArray(shape: [Int], fill: Float = 0.0) -> MLXArray {
        if fill == 0.0 {
            return MLXArray.zeros(shape)
        } else if fill == 1.0 {
            return MLXArray.ones(shape)
        } else {
            return MLXArray.ones(shape) * fill
        }
    }

    /// Create a test array with random values
    /// - Parameter shape: The shape of the array
    /// - Returns: A new MLXArray with random values
    static func randomTestArray(shape: [Int]) -> MLXArray {
        return MLXRandom.uniform(low: -1.0, high: 1.0, shape)
    }

    /// Create a test array from a Swift array
    /// - Parameter values: The values
    /// - Returns: A new MLXArray
    static func fromValues(_ values: [Float]) -> MLXArray {
        return MLXArray(values)
    }

    /// Create a 2D test array from nested arrays
    /// - Parameter values: The 2D values
    /// - Returns: A new MLXArray
    static func fromValues2D(_ values: [[Float]]) -> MLXArray {
        let flat = values.flatMap { $0 }
        let shape = [values.count, values[0].count]
        return MLXArray(flat).reshaped(shape)
    }
}

// MARK: - Custom Test Assertions

/// Assert that two MLXArrays have the same shape
/// - Parameters:
///   - array: The array to check
///   - expected: The expected shape
///   - file: Source file for error reporting
///   - line: Line number for error reporting
func assertShapeEqual(
    _ array: MLXArray,
    expected: [Int],
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(
        array.shape,
        expected,
        "Shape mismatch: got \(array.shape), expected \(expected)",
        file: file,
        line: line
    )
}

/// Assert that two MLXArrays have values close to each other
/// - Parameters:
///   - a: First array
///   - b: Second array
///   - tolerance: Maximum allowed difference (default 1e-5)
///   - file: Source file for error reporting
///   - line: Line number for error reporting
func assertValuesClose(
    _ a: MLXArray,
    _ b: MLXArray,
    tolerance: Float = 1e-5,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(a.shape, b.shape, "Arrays must have same shape", file: file, line: line)

    let diff = abs(a - b)
    let maxDiff = diff.max().item(Float.self)

    XCTAssertLessThanOrEqual(
        maxDiff,
        tolerance,
        "Values differ by \(maxDiff), tolerance is \(tolerance)",
        file: file,
        line: line
    )
}

/// Assert that an MLXArray contains expected values
/// - Parameters:
///   - array: The array to check
///   - expected: Expected values as a flat array
///   - tolerance: Maximum allowed difference (default 1e-5)
///   - file: Source file for error reporting
///   - line: Line number for error reporting
func assertArrayValues(
    _ array: MLXArray,
    expected: [Float],
    tolerance: Float = 1e-5,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let flat = array.reshaped([-1])
    XCTAssertEqual(flat.shape[0], expected.count, "Array size mismatch", file: file, line: line)

    for i in 0..<expected.count {
        let actual = flat[i].item(Float.self)
        XCTAssertEqual(
            actual,
            expected[i],
            accuracy: tolerance,
            "Value at index \(i): got \(actual), expected \(expected[i])",
            file: file,
            line: line
        )
    }
}

/// Assert that an MLXArray has values within a range
/// - Parameters:
///   - array: The array to check
///   - min: Minimum expected value
///   - max: Maximum expected value
///   - file: Source file for error reporting
///   - line: Line number for error reporting
func assertValuesInRange(
    _ array: MLXArray,
    min: Float,
    max: Float,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let actualMin = array.min().item(Float.self)
    let actualMax = array.max().item(Float.self)

    XCTAssertGreaterThanOrEqual(
        actualMin,
        min,
        "Min value \(actualMin) is less than expected \(min)",
        file: file,
        line: line
    )
    XCTAssertLessThanOrEqual(
        actualMax,
        max,
        "Max value \(actualMax) is greater than expected \(max)",
        file: file,
        line: line
    )
}

/// Assert that an MLXArray has the expected dtype
/// - Parameters:
///   - array: The array to check
///   - expected: The expected dtype
///   - file: Source file for error reporting
///   - line: Line number for error reporting
func assertDType(
    _ array: MLXArray,
    expected: DType,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(
        array.dtype,
        expected,
        "DType mismatch: got \(array.dtype), expected \(expected)",
        file: file,
        line: line
    )
}

// MARK: - Test Data Generators

/// Generate a test mel spectrogram-like array
/// - Parameters:
///   - numFrames: Number of time frames
///   - numMels: Number of mel bins (default 128)
/// - Returns: A 2D MLXArray [numFrames, numMels]
func generateTestMelSpectrogram(numFrames: Int, numMels: Int = 128) -> MLXArray {
    return MLXRandom.uniform(low: -10.0, high: 0.0, [numFrames, numMels])
}

/// Generate test attention mask
/// - Parameters:
///   - batchSize: Batch size
///   - seqLen: Sequence length
/// - Returns: A 2D MLXArray [batchSize, seqLen] with 1s and 0s
func generateTestAttentionMask(batchSize: Int, seqLen: Int) -> MLXArray {
    return MLXArray.ones([batchSize, seqLen])
}

/// Generate test token IDs
/// - Parameters:
///   - batchSize: Batch size
///   - seqLen: Sequence length
///   - vocabSize: Vocabulary size for random token generation
/// - Returns: A 2D MLXArray [batchSize, seqLen] of Int32
func generateTestTokenIds(batchSize: Int, seqLen: Int, vocabSize: Int = 32000) -> MLXArray {
    return MLXRandom.randInt(low: 0, high: vocabSize, [batchSize, seqLen])
}
