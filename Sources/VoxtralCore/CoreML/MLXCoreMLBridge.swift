/**
 * MLXCoreMLBridge - Bridge between MLX and Core ML tensor types
 *
 * This module provides efficient conversion utilities between:
 * - MLXArray (MLX Swift framework)
 * - MLMultiArray (Core ML framework)
 *
 * Used for the hybrid architecture where:
 * - Audio encoder runs on Core ML (ANE)
 * - LLM decoder runs on MLX (GPU)
 */

import Foundation
import CoreML
import MLX

// MARK: - Float16 Bit Conversion Helpers

/// Convert Float to Float16 bit representation (IEEE 754 half-precision)
/// This avoids using Float16 type directly which has availability issues in Release builds
@inline(__always)
private func floatToFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = (bits >> 16) & 0x8000
    let exp = Int((bits >> 23) & 0xFF) - 127 + 15
    let mantissa = bits & 0x007FFFFF

    if exp <= 0 {
        // Denormalized or zero
        if exp < -10 {
            return UInt16(sign)
        }
        let m = (mantissa | 0x00800000) >> (1 - exp + 13)
        return UInt16(sign | (m >> 13))
    } else if exp >= 31 {
        // Infinity or NaN
        if mantissa != 0 {
            return UInt16(sign | 0x7FFF)  // NaN
        }
        return UInt16(sign | 0x7C00)  // Infinity
    }

    return UInt16(sign | UInt32(exp << 10) | (mantissa >> 13))
}

/// Convert Float16 bit representation to Float
@inline(__always)
private func float16BitsToFloat(_ bits: UInt16) -> Float {
    let sign = UInt32(bits & 0x8000) << 16
    let exp = UInt32((bits >> 10) & 0x1F)
    let mantissa = UInt32(bits & 0x03FF)

    if exp == 0 {
        if mantissa == 0 {
            return Float(bitPattern: sign)  // Zero
        }
        // Denormalized
        var m = mantissa
        var e: UInt32 = 0
        while (m & 0x0400) == 0 {
            m <<= 1
            e += 1
        }
        let newExp = (127 - 15 - e) << 23
        let newMantissa = (m & 0x03FF) << 13
        return Float(bitPattern: sign | newExp | newMantissa)
    } else if exp == 31 {
        // Infinity or NaN
        if mantissa != 0 {
            return Float.nan
        }
        return sign == 0 ? Float.infinity : -Float.infinity
    }

    let newExp = (exp + 127 - 15) << 23
    let newMantissa = mantissa << 13
    return Float(bitPattern: sign | newExp | newMantissa)
}

/// Bridge utilities for MLX <-> Core ML tensor conversion
@available(macOS 13.0, iOS 16.0, *)
public struct MLXCoreMLBridge {

    // MARK: - MLXArray to MLMultiArray

    /// Convert MLXArray to MLMultiArray for Core ML
    /// - Parameter mlxArray: Source MLXArray
    /// - Returns: Equivalent MLMultiArray
    /// - Throws: Error if conversion fails
    public static func toMLMultiArray(_ mlxArray: MLXArray) throws -> MLMultiArray {
        // Ensure array is evaluated
        eval(mlxArray)

        // Get shape
        let shape = mlxArray.shape.map { NSNumber(value: $0) }

        // Determine data type
        let dataType: MLMultiArrayDataType
        switch mlxArray.dtype {
        case .float32:
            dataType = .float32
        case .float16:
            dataType = .float16
        case .int32:
            dataType = .int32
        default:
            // Convert to float32 for unsupported types
            let converted = mlxArray.asType(.float32)
            return try toMLMultiArray(converted)
        }

        // Create MLMultiArray
        let multiArray = try MLMultiArray(shape: shape, dataType: dataType)

        // Copy data efficiently
        let count = mlxArray.size

        switch dataType {
        case .float32:
            let floatArray = mlxArray.asArray(Float.self)
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
            floatArray.withUnsafeBufferPointer { buffer in
                for i in 0..<count {
                    pointer[i] = buffer[i]
                }
            }

        case .float16:
            // For Float16, convert through Float and use raw memory copy
            // Float16 is only available on arm64, so we handle it via raw bytes
            let floatArray = mlxArray.asType(.float32).asArray(Float.self)
            let destPointer = multiArray.dataPointer.assumingMemoryBound(to: UInt16.self)
            floatArray.withUnsafeBufferPointer { buffer in
                for i in 0..<count {
                    // Convert Float to Float16 representation (IEEE 754 half-precision)
                    destPointer[i] = floatToFloat16Bits(buffer[i])
                }
            }

        case .int32:
            let intArray = mlxArray.asArray(Int32.self)
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Int32.self)
            intArray.withUnsafeBufferPointer { buffer in
                for i in 0..<count {
                    pointer[i] = buffer[i]
                }
            }

        default:
            throw MLXCoreMLBridgeError.unsupportedDataType(String(describing: dataType))
        }

        return multiArray
    }

    // MARK: - MLMultiArray to MLXArray

    /// Convert MLMultiArray to MLXArray for MLX
    /// - Parameter multiArray: Source MLMultiArray
    /// - Returns: Equivalent MLXArray
    public static func toMLXArray(_ multiArray: MLMultiArray) -> MLXArray {
        // Get shape
        let shape = multiArray.shape.map { $0.intValue }
        let count = shape.reduce(1, *)

        // Get data type and convert
        switch multiArray.dataType {
        case .int8:
            // Convert int8 to Float
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Int8.self)
            var floatArray = [Float](repeating: 0, count: count)
            for i in 0..<count {
                floatArray[i] = Float(pointer[i])
            }
            return MLXArray(floatArray).reshaped(shape)

        case .float32:
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
            let array = Array(UnsafeBufferPointer(start: pointer, count: count))
            return MLXArray(array).reshaped(shape)

        case .float16:
            // Read Float16 as raw UInt16 bits and convert to Float
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: UInt16.self)
            var floatArray = [Float](repeating: 0, count: count)
            for i in 0..<count {
                floatArray[i] = float16BitsToFloat(pointer[i])
            }
            return MLXArray(floatArray).reshaped(shape).asType(.float16)

        case .int32:
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Int32.self)
            let array = Array(UnsafeBufferPointer(start: pointer, count: count))
            return MLXArray(array).reshaped(shape)

        case .double:
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Double.self)
            let array = Array(UnsafeBufferPointer(start: pointer, count: count)).map { Float($0) }
            return MLXArray(array).reshaped(shape)

        @unknown default:
            // Fallback: try to read as Float32
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
            let array = Array(UnsafeBufferPointer(start: pointer, count: count))
            return MLXArray(array).reshaped(shape)
        }
    }

    // MARK: - Optimized Zero-Copy Conversion (when possible)

    /// Convert MLXArray to MLMultiArray using pointer if possible
    /// This avoids a copy when the MLXArray has contiguous memory layout
    /// - Warning: The returned MLMultiArray shares memory with the input
    /// - Parameter mlxArray: Source MLXArray (must outlive the returned MLMultiArray)
    /// - Returns: MLMultiArray sharing memory with input
    public static func toMLMultiArrayNoCopy(_ mlxArray: MLXArray) throws -> MLMultiArray {
        // Evaluate to ensure data is materialized
        eval(mlxArray)

        // Note: True zero-copy requires MLX to expose raw pointers
        // For now, fall back to regular conversion
        // In future, with MLX updates, we could use:
        // return try MLMultiArray(dataPointer: mlxArray.rawPointer, ...)

        // Fall back to copy-based conversion
        return try toMLMultiArray(mlxArray)
    }

    // MARK: - Batch Conversion

    /// Convert multiple MLXArrays to MLMultiArrays
    /// - Parameter arrays: Array of MLXArrays to convert
    /// - Returns: Array of equivalent MLMultiArrays
    public static func toMLMultiArrayBatch(_ arrays: [MLXArray]) throws -> [MLMultiArray] {
        try arrays.map { try toMLMultiArray($0) }
    }

    /// Convert multiple MLMultiArrays to MLXArrays
    /// - Parameter arrays: Array of MLMultiArrays to convert
    /// - Returns: Array of equivalent MLXArrays
    public static func toMLXArrayBatch(_ arrays: [MLMultiArray]) -> [MLXArray] {
        arrays.map { toMLXArray($0) }
    }

    // MARK: - Validation

    /// Validate that two tensors have compatible shapes
    /// - Parameters:
    ///   - mlxShape: Shape of MLXArray
    ///   - mlShape: Shape of MLMultiArray
    /// - Returns: true if shapes match
    public static func validateShapes(_ mlxShape: [Int], _ mlShape: [NSNumber]) -> Bool {
        guard mlxShape.count == mlShape.count else { return false }
        for (a, b) in zip(mlxShape, mlShape) {
            if a != b.intValue { return false }
        }
        return true
    }
}

// MARK: - Errors

/// Errors that can occur during MLX-CoreML bridging
public enum MLXCoreMLBridgeError: Error, LocalizedError {
    case unsupportedDataType(String)
    case shapeMismatch(mlxShape: [Int], mlShape: [Int])
    case conversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedDataType(let type):
            return "Unsupported data type for conversion: \(type)"
        case .shapeMismatch(let mlxShape, let mlShape):
            return "Shape mismatch: MLX \(mlxShape) vs CoreML \(mlShape)"
        case .conversionFailed(let reason):
            return "Conversion failed: \(reason)"
        }
    }
}

// MARK: - MLXArray Extensions

@available(macOS 13.0, iOS 16.0, *)
extension MLXArray {

    /// Convert this MLXArray to Core ML MLMultiArray
    /// - Returns: Equivalent MLMultiArray
    public func toMLMultiArray() throws -> MLMultiArray {
        try MLXCoreMLBridge.toMLMultiArray(self)
    }
}

// MARK: - MLMultiArray Extensions

@available(macOS 13.0, iOS 16.0, *)
extension MLMultiArray {

    /// Convert this MLMultiArray to MLX MLXArray
    /// - Returns: Equivalent MLXArray
    public func toMLXArray() -> MLXArray {
        MLXCoreMLBridge.toMLXArray(self)
    }
}

// MARK: - Performance Utilities

@available(macOS 13.0, iOS 16.0, *)
extension MLXCoreMLBridge {

    /// Measure conversion time for benchmarking
    /// - Parameter mlxArray: Input to convert
    /// - Returns: Tuple of (result, timeInMilliseconds)
    public static func toMLMultiArrayWithTiming(_ mlxArray: MLXArray) throws -> (MLMultiArray, Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try toMLMultiArray(mlxArray)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        return (result, elapsed)
    }

    /// Measure conversion time for benchmarking
    /// - Parameter multiArray: Input to convert
    /// - Returns: Tuple of (result, timeInMilliseconds)
    public static func toMLXArrayWithTiming(_ multiArray: MLMultiArray) -> (MLXArray, Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = toMLXArray(multiArray)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        return (result, elapsed)
    }
}
