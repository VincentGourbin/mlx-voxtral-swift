/**
 * VoxtralBenchmark - Performance benchmark for Float16 conversion
 *
 * Compares:
 * 1. Current: Float16 bit-level conversion (IEEE 754)
 * 2. Alternative: Float32 pass-through
 * 3. Real MLX ↔ CoreML bridge operations
 */

import Foundation
import VoxtralCore
import MLX
import CoreML
import ArgumentParser

@main
struct BenchmarkCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "VoxtralBenchmark",
        abstract: "Benchmark Float16 conversion performance"
    )

    @Option(name: .shortAndLong, help: "Number of iterations")
    var iterations: Int = 100

    @Option(name: .shortAndLong, help: "Sequence length (encoder output frames)")
    var sequenceLength: Int = 1500

    @Flag(name: .long, help: "Run real MLX-CoreML bridge test")
    var realBridge: Bool = false

    func run() throws {
        print(String(repeating: "=", count: 60))
        print("Float16 Conversion Benchmark")
        print(String(repeating: "=", count: 60))
        print()

        let hiddenDim = 1280  // Voxtral encoder hidden dimension
        let totalElements = sequenceLength * hiddenDim

        print("Configuration:")
        print("  - Array size: \(sequenceLength) x \(hiddenDim) = \(totalElements) elements")
        print("  - Memory: \(totalElements * 4 / 1024 / 1024) MB (Float32)")
        print("  - Iterations: \(iterations)")
        print()

        // Generate test data
        print("Generating test data...")
        var testFloats = [Float](repeating: 0, count: totalElements)
        for i in 0..<totalElements {
            testFloats[i] = Float.random(in: -10...10)
        }

        // Pre-allocate output arrays
        var float16Output = [UInt16](repeating: 0, count: totalElements)
        var float32Output = [Float](repeating: 0, count: totalElements)

        print()
        print("Pure Swift Benchmarks:")
        print(String(repeating: "-", count: 40))

        // Benchmark 1: Float → Float16 bits (current implementation)
        let float16WriteTime = benchmark("Float → Float16 bits") {
            for i in 0..<totalElements {
                float16Output[i] = floatToFloat16Bits(testFloats[i])
            }
        }

        // Benchmark 2: Float → Float (pass-through, baseline)
        let float32CopyTime = benchmark("Float → Float32 copy") {
            for i in 0..<totalElements {
                float32Output[i] = testFloats[i]
            }
        }

        // Populate Float16 data for read test
        for i in 0..<totalElements {
            float16Output[i] = floatToFloat16Bits(testFloats[i])
        }

        // Benchmark 3: Float16 bits → Float (read back)
        let float16ReadTime = benchmark("Float16 bits → Float") {
            for i in 0..<totalElements {
                float32Output[i] = float16BitsToFloat(float16Output[i])
            }
        }

        // Benchmark 4: Full round-trip
        let roundTripTime = benchmark("Round-trip (F32→F16→F32)") {
            for i in 0..<totalElements {
                let f16 = floatToFloat16Bits(testFloats[i])
                float32Output[i] = float16BitsToFloat(f16)
            }
        }

        if realBridge {
            print()
            print("Real MLX-CoreML Bridge Benchmarks:")
            print(String(repeating: "-", count: 40))

            try runRealBridgeBenchmark(testFloats: testFloats, shape: [1, sequenceLength, hiddenDim])
        }

        print()
        print(String(repeating: "=", count: 60))
        print("Summary")
        print(String(repeating: "=", count: 60))
        print()

        let overhead = float16WriteTime - float32CopyTime
        print("Per-operation overhead vs baseline:")
        print("  Float16 write: \(String(format: "%+.3f", overhead)) ms")
        print("  Float16 read:  \(String(format: "%.3f", float16ReadTime)) ms")
        print("  Round-trip:    \(String(format: "%.3f", roundTripTime)) ms")
        print()

        // Context: typical transcription
        let typicalInferenceMs = 30000.0  // 30 seconds for a transcription
        let overheadPercent = (roundTripTime / typicalInferenceMs) * 100

        print("In context of typical transcription (~30s):")
        print("  Conversion overhead: \(String(format: "%.3f", roundTripTime)) ms")
        print("  Percentage of total: \(String(format: "%.4f", overheadPercent))%")
        print()

        if overheadPercent < 0.01 {
            print("✅ Impact: NEGLIGIBLE (< 0.01% of inference time)")
        } else if overheadPercent < 0.1 {
            print("✅ Impact: MINIMAL (< 0.1% of inference time)")
        } else if overheadPercent < 1.0 {
            print("⚠️  Impact: LOW (< 1% of inference time)")
        } else {
            print("❌ Impact: SIGNIFICANT (> 1% of inference time)")
        }
    }

    // MARK: - Benchmark Helper

    func benchmark(_ name: String, operation: () -> Void) -> Double {
        // Warmup
        for _ in 0..<10 {
            operation()
        }

        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            operation()
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0

        print("  \(name): \(String(format: "%.4f", avgMs)) ms/iter")
        return avgMs
    }

    // MARK: - Real Bridge Benchmark

    @available(macOS 13.0, *)
    func runRealBridgeBenchmark(testFloats: [Float], shape: [Int]) throws {
        // Create MLXArray
        let mlxArray = MLXArray(testFloats).reshaped(shape).asType(.float16)
        eval(mlxArray)

        // Benchmark MLX → CoreML
        var multiArray: MLMultiArray?
        let mlxToCoreML = benchmark("MLXArray → MLMultiArray") {
            multiArray = try? mlxArray.toMLMultiArray()
        }

        guard let coreMLArray = multiArray else {
            print("  ⚠️  MLMultiArray creation failed")
            return
        }

        // Benchmark CoreML → MLX
        var resultMLX: MLXArray?
        let coreMLToMLX = benchmark("MLMultiArray → MLXArray") {
            resultMLX = coreMLArray.toMLXArray()
            eval(resultMLX!)
        }

        print()
        print("  Real bridge total: \(String(format: "%.3f", mlxToCoreML + coreMLToMLX)) ms")
    }
}

// MARK: - Float16 Bit Conversion (same as MLXCoreMLBridge)

@inline(__always)
func floatToFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = (bits >> 16) & 0x8000
    let exp = Int((bits >> 23) & 0xFF) - 127 + 15
    let mantissa = bits & 0x007FFFFF

    if exp <= 0 {
        if exp < -10 {
            return UInt16(sign)
        }
        let m = (mantissa | 0x00800000) >> (1 - exp + 13)
        return UInt16(sign | (m >> 13))
    } else if exp >= 31 {
        if mantissa != 0 {
            return UInt16(sign | 0x7FFF)
        }
        return UInt16(sign | 0x7C00)
    }

    return UInt16(sign | UInt32(exp << 10) | (mantissa >> 13))
}

@inline(__always)
func float16BitsToFloat(_ bits: UInt16) -> Float {
    let sign = UInt32(bits & 0x8000) << 16
    let exp = UInt32((bits >> 10) & 0x1F)
    let mantissa = UInt32(bits & 0x03FF)

    if exp == 0 {
        if mantissa == 0 {
            return Float(bitPattern: sign)
        }
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
        if mantissa != 0 {
            return Float.nan
        }
        return sign == 0 ? Float.infinity : -Float.infinity
    }

    let newExp = (exp + 127 - 15) << 23
    let newMantissa = mantissa << 13
    return Float(bitPattern: sign | newExp | newMantissa)
}
