/**
 * VoxtralVoiceSLERPTests - Tests for voice SLERP blending and ZeroVoice
 */

import XCTest
import MLX
@testable import VoxtralCore

final class VoxtralVoiceSLERPTests: XCTestCase {

    // MARK: - Length Alignment Tests

    func testAlignSameLength() {
        let a = MLXArray.ones([10, 4])
        let b = MLXArray.ones([10, 4]) * 2.0
        let (aA, bA) = alignVoiceLengths(a, b)
        XCTAssertEqual(aA.dim(0), 10)
        XCTAssertEqual(bA.dim(0), 10)
    }

    func testAlignDifferentLengths() {
        let a = MLXArray.ones([5, 4])
        let b = MLXArray.ones([10, 4])
        let (aA, bA) = alignVoiceLengths(a, b)
        XCTAssertEqual(aA.dim(0), 10)
        XCTAssertEqual(bA.dim(0), 10)
        XCTAssertEqual(aA.dim(1), 4)
    }

    func testResamplePreservesEndpoints() {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let t = MLXArray(values).reshaped(5, 1)
        let resampled = resampleVoiceEmbedding(t, targetLen: 10)
        XCTAssertEqual(resampled.dim(0), 10)
        // First and last should match
        let first = resampled[0, 0].item(Float.self)
        let last = resampled[9, 0].item(Float.self)
        XCTAssertEqual(first, 1.0, accuracy: 0.01)
        XCTAssertEqual(last, 5.0, accuracy: 0.01)
    }

    // MARK: - SLERP Tests

    func testSlerpT0ReturnsA() {
        let a = MLXArray.ones([5, 8])
        let b = MLXArray.ones([5, 8]) * 3.0
        let result = slerpVoices(a, b, t: 0.0)
        // t=0 should return a
        let diff = abs(result - a.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(diff, 0.01)
    }

    func testSlerpT1ReturnsB() {
        let a = MLXArray.ones([5, 8])
        let b = MLXArray.ones([5, 8]) * 3.0
        let result = slerpVoices(a, b, t: 1.0)
        let diff = abs(result - b.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(diff, 0.01)
    }

    func testSlerpPreservesShape() {
        let a = MLXArray.ones([10, 16])
        let b = MLXArray.ones([10, 16]) * 2.0
        let result = slerpVoices(a, b, t: 0.5)
        XCTAssertEqual(result.shape, [10, 16])
    }

    func testSlerpMidpointMagnitudeInterpolated() {
        let a = MLXArray.ones([3, 4]) * 2.0
        let b = MLXArray.ones([3, 4]) * 4.0
        let result = slerpVoices(a, b, t: 0.5)
        // For parallel vectors, SLERP degrades to LERP
        // Midpoint of norms 2*sqrt(4)=4 and 4*sqrt(4)=8 → target ~6
        let meanNorm = MLX.sqrt((result * result).sum(axis: -1)).mean().item(Float.self)
        XCTAssertGreaterThan(meanNorm, 3.0)
        XCTAssertLessThan(meanNorm, 9.0)
    }

    // MARK: - Norm Calibration Tests

    func testCalibrateNorms() {
        let input = MLXArray.ones([10, 8]) * 10.0  // high norms
        let calibrated = calibrateVoiceNorms(input, targetMeanNorm: 4.48)
        let rowNorms = MLX.sqrt((calibrated * calibrated).sum(axis: -1))
        let meanNorm = rowNorms.mean().item(Float.self)
        XCTAssertEqual(meanNorm, 4.48, accuracy: 0.1)
    }

    // MARK: - Blend Convenience Tests

    func testBlendVoices() {
        let a = MLXArray.ones([5, 8]) * 2.0
        let b = MLXArray.ones([10, 8]) * 3.0
        let blended = blendVoices(voiceA: a, voiceB: b, t: 0.3)
        // Should be aligned to max length and calibrated
        XCTAssertEqual(blended.dim(0), 10)
        XCTAssertEqual(blended.dim(1), 8)
    }

    func testBlendClampsTParam() {
        let a = MLXArray.ones([5, 4])
        let b = MLXArray.ones([5, 4]) * 2.0
        // Should not crash with out-of-range t
        let _ = blendVoices(voiceA: a, voiceB: b, t: -0.5)
        let _ = blendVoices(voiceA: a, voiceB: b, t: 1.5)
    }

    // MARK: - ZeroVoice Tests

    func testVoiceFamilyFromZ() {
        XCTAssertEqual(VoiceFamily.fromZ(50), .english)
        XCTAssertEqual(VoiceFamily.fromZ(99), .english)
        XCTAssertEqual(VoiceFamily.fromZ(100), .european)
        XCTAssertEqual(VoiceFamily.fromZ(150), .european)
        XCTAssertEqual(VoiceFamily.fromZ(200), .asianArabic)
        XCTAssertEqual(VoiceFamily.fromZ(999), .asianArabic)
    }

    func testVoiceFamilyVoiceCount() {
        XCTAssertEqual(VoiceFamily.english.voices.count, 5)
        XCTAssertEqual(VoiceFamily.european.voices.count, 12)
        XCTAssertEqual(VoiceFamily.asianArabic.voices.count, 3)
    }

    func testCrossFamilySelection() {
        XCTAssertEqual(VoiceFamily.english.crossFamily(z: 50), .european)
        XCTAssertEqual(VoiceFamily.asianArabic.crossFamily(z: 200), .european)
    }

    func testZeroVoiceRecipeDeterministic() {
        let embeddings: [String: MLXArray] = [:]
        let zv = VoxtralZeroVoice(voiceEmbeddings: embeddings)

        let recipe1 = zv.voiceRecipe(x: 50, y: 50, z: 50)
        let recipe2 = zv.voiceRecipe(x: 50, y: 50, z: 50)

        XCTAssertEqual(recipe1.voiceA, recipe2.voiceA)
        XCTAssertEqual(recipe1.voiceB, recipe2.voiceB)
        XCTAssertEqual(recipe1.blendWeight, recipe2.blendWeight)
    }

    func testZeroVoiceRecipeCrossFamilyPairing() {
        let embeddings: [String: MLXArray] = [:]
        let zv = VoxtralZeroVoice(voiceEmbeddings: embeddings)

        // z=50 → English primary, European secondary
        let recipe = zv.voiceRecipe(x: 10, y: 20, z: 50)
        let familyA = VoiceFamily.english.voices
        let familyB = VoiceFamily.european.voices

        XCTAssertTrue(familyA.contains(recipe.voiceA))
        XCTAssertTrue(familyB.contains(recipe.voiceB))
    }

    func testZeroVoiceBlendWeightCapped() {
        let embeddings: [String: MLXArray] = [:]
        let zv = VoxtralZeroVoice(voiceEmbeddings: embeddings)

        // Test many coordinates, blend weight should always be <= 0.20
        for x in stride(from: 0, to: 100, by: 17) {
            for y in stride(from: 0, to: 100, by: 23) {
                let recipe = zv.voiceRecipe(x: x, y: y, z: 50)
                XCTAssertLessThanOrEqual(recipe.blendWeight, 0.20 + 0.001)
                XCTAssertGreaterThanOrEqual(recipe.blendWeight, 0.0)
            }
        }
    }

    func testZeroVoiceDifferentCoordinatesProduceDifferentRecipes() {
        let embeddings: [String: MLXArray] = [:]
        let zv = VoxtralZeroVoice(voiceEmbeddings: embeddings)

        let recipe1 = zv.voiceRecipe(x: 0, y: 0, z: 0)
        let recipe2 = zv.voiceRecipe(x: 1000, y: 1000, z: 200)

        // At least something should differ
        let different = recipe1.voiceA != recipe2.voiceA
            || recipe1.voiceB != recipe2.voiceB
            || recipe1.blendWeight != recipe2.blendWeight
        XCTAssertTrue(different)
    }

    func testZeroVoiceReturnsNilWithoutEmbeddings() {
        let zv = VoxtralZeroVoice(voiceEmbeddings: [:])
        let voice = zv.voiceAt(x: 50, y: 50, z: 50)
        XCTAssertNil(voice)
    }
}
