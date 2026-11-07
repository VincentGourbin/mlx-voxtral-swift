/**
 * MemoryManagerTests - the limits `lean` sizes, the defaults, the GPU gate, weight residency, KV cache and prefill slices
 *
 * No real model: the generation tests run on a reduced random one.
 */

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import VoxtralCore

final class MemoryManagerTests: XCTestCase {

    private let mb = 1_048_576

    private func prompt(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { Int32(30 + $0 % 60) }).reshaped([1, n])  // never the audio token (24)
    }

    // MARK: - Limits

    func testLeanLimitsFollowTheAvailableMemory() {
        let phone = VoxtralMemoryLimits.lean(availableBytes: 6144 * mb)
        XCTAssertEqual(phone.cacheBytes, 1024 * mb)
        XCTAssertEqual(phone.memoryBytes, 4864 * mb)
        XCTAssertTrue(phone.clearsCacheBetweenStages)

        let tight = VoxtralMemoryLimits.lean(availableBytes: 3072 * mb)
        XCTAssertEqual(tight.cacheBytes, 512 * mb)
        XCTAssertEqual(tight.memoryBytes, 3072 * mb, "never below 3 GB")

        XCTAssertEqual(VoxtralMemoryLimits.lean(availableBytes: 0).cacheBytes, 256 * mb, "never below 256 MB")
    }

    func testLoadingStaysUnderOneGigabyte() {
        let limits = VoxtralMemoryLimits(cacheBytes: 2048 * mb)
        XCTAssertEqual(limits.cacheBytes(for: .load), 1024 * mb)
        XCTAssertEqual(limits.cacheBytes(for: .generate), 2048 * mb)
        XCTAssertEqual(limits.cacheBytes(for: .codec), 2048 * mb)
        XCTAssertEqual(VoxtralMemoryLimits(cacheBytes: 256 * mb).cacheBytes(for: .load), 256 * mb)
        XCTAssertNil(VoxtralMemoryLimits.none.cacheBytes(for: .load))
    }

    func testLeanProfileSetsEveryLever() {
        let lean = MemoryOptimizationConfig.lean
        XCTAssertEqual(lean.limits, .lean())
        XCTAssertEqual(lean.kvCacheBits, 8)
        XCTAssertEqual(lean.prefillStepSize, 256)
        XCTAssertTrue(lean.releasesAudioTower)
        XCTAssertNil(lean.maxKVCacheSize, "no window: the LM has none (K-2)")
        XCTAssertEqual(VoxtralTTSPipeline.Configuration.lean.memoryLimits, .lean())
        XCTAssertEqual(VoxtralRealtimePipeline.Configuration.lean.memoryLimits, .lean())
    }

    // MARK: - Defaults

    func testTTSHasATwoGigabyteCeilingByDefaultThatCanBeRemoved() {
        XCTAssertEqual(VoxtralTTSPipeline.Configuration.default.cacheLimitBytes, 2048 * mb)
        XCTAssertEqual(VoxtralTTSPipeline.Configuration().cacheLimitBytes, 2048 * mb)
        XCTAssertNil(VoxtralTTSPipeline.Configuration(cacheLimitBytes: nil).cacheLimitBytes)
    }

    func testRealtimeHasNoCeilingByDefault() {
        XCTAssertNil(VoxtralRealtimePipeline.Configuration.default.cacheLimitBytes)
    }

    func testSTTCeilingFollowsTheMachine() {
        XCTAssertEqual(MemoryOptimizationConfig.recommended(forRAMGB: 16).cacheLimitBytes, 2048 * mb)
        XCTAssertEqual(MemoryOptimizationConfig.recommended(forRAMGB: 31).cacheLimitBytes, 2048 * mb)
        XCTAssertNil(MemoryOptimizationConfig.recommended(forRAMGB: 32).cacheLimitBytes)
        XCTAssertNil(MemoryOptimizationConfig.fast.cacheLimitBytes)
    }

    func testSmallHasTheCeilingUpTo32GigabytesAndCanGoWithout() {
        typealias Configuration = VoxtralPipeline.Configuration
        for model in [VoxtralPipeline.Model.small4bit, .small24b8bit, .small24b] {
            XCTAssertEqual(
                Configuration.recommended(for: model, ramGB: 32).memoryOptimization.cacheLimitBytes, 2048 * mb, model.rawValue)
            XCTAssertNil(Configuration.recommended(for: model, ramGB: 33).memoryOptimization.cacheLimitBytes, model.rawValue)
        }
        XCTAssertNil(Configuration.recommended(for: .mini3b8bit, ramGB: 32).memoryOptimization.cacheLimitBytes)

        var explicit = Configuration.default
        explicit.memoryOptimization = .fast
        let pipeline = VoxtralPipeline(model: .small4bit, configuration: explicit)
        XCTAssertNil(pipeline.configuration.memoryOptimization.cacheLimitBytes, "a configuration given by the host stays")
    }

    @available(*, deprecated)
    func testDeprecatedPresetsStillCompileAndNameTheNewSettings() {
        XCTAssertEqual(MemoryOptimizationConfig.light, .fast)
        XCTAssertEqual(MemoryOptimizationConfig.ultra.cacheLimitBytes, 2048 * mb)
        var config = MemoryOptimizationConfig(evalFrequency: 8, clearCacheOnEval: true, cacheLimitBytes: 256 * mb)
        config.evalFrequency = 2
        XCTAssertEqual(config.evalFrequency, 0)
        XCTAssertEqual(config.cacheLimitBytes, 256 * mb)
    }

    // MARK: - Report

    func testReportCarriesTheFootprint() throws {
        let report = VoxtralMemoryManager.report()
        XCTAssertGreaterThan(try XCTUnwrap(report.footprintBytes), 0)
        XCTAssertEqual(report.availableBytes, VoxtralMemoryManager.availableBytes)
        XCTAssertTrue(report.description.contains("footprint"), report.description)
    }

    // MARK: - GPU gate

    func testClosedGateParksUntilResume() {
        let gate = VoxtralGPUGate()
        gate.suspend()
        XCTAssertTrue(gate.isSuspended)
        let passed = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            gate.wait()
            passed.signal()
        }
        XCTAssertEqual(passed.wait(timeout: .now() + 0.3), .timedOut, "closed: the caller stays parked")
        gate.resume()
        XCTAssertEqual(passed.wait(timeout: .now() + 5), .success)
    }

    func testCancelledCallerLeavesAClosedGate() async {
        let gate = VoxtralGPUGate()
        gate.suspend()
        let task = Task.detached { gate.wait() }
        task.cancel()
        await task.value
        XCTAssertTrue(gate.isSuspended)
    }

    func testDecodingParksAtTheGateAndResumes() throws {
        let model = try makeReducedVoxtralModel()
        nonisolated(unsafe) let unsafeModel = model
        let input = prompt(16)
        nonisolated(unsafe) let unsafeInput = input
        let tokens = Locked<[Int]?>(nil)
        let finished = DispatchSemaphore(value: 0)

        VoxtralGPUGate.shared.suspend()
        defer { VoxtralGPUGate.shared.resume() }
        DispatchQueue.global().async {
            tokens.set(try? unsafeModel.generateStream(inputIds: unsafeInput, maxNewTokens: 2, temperature: 0))
            finished.signal()
        }
        XCTAssertEqual(finished.wait(timeout: .now() + 0.3), .timedOut, "no GPU work while the gate is closed")
        VoxtralGPUGate.shared.resume()
        XCTAssertEqual(finished.wait(timeout: .now() + 30), .success)
        XCTAssertEqual(tokens.get()?.isEmpty, false)
    }

    // MARK: - Prefill slices and KV cache

    /// Logits of the prefill's last position under `memory`
    private func prefillLogits(
        _ model: VoxtralForConditionalGeneration, _ tokens: Int, _ memory: MemoryOptimizationConfig
    ) throws -> MLXArray {
        var logits: MLXArray?
        model.prefillLogitsObserver = { logits = $0 }
        defer { model.prefillLogitsObserver = nil }
        _ = try model.generateStream(inputIds: prompt(tokens), maxNewTokens: 1, temperature: 0, memoryOptimization: memory)
        return try XCTUnwrap(logits).asType(.float32)
    }

    func testPrefillSliceDoesNotChangeTheResult() throws {
        let model = try makeReducedVoxtralModel()
        var sliced = MemoryOptimizationConfig.fast
        sliced.prefillStepSize = 64
        let reference = try prefillLogits(model, 300, .fast)
        let logits = try prefillLogits(model, 300, sliced)
        XCTAssertTrue(allClose(reference, logits, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testPrefillSliceMustBePositive() throws {
        let model = try makeReducedVoxtralModel()
        var config = MemoryOptimizationConfig.fast
        config.prefillStepSize = 0
        XCTAssertThrowsError(try model.generateStream(inputIds: prompt(8), maxNewTokens: 1, memoryOptimization: config))
    }

    func testEightBitKVCacheStaysCloseToTheFullOne() throws {
        let model = try makeReducedVoxtralModel(headDim: 32)
        var quantized = MemoryOptimizationConfig.fast
        quantized.kvCacheBits = 8
        quantized.prefillStepSize = 128  // several slices: the second one attends on packed keys

        let reference = try prefillLogits(model, 300, .fast)
        let logits = try prefillLogits(model, 300, quantized)
        let error = abs(reference - logits).max().item(Float.self) / abs(reference).max().item(Float.self)
        XCTAssertLessThan(error, 0.05)
        XCTAssertGreaterThan(error, 0, "the cache really was quantized")
    }

    func testLeanGeneratesOnTheReducedModel() throws {
        let model = try makeReducedVoxtralModel(headDim: 32)
        let tokens = try model.generateStream(
            inputIds: prompt(600), maxNewTokens: 3, temperature: 0, repetitionPenalty: 1.0, memoryOptimization: .lean)
        XCTAssertFalse(tokens.isEmpty)
    }

    // MARK: - Weight residency

    /// A checkpoint holding the reduced model's audio tower
    private func towerCheckpoint(_ model: VoxtralForConditionalGeneration) throws -> (URL, [String: MLXArray]) {
        let tower = try XCTUnwrap(model.standardModel?.audioTower)
        let weights = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
        MLX.eval(Array(weights.values))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-tower-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let renamed = Dictionary(uniqueKeysWithValues: weights.map { ("audioTower.\($0.key)", $0.value) })
        try MLX.save(arrays: renamed, url: directory.appendingPathComponent("model.safetensors"))
        return (directory, weights)
    }

    func testReleasedAudioTowerComesBackFromTheCheckpoint() throws {
        let model = try makeReducedVoxtralModel()
        let (directory, original) = try towerCheckpoint(model)
        let tower = try XCTUnwrap(model.standardModel?.audioTower)

        XCTAssertGreaterThan(model.releaseAudioTower(), 0)
        XCTAssertTrue(model.isAudioTowerReleased)
        XCTAssertEqual(model.releaseAudioTower(), 0, "already released")
        for (key, value) in tower.parameters().flattened() {
            XCTAssertEqual(abs(value).max().item(Float.self), 0, key)
        }

        try model.reloadAudioTower(from: directory)
        XCTAssertFalse(model.isAudioTowerReleased)
        for (key, value) in tower.parameters().flattened() {
            XCTAssertTrue(arrayEqual(value, try XCTUnwrap(original[key])).item(Bool.self), key)
        }
    }

    func testEncodingRunsBeforeTheTowerGoes() throws {
        let model = try makeReducedVoxtralModel()
        let audioTokenId = Int32(model.config.audio_token_id)
        let ids = [Int32(30), 31] + Array(repeating: audioTokenId, count: 375) + [Int32(32), 33]
        let inputIds = MLXArray(ids).reshaped([1, ids.count])
        let mel = MLXRandom.normal([1, 128, 3000], key: MLXRandom.key(7))
        let expected = model.getAudioEmbeds(mel)

        var released = false
        var text: [Int] = []
        text = try model.generateStream(
            inputIds: inputIds, inputFeatures: mel, maxNewTokens: 1, temperature: 0,
            onEncoded: {
                released = true
                model.releaseAudioTower()
            })
        XCTAssertTrue(released)
        XCTAssertFalse(text.isEmpty)
        XCTAssertTrue(model.isAudioTowerReleased)
        XCTAssertGreaterThan(abs(expected).max().item(Float.self), 0)
    }

    func testTextOnlyRequestLeavesTheTowerAlone() throws {
        let model = try makeReducedVoxtralModel()
        var called = false
        _ = try model.generateStream(inputIds: prompt(8), maxNewTokens: 1, onEncoded: { called = true })
        XCTAssertFalse(called)
    }

    func testMaterializingLeavesTheModelUsable() throws {
        let model = try makeReducedVoxtralModel()
        model.materializeWeights(audioTower: false)
        model.materializeWeights(audioTower: true)
        XCTAssertFalse(try model.generateStream(inputIds: prompt(8), maxNewTokens: 1, temperature: 0).isEmpty)
    }
}
