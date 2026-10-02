/**
 * PipelineStateStressTests - K-11 (S-10)
 *
 * State transitions are atomic and one operation holds a pipeline at a time. Run under TSan:
 *   xcodebuild test … -enableThreadSanitizer YES -only-testing:VoxtralCoreTests/PipelineStateStressTests
 * No model is needed: loads fail fast on a folder without tekken.json, and "ready" is set
 * through the gate to exercise the refusals and the generation token.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class PipelineStateStressTests: XCTestCase {

    private let iterations = 10
    private let fm = FileManager.default
    private var sandbox: URL!
    private var savedCustomDir: URL?

    override func setUp() {
        super.setUp()
        sandbox = fm.temporaryDirectory.appendingPathComponent("voxtral-state-stress-\(UUID().uuidString)")
        try? fm.createDirectory(at: sandbox, withIntermediateDirectories: true)
        savedCustomDir = VoxtralModelDownloader.customModelsDirectory
        VoxtralModelDownloader.customModelsDirectory = sandbox
    }

    override func tearDown() {
        VoxtralModelDownloader.customModelsDirectory = savedCustomDir
        try? fm.removeItem(at: sandbox)
        super.tearDown()
    }

    private func isBusy(_ error: Error) -> Bool {
        if case VoxtralTTSError.busy = error { return true }
        return false
    }

    /// Two concurrent loads: never two at once; the pipeline ends in `.error`, free.
    func testConcurrentLoadsAreExclusive() async throws {
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit"))
        let folder = sandbox.appendingPathComponent(info.repoId)
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        let params = Data("{}".utf8)
        try params.write(to: folder.appendingPathComponent("params.json"))
        try Data("{\"version\":1,\"repoId\":\"x\",\"files\":[{\"path\":\"params.json\",\"size\":\(params.count)}]}".utf8)
            .write(to: folder.appendingPathComponent(VoxtralModelDownloader.manifestFileName))

        for _ in 0 ..< iterations {
            let pipeline = VoxtralTTSPipeline()
            let errors = await withTaskGroup(of: Error?.self) { group -> [Error] in
                for _ in 0 ..< 2 {
                    group.addTask {
                        do { try await pipeline.loadModel(modelInfo: info); return nil } catch { return error }
                    }
                }
                var all: [Error] = []
                for await error in group { if let error { all.append(error) } }
                return all
            }
            XCTAssertEqual(errors.count, 2, "both loads fail (no tekken.json)")
            XCTAssertTrue(errors.allSatisfy { error in
                if case VoxtralError.fileNotFound = error { return true }
                if case VoxtralTTSError.invalidConfiguration = error { return true }
                return isBusy(error)
            }, "\(errors)")
            guard case .error = pipeline.state else { return XCTFail("final state \(pipeline.state)") }
            XCTAssertNil(pipeline.gate.operation)
        }
    }

    /// `unload()` while an operation runs: the operation's late `end` must not resurrect `.ready`.
    func testUnloadDuringOperationKeepsUnloaded() throws {
        for _ in 0 ..< iterations {
            let pipeline = VoxtralTTSPipeline()
            pipeline.gate.reset(.ready)
            let generation = try pipeline.gate.begin(
                "streaming synthesis", accepts: { $0.isReady }, refusal: VoxtralTTSError.invalidConfiguration("x"),
                busy: VoxtralTTSError.busy, state: .synthesizing)
            DispatchQueue.concurrentPerform(iterations: 2) { i in
                if i == 0 { pipeline.unload() } else { _ = pipeline.state }
            }
            pipeline.gate.end(generation, state: .ready)  // the stale stream Task finishing
            guard case .unloaded = pipeline.state else { return XCTFail("final state \(pipeline.state)") }
            XCTAssertNil(pipeline.gate.operation)
        }
    }

    /// Syntheses started while an enrollment holds the pipeline are refused `busy`.
    func testSynthesisDuringEnrollmentIsRefusedBusy() async throws {
        for _ in 0 ..< iterations {
            let pipeline = VoxtralTTSPipeline()
            pipeline.gate.reset(.ready)
            let generation = try pipeline.gate.begin(
                "enrollment", accepts: { $0.isReady }, refusal: VoxtralTTSError.invalidConfiguration("x"),
                busy: VoxtralTTSError.busy)
            let refused = await withTaskGroup(of: Bool.self) { group -> Int in
                for _ in 0 ..< 8 {
                    group.addTask {
                        do { _ = try await pipeline.synthesize(text: "x"); return false }
                        catch {
                            if case VoxtralTTSError.busy = error { return true }
                            return false
                        }
                    }
                }
                var count = 0
                for await busy in group where busy { count += 1 }
                return count
            }
            XCTAssertEqual(refused, 8)
            var streamError: Error?
            do { for try await _ in pipeline.synthesizeStreaming(text: "x", voiceEmbedding: .zeros([1, 1])) {} }
            catch { streamError = error }
            XCTAssertTrue(streamError.map(isBusy) ?? false, "\(String(describing: streamError))")
            pipeline.gate.end(generation)
            guard case .ready = pipeline.state else { return XCTFail("final state \(pipeline.state)") }
            XCTAssertNil(pipeline.gate.operation)
        }
    }

    /// Many threads claiming the gate: at most one holder at any time.
    func testGateGrantsOneOperationAtATime() {
        final class Counter: @unchecked Sendable {
            let lock = NSLock(); var holders = 0; var maxHolders = 0; var granted = 0
        }
        for _ in 0 ..< iterations {
            let gate = PipelineGate<VoxtralTTSPipeline.State>(.ready)
            let counter = Counter()
            DispatchQueue.concurrentPerform(iterations: 64) { _ in
                guard let generation = try? gate.begin(
                    "op", accepts: { $0.isReady }, refusal: VoxtralTTSError.invalidConfiguration("x"),
                    busy: VoxtralTTSError.busy, state: .synthesizing) else { return }
                counter.lock.withLock {
                    counter.holders += 1; counter.granted += 1
                    counter.maxHolders = max(counter.maxHolders, counter.holders)
                }
                counter.lock.withLock { counter.holders -= 1 }
                gate.end(generation, state: .ready)
            }
            XCTAssertEqual(counter.maxHolders, 1)
            XCTAssertGreaterThan(counter.granted, 0)
            guard case .ready = gate.state else { return XCTFail("final state \(gate.state)") }
            XCTAssertNil(gate.operation)
        }
    }
}
