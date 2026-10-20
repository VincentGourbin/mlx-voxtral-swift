/**
 * StrictKeysLoadingTests - K-88 (suite of K-7)
 *
 * `updateVerified` refuses a weight that matches no model parameter, naming it, instead of dropping it silently
 * (`verify: [.all]`). The fixture test runs everywhere; the pack test loads, with the three live loaders, every
 * registry pack present on disk (`TEST_RUNNER_VOXTRAL_ALL_PACKS=1`: real models, several GB each).
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

final class StrictKeysLoadingTests: XCTestCase {

    private final class Tiny: Module {
        @ModuleInfo var proj: Linear
        override init() {
            _proj.wrappedValue = Linear(4, 2)
            super.init()
        }
    }

    private func parameters(extra: [String: MLXArray] = [:]) -> ModuleParameters {
        var flat: [String: MLXArray] = ["proj.weight": MLXArray.zeros([2, 4]), "proj.bias": MLXArray.zeros([2])]
        flat.merge(extra) { $1 }
        return ModuleParameters.unflattened(flat)
    }

    func testCompleteWeightsLoad() throws {
        XCTAssertNoThrow(try Tiny().updateVerified(parameters: parameters()))
    }

    func testExtraKeyIsRefusedAndNamed() {
        XCTAssertThrowsError(try Tiny().updateVerified(
            parameters: parameters(extra: ["proj.weight_alias": MLXArray.zeros([2, 4])]))) { error in
            XCTAssertTrue("\(error)".contains("proj.weight_alias"), "the error must name the extra key: \(error)")
        }
    }

    func testMissingKeyIsStillNamed() {
        XCTAssertThrowsError(try Tiny().updateVerified(
            parameters: ModuleParameters.unflattened(["proj.weight": MLXArray.zeros([2, 4])]))) { error in
            guard case VoxtralError.missingWeights(let keys) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(keys, ["proj.bias"])
        }
    }

    /// Every pack of the three registries that is on disk loads with 0 missing and 0 extra key.
    func testEveryPresentPackLoadsStrictly() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_ALL_PACKS"] == "1",
                          "Set VOXTRAL_ALL_PACKS=1 to load every pack on disk")
        var loaded: [String] = [], failed: [String] = [], absent: [String] = []
        func attempt(_ id: String, _ load: () throws -> Void) {
            do {
                try autoreleasepool { try load() }
                loaded.append(id)
                print("[k88] LOAD \(id) : ok")
            } catch {
                failed.append(id)
                print("[k88] LOAD \(id) : \(String("\(error)".prefix(600)))")
            }
            Memory.clearCache()
        }
        for info in VoxtralModelRegistry.models {
            guard let path = VoxtralModelDownloader.findModelPath(for: info) else { absent.append(info.id); continue }
            attempt(info.id) { _ = try loadVoxtralStandardModel(modelPath: path.path) }
        }
        for info in VoxtralTTSRegistry.models {
            guard let path = VoxtralModelDownloader.findTTSModelPath(for: info) else { absent.append(info.id); continue }
            attempt(info.id) { _ = try loadVoxtralTTSModel(from: path) }
        }
        for info in VoxtralRealtimeRegistry.models {
            guard let path = VoxtralModelDownloader.findRealtimeModelPath(for: info) else { absent.append(info.id); continue }
            attempt(info.id) { _ = try loadVoxtralRealtimeModel(from: path) }
        }
        print("[k88] STRICT LOAD \(loaded.count) packs ok, \(failed.count) failed: \(failed) ; not on disk: \(absent)")
        XCTAssertEqual(failed, [])
        XCTAssertGreaterThanOrEqual(loaded.count, 9, "the 9 packs of K-7 at least")
    }
}
