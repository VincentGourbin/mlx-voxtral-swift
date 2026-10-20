/**
 * VoxtralRealtimeRegistryTests - Tests for Realtime model registry
 */

import XCTest
@testable import VoxtralCore

final class VoxtralRealtimeRegistryTests: XCTestCase {

    func testRegistryNotEmpty() {
        XCTAssertFalse(VoxtralRealtimeRegistry.models.isEmpty)
    }

    func testRegistryHasExpectedCount() {
        XCTAssertEqual(VoxtralRealtimeRegistry.models.count, 5)
    }

    func testAllModelsHaveRequiredFields() {
        for model in VoxtralRealtimeRegistry.models {
            XCTAssertFalse(model.id.isEmpty)
            XCTAssertFalse(model.repoId.isEmpty)
            XCTAssertFalse(model.name.isEmpty)
            XCTAssertFalse(model.description.isEmpty)
        }
    }

    func testAllIdsUnique() {
        let ids = VoxtralRealtimeRegistry.models.map { $0.id }
        XCTAssertEqual(ids.count, Set(ids).count)
    }

    func testLookupById() {
        let model = VoxtralRealtimeRegistry.model(withId: "realtime-4b-4bit")
        XCTAssertNotNil(model)
        XCTAssertEqual(model?.repoId, "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
    }

    /// voxmlx packs referenced at a pinned revision (the downloader checks each file's SHA-256 at that revision; K-91)
    func testVoxmlxPacksArePinned() {
        let six = VoxtralRealtimeRegistry.model(withId: "realtime-4b-6bit")
        XCTAssertEqual(six?.repoId, "mlx-community/Voxtral-Mini-4B-Realtime-6bit")
        XCTAssertEqual(six?.revision, "02eb0caeb9dafb554c17a72b93dbf40cd3736c31")
        XCTAssertEqual(six?.approximateBytes, 3_609_304_614)
        let eight = VoxtralRealtimeRegistry.model(withId: "realtime-4b-8bit")
        XCTAssertEqual(eight?.repoId, "ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx")
        XCTAssertEqual(eight?.revision, "50e90282809c024a2b7b9b4718882e1dd69efb69")
        XCTAssertEqual(eight?.approximateBytes, 4_714_618_595)
        XCTAssertEqual(VoxtralRealtimeRegistry.defaultModel.id, "realtime-4b-4bit", "default unchanged")
    }

    func testLookupByIdFP16() {
        let model = VoxtralRealtimeRegistry.model(withId: "realtime-4b-fp16")
        XCTAssertNotNil(model)
    }

    func testLookupByIdOriginal() {
        let model = VoxtralRealtimeRegistry.model(withId: "realtime-4b")
        XCTAssertNotNil(model)
        XCTAssertEqual(model?.repoId, "mistralai/Voxtral-Mini-4B-Realtime-2602")
    }

    func testLookupByIdNotFound() {
        XCTAssertNil(VoxtralRealtimeRegistry.model(withId: "nonexistent"))
    }

    func testDefaultModelIsRecommended() {
        let defaultModel = VoxtralRealtimeRegistry.defaultModel
        XCTAssertTrue(defaultModel.recommended)
        XCTAssertEqual(defaultModel.id, "realtime-4b-4bit")
    }

    func testAllModelsAre4B() {
        for model in VoxtralRealtimeRegistry.models {
            XCTAssertEqual(model.parameters, "4B")
        }
    }
}
