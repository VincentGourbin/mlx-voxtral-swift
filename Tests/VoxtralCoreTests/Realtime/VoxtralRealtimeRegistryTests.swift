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
        XCTAssertEqual(VoxtralRealtimeRegistry.models.count, 3)
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
