/**
 * AttentionMaskTests - K-3 (P-02, P-17, MLX-019)
 *
 * The STT decoder's attention masks come from the cache (boolean, shaped like the keys it presents), not from a
 * hand-made additive fp32 mask: a bf16 model no longer fails with "Mask type must promote to output type", a
 * wrapped RotatingKVCache prefilled in two chunks gets the right shape. (The legacy-decoder case went with the
 * legacy loaders, removed in 3.0, K-31.)
 */

import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import VoxtralCore

final class AttentionMaskTests: XCTestCase {

    func testBF16PrefillDoesNotFailMaskPromotion() throws {
        let model = try makeReducedVoxtralModel()
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let inputIds = MLXArray((0 ..< 16).map { Int32(30 + $0) }).reshaped([1, 16])
        XCTAssertNoThrow(try model.generateStream(inputIds: inputIds, maxNewTokens: 1, memoryOptimization: .disabled))
    }

    func testMaskIsBooleanAndShapedLikeTheKeys() {
        let cache = KVCacheSimple()
        _ = cache.update(keys: MLXArray.zeros([1, 2, 5, 16]), values: MLXArray.zeros([1, 2, 5, 16]))
        let mask = LlamaStandardModel.causalMask(n: 3, cache: cache)
        XCTAssertEqual(mask?.dtype, .bool)
        XCTAssertEqual(mask?.shape, [3, 8])  // offset 5 + 3 new positions
        XCTAssertNil(LlamaStandardModel.causalMask(n: 1, cache: cache))
    }

    func testWrappedRotatingCacheTwoChunkPrefill() throws {
        let model = try makeReducedVoxtralModel()
        let language = try XCTUnwrap(model.language_model as? LlamaStandardModel)
        let caches: [any KVCache] = (0 ..< 2).map { _ in RotatingKVCache(maxSize: 8, keep: 0) }
        // 10 + 6 positions through a window of 8: at the 2nd chunk the wrapped cache presents fewer keys (13) than
        // offset + T (16), the width of the former hand-made mask (K-3, discriminating since 2026-10-03)
        let embeddings = MLXRandom.normal([1, 16, 64], key: MLXRandom.key(3))
        var keysShapes: [[Int]] = []
        try withMLXErrors { errors in
            for chunk in [embeddings[0..., 0 ..< 10, 0...], embeddings[0..., 10 ..< 16, 0...]] {
                let mask = try XCTUnwrap(LlamaStandardModel.causalMask(n: chunk.dim(1), cache: caches[0]))
                XCTAssertEqual(mask.dtype, .bool)
                let out = language(inputs: nil, mask: nil, cache: caches, inputsEmbeds: chunk)
                MLX.eval(out)
                try errors.check()
                keysShapes.append([mask.dim(0), mask.dim(1), caches[0].state[0].dim(2)])
            }
        }
        print("[mask] rotating chunks (mask rows, mask cols, keys) = \(keysShapes)")
        XCTAssertEqual(keysShapes, [[10, 10, 10], [6, 13, 13]])
        XCTAssertLessThan(keysShapes[1][2], 16, "the wrapped cache presents fewer keys than offset + T")
        XCTAssertEqual(keysShapes[1][1], keysShapes[1][2], "the mask spans the keys of the wrapped cache")
    }

}
