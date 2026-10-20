/**
 * TTS8bitPackTests — K-90 (ASK-19 = A): the 8-bit TTS pack exported by `Scripts/export_tts_8bit.py` loads with every
 * model parameter covered and no key the model lacks, LLM and flow-matching transformer in 8-bit g64 affine, codec bf16.
 *
 * `updateVerified` throws on a missing parameter but ignores a key the model does not have (K-88): the pack's keys are
 * compared with the model's here. Skipped when `tts-4b-8bit` is not on disk.
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

final class TTS8bitPackTests: XCTestCase {

    func testEightBitPackLoadsEveryWeight() throws {
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-8bit"))
        guard let directory = VoxtralModelDownloader.findTTSModelPath(for: info) else {
            throw XCTSkip("tts-4b-8bit is not downloaded")
        }
        let quantization = try XCTUnwrap(try loadQuantizationConfig(from: directory)?.defaults)
        XCTAssertEqual(quantization.bits, 8)
        XCTAssertEqual(quantization.groupSize, 64)
        XCTAssertEqual(quantization.mode, .affine)

        let model = try loadVoxtralTTSModel(from: directory)
        let pack = Set(sanitizeTTSWeights(try loadAllTTSWeights(from: directory)).keys)
        let own = Set(model.parameters().flattened().map { $0.0 })
        let missing = own.subtracting(pack).sorted(), unexpected = pack.subtracting(own).sorted()
        XCTAssertEqual(missing, [])
        XCTAssertEqual(unexpected, [])

        let quantized = model.leafModules().flattened().compactMap { path, module -> (String, Int)? in
            if let linear = module as? QuantizedLinear { return (path, linear.bits) }
            if let embedding = module as? QuantizedEmbedding { return (path, embedding.bits) }
            return nil
        }
        XCTAssertEqual(Set(quantized.map(\.1)), [8], "every quantized module in 8-bit")
        let roots = Set(quantized.map { String($0.0.split(separator: ".")[0]) })
        XCTAssertFalse(roots.isEmpty)
        XCTAssertFalse(roots.contains { $0.contains("codec") || $0.contains("audio_tokenizer") }, "codec stays bf16: \(roots)")
        let size = try XCTUnwrap(
            try FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent("model.safetensors").path)[.size]
                as? Int64)
        XCTAssertEqual(size, info.approximateBytes)
        print("[k90] LOAD tts-4b-8bit : missing \(missing.count), unexpected \(unexpected.count), size \(size); "
              + "quantized modules \(quantized.count) in \(roots.sorted())")
    }
}
