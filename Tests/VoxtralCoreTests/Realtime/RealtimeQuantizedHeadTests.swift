/**
 * RealtimeQuantizedHeadTests — K-46 (P-61 step 2): the Realtime tied head quantized at load (8 or 4 bits,
 * `VOXTRAL_REALTIME_HEAD_BITS`), logits by `QuantizedEmbedding.asLinear` (`quantizedMM`), one row de-quantized per
 * token embedding; the compute dtype is still read as the 16-bit one (K-38 note: the packed table is uint32).
 *
 * Small random decoder in float16, like the packs' table.
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class RealtimeQuantizedHeadTests: XCTestCase {

    private func decoder() throws -> VoxtralRealtimeDecoder {
        let json = """
        {"dim": 128, "n_layers": 1, "head_dim": 32, "hidden_dim": 256, "n_heads": 4, "n_kv_heads": 2,
         "vocab_size": 512, "norm_eps": 1e-5, "rope_theta": 1000000, "sliding_window": 64,
         "tied_embeddings": true, "ada_rms_norm_t_cond": false, "ada_rms_norm_t_cond_dim": 32}
        """
        let config = try JSONDecoder().decode(RealtimeDecoderConfig.self, from: Data(json.utf8))
        let decoder = VoxtralRealtimeDecoder(config: config)
        MLXRandom.seed(46)
        decoder.update(parameters: decoder.parameters().mapValues { (MLXRandom.normal($0.shape) * 0.05).asType(.float16) })
        return decoder
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    /// Without the switch the head is the 16-bit table, bit for bit the former formulas.
    func testUnquantizedHeadMatchesTheFormerFormulas() throws {
        let decoder = try decoder()
        let table = decoder.tokEmbeddings.weight
        let h = (MLXRandom.normal([3, 128], key: MLXRandom.key(1)) * 0.5).asType(.float16)
        XCTAssertEqual(maxAbs(decoder.logits(h), MLX.matmul(h, table.transposed())), 0)
        XCTAssertEqual(maxAbs(decoder.embedToken(17), table[17]), 0)
        XCTAssertEqual(decoder.embedToken(17).shape, [128])
    }

    func testQuantizedHeadKeepsTheComputeDTypeAndApproximatesTheTable() throws {
        for bits in [8, 4] {
            let decoder = try decoder()
            let table = decoder.tokEmbeddings.weight
            let h = (MLXRandom.normal([3, 128], key: MLXRandom.key(1)) * 0.5).asType(.float16)
            let reference = MLX.matmul(h, table.transposed())
            quantizeRealtimeHead(decoder, bits: bits)
            let head = try XCTUnwrap(decoder.tokEmbeddings as? QuantizedEmbedding, "\(bits) bits")
            XCTAssertEqual(head.bits, bits)
            XCTAssertEqual(head.weight.dtype, .uint32)
            XCTAssertEqual(realtimeComputeDType(decoder), .float16, "\(bits) bits: dtype read on the scales")
            XCTAssertFalse(decoder.parameters().flattened().contains { $0.1.dtype == .float32 })
            // Logits by quantizedMM, in the compute dtype, close to the 16-bit table
            let logits = decoder.logits(h)
            XCTAssertEqual(logits.shape, reference.shape)
            XCTAssertEqual(logits.dtype, .float16)
            let relative = maxAbs(logits, reference) / MLX.max(MLX.abs(reference)).item(Float.self)
            print("[k46] head \(bits) bits: logits |Δ|max / max = \(relative)")
            XCTAssertLessThan(relative, bits == 8 ? 0.01 : 0.15)
            // One row de-quantized: the quantized table's own row
            let row = decoder.embedToken(17)
            XCTAssertEqual(row.shape, [128])
            XCTAssertEqual(row.dtype, .float16)
            let dequantized = MLX.dequantized(head.weight, scales: head.scales, biases: head.biases,
                                              groupSize: head.groupSize, bits: bits)
            XCTAssertEqual(maxAbs(row, dequantized[17]), 0)
            XCTAssertEqual(maxAbs(decoder.embedTokens(MLXArray([Int32(17)]))[0], row), 0)
            // Idempotent
            quantizeRealtimeHead(decoder, bits: bits)
            XCTAssertTrue(decoder.tokEmbeddings === head)
        }
    }

    func testHeadBitsSwitch() {
        XCTAssertNil(realtimeHeadBits([:]))
        XCTAssertEqual(realtimeHeadBits(["VOXTRAL_REALTIME_HEAD_BITS": "8"]), 8)
        XCTAssertEqual(realtimeHeadBits(["VOXTRAL_REALTIME_HEAD_BITS": "4"]), 4)
        XCTAssertNil(realtimeHeadBits(["VOXTRAL_REALTIME_HEAD_BITS": "6"]))
        XCTAssertNil(realtimeHeadBits(["VOXTRAL_REALTIME_HEAD_BITS": "x"]))
    }
}
