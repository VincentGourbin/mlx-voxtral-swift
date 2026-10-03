/**
 * RealtimeSlidingWindowTests - K-13 (P-62, P-63)
 *
 * The Realtime encoder attends within its sliding window (750 positions = 15 s in the real model),
 * like the mlx-audio reference: beyond the window it must equal a full-sequence pass with a banded
 * causal mask, not a full causal attention (which diverges past 15 s and costs O(n²)).
 * A tiny encoder (window 8) keeps the check fast.
 */

import Foundation
import MLX
import MLXRandom
import XCTest
@testable import VoxtralCore

final class RealtimeSlidingWindowTests: XCTestCase {

    private let window = 8

    private func tinyEncoder() throws -> VoxtralRealtimeEncoder {
        let json = """
        {"dim": 32, "n_layers": 2, "head_dim": 8, "hidden_dim": 64, "n_heads": 4, "n_kv_heads": 4,
         "use_biases": true, "rope_theta": 1000000.0, "norm_eps": 1e-5, "sliding_window": \(window),
         "downsample_factor": 4,
         "audio_encoding_args": {"sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 16, "hop_length": 160,
                                 "window_size": 400, "global_log_mel_max": 1.5, "transcription_format": "streaming"}}
        """
        let config = try JSONDecoder().decode(RealtimeEncoderConfig.self, from: Data(json.utf8))
        MLXRandom.seed(13)
        return VoxtralRealtimeEncoder(config: config, decoderDim: 32)
    }

    /// Full-sequence pass with a banded causal mask: query i sees keys i-window+1 … i
    private func bandedReference(_ encoder: VoxtralRealtimeEncoder, mel: MLXArray) -> MLXArray {
        let convOut = encoder.convStem(mel)
        let n = convOut.dim(0)
        let (cos, sin) = computeRoPEFreqs(positions: MLXArray(0 ..< Int32(n)), headDim: 8, theta: 1_000_000)
        var maskValues = [Float](repeating: -Float.infinity, count: n * n)
        for i in 0 ..< n {
            for j in max(0, i - window + 1) ... i { maskValues[i * n + j] = 0 }
        }
        let mask = MLXArray(maskValues, [n, n])
        var x = convOut
        for layer in encoder.layers {
            x = layer(x, ropeCos: cos, ropeSin: sin, mask: mask)
        }
        return encoder.downsampleAndProject(encoder.transformerNorm(x))
    }

    private func relativeL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        (MLX.sqrt(MLX.sum((a - b) * (a - b))) / MLX.sqrt(MLX.sum(b * b))).item(Float.self)
    }

    func testBeyondWindowMatchesBandedAttention() throws {
        let encoder = try tinyEncoder()
        let mel = MLXRandom.normal([16, 80])  // 40 conv positions = 5 windows of 8
        let out = encoder(mel)
        let ref = bandedReference(encoder, mel: mel)
        XCTAssertEqual(out.shape, ref.shape)
        let l2 = relativeL2(out, ref)
        print("[sliding-window] beyond window: L2 rel = \(l2)")
        XCTAssertLessThan(l2, 1e-3, "the encoder must attend within its sliding window")
    }

    func testWithinWindowUnchanged() throws {
        let encoder = try tinyEncoder()
        let mel = MLXRandom.normal([16, 16])  // 8 conv positions = one window
        let l2 = relativeL2(encoder(mel), bandedReference(encoder, mel: mel))
        print("[sliding-window] within window: L2 rel = \(l2)")
        XCTAssertLessThan(l2, 1e-3)
    }

    /// Within the window, the chunked path (one chunk, rotating cache) equals `encodeFull`
    func testChunkedEqualsFullWithinWindow() throws {
        let encoder = try tinyEncoder()
        let convOut = encoder.convStem(MLXRandom.normal([16, 16]))
        let chunked = encoder.downsampleAndProject(encoder.encodeChunked(convOut))
        let l2 = relativeL2(chunked, encoder.encodeFull(convOut))
        print("[sliding-window] EQUIV within window: L2 rel = \(l2)")
        XCTAssertLessThan(l2, 1e-3)
    }

    // The conv stem evaluated in slices (K-15: cancellable on long audio) equals the one-pass conv stem, for slice
    // sizes that do and do not divide the length
    func testChunkedConvStemMatchesOnePass() throws {
        let encoder = try tinyEncoder()
        let mel = MLXRandom.normal([16, 98])
        let reference = encoder.convStem(mel)
        for chunk in [2, 6, 10, 98, 200] {
            let chunked = try XCTUnwrap(encoder.convStemChunked(mel, chunkFrames: chunk))
            XCTAssertEqual(chunked.shape, reference.shape, "chunk \(chunk)")
            let l2 = relativeL2(chunked, reference)
            print("[conv-stem] chunk \(chunk): L2 rel = \(l2)")
            XCTAssertLessThan(l2, 1e-6, "chunk \(chunk)")
        }
    }
}
