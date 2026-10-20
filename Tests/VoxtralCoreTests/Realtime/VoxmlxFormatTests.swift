/**
 * VoxmlxFormatTests — K-91 (P-70, ASK-20 = A): the Realtime loader reads the voxmlx layout (format C) and refuses,
 * with `VoxtralRealtimePackError`, a pack whose keys do not cover the model or hold keys it does not have.
 *
 * Before K-91 a voxmlx pack (`mlx-community/Voxtral-Mini-4B-Realtime-6bit`, `ellamind/…-8bit-mlx`) was not read, an
 * uncovered pack threw an untyped error naming no format, and keys the model lacks were ignored.
 * Fixtures: a small random model (8-bit, group 64, like the packs), saved under voxmlx names or mlx-audio names, with a
 * params.json-style config.json (voxmlx writes one, `quantization` included).
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

final class VoxmlxFormatTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("voxmlx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func configJSON(quantized: Bool) -> String {
        """
        {"dim": 64, "n_layers": 1, "head_dim": 32, "hidden_dim": 128, "n_heads": 2, "n_kv_heads": 1,
         "vocab_size": 256, "norm_eps": 1e-5, "rope_theta": 1000000.0, "sliding_window": 64, "tied_embeddings": true,
         "ada_rms_norm_t_cond": true, "ada_rms_norm_t_cond_dim": 32,
         "multimodal": {"whisper_model_args": {
           "encoder_args": {"audio_encoding_args": {"sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 128,
             "hop_length": 160, "window_size": 400, "chunk_length_s": null, "global_log_mel_max": 1.5,
             "transcription_format": "streaming"},
             "dim": 64, "n_layers": 1, "head_dim": 32, "hidden_dim": 128, "n_heads": 2, "n_kv_heads": 2,
             "use_biases": true, "rope_theta": 1000000.0, "norm_eps": 1e-5, "sliding_window": 64},
           "downsample_args": {"downsample_factor": 4}}}\(quantized ? ", \"quantization\": {\"group_size\": 64, \"bits\": 8}" : "")}
        """
    }

    /// The model's own names → voxmlx names (independent of the loader's table)
    private func voxmlxName(_ key: String) -> String {
        var k = key
        let prefixes: [(String, String)] = [
            ("encoder.transformer_layers.", "encoder.layers."), ("encoder.conv_layers_0_conv.conv.", "encoder.conv1."),
            ("encoder.conv_layers_1_conv.conv.", "encoder.conv2."), ("encoder.transformer_norm.", "encoder.norm."),
            ("encoder.audio_language_projection_0.", "adapter.w_in."),
            ("encoder.audio_language_projection_2.", "adapter.w_out."),
            ("decoder.tok_embeddings.", "language_model.embed_tokens."), ("decoder.norm.", "language_model.norm."),
            ("decoder.layers.", "language_model.layers."),
        ]
        for (from, to) in prefixes where k.hasPrefix(from) {
            k = to + k.dropFirst(from.count)
            break
        }
        let parts: [(String, String)] = [
            (".attention.wq.", ".attention.q_proj."), (".attention.wk.", ".attention.k_proj."),
            (".attention.wv.", ".attention.v_proj."), (".attention.wo.", ".attention.o_proj."),
            (".attention_norm.", ".attn_norm."), (".feed_forward_w1.", ".mlp.gate_proj."),
            (".feed_forward_w2.", ".mlp.down_proj."), (".feed_forward_w3.", ".mlp.up_proj."),
            (".ada_rms_norm_t_cond.ada_down.", ".ada_norm.linear_in."),
            (".ada_rms_norm_t_cond.ada_up.", ".ada_norm.linear_out."),
        ]
        return parts.reduce(k) { $0.replacingOccurrences(of: $1.0, with: $1.1) }
    }

    /// The model's own names → mlx-audio names (decoder attention in Mistral naming)
    private func mlxAudioName(_ key: String) -> String {
        guard key.hasPrefix("decoder.layers.") else { return key }
        return [("q", "wq"), ("k", "wk"), ("v", "wv"), ("o", "wo")].reduce(key) {
            $0.replacingOccurrences(of: ".attention.\($1.0)_proj.", with: ".attention.\($1.1).")
        }
    }

    /// Writes a pack of a small random model; returns the weights under the model's own names
    @discardableResult
    private func writePack(
        quantized: Bool, name: (String) -> String, edit: (inout [String: MLXArray]) -> Void = { _ in }
    ) throws -> [String: MLXArray] {
        try configJSON(quantized: quantized).write(
            to: directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let model = VoxtralRealtimeModel(config: try loadRealtimeConfig(from: directory))
        if quantized {
            // voxmlx quantizes every Linear and Embedding it can (input width a multiple of the group)
            MLXNN.quantize(model: model) { _, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                if let linear = module as? Linear, linear.weight.dim(-1) % 64 == 0 { return (64, 8, .affine) }
                if let embedding = module as? Embedding, embedding.weight.dim(-1) % 64 == 0 { return (64, 8, .affine) }
                return nil
            }
        }
        let own = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var pack = Dictionary(uniqueKeysWithValues: own.map { (name($0.key), $0.value) })
        edit(&pack)
        try MLX.save(arrays: pack, url: directory.appendingPathComponent("model.safetensors"))
        return own
    }

    private func packError(_ body: () throws -> Void) -> VoxtralRealtimePackError? {
        do {
            try body()
            return nil
        } catch {
            return error as? VoxtralRealtimePackError
        }
    }

    // MARK: - Format C

    func testVoxmlxPackLoadsEveryWeight() throws {
        let own = try writePack(quantized: true, name: voxmlxName)
        XCTAssertNotNil(own["decoder.tok_embeddings.scales"], "fixture: tied head quantized, as in the voxmlx packs")
        XCTAssertNotNil(own["decoder.layers.0.ada_rms_norm_t_cond.ada_down.scales"])
        XCTAssertNil(own["decoder.layers.0.ada_rms_norm_t_cond.ada_up.scales"], "32 columns: not quantizable")
        let model = try loadVoxtralRealtimeModel(from: directory)
        XCTAssertTrue(model.decoder.tokEmbeddings is QuantizedEmbedding)
        let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(Set(loaded.keys), Set(own.keys))
        let different = own.filter { key, value in
            guard let other = loaded[key], other.shape == value.shape else { return true }
            return !MLX.arrayEqual(other, value).item(Bool.self)
        }.map(\.key).sorted()
        XCTAssertEqual(different, [], "every parameter read from the voxmlx pack")
        print("[k91] voxmlx fixture: \(own.count) parameters loaded, \(different.count) different")
    }

    func testVoxmlxNamesMapToTheModel() {
        let weights = [
            "adapter.w_in.weight": "encoder.audio_language_projection_0.weight",
            "encoder.conv2.bias": "encoder.conv_layers_1_conv.conv.bias",
            "encoder.layers.3.attention.o_proj.bias": "encoder.transformer_layers.3.attention.wo.bias",
            "encoder.layers.3.mlp.down_proj.scales": "encoder.transformer_layers.3.feed_forward_w2.scales",
            "encoder.layers.3.attn_norm.weight": "encoder.transformer_layers.3.attention_norm.weight",
            "language_model.layers.7.attention.k_proj.biases": "decoder.layers.7.attention.k_proj.biases",
            "language_model.layers.7.mlp.gate_proj.weight": "decoder.layers.7.feed_forward_w1.weight",
            "language_model.layers.7.mlp.up_proj.weight": "decoder.layers.7.feed_forward_w3.weight",
            "language_model.layers.7.ada_norm.linear_in.scales": "decoder.layers.7.ada_rms_norm_t_cond.ada_down.scales",
            "language_model.layers.7.ada_norm.linear_out.weight": "decoder.layers.7.ada_rms_norm_t_cond.ada_up.weight",
            "language_model.embed_tokens.biases": "decoder.tok_embeddings.biases",
            "language_model.norm.weight": "decoder.norm.weight",
            "encoder.norm.weight": "encoder.transformer_norm.weight",
        ]
        let sanitized = sanitizeVoxmlxWeights(weights.mapValues { _ in MLXArray(0) })
        XCTAssertEqual(Set(sanitized.keys), Set(weights.values))
        XCTAssertEqual(RealtimePackFormat.detect(weights.keys), .voxmlx)
    }

    // MARK: - Explicit refusal

    func testTruncatedVoxmlxPackIsRefused() throws {
        try writePack(quantized: true, name: voxmlxName) { pack in
            pack["language_model.layers.0.attention.q_proj.scales"] = nil
            pack["adapter.w_out.weight"] = nil
        }
        let error = packError { _ = try loadVoxtralRealtimeModel(from: directory) }
        XCTAssertEqual(error?.format, "voxmlx")
        XCTAssertEqual(error?.missing, ["encoder.audio_language_projection_2.weight"])
        // Without its scales the layer is not quantized (the loader follows the pack's scales): its biases are extra
        XCTAssertEqual(error?.unexpected, ["decoder.layers.0.attention.q_proj.biases"])
        print("[k91] truncated voxmlx pack: \(error.map { "\($0)" } ?? "loaded without error")")
    }

    func testUnknownFormatIsRefused() throws {
        try writePack(quantized: false, name: { "model." + $0 })
        let error = packError { _ = try loadVoxtralRealtimeModel(from: directory) }
        XCTAssertEqual(error?.format, "unknown")
        XCTAssertFalse(error?.unexpected.isEmpty ?? true, "the first keys of the pack are named")
        print("[k91] unknown pack: \(error?.errorDescription ?? "loaded without error")")
    }

    /// A key the model does not have was ignored before K-91: the pack loaded with part of its weights unread
    func testUnexpectedKeyIsRefused() throws {
        try writePack(quantized: false, name: mlxAudioName) { pack in
            pack["decoder.layers.0.attention.q_norm.weight"] = MLXArray.ones([32])
        }
        let error = packError { _ = try loadVoxtralRealtimeModel(from: directory) }
        XCTAssertEqual(error?.format, "mlx-audio")
        XCTAssertEqual(error?.missing, [])
        XCTAssertEqual(error?.unexpected, ["decoder.layers.0.attention.q_norm.weight"])
    }

    func testMLXAudioPackStillLoads() throws {
        try writePack(quantized: false, name: mlxAudioName)
        XCTAssertNoThrow(try loadVoxtralRealtimeModel(from: directory))
    }
}
