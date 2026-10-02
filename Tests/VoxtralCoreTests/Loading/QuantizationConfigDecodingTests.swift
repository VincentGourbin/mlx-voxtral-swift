/**
 * QuantizationConfigDecodingTests - K-8
 *
 * The quantization block is decoded like MLXLMCommon (mode, per-layer, `false`, metadata keys) by the three loaders.
 * Fixtures: the config.json of five packs (Tests/VoxtralCoreTests/Fixtures/quantization/SOURCES.md) and a synthetic
 * mxfp4 one. Non-affine modes load as experimental (ASK-21 = B) and reach `quantize`; an NVFP4 global scale, absent
 * from mlx-swift 0.31.6, is refused. TTS: `quantization_config` read when `quantization` is absent; shards from the
 * index.
 */

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest
@testable import VoxtralCore

final class QuantizationConfigDecodingTests: XCTestCase {

    private let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/quantization")

    private func data(_ name: String) throws -> Data {
        try Data(contentsOf: fixtures.appendingPathComponent("\(name).json"))
    }

    private func decode(_ name: String) throws -> PackQuantization {
        try XCTUnwrap(PackQuantization.decode(
            configData: data(name), source: name, makeError: VoxtralError.invalidConfiguration))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("k8-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // mzbac 4 b mixed: 4 b g64 by default, encoder in 6 b, skipped convolutions, lm_head 6 b g128
    func testMzbacMixed() throws {
        let q = try decode("mzbac_voxtral-mini-3b-4bit-mixed")
        XCTAssertEqual(q.defaults, BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        XCTAssertEqual(q.quantization(layer: "audio_tower.layers.0.self_attn.k_proj")?.bits, 6)
        XCTAssertNil(q.quantization(layer: "audio_tower.conv1"))
        XCTAssertEqual(q.quantization(layer: "lm_head").map { [$0.groupSize, $0.bits] }, [128, 6])
        XCTAssertEqual(q.modes, [.affine])
        XCTAssertNoThrow(try JSONDecoder().decode(VoxtralStandardConfiguration.self, from: data("mzbac_voxtral-mini-3b-4bit-mixed")))
    }

    // VincentGOURBIN 8 b: per-layer `true` takes the defaults
    func testVincentGourbin8bit() throws {
        let q = try decode("VincentGOURBIN_voxtral-small-8bit")
        XCTAssertEqual(q.defaults, BaseConfiguration.Quantization(groupSize: 64, bits: 8))
        XCTAssertEqual(q.quantization(layer: "audio_tower.layers.0.self_attn.k_proj"), q.defaults)
        XCTAssertNil(q.quantization(layer: "audio_tower.embed_positions"))
    }

    // aufklarer: `"mode": "affine"` no longer fails the whole config
    func testAufklarerMode() throws {
        let q = try decode("aufklarer_Voxtral-Mini-3B-2507-MLX-8bit")
        XCTAssertEqual(q.defaults?.mode, .affine)
        XCTAssertEqual(q.defaults?.bits, 8)
        let configuration = try JSONDecoder().decode(
            VoxtralStandardConfiguration.self, from: data("aufklarer_Voxtral-Mini-3B-2507-MLX-8bit"))
        XCTAssertNotNil(configuration.quantization)
    }

    // Markus: per-layer LM entries + mode; the bf16 encoder has no scales, so it stays unquantized
    func testMarkusPerLayerMode() throws {
        let q = try decode("MarkusKaemmerer_Voxtral-Mini-3B-2507-8bit-dense-encoder")
        XCTAssertEqual(q.quantization(layer: "language_model.embed_tokens")?.bits, 8)
        XCTAssertNoThrow(try JSONDecoder().decode(
            VoxtralStandardConfiguration.self, from: data("MarkusKaemmerer_Voxtral-Mini-3B-2507-8bit-dense-encoder")))
        let modules = detectQuantizedModules(
            weightKeys: ["language_model.layers.0.self_attn.q_proj.scales", "audio_tower.layers.0.fc1.weight"],
            quantization: q)
        XCTAssertEqual(modules.count, 1)
        XCTAssertEqual(modules.values.first?.bits, 8)
    }

    // mxfp4 (synthetic): decoded, experimental, and the mode reaches quantize (no longer forced to affine)
    func testMXFP4ReachesQuantize() throws {
        let q = try decode("synthetic_mxfp4")
        XCTAssertEqual(q.modes, [.mxfp4])
        let configuration = try JSONDecoder().decode(VoxtralStandardConfiguration.self, from: Data("""
            {"model_type": "voxtral", "audio_token_id": 24, "projector_hidden_act": "gelu",
             "text_config": {"vocab_size": 128, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
               "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 512,
               "rms_norm_eps": 1e-5, "rope_theta": 1000000.0, "hidden_act": "silu", "attention_bias": false,
               "mlp_bias": false},
             "audio_config": {"hidden_size": 32, "intermediate_size": 128, "num_hidden_layers": 1,
               "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16, "max_source_positions": 1500,
               "num_mel_bins": 128, "vocab_size": 128}}
            """.utf8))
        let model = VoxtralStandardModel(configuration: configuration)
        let weights = ["language_model.layers.0.self_attn.q_proj.scales": MLXArray.zeros([1])]
        _ = loadQuantizedVoxtral(model: model, weights: weights, quantization: q)
        let quantized = model.leafModules().flattened().compactMap { $0.1 as? QuantizedLinear }
        XCTAssertEqual(quantized.count, 1)
        XCTAssertEqual(quantized.first?.mode, .mxfp4)
        XCTAssertEqual(quantized.first?.groupSize, 32)
    }

    // Refused explicitly: an unknown mode, and an NVFP4 global scale (absent from mlx-swift 0.31.6)
    func testUnsupportedQuantizationThrows() throws {
        let unknown = Data(#"{"quantization": {"group_size": 32, "bits": 4, "mode": "nvfp9"}}"#.utf8)
        XCTAssertThrowsError(try PackQuantization.decode(
            configData: unknown, source: "nvfp9", makeError: VoxtralError.invalidConfiguration)) { error in
            guard case VoxtralError.invalidConfiguration(let message) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(message.contains("mode"), message)
        }
        XCTAssertThrowsError(try PackQuantization.checkSupported(
            weightKeys: ["layers.0.q_proj.weight", "layers.0.q_proj.global_scale"], source: "nvfp4",
            makeError: VoxtralError.invalidConfiguration)) { error in
            guard case VoxtralError.invalidConfiguration(let message) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(message.contains("global_scale"), message)
        }
    }

    // TTS: `quantization_config` only (majentik) is read
    func testTTSQuantizationConfigOnly() throws {
        let directory = try temporaryDirectory()
        try data("majentik_Voxtral-4B-TTS-2603-TurboQuant-MLX-8bit")
            .write(to: directory.appendingPathComponent("config.json"))
        let q = try XCTUnwrap(loadQuantizationConfig(from: directory))
        XCTAssertEqual(q.defaults, BaseConfiguration.Quantization(groupSize: 64, bits: 8))
    }

    // TTS: three shards named by the index are all loaded
    func testTTSThreeShardsFromIndex() throws {
        let directory = try temporaryDirectory()
        var weightMap: [String: String] = [:]
        for shard in 1 ... 3 {
            let name = "model-0000\(shard)-of-00003.safetensors"
            let key = "layers.\(shard).weight"
            try MLX.save(arrays: [key: MLXArray.ones([2, 2]) * Float(shard)], url: directory.appendingPathComponent(name))
            weightMap[key] = name
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let weights = try loadAllTTSWeights(from: directory)
        XCTAssertEqual(Set(weights.keys), Set(weightMap.keys))
        XCTAssertEqual(weights["layers.3.weight"]?.sum().item(Float.self), 12)
    }
}

/// Real pack `MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder` (per-layer + `mode`): loaded with
/// `verify: [.all]` and transcribed greedy on C-court EN. Skipped unless VOXTRAL_K8_PACK_DIR points to the pack.
final class QuantizedThirdPartyPackTests: XCTestCase {

    private var packDirectory: URL? {
        ProcessInfo.processInfo.environment["VOXTRAL_K8_PACK_DIR"].map { URL(fileURLWithPath: $0) }
    }

    func testLoadsWithEveryKeyVerified() throws {
        guard let directory = packDirectory else { throw XCTSkip("Set VOXTRAL_K8_PACK_DIR") }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(VoxtralStandardConfiguration.self, from: configData)
        let quantization = try XCTUnwrap(PackQuantization.decode(
            configData: configData, source: "config.json", makeError: VoxtralError.invalidConfiguration))
        let model = VoxtralStandardModel(configuration: configuration)
        let weights = try loadWeights(from: directory)
        _ = loadQuantizedVoxtral(model: model, weights: weights, quantization: quantization)
        let sanitized = removingEmbeddingAlias(try model.sanitize(weights))

        let expected = Set(model.parameters().flattened().map(\.0))
        let provided = Set(sanitized.keys)
        let missing = expected.subtracting(provided), unused = provided.subtracting(expected)
        print("[K-8] LOAD MarkusKaemmerer/…-8bit-dense-encoder verify [.all] : \(missing.count) missing, "
              + "\(unused.count) unused (\(expected.count) keys)")
        XCTAssertNoThrow(try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all]))
        XCTAssertEqual(missing, [])
        XCTAssertEqual(unused, [])
    }

    @MainActor
    func testGreedyTranscription() async throws {
        guard let directory = packDirectory else { throw XCTSkip("Set VOXTRAL_K8_PACK_DIR") }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let pipeline = VoxtralPipeline(model: .mini3b8bit, backend: .mlx)
        pipeline.modelDirectoryOverride = directory
        try await pipeline.loadModel()
        let text = try await pipeline.transcribe(
            audio: root.appendingPathComponent("docs/examples/fluxforge_short_en_6bit.wav"), language: "en")
        print("[K-8] SWIFT: \(text)")
    }
}

