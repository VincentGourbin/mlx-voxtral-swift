/**
 * TTSOriginalPackLoadingTests — K-86: the original Mistral pack `tts-4b` loads its preset voices (`.pt` files written
 * by `torch.save`), and a TTS folder with N shards and no index loads every shard.
 *
 * The `.pt` fixtures are built here byte for byte like `torch.save` writes a bf16 tensor (zip of stored entries, the
 * same pickle opcodes as `voice_embedding/neutral_female.pt` of mistralai/Voxtral-4B-TTS-2603). Before K-86 the
 * pipeline read only `.safetensors` voices ("0 voices loaded"), and a folder of 3 shards without index threw
 * "No safetensors files found" (fixed two-shard fallback).
 * testRealPresetMatchesTheMLXPack (skipped when the packs are not downloaded): the real `.pt` voice equals the
 * `.safetensors` voice of the bf16 MLX pack converted from it.
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class TTSOriginalPackLoadingTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-k86-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // MARK: - torch.save fixture

    /// Pickle of `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, False, OrderedDict())`, as written
    /// by torch.save (protocol 2).
    private func tensorPickle(storage: String, rows: Int, cols: Int, offset: Int = 0, stride: [Int]? = nil) -> [UInt8] {
        var p: [UInt8] = [0x80, 0x02]
        func global(_ module: String, _ name: String) { p += [0x63] + Array("\(module)\n\(name)\n".utf8) }
        func unicode(_ s: String) {
            let b = Array(s.utf8); let n = UInt32(b.count)
            p += [0x58, UInt8(n & 0xFF), UInt8(n >> 8 & 0xFF), UInt8(n >> 16 & 0xFF), UInt8(n >> 24)] + b
        }
        func int(_ v: Int) {
            if v < 256 { p += [0x4B, UInt8(v)] } else if v < 65536 { p += [0x4D, UInt8(v & 0xFF), UInt8(v >> 8)] }
            else { p += [0x4A, UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v >> 16 & 0xFF), UInt8(v >> 24)] }
        }
        func put(_ i: UInt8) { p += [0x71, i] }
        global("torch._utils", "_rebuild_tensor_v2"); put(0)
        p += [0x28, 0x28]                                  // MARK MARK
        unicode("storage"); put(1); global("torch", storage); put(2); unicode("0"); put(3); unicode("cpu"); put(4)
        int(offset + rows * cols); p += [0x74]; put(5)     // numel, TUPLE
        p += [0x51]                                        // BINPERSID
        int(offset); int(rows); int(cols); p += [0x86]; put(6)
        let s = stride ?? [cols, 1]
        int(s[0]); int(s[1]); p += [0x86]; put(7)
        p += [0x89]                                        // NEWFALSE
        global("collections", "OrderedDict"); put(8); p += [0x29, 0x52]; put(9)  // EMPTY_TUPLE REDUCE
        p += [0x74]; put(10); p += [0x52]; put(11); p += [0x2E]                 // TUPLE REDUCE STOP
        return p
    }

    /// Zip archive of stored (uncompressed) entries, CRC left at 0 (the reader does not check it).
    private func storedZip(_ entries: [(String, [UInt8])], method: Int = 0) -> Data {
        var out: [UInt8] = [], central: [UInt8] = []
        func u16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)] }
        func u32(_ v: Int) -> [UInt8] { u16(v & 0xFFFF) + u16(v >> 16 & 0xFFFF) }
        for (name, bytes) in entries {
            let n = Array(name.utf8), offset = out.count
            out += u32(0x0403_4b50) + u16(20) + u16(0) + u16(0) + u16(0) + u16(0) + u32(0)
            out += u32(bytes.count) + u32(bytes.count) + u16(n.count) + u16(0) + n + bytes
            central += u32(0x0201_4b50) + u16(20) + u16(20) + u16(0) + u16(method) + u16(0) + u16(0) + u32(0)
            central += u32(bytes.count) + u32(bytes.count) + u16(n.count) + u16(0) + u16(0) + u16(0) + u16(0)
            central += u32(0) + u32(offset) + n
        }
        let start = out.count
        out += central
        out += u32(0x0605_4b50) + u16(0) + u16(0) + u16(entries.count) + u16(entries.count)
        out += u32(central.count) + u32(start) + u16(0)
        return Data(out)
    }

    /// bf16 values exactly representable, row-major [rows, cols].
    private func values(rows: Int, cols: Int) -> [Float] {
        (0 ..< rows * cols).map { Float($0 % 17) * 0.25 - 2 }
    }

    private func writeTorchVoice(_ url: URL, rows: Int, cols: Int, storage: String = "BFloat16Storage",
                                 method: Int = 0, stride: [Int]? = nil) throws {
        let raw = values(rows: rows, cols: cols).flatMap { v -> [UInt8] in
            let b = v.bitPattern >> 16
            return [UInt8(b & 0xFF), UInt8(b >> 8 & 0xFF)]
        }
        let zip = storedZip([
            ("voice_embed/data.pkl", tensorPickle(storage: storage, rows: rows, cols: cols, stride: stride)),
            ("voice_embed/byteorder", Array("little".utf8)),
            ("voice_embed/data/0", raw),
            ("voice_embed/version", Array("3\n".utf8)),
        ], method: method)
        try zip.write(to: url)
    }

    // MARK: - .pt voices

    func testTorchSaveTensorIsRead() throws {
        let url = try temporaryDirectory().appendingPathComponent("v.pt")
        try writeTorchVoice(url, rows: 5, cols: 6)
        let tensor = try TorchTensorReader.loadTensor(url: url)
        XCTAssertEqual(tensor.shape, [5, 6])
        XCTAssertEqual(tensor.dtype, .bfloat16)
        XCTAssertEqual(tensor.asType(.float32).asArray(Float.self), values(rows: 5, cols: 6))
    }

    func testPresetVoicesLoadFromPtFiles() throws {
        let dir = try temporaryDirectory()
        try writeTorchVoice(dir.appendingPathComponent("neutral_female.pt"), rows: 7, cols: 3072)
        try writeTorchVoice(dir.appendingPathComponent("fr_male.pt"), rows: 4, cols: 3072)
        let voices = try VoxtralTTSPipeline.loadVoiceEmbeddings(from: dir)
        XCTAssertEqual(Set(voices.keys), ["neutral_female", "fr_male"])
        XCTAssertEqual(voices["neutral_female"]?.shape, [7, 3072])
        XCTAssertEqual(voices["fr_male"]?.asType(.float32).asArray(Float.self), values(rows: 4, cols: 3072))
    }

    func testUnreadablePtFileNamesTheFormat() throws {
        let dir = try temporaryDirectory()
        let cases: [(String, (URL) throws -> Void)] = [
            ("not a zip", { try Data("not a zip archive at all, just text".utf8).write(to: $0) }),
            ("storage type", { try self.writeTorchVoice($0, rows: 2, cols: 3, storage: "LongStorage") }),
            ("non-contiguous", { try self.writeTorchVoice($0, rows: 2, cols: 3, stride: [1, 2]) }),
            ("compressed entry", { try self.writeTorchVoice($0, rows: 2, cols: 3, method: 8) }),
        ]
        for (label, write) in cases {
            let url = dir.appendingPathComponent("neutral_female.pt")
            try write(url)
            XCTAssertThrowsError(try VoxtralTTSPipeline.loadVoiceEmbeddings(from: dir), label) { error in
                let text = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                XCTAssertTrue(text.contains(".pt") && text.contains("neutral_female.pt"), "\(label): \(text)")
            }
        }
    }

    func testRealPresetMatchesTheMLXPack() throws {
        guard let original = VoxtralTTSRegistry.model(withId: "tts-4b").flatMap(VoxtralModelDownloader.findTTSModelPath(for:)),
              let mlx = VoxtralTTSRegistry.model(withId: "tts-4b-mlx").flatMap(VoxtralModelDownloader.findTTSModelPath(for:)) else {
            throw XCTSkip("tts-4b and tts-4b-mlx are not both downloaded")
        }
        let pt = try VoxtralTTSPipeline.loadVoiceEmbeddings(from: original.appendingPathComponent("voice_embedding"))
        let st = try VoxtralTTSPipeline.loadVoiceEmbeddings(from: mlx.appendingPathComponent("voice_embedding"))
        print("[k86] tts-4b: \(pt.count) voices from .pt ; tts-4b-mlx: \(st.count)")
        XCTAssertEqual(pt.count, VoxtralVoice.allCases.count)
        for (name, a) in pt {
            let b = try XCTUnwrap(st[name], name)
            XCTAssertEqual(a.shape, b.shape, name)
            XCTAssertTrue(MLX.arrayEqual(a.asType(.float32), b.asType(.float32)).item(Bool.self), name)
        }
    }

    // MARK: - Shards without index

    func testThreeShardsWithoutIndexAllLoad() throws {
        let dir = try temporaryDirectory()
        for shard in 1 ... 3 {
            try MLX.save(arrays: ["layers.\(shard).weight": MLXArray.ones([2, 2]) * Float(shard)],
                         url: dir.appendingPathComponent("model-0000\(shard)-of-00003.safetensors"))
        }
        let weights = try loadAllTTSWeights(from: dir)
        XCTAssertEqual(Set(weights.keys), ["layers.1.weight", "layers.2.weight", "layers.3.weight"])
        XCTAssertEqual(weights["layers.3.weight"]?.sum().item(Float.self), 12)
    }

    func testMissingShardWithoutIndexIsNamed() throws {
        let dir = try temporaryDirectory()
        for shard in [1, 3] {
            try MLX.save(arrays: ["layers.\(shard).weight": MLXArray.ones([2, 2])],
                         url: dir.appendingPathComponent("model-0000\(shard)-of-00003.safetensors"))
        }
        XCTAssertThrowsError(try loadAllTTSWeights(from: dir)) { error in
            XCTAssertTrue("\(error)".contains("model-00002-of-00003.safetensors"), "\(error)")
        }
    }
}
