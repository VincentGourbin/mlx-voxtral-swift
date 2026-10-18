/**
 * TorchTensorReader — reads one tensor saved by `torch.save` (K-86): the voice presets of the original `tts-4b` pack
 * (`.pt` files of `voice_embedding`, bf16 `[T, 3072]`), without Python.
 *
 * Format: a zip archive whose entries are stored uncompressed — `<prefix>/data.pkl` (a pickle that rebuilds the
 * tensor through `torch._utils._rebuild_tensor_v2`), `<prefix>/data/<key>` (the raw storage), `<prefix>/byteorder`.
 * The pickle is run by a minimal virtual machine that knows only the opcodes such a file uses; anything else (another
 * object, a compressed entry, a non-contiguous or big-endian tensor) throws an error that names the `.pt` file.
 */

import Foundation
import MLX

enum TorchTensorReader {

    /// Load the single tensor of a `torch.save` file.
    static func loadTensor(url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        func fail(_ why: String) -> VoxtralTTSError {
            .modelLoadingFailed("unsupported PyTorch .pt file \(url.lastPathComponent): \(why)")
        }
        let entries = try zipEntries(data, fail: fail)
        guard let pickleName = entries.keys.first(where: { $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }),
              let pickle = entries[pickleName] else {
            throw fail("no data.pkl entry (not a torch.save zip archive)")
        }
        let prefix = String(pickleName.dropLast("data.pkl".count))
        if let order = entries[prefix + "byteorder"], String(decoding: order, as: UTF8.self) != "little" {
            throw fail("byte order \(String(decoding: order, as: UTF8.self))")
        }
        let spec = try TensorSpec(pickle: try runPickle(pickle, fail: fail), fail: fail)
        guard let storage = entries[prefix + "data/" + spec.storageKey] else {
            throw fail("storage data/\(spec.storageKey) missing")
        }
        let count = spec.shape.reduce(1, *)
        let start = spec.offset * spec.itemSize
        let end = start + count * spec.itemSize
        guard end <= storage.count else {
            throw fail("storage holds \(storage.count) bytes, the tensor needs \(end)")
        }
        return MLXArray(storage.subdata(in: start ..< end), spec.shape, dtype: spec.dtype)
    }

    // MARK: - Zip (stored entries, central directory)

    private static func zipEntries(_ d: Data, fail: (String) -> VoxtralTTSError) throws -> [String: Data] {
        func u16(_ o: Int) -> Int { Int(d[d.startIndex + o]) | Int(d[d.startIndex + o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        // End of central directory: last 22 bytes, or earlier when the archive has a comment
        guard d.count >= 22 else { throw fail("file too short for a zip archive") }
        var eocd = d.count - 22
        while eocd >= 0 && u32(eocd) != 0x0605_4b50 { eocd -= 1 }
        guard eocd >= 0 else { throw fail("no zip end-of-central-directory record") }
        let entryCount = u16(eocd + 10)
        var p = u32(eocd + 16)
        var entries: [String: Data] = [:]
        for _ in 0 ..< entryCount {
            guard p + 46 <= d.count, u32(p) == 0x0201_4b50 else { throw fail("corrupt zip central directory") }
            let method = u16(p + 10), size = u32(p + 20), nameLength = u16(p + 28)
            let extraLength = u16(p + 30), commentLength = u16(p + 32), local = u32(p + 42)
            guard p + 46 + nameLength <= d.count else { throw fail("corrupt zip central directory") }
            let name = String(decoding: d[(d.startIndex + p + 46) ..< (d.startIndex + p + 46 + nameLength)], as: UTF8.self)
            guard size != 0xFFFF_FFFF, local != 0xFFFF_FFFF else { throw fail("zip64 archive") }
            guard method == 0 else { throw fail("entry \(name) is compressed (method \(method))") }
            guard local + 30 <= d.count, u32(local) == 0x0403_4b50 else { throw fail("corrupt zip local header") }
            let dataStart = local + 30 + u16(local + 26) + u16(local + 28)
            guard dataStart + size <= d.count else { throw fail("entry \(name) runs past the end of the file") }
            entries[name] = d.subdata(in: (d.startIndex + dataStart) ..< (d.startIndex + dataStart + size))
            p += 46 + nameLength + extraLength + commentLength
        }
        return entries
    }

    // MARK: - Pickle (the opcodes of a saved tensor)

    indirect enum Value: Equatable {
        case none, bool(Bool), int(Int), string(String)
        case tuple([Value]), list([Value]), dict
        case global(String, String)
        case persistent(Value)
        case reduce(Value, Value)
        case mark
    }

    static func runPickle(_ bytes: Data, fail: (String) -> VoxtralTTSError) throws -> Value {
        let b = [UInt8](bytes)
        var i = 0
        var stack: [Value] = []
        var memo: [Int: Value] = [:]
        func take(_ n: Int) throws -> ArraySlice<UInt8> {
            guard i + n <= b.count else { throw fail("truncated pickle") }
            defer { i += n }
            return b[i ..< i + n]
        }
        func le(_ s: ArraySlice<UInt8>) -> Int { s.reversed().reduce(0) { $0 << 8 | Int($1) } }
        func line() throws -> String {
            guard let end = b[i...].firstIndex(of: 0x0A) else { throw fail("truncated pickle") }
            defer { i = end + 1 }
            return String(decoding: b[i ..< end], as: UTF8.self)
        }
        func pop() throws -> Value {
            guard let v = stack.popLast() else { throw fail("pickle stack underflow") }
            return v
        }
        func popToMark() throws -> [Value] {
            guard let m = stack.lastIndex(of: .mark) else { throw fail("pickle MARK missing") }
            let items = Array(stack[(m + 1)...])
            stack.removeSubrange(m...)
            return items
        }
        while true {
            let op = try take(1).first!
            switch op {
            case 0x80: _ = try take(1)                                    // PROTO
            case 0x95: _ = try take(8)                                    // FRAME
            case 0x63: stack.append(.global(try line(), try line()))      // GLOBAL
            case 0x28: stack.append(.mark)                                // MARK
            case 0x58: stack.append(.string(String(decoding: try take(le(try take(4))), as: UTF8.self)))  // BINUNICODE
            case 0x8C: stack.append(.string(String(decoding: try take(le(try take(1))), as: UTF8.self)))  // SHORT_BINUNICODE
            case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: le(try take(4))))))  // BININT
            case 0x4B: stack.append(.int(le(try take(1))))                // BININT1
            case 0x4D: stack.append(.int(le(try take(2))))                // BININT2
            case 0x8A:                                                    // LONG1 (non-negative, ≤ 8 bytes)
                let n = le(try take(1))
                guard n <= 8 else { throw fail("integer too large") }
                stack.append(.int(le(try take(n))))
            case 0x4E: stack.append(.none)                                // NONE
            case 0x88: stack.append(.bool(true))                          // NEWTRUE
            case 0x89: stack.append(.bool(false))                         // NEWFALSE
            case 0x29: stack.append(.tuple([]))                           // EMPTY_TUPLE
            case 0x74: stack.append(.tuple(try popToMark()))              // TUPLE
            case 0x85: stack.append(.tuple([try pop()]))                  // TUPLE1
            case 0x86: let y = try pop(), x = try pop(); stack.append(.tuple([x, y]))            // TUPLE2
            case 0x87: let z = try pop(), y = try pop(), x = try pop(); stack.append(.tuple([x, y, z]))  // TUPLE3
            case 0x5D: stack.append(.list([]))                            // EMPTY_LIST
            case 0x7D: stack.append(.dict)                                // EMPTY_DICT
            case 0x75: _ = try popToMark()                                // SETITEMS (metadata dict, dropped)
            case 0x73: _ = try pop(); _ = try pop()                       // SETITEM
            case 0x51: stack.append(.persistent(try pop()))               // BINPERSID
            case 0x52: let args = try pop(), fn = try pop(); stack.append(.reduce(fn, args))     // REDUCE
            case 0x71: memo[le(try take(1))] = stack.last                 // BINPUT
            case 0x72: memo[le(try take(4))] = stack.last                 // LONG_BINPUT
            case 0x94: memo[memo.count] = stack.last                      // MEMOIZE
            case 0x68, 0x6A:                                              // BINGET, LONG_BINGET
                let key = le(try take(op == 0x68 ? 1 : 4))
                guard let v = memo[key] else { throw fail("pickle memo \(key) missing") }
                stack.append(v)
            case 0x2E: return try pop()                                   // STOP
            default: throw fail(String(format: "pickle opcode 0x%02X not supported", op))
            }
        }
    }

    // MARK: - Tensor

    struct TensorSpec {
        let storageKey: String
        let dtype: DType
        let itemSize: Int
        let offset: Int
        let shape: [Int]

        init(pickle: Value, fail: (String) -> VoxtralTTSError) throws {
            guard case .reduce(.global("torch._utils", "_rebuild_tensor_v2"), .tuple(let args)) = pickle, args.count >= 4,
                  case .persistent(.tuple(let pid)) = args[0], pid.count >= 5,
                  case .string("storage") = pid[0], case .global(_, let storageType) = pid[1],
                  case .string(let key) = pid[2],
                  case .int(let offset) = args[1], case .tuple(let size) = args[2], case .tuple(let stride) = args[3]
            else { throw fail("the pickle is not a single tensor") }
            let shape = try size.map { v -> Int in
                guard case .int(let n) = v else { throw fail("tensor size is not an integer tuple") }
                return n
            }
            let strides = try stride.map { v -> Int in
                guard case .int(let n) = v else { throw fail("tensor stride is not an integer tuple") }
                return n
            }
            var expected = 1
            for (n, s) in zip(shape, strides).reversed() {
                guard n == 1 || s == expected else { throw fail("non-contiguous tensor (stride \(strides))") }
                expected *= n
            }
            switch storageType {
            case "BFloat16Storage": (dtype, itemSize) = (.bfloat16, 2)
            case "HalfStorage": (dtype, itemSize) = (.float16, 2)
            case "FloatStorage": (dtype, itemSize) = (.float32, 4)
            default: throw fail("storage type \(storageType)")
            }
            self.storageKey = key
            self.offset = offset
            self.shape = shape
        }
    }
}
