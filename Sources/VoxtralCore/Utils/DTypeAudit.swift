/**
 * DTypeAudit - `VOXTRAL_DTYPE_AUDIT=1` prints the dtype of the arrays that decide the compute
 * precision (mel, encoder output, KV cache, logits): quick evidence for fp32 leaks (P-01, P-60).
 */

import Foundation
import MLX
import MLXLMCommon

enum DTypeAudit {
    static var isEnabled: Bool { ProcessInfo.processInfo.environment["VOXTRAL_DTYPE_AUDIT"] == "1" }

    static func report(_ pipeline: String, _ label: String, _ array: MLXArray?) {
        guard isEnabled, let array else { return }
        VoxtralDebug.console("DTYPE \(pipeline) \(label)=\(array.dtype) shape=\(array.shape)")
    }

    static func report(_ pipeline: String, cache: [any KVCache]?) {
        guard isEnabled, let first = cache?.first else { return }
        report(pipeline, "kv_cache", first.state.first)
    }
}
