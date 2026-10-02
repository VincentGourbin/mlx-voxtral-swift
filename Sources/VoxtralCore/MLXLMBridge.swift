/**
 * MLXLMBridge - the model type the pipeline holds. The Python-port helpers that lived here (masks, attention, RoPE,
 * offline quantization, `init(path:)`) were deprecated in 2.3 and removed in 3.0 (ASK-23, K-31).
 */

typealias VoxtralModel = VoxtralForConditionalGeneration
