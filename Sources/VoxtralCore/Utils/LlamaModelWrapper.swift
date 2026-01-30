/**
 * LlamaModelWrapper - Minimal wrapper pour compatibilité legacy
 *
 * Cette classe simple remplace l'ancien VoxtralMLXLMWrapper complexe qui a été supprimé.
 * Elle fournit juste l'interface minimale nécessaire pour les références existantes.
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/**
 * Wrapper minimal pour les anciens codes qui référencent LlamaModelWrapper
 */
public class LlamaModelWrapper: Module {
    // @ModuleInfo required for weight loading and quantization
    @ModuleInfo(key: "embed_tokens") public var embed_tokens: Embedding
    @ModuleInfo public var layers: [LlamaStandardBlock]

    public init(embedTokens: Embedding, layers: [LlamaStandardBlock]) {
        self._embed_tokens.wrappedValue = embedTokens
        self._layers.wrappedValue = layers
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        // Simple forward pass
        var hiddenStates = embed_tokens(inputs)

        for (i, layer) in layers.enumerated() {
            let layerCache: (any KVCache)? = cache?[i]
            hiddenStates = layer(hiddenStates, cache: layerCache)
        }

        return hiddenStates
    }
}