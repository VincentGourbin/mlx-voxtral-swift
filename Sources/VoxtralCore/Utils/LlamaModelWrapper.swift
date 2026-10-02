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
@available(*, deprecated, message: "Legacy Python-port path, removed in 3.0 (ASK-23). Never built by the loaders; the decoder is LlamaStandardModel.")
public class LlamaModelWrapper: Module {
    let embed_tokens: Embedding
    let layers: [LlamaStandardBlock]

    public init(embedTokens: Embedding, layers: [LlamaStandardBlock]) {
        self.embed_tokens = embedTokens
        self.layers = layers
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