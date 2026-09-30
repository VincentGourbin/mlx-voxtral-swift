/**
 * VerifiedWeights - apply loaded weights only if they cover the whole model (K-7, S-04, MLX-018)
 *
 * `Module.update(parameters:)` is `try! update(parameters:, verify: .none)`: a missing key keeps
 * its random initialization and a wrong shape surfaces later in a matmul. The live loaders use
 * this instead.
 */

import MLX
import MLXNN

extension Module {
    /// Applies `parameters`, throwing `VoxtralError.missingWeights` (the missing keys, sorted) when a
    /// model parameter is absent, and `VoxtralError.loadingFailed` on a shape or structure mismatch.
    /// Extra keys are tolerated: the sanitizers add aliases (e.g. a duplicated embedding).
    func updateVerified(parameters: ModuleParameters) throws {
        let provided = Set(parameters.flattened().map { $0.0 })
        let missing = self.parameters().flattened().map { $0.0 }.filter { !provided.contains($0) }
        guard missing.isEmpty else {
            throw VoxtralError.missingWeights(missing.sorted())
        }
        do {
            try update(parameters: parameters, verify: [.allModelKeysSet, .shapeMismatch])
        } catch let error as UpdateError {
            throw VoxtralError.loadingFailed("Weights do not match the model: \(error.localizedDescription)")
        }
    }
}
