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
    /// model parameter is absent, and `VoxtralError.loadingFailed` naming them when a key matches no model
    /// parameter (K-88: a misnamed weight is not silently dropped; the sanitizers remove known aliases first)
    /// or on a shape or structure mismatch.
    func updateVerified(parameters: ModuleParameters) throws {
        let provided = Set(parameters.flattened().map { $0.0 })
        let own = Set(self.parameters().flattened().map { $0.0 })
        let missing = own.subtracting(provided)
        guard missing.isEmpty else {
            throw VoxtralError.missingWeights(missing.sorted())
        }
        let unexpected = provided.subtracting(own).sorted()
        guard unexpected.isEmpty else {
            throw VoxtralError.loadingFailed(
                "\(unexpected.count) weight(s) match no model parameter: \(unexpected.prefix(20).joined(separator: ", "))")
        }
        do {
            try update(parameters: parameters, verify: [.all])
        } catch let error as UpdateError {
            throw VoxtralError.loadingFailed("Weights do not match the model: \(error.localizedDescription)")
        }
    }
}
