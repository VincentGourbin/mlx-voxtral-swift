/**
 * CoreMLEncoderPathTests - K-25 (A-04, A-13)
 *
 * The Core ML encoder lives under `customModelsDirectory/<org>/<repo>/<name>` with the
 * K-6 completeness manifest, reloads with the network cut, writes nothing to
 * ~/.cache/huggingface; a mini encoder under a small configuration and an MLX encoder
 * without loaded weights raise instead of producing noise.
 *
 * Heavy tests (download the real mini encoder, ~1.3 GB, into <tmp>/VoxtralModels): skipped
 * unless VOXTRAL_COREML=1 (`TEST_RUNNER_VOXTRAL_COREML=1 xcodebuild test …`).
 * `testEncoderUnderCustomDirectoryReloadsOffline` runs first and leaves the encoder in
 * place for `testMiniEncoderUnderSmallConfigThrows`.
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

/// Fails every huggingface.co request as if the network were down, and counts them.
final class OfflineHubProtocol: URLProtocol {
    nonisolated(unsafe) static var requests = 0

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "huggingface.co" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}
    override func startLoading() {
        Self.requests += 1
        client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
    }
}

final class CoreMLEncoderPathTests: XCTestCase {

    private let fm = FileManager.default
    /// Fixed so the second heavy test finds what the first one downloaded
    private var modelsRoot: URL { fm.temporaryDirectory.appendingPathComponent("VoxtralModels") }
    private var miniEncoderPath: URL {
        modelsRoot.appendingPathComponent(VoxtralCoreMLVariant.mini.huggingFaceRepo)
            .appendingPathComponent(VoxtralCoreMLVariant.mini.modelName)
    }

    private func requireHeavy() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_COREML"] == "1",
                          "Set VOXTRAL_COREML=1 to run the Core ML encoder download tests")
    }

    /// Bytes under a directory (0 if absent), following nothing: the Hub cache's own files.
    private func bytes(under url: URL) -> Int64 {
        guard let e = fm.enumerator(atPath: url.path) else { return 0 }
        var total: Int64 = 0
        for case let rel as String in e {
            if let size = (try? fm.attributesOfItem(atPath: url.appendingPathComponent(rel).path))?[.size] as? Int64 {
                total += size
            }
        }
        return total
    }

    // MARK: - Path, offline reload, Hub cache untouched

    func testEncoderUnderCustomDirectoryReloadsOffline() async throws {
        try requireHeavy()
        let hubCache = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
        let hubBefore = bytes(under: hubCache)

        try? fm.removeItem(at: modelsRoot)
        let saved = ModelDownloader.customModelsDirectory
        ModelDownloader.customModelsDirectory = modelsRoot
        defer { ModelDownloader.customModelsDirectory = saved }

        // 1st hybrid load: downloads under the app-chosen root
        let first = try await VoxtralHybridEncoder.withHuggingFaceDownload(variant: .mini, preferredBackend: .coreML)
        XCTAssertTrue(first.status.coreMLAvailable)
        XCTAssertTrue(fm.fileExists(atPath: miniEncoderPath.appendingPathComponent("weights/weight.bin").path),
                      "encoder must live under customModelsDirectory/<org>/<repo>/<name>")

        // 2nd hybrid load with the network cut
        OfflineHubProtocol.requests = 0
        URLProtocol.registerClass(OfflineHubProtocol.self)
        defer { URLProtocol.unregisterClass(OfflineHubProtocol.self) }
        let second = try await VoxtralHybridEncoder.withHuggingFaceDownload(variant: .mini, preferredBackend: .coreML)
        print("[coreml-path] offline reload: Core ML available: \(second.status.coreMLAvailable), requests: \(OfflineHubProtocol.requests)")
        XCTAssertTrue(second.status.coreMLAvailable)
        XCTAssertEqual(OfflineHubProtocol.requests, 0)

        let hubAfter = bytes(under: hubCache)
        print("[coreml-path] ~/.cache/huggingface: \(hubBefore) -> \(hubAfter) bytes (+\(hubAfter - hubBefore))")
        XCTAssertEqual(hubAfter, hubBefore, "nothing may be written to ~/.cache/huggingface")
    }

    // MARK: - Small configuration + mini encoder

    func testMiniEncoderUnderSmallConfigThrows() throws {
        try requireHeavy()
        guard fm.fileExists(atPath: miniEncoderPath.path) else {
            throw XCTSkip("Run testEncoderUnderCustomDirectoryReloadsOffline first (mini encoder not in \(modelsRoot.path))")
        }
        XCTAssertThrowsError(try VoxtralCoreMLEncoder(modelURL: miniEncoderPath, config: .small),
                             "a 3072-wide encoder must be refused for the small (5120) variant")
        XCTAssertNoThrow(try VoxtralCoreMLEncoder(modelURL: miniEncoderPath, config: .mini))
    }

    // MARK: - MLX encoder without loaded weights

    func testUninitializedMLXEncoderThrows() throws {
        let hybrid = VoxtralHybridEncoder(preferredBackend: .mlx)
        XCTAssertThrowsError(try hybrid.encode(MLXArray.zeros([1, 128, 3000])),
                             "an MLX encoder with random weights must refuse to encode")
    }
}
