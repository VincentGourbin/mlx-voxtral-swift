/**
 * DeprecatedNames - the 2.x names of the renamed public types (K-31, ASK-25): generic names collided with the
 * hosts' own types (FluxForge `ModelRegistry`, `ModelDownloader`; LTX `RuntimeBeacon`). Kept through 3.x.
 */

import Foundation

@available(*, deprecated, renamed: "VoxtralModelRegistry")
public typealias ModelRegistry = VoxtralModelRegistry

@available(*, deprecated, renamed: "VoxtralModelDownloader")
public typealias ModelDownloader = VoxtralModelDownloader

@available(*, deprecated, renamed: "VoxtralModelDownloaderError")
public typealias ModelDownloaderError = VoxtralModelDownloaderError

@available(*, deprecated, renamed: "VoxtralDownloadProgressCallback")
public typealias DownloadProgressCallback = VoxtralDownloadProgressCallback

@available(*, deprecated, renamed: "VoxtralRuntimeBeacon")
public typealias RuntimeBeacon = VoxtralRuntimeBeacon
