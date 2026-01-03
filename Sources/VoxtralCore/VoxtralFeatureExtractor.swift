/**
 * VoxtralFeatureExtractor - Swift equivalent of mlx.voxtral/audio_processing.py
 * 
 * Pure conversion of Python audio_processing.py functionality.
 */

import Foundation
import MLX
import MLXNN
import AVFoundation

// Constants - Direct Python equivalents
let SAMPLE_RATE: Int = 16000
let N_FFT: Int = 400
let HOP_LENGTH: Int = 160
let CHUNK_LENGTH: Int = 30  // seconds
let N_SAMPLES: Int = CHUNK_LENGTH * SAMPLE_RATE  // 480000
let N_FRAMES: Int = N_SAMPLES / HOP_LENGTH  // 3000
let N_MELS: Int = 128

/**
 * Direct Python equivalent: def load_audio(file: str) -> np.ndarray
 */
func loadAudio(_ file: String) throws -> MLXArray {
    // Python equivalent: soundfile.read(file, dtype="float32")
    // Use AVFoundation instead of ffmpeg for native audio loading
    
    let fileURL = URL(fileURLWithPath: file)
    
    // Load audio file using AVFoundation
    let audioFile = try AVAudioFile(forReading: fileURL)
    let format = audioFile.processingFormat
    let frameCount = UInt32(audioFile.length)
    
    // Create buffer for audio data
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw VoxtralError.audioProcessingFailed("Failed to create audio buffer")
    }
    
    // Read audio data into buffer
    try audioFile.read(into: buffer)
    buffer.frameLength = frameCount
    
    // Get audio data as Float32 array
    guard let floatChannelData = buffer.floatChannelData else {
        throw VoxtralError.audioProcessingFailed("Failed to get float channel data")
    }
    
    let channelCount = Int(format.channelCount)
    let samples = Int(frameCount)
    var audioData: [Float] = []
    
    // Convert to mono if needed (average channels)
    if channelCount > 1 {
        // Average all channels to mono
        audioData = Array(repeating: 0, count: samples)
        for sample in 0..<samples {
            var sum: Float = 0
            for channel in 0..<channelCount {
                sum += floatChannelData[channel][sample]
            }
            audioData[sample] = sum / Float(channelCount)
        }
    } else {
        // Already mono
        audioData = Array(UnsafeBufferPointer(start: floatChannelData[0], count: samples))
    }
    
    // Resample to SAMPLE_RATE (16000 Hz) if needed
    let currentSampleRate = format.sampleRate
    if currentSampleRate != Double(SAMPLE_RATE) {
        // Simple linear interpolation resampling
        let ratio = Double(SAMPLE_RATE) / currentSampleRate
        let newLength = Int(Double(audioData.count) * ratio)
        var resampled = Array<Float>(repeating: 0, count: newLength)
        
        for i in 0..<newLength {
            let srcIndex = Double(i) / ratio
            let index = Int(srcIndex)
            let fraction = Float(srcIndex - Double(index))
            
            if index < audioData.count - 1 {
                resampled[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction
            } else if index < audioData.count {
                resampled[i] = audioData[index]
            }
        }
        
        audioData = resampled
    }
    
    return MLXArray(audioData)
}

/**
 * Direct Python equivalent: def pad_to_multiple(x: mx.array, multiple: int, axis: int = 0) -> mx.array
 */
func padToMultiple(_ x: MLXArray, multiple: Int, axis: Int = 0) -> MLXArray {
    // Python: 
    // remainder = x.shape[axis] % multiple
    // if remainder != 0:
    //     pad_amount = multiple - remainder
    //     pad_widths = [(0, 0)] * x.ndim
    //     pad_widths[axis] = (0, pad_amount)
    //     x = mx.pad(x, pad_widths)
    // return x
    
    let remainder = x.shape[axis] % multiple
    if remainder != 0 {
        let padAmount = multiple - remainder
        // MLX Swift uses IntOrPair - [(start, end), (start, end), ...] for each dimension
        var padWidths: [IntOrPair] = Array(repeating: IntOrPair((0, 0)), count: x.ndim)
        padWidths[axis] = IntOrPair((0, padAmount))  // Set end padding for axis
        return padded(x, widths: padWidths)
    }
    return x
}

/**
 * Direct Python equivalent: def hanning(size: int) -> mx.array
 */
func hanning(_ size: Int) -> MLXArray {
    // Python: window_np = np.hanning(size + 1)[:-1].astype(np.float32)
    //         return mx.array(window_np)
    
    // Utiliser MLX Swift cos au lieu de Foundation cos
    let indices = MLXArray(Array(0..<size).map(Float.init))
    let piValues = 2.0 * Float.pi * indices / Float(size)
    return 0.5 - 0.5 * cos(piValues)
}

/**
 * Direct Python equivalent: def stft_mlx(x: mx.array, window: mx.array, nperseg: int, noverlap: Optional[int], nfft: Optional[int], center: bool) -> mx.array
 */
func stftMlx(
    _ x: MLXArray, 
    window: MLXArray, 
    nperseg: Int = 256, 
    noverlap: Int? = nil, 
    nfft: Int? = nil, 
    center: Bool = true
) -> MLXArray {
    // Python implementation:
    // if nfft is None: nfft = nperseg
    // if noverlap is None: noverlap = nfft // 4
    // hop_length = nperseg - noverlap
    // if center: [reflection padding logic]
    // n_frames = (x.size - nperseg) // hop_length + 1
    // x_strided = mx.as_strided(x, shape=[n_frames, nfft], strides=[hop_length, 1])
    // x_windowed = x_strided[:, :nperseg] * window
    // if nfft > nperseg: [zero padding]
    // return mx.fft.rfft(x_windowed)
    
    let nfftValue = nfft ?? nperseg
    let noverlapValue = noverlap ?? (nfftValue / 4)
    let hopLength = nperseg - noverlapValue
    
    var processedX = x
    if center {
        let padding = nperseg / 2
        // Python: prefix = x[padding:0:-1], suffix = x[-2:-padding-2:-1]
        let prefix = x[.stride(from: padding, to: 0, by: -1)]
        let suffix = x[.stride(from: -2, to: -padding - 2, by: -1)]
        // Python: x = mx.concatenate([prefix, x, suffix])
        processedX = concatenated([prefix, x, suffix], axis: 0)
    }
    
    let nFrames = (processedX.shape[0] - nperseg) / hopLength + 1
    
    // Python: x_strided = mx.as_strided(x, shape=[n_frames, nfft], strides=[hop_length, 1])
    let xStrided = asStrided(processedX, [nFrames, nperseg], strides: [hopLength, 1])
    var xWindowed = xStrided * window
    
    if nfftValue > nperseg {
        let padWidth = [(0, 0), (0, nfftValue - nperseg)]
        xWindowed = padded(xWindowed, widths: padWidth.map { IntOrPair($0) })
    }
    
    return rfft(xWindowed, axis: -1)
}

/**
 * Direct Python equivalent: def mel_filter_bank_slaney(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None) -> mx.array
 */
func melFilterBankSlaney(sr: Int, nFft: Int, nMels: Int, fmin: Float = 0.0, fmax: Float? = nil) -> MLXArray {
    // Python: if fmax is None: fmax = sr / 2
    let fmaxValue = fmax ?? Float(sr) / 2.0
    
    // Python: def hz_to_mel(freq)
    func hzToMel(_ freq: Float) -> Float {
        let minLogHz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstep: Float = 27.0 / log(6.4)
        
        if freq >= minLogHz {
            return minLogMel + log(freq / minLogHz) * logstep
        } else {
            return 3.0 * freq / 200.0
        }
    }
    
    // Python: def mel_to_hz(mels)
    func melToHz(_ mels: Float) -> Float {
        let minLogHz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstep: Float = log(6.4) / 27.0
        
        if mels >= minLogMel {
            return minLogHz * exp(logstep * (mels - minLogMel))
        } else {
            return 200.0 * mels / 3.0
        }
    }
    
    // Python: mel_min = hz_to_mel(fmin), mel_max = hz_to_mel(fmax)
    let melMin = hzToMel(fmin)
    let melMax = hzToMel(fmaxValue)
    
    // Python: mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    let melPoints = (0..<(nMels + 2)).map { i in
        melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
    }
    let hzPoints = melPoints.map(melToHz)
    
    // Python: fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    let fftFreqs = (0..<(nFft / 2 + 1)).map { i in
        Float(i) * Float(sr) / Float(2 * (nFft / 2))
    }
    
    // Python: filters = np.zeros((n_mels, n_fft // 2 + 1))
    var filters = Array(repeating: Array(repeating: Float(0), count: nFft / 2 + 1), count: nMels)
    
    // Python: for i in range(n_mels): [triangular filter computation]
    for i in 0..<nMels {
        let left = hzPoints[i]
        let center = hzPoints[i + 1]
        let right = hzPoints[i + 2]
        
        for (j, freq) in fftFreqs.enumerated() {
            let rising = max(0, min(1, (freq - left) / (center - left)))
            let falling = max(0, min(1, (right - freq) / (right - center)))
            filters[i][j] = rising * falling
            
            // Python: enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
            let enorm = 2.0 / (hzPoints[i + 2] - hzPoints[i])
            filters[i][j] *= enorm
        }
    }
    
    // Convert to MLXArray
    let flatFilters = filters.flatMap { $0 }
    return MLXArray(flatFilters, [nMels, nFft / 2 + 1])
}

// Python: _mel_filters_cache = {}
var _melFiltersCache: [Int: MLXArray] = [:]

/**
 * Direct Python equivalent: def get_mel_filters(n_mels: int = N_MELS) -> mx.array
 */
func getMelFilters(nMels: Int = N_MELS) -> MLXArray {
    // Python: if n_mels not in _mel_filters_cache:
    //             _mel_filters_cache[n_mels] = mel_filter_bank_slaney(SAMPLE_RATE, N_FFT, n_mels, fmax=8000)
    //         return _mel_filters_cache[n_mels]
    
    if let cached = _melFiltersCache[nMels] {
        return cached
    }
    
    let filters = melFilterBankSlaney(
        sr: SAMPLE_RATE, 
        nFft: N_FFT, 
        nMels: nMels, 
        fmax: 8000.0
    )
    _melFiltersCache[nMels] = filters
    return filters
}

/**
 * Direct Python equivalent: def log_mel_spectrogram(audio: mx.array, n_mels: int, n_fft: int, hop_length: int, global_max: Optional[float]) -> Union[mx.array, Tuple[mx.array, float]]
 */
func logMelSpectrogram(
    _ audio: MLXArray, 
    nMels: Int = N_MELS, 
    nFft: Int = N_FFT, 
    hopLength: Int = HOP_LENGTH,
    globalMax: Float? = nil
) -> (MLXArray, Float?) {
    // Python:
    // window = hanning(n_fft)
    // freqs = stft_mlx(audio, window, nperseg=n_fft, noverlap=n_fft - hop_length)
    // freqs = freqs[:-1, :]
    // magnitudes = mx.abs(freqs) ** 2
    // filters = get_mel_filters(n_mels)
    // mel_spec = magnitudes @ filters.T
    // log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    // if global_max is None: log_max = mx.max(log_spec); return_log_max = True
    // else: log_max = global_max; return_log_max = False
    // log_spec = mx.maximum(log_spec, log_max - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    // return log_spec.T, log_max if return_log_max else log_spec.T
    
    let window = hanning(nFft)
    let freqs = stftMlx(audio, window: window, nperseg: nFft, noverlap: nFft - hopLength)
    
    // Python: freqs = freqs[:-1, :]
    let trimmedFreqs = freqs[0..<(freqs.shape[0] - 1), 0..<freqs.shape[1]]
    
    // Python: magnitudes = mx.abs(freqs) ** 2
    let magnitudes = pow(abs(trimmedFreqs), MLXArray(2.0))
    
    // Python: filters = get_mel_filters(n_mels)
    let filters = getMelFilters(nMels: nMels)
    
    // Python: mel_spec = magnitudes @ filters.T
    let melSpec = matmul(magnitudes, filters.T)
    
    // Python: log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    let logSpec = log10(maximum(melSpec, MLXArray(1e-10)))
    
    let logMax: Float
    let returnLogMax: Bool
    if let globalMaxValue = globalMax {
        logMax = globalMaxValue
        returnLogMax = false
    } else {
        logMax = logSpec.max().item(Float.self)
        returnLogMax = true
    }
    
    // Python: log_spec = mx.maximum(log_spec, log_max - 8.0)
    let clampedSpec = maximum(logSpec, MLXArray(logMax - 8.0))
    
    // Python: log_spec = (log_spec + 4.0) / 4.0
    let normalizedSpec = (clampedSpec + 4.0) / 4.0
    
    // Python: return log_spec.T
    let result = normalizedSpec.T
    
    return (result, returnLogMax ? logMax : nil)
}

/**
 * Direct Python equivalent: def process_audio_for_voxtral(audio_path: Union[str, Path], chunk_length_s: int, normalize_audio: bool, return_attention_mask: bool) -> Dict[str, Union[mx.array, np.ndarray, int, float]]
 */
func processAudioForVoxtral(
    _ audioPath: String,
    chunkLengthS: Int = 30,
    normalizeAudio: Bool = false,
    returnAttentionMask: Bool = false
) throws -> [String: Any] {
    // Python:
    // Process audio file for Voxtral model input with automatic chunking.
    // - Loading and resampling to 16kHz
    // - Padding to multiples of 30 seconds  
    // - Computing log-mel spectrograms with global normalization
    // - Automatic chunking for long audio
    
    let audio = try loadAudio(audioPath)
    var processedAudio = audio
    
    // Python: if normalize_audio: [normalize logic]
    if normalizeAudio {
        let audioMax = abs(processedAudio).max()
        if audioMax.item(Float.self) > 0 {
            processedAudio = processedAudio / audioMax
        }
    }
    
    let chunkSamples = chunkLengthS * SAMPLE_RATE
    let audioPadded = padToMultiple(processedAudio, multiple: chunkSamples)
    
    let numChunks = audioPadded.shape[0] / chunkSamples
    var allLogMelSpecs: [MLXArray] = []
    var globalLogMax: Float? = nil
    
    // First pass: compute global max for normalization consistency
    for chunkIdx in 0..<numChunks {
        let startIdx = chunkIdx * chunkSamples
        let endIdx = min(startIdx + chunkSamples, audioPadded.shape[0])
        let chunk = audioPadded[startIdx..<endIdx]
        
        let (_, logMax) = logMelSpectrogram(chunk)
        if let logMax = logMax {
            globalLogMax = max(globalLogMax ?? -Float.infinity, logMax)
        }
    }
    
    // Second pass: compute spectrograms with global normalization
    for chunkIdx in 0..<numChunks {
        let startIdx = chunkIdx * chunkSamples
        let endIdx = min(startIdx + chunkSamples, audioPadded.shape[0])
        let chunk = audioPadded[startIdx..<endIdx]
        
        let (logMel, _) = logMelSpectrogram(chunk, globalMax: globalLogMax)
        // Python: chunk_mel[None, :, :] - add batch dimension
        let logMelWithBatch = expandedDimensions(logMel, axis: 0)
        allLogMelSpecs.append(logMelWithBatch)
    }
    
    let finalLogMel: MLXArray
    if allLogMelSpecs.count == 1 {
        finalLogMel = allLogMelSpecs[0]
    } else {
        // Python: mx.concatenate(mel_chunks, axis=0) - concatenate along batch dimension
        finalLogMel = concatenated(allLogMelSpecs, axis: 0)
    }
    
    var result: [String: Any] = [
        "input_features": finalLogMel,
        "num_chunks": numChunks,
        "chunk_length_s": chunkLengthS
    ]
    
    if returnAttentionMask {
        // Create attention mask (all ones for valid audio)
        let attentionMask = MLXArray.ones(finalLogMel.shape)
        result["attention_mask"] = attentionMask
    }
    
    return result
}

/**
 * Direct Python equivalent: class VoxtralFeatureExtractor
 */
public class VoxtralFeatureExtractor {
    
    let featureSize: Int
    let samplingRate: Int
    let hopLength: Int
    let chunkLength: Int
    let nFft: Int
    let paddingValue: Float
    let nbMaxFrames: Int
    
    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        hopLength: Int = 160,
        chunkLength: Int = 30,
        nFft: Int = 400,
        paddingValue: Float = 0.0
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.hopLength = hopLength
        self.chunkLength = chunkLength
        self.nFft = nFft
        self.paddingValue = paddingValue
        self.nbMaxFrames = chunkLength * samplingRate / hopLength
    }
    
    /**
     * Direct Python equivalent: def __call__(self, raw_speech, sampling_rate=None, return_tensors="np")
     */
    public func callAsFunction(
        rawSpeech: Any,
        samplingRate: Int? = nil,
        returnTensors: String = "np"
    ) throws -> [String: Any] {
        // Python equivalent: VoxtralFeatureExtractor.__call__
        // When input is a file path, use process_audio_for_voxtral
        // which returns [n_chunks, 128, 3000] with proper chunking
        
        if let audioPath = rawSpeech as? String {
            // Python: result = process_audio_for_voxtral(raw_speech)
            // This returns features with shape [n_chunks, 128, 3000]
            let result = try processAudioForVoxtral(
                audioPath,
                chunkLengthS: 30,
                normalizeAudio: false,
                returnAttentionMask: false
            )
            return result
        } else if let audioArray = rawSpeech as? [Float] {
            // Python: mel_features = self._process_audio_array_with_chunking(audio_array)
            let audio = MLXArray(audioArray)
            let chunkSamples = 30 * 16000  // 30 seconds at 16kHz
            let nSamples = audio.shape[0]
            let nChunks = Int((Float(nSamples) / Float(chunkSamples)).rounded(.up))
            
            var allFeatures: [MLXArray] = []
            for i in 0..<nChunks {
                let start = i * chunkSamples
                let end = min(start + chunkSamples, nSamples)
                let chunk = audio[start..<end]
                
                // Python: process_audio_chunk(chunk) - pad to N_SAMPLES=480000 then log_mel_spectrogram
                let chunkPadded: MLXArray
                if chunk.shape[0] > 480000 {
                    // Python: if audio_array.shape[0] > N_SAMPLES: audio_array = audio_array[:N_SAMPLES] 
                    chunkPadded = chunk[0..<480000]
                } else if chunk.shape[0] < 480000 {
                    // Python: elif audio_array.shape[0] < N_SAMPLES: padding = N_SAMPLES - audio_array.shape[0]; audio_array = mx.pad(audio_array, [(0, padding)])
                    let padding = 480000 - chunk.shape[0]
                    chunkPadded = padded(chunk, widths: [IntOrPair((0, padding))], value: MLXArray(0.0))
                } else {
                    chunkPadded = chunk
                }
                
                // Python: mel_features, _ = log_mel_spectrogram(audio_array)
                let (logMel, _) = logMelSpectrogram(chunkPadded)
                allFeatures.append(logMel)
            }
            
            // Python: return np.stack(all_features, axis=0)
            let stackedFeatures = stacked(allFeatures, axis: 0)
            return ["input_features": stackedFeatures]
        } else if let mlxAudio = rawSpeech as? MLXArray {
            // For MLXArray input, process with chunking
            let (logMel, _) = logMelSpectrogram(mlxAudio)
            // Add batch dimension
            let logMelWithBatch = expandedDimensions(logMel, axis: 0)
            return ["input_features": logMelWithBatch]
        } else {
            throw VoxtralError.audioProcessingFailed("Unsupported raw_speech type")
        }
    }
}

/**
 * Direct Python equivalent: def process_audio_chunk(chunk: mx.array, chunk_length: int = CHUNK_LENGTH, hop_length: int = HOP_LENGTH) -> mx.array
 */
func processAudioChunk(_ chunk: MLXArray, chunkLength: Int = CHUNK_LENGTH, hopLength: Int = HOP_LENGTH) -> MLXArray {
    // Python:
    // expected_length = chunk_length * SAMPLE_RATE
    // if chunk.shape[0] < expected_length:
    //     chunk = mx.pad(chunk, [(0, expected_length - chunk.shape[0])])
    // elif chunk.shape[0] > expected_length:
    //     chunk = chunk[:expected_length]
    // return log_mel_spectrogram(chunk)
    
    let expectedLength = chunkLength * SAMPLE_RATE
    
    let adjustedChunk: MLXArray
    if chunk.shape[0] < expectedLength {
        let padAmount = expectedLength - chunk.shape[0]
        adjustedChunk = padded(chunk, widths: [IntOrPair((0, padAmount))])
    } else if chunk.shape[0] > expectedLength {
        adjustedChunk = chunk[0..<expectedLength]
    } else {
        adjustedChunk = chunk
    }
    
    let (result, _) = logMelSpectrogram(adjustedChunk)
    return result
}