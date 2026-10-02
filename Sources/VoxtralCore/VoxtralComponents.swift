/**
 * VoxtralComponents.swift
 * 
 * COMPOSANTS VOXTRAL VALIDÉS - Version Production
 * ==============================================
 * 
 * Les 3 composants core de Voxtral validés et fonctionnels :
 * 1. TekkenTokenizer - Tokenisation texte BPE
 * 2. AudioEncoder - Audio tokenisation 75fps  
 * 3. ChatTemplateProcessor - Application template Voxtral
 * 
 * USAGE :
 * ```swift
 * let tokenizer = TekkenTokenizer()
 * let audioEncoder = AudioEncoder()
 * let chatProcessor = ChatTemplateProcessor()
 * 
 * let textTokens = tokenizer.encode("résume moi cet audio")
 * let audioTokens = audioEncoder.encode(audioData: wavData)
 * let formatted = chatProcessor.apply(conversation: messages)
 * ```
 */

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - 1. TEKKEN TOKENIZER (VALIDÉ)

/**
 * TekkenTokenizer - Implementation BPE exacte basée sur tiktoken (comme Python mistral-common)
 * Équivalent exact de mistral_common.tokens.tokenizers.tekken.Tekkenizer
 */
class TekkenTokenizer {
    
    // Vocabulaire principal (mergeable_ranks dans tiktoken)
    private var mergeableRanks: [Data: Int] = [:]  // byte sequences -> rank
    private var reverseVocabulary: [Int: String] = [:]
    private var modelPath: String?
    
    // Regex pattern pour découper le texte (pat_str dans tiktoken) 
    private var regexPattern: String = ""
    private var compiledRegex: NSRegularExpression?
    
    // Configuration
    private var numSpecialTokens: Int = 0
    
    // Pas besoin d'avoidTokens : on utilise la logique Python exacte (troncature vocab)
    
    // Tokens spéciaux (chargés depuis les fichiers de config)
    private var bosTokenId = 1
    private var eosTokenId = 2
    private let unkTokenId = 0  // Always 0 for unknown tokens
    private var padTokenId = 11  // Default, will be loaded from config
    private var audioTokenIdInternal = 24  // Default, will be loaded from config
    
    // Public access to vocabulary for compatibility
    var vocab: [String: Int] { 
        var result: [String: Int] = [:]
        for (bytes, rank) in mergeableRanks {
            if let string = String(data: bytes, encoding: .utf8) {
                result[string] = rank  // Rank déjà correct
            }
        }
        return result
    }
    
    // Structure pour parser le JSON tekken.json
    private struct TekkenVocab: Codable {
        let config: TekkenConfig
        let vocab: [TekkenToken]
        let special_tokens: [TekkenSpecialToken]?
    }
    
    private struct GenerationConfig: Codable {
        let bos_token_id: Int?
        let eos_token_id: Int?
        let pad_token_id: Int?
    }

    private struct ModelConfig: Codable {
        let audio_token_id: Int?
    }
    
    private struct TekkenConfig: Codable {
        let pattern: String
        let num_vocab_tokens: Int
        let default_vocab_size: Int
        let default_num_special_tokens: Int
        let version: String
    }
    
    private struct TekkenToken: Codable {
        let rank: Int
        let token_bytes: String  // base64 encoded
        let token_str: String?
    }
    
    private struct TekkenSpecialToken: Codable {
        let rank: Int
        let token_str: String
        let is_control: Bool
    }
    
    /// Progress callback type for tokenizer loading
    typealias TokenizerProgressCallback = (Double, String) -> Void

    /// Loads `tekken.json` from the model directory or throws: a missing or unreadable file, an
    /// empty vocabulary or a split pattern that does not compile is an error, never a silent
    /// byte-level demo tokenizer (S-05).
    static func load(modelPath: String, progress: TokenizerProgressCallback? = nil) throws -> TekkenTokenizer {
        let tekkenPath = "\(modelPath)/tekken.json"
        guard FileManager.default.fileExists(atPath: tekkenPath) else {
            throw VoxtralError.fileNotFound(tekkenPath)
        }
        let tokenizer = TekkenTokenizer(unloadedModelPath: modelPath)
        try tokenizer.loadTekkenStrict(modelPath: modelPath, progress: progress)
        guard tokenizer.compiledRegex != nil, !tokenizer.mergeableRanks.isEmpty else {
            throw VoxtralError.invalidTokenizer("\(tekkenPath): empty vocabulary or invalid split pattern")
        }
        return tokenizer
    }

    /// Byte-level demo tokenizer, for tests only: never a stand-in for a model's `tekken.json`.
    static func demo() -> TekkenTokenizer {
        let tokenizer = TekkenTokenizer(unloadedModelPath: nil)
        tokenizer.loadDemoTokenizerData()
        return tokenizer
    }

    private init(unloadedModelPath modelPath: String?) {
        self.modelPath = modelPath
    }

    private func loadTokenizerData(progress: TokenizerProgressCallback? = nil) {
        if let modelPath = modelPath {
            loadTekkenTokenizerFromFile(modelPath: modelPath, progress: progress)
        } else {
            loadDemoTokenizerData()
        }
    }

    /// Non-throwing load kept for the deprecated `init`: on failure it falls back to the demo tokenizer.
    func loadTekkenTokenizerFromFile(modelPath: String, progress: TokenizerProgressCallback? = nil) {
        do {
            try loadTekkenStrict(modelPath: modelPath, progress: progress)
        } catch {
            VoxtralDebug.log("Cannot load \(modelPath)/tekken.json (\(error)), using demo tokenizer")
            loadDemoTokenizerData()
        }
    }

    /// Loads the vocabulary from `tekken.cache` or `tekken.json`, throwing on an unreadable file.
    func loadTekkenStrict(modelPath: String, progress: TokenizerProgressCallback? = nil) throws {
        // Reset tokenizer state before loading new data
        mergeableRanks.removeAll()
        reverseVocabulary.removeAll()
        numSpecialTokens = 0

        let tekkenPath = "\(modelPath)/tekken.json"
        let cachePath = "\(modelPath)/tekken.cache"

        // Try to load from binary cache first (10-100x faster)
        if loadFromCache(cachePath: cachePath, progress: progress) {
            try loadSpecialTokens(modelPath: modelPath)
            progress?(1.0, "Tokenizer loaded from cache")
            return
        }

        progress?(0.0, "Loading tokenizer vocabulary...")

        let jsonData: Data
        do {
            jsonData = try Data(contentsOf: URL(fileURLWithPath: tekkenPath))
        } catch {
            throw VoxtralError.fileNotFound(tekkenPath)
        }

        do {
            progress?(0.1, "Parsing tokenizer JSON...")
            let tekkenVocab: TekkenVocab
            do {
                tekkenVocab = try JSONDecoder().decode(TekkenVocab.self, from: jsonData)
            } catch {
                throw VoxtralError.invalidTokenizer("\(tekkenPath): \(error.localizedDescription)")
            }

            // 1. Charger la regex pattern (équivalent pat_str dans tiktoken)
            regexPattern = tekkenVocab.config.pattern
            do {
                compiledRegex = try NSRegularExpression(pattern: regexPattern, options: [])
            } catch {
                throw VoxtralError.invalidTokenizer("\(tekkenPath): pattern does not compile (\(error.localizedDescription))")
            }

            // 2. LOGIQUE PYTHON EXACTE : Tronquer le vocabulaire
            numSpecialTokens = tekkenVocab.config.default_num_special_tokens
            let defaultVocabSize = tekkenVocab.config.default_vocab_size
            let maxVocab = defaultVocabSize - numSpecialTokens  // 131072 - 1000 = 130072

            // Ne charger que les premiers maxVocab tokens (comme Python)
            let truncatedVocab = Array(tekkenVocab.vocab.prefix(maxVocab))

            // Pre-allocate dictionaries for better performance (2-3x faster)
            mergeableRanks.reserveCapacity(maxVocab)
            reverseVocabulary.reserveCapacity(maxVocab)

            progress?(0.2, "Building vocabulary (\(maxVocab) tokens)...")

            let progressInterval = maxVocab / 10  // Report every 10%
            for (index, token) in truncatedVocab.enumerated() {
                // Décoder token_bytes base64 -> Data
                if let tokenData = Data(base64Encoded: token.token_bytes) {
                    // Store original ranks from vocab
                    mergeableRanks[tokenData] = token.rank

                    // Pour decode: rank avec offset de special tokens
                    if let tokenString = token.token_str {
                        reverseVocabulary[token.rank + numSpecialTokens] = tokenString
                    } else if let decodedString = String(data: tokenData, encoding: .utf8) {
                        reverseVocabulary[token.rank + numSpecialTokens] = decodedString
                    }
                }

                // Report progress every 10%
                if progressInterval > 0 && index % progressInterval == 0 {
                    let pct = 0.2 + (Double(index) / Double(maxVocab)) * 0.7
                    progress?(pct, "Building vocabulary (\(index)/\(maxVocab))...")
                }
            }

            // Load special token IDs from config files
            try loadSpecialTokens(modelPath: modelPath)

            progress?(0.95, "Saving tokenizer cache...")

            // Save binary cache for next time
            saveToCache(cachePath: cachePath)

            progress?(1.0, "Tokenizer ready")

        }
    }

    // MARK: - Binary Cache for Fast Loading

    /// Binary cache format version
    private let cacheVersion: UInt32 = 1

    /// Load tokenizer from binary cache (10-100x faster than JSON parsing)
    private func loadFromCache(cachePath: String, progress: TokenizerProgressCallback?) -> Bool {
        guard FileManager.default.fileExists(atPath: cachePath),
              let cacheData = try? Data(contentsOf: URL(fileURLWithPath: cachePath)) else {
            return false
        }

        progress?(0.0, "Loading tokenizer from cache...")

        var offset = 0

        // Read version
        guard cacheData.count >= 4 else { return false }
        let version = cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        offset += 4

        guard version == cacheVersion else {
            VoxtralDebug.log("Cache version mismatch, rebuilding...")
            return false
        }

        // Read numSpecialTokens
        guard cacheData.count >= offset + 4 else { return false }
        numSpecialTokens = Int(cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) })
        offset += 4

        // Read regex pattern length and string
        guard cacheData.count >= offset + 4 else { return false }
        let patternLength = Int(cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) })
        offset += 4

        guard cacheData.count >= offset + patternLength else { return false }
        let patternData = cacheData.subdata(in: offset..<(offset + patternLength))
        regexPattern = String(data: patternData, encoding: .utf8) ?? ""
        compiledRegex = try? NSRegularExpression(pattern: regexPattern, options: [])
        offset += patternLength

        // Read vocabulary count
        guard cacheData.count >= offset + 4 else { return false }
        let vocabCount = Int(cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) })
        offset += 4

        progress?(0.3, "Loading \(vocabCount) tokens...")

        // Pre-allocate
        mergeableRanks.reserveCapacity(vocabCount)
        reverseVocabulary.reserveCapacity(vocabCount)

        // Read each entry: [keyLength: UInt16][keyData: Data][rank: Int32][strLength: UInt16][strData: Data]
        for i in 0..<vocabCount {
            guard cacheData.count >= offset + 2 else { return false }
            let keyLength = Int(cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self) })
            offset += 2

            guard cacheData.count >= offset + keyLength else { return false }
            let keyData = cacheData.subdata(in: offset..<(offset + keyLength))
            offset += keyLength

            guard cacheData.count >= offset + 4 else { return false }
            let rank = Int(cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Int32.self) })
            offset += 4

            guard cacheData.count >= offset + 2 else { return false }
            let strLength = Int(cacheData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self) })
            offset += 2

            var tokenString: String? = nil
            if strLength > 0 {
                guard cacheData.count >= offset + strLength else { return false }
                let strData = cacheData.subdata(in: offset..<(offset + strLength))
                tokenString = String(data: strData, encoding: .utf8)
                offset += strLength
            }

            mergeableRanks[keyData] = rank
            if let str = tokenString {
                reverseVocabulary[rank + numSpecialTokens] = str
            }

            // Progress every 10%
            if i % (vocabCount / 10 + 1) == 0 {
                progress?(0.3 + Double(i) / Double(vocabCount) * 0.6, "Loading tokens (\(i)/\(vocabCount))...")
            }
        }

        progress?(0.95, "Tokenizer cache loaded")
        return true
    }

    /// Save tokenizer to binary cache
    private func saveToCache(cachePath: String) {
        var cacheData = Data()

        // Write version
        var version = cacheVersion
        cacheData.append(Data(bytes: &version, count: 4))

        // Write numSpecialTokens
        var numSpecial = UInt32(numSpecialTokens)
        cacheData.append(Data(bytes: &numSpecial, count: 4))

        // Write regex pattern
        let patternData = regexPattern.data(using: .utf8) ?? Data()
        var patternLength = UInt32(patternData.count)
        cacheData.append(Data(bytes: &patternLength, count: 4))
        cacheData.append(patternData)

        // Write vocabulary count
        var vocabCount = UInt32(mergeableRanks.count)
        cacheData.append(Data(bytes: &vocabCount, count: 4))

        // Write each entry
        for (keyData, rank) in mergeableRanks {
            // Key length and data
            var keyLength = UInt16(keyData.count)
            cacheData.append(Data(bytes: &keyLength, count: 2))
            cacheData.append(keyData)

            // Rank
            var rankInt32 = Int32(rank)
            cacheData.append(Data(bytes: &rankInt32, count: 4))

            // String value (from reverseVocabulary)
            let strValue = reverseVocabulary[rank + numSpecialTokens]
            let strData = strValue?.data(using: .utf8) ?? Data()
            var strLength = UInt16(strData.count)
            cacheData.append(Data(bytes: &strLength, count: 2))
            if strData.count > 0 {
                cacheData.append(strData)
            }
        }

        // Write cache file
        try? cacheData.write(to: URL(fileURLWithPath: cachePath))
        VoxtralDebug.log("Tokenizer cache saved: \(cachePath) (\(cacheData.count) bytes)")
    }
    
    /// Special token ids from generation_config.json and config.json when present: an absent file keeps the
    /// defaults, an unreadable one throws instead of silently keeping wrong end tokens (K-27)
    private func loadSpecialTokens(modelPath: String) throws {
        func decode<T: Decodable>(_ type: T.Type, _ name: String) throws -> T? {
            let url = URL(fileURLWithPath: "\(modelPath)/\(name)")
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            do {
                return try JSONDecoder().decode(type, from: Data(contentsOf: url))
            } catch {
                throw VoxtralError.invalidTokenizer("\(url.path): \(error.localizedDescription)")
            }
        }
        if let generationConfig = try decode(GenerationConfig.self, "generation_config.json") {
            if let bos = generationConfig.bos_token_id { bosTokenId = bos }
            if let eos = generationConfig.eos_token_id { eosTokenId = eos }
            if let pad = generationConfig.pad_token_id { padTokenId = pad }
        }
        if let modelConfig = try decode(ModelConfig.self, "config.json"), let audio = modelConfig.audio_token_id {
            audioTokenIdInternal = audio
        }
    }
    
    private func loadDemoTokenizerData() {
        // Use a byte-level vocabulary so that any text round-trips correctly.
        // Each of the 256 possible byte values gets its own token rank (0-255).
        // This mirrors the fallback behaviour of real BPE tokenizers: every byte
        // is representable, guaranteeing encode→decode fidelity even without the
        // actual model file.
        numSpecialTokens = 1000

        // Pattern regex basique pour demo
        regexPattern = "[\\w]+|[^\\w\\s]|\\s"
        compiledRegex = try? NSRegularExpression(pattern: regexPattern, options: [])

        // Build a byte-level vocab: rank i maps to the single byte with value i.
        for byteValue in 0..<256 {
            let byteData = Data([UInt8(byteValue)])
            mergeableRanks[byteData] = byteValue
            // Decode side: token ID = byteValue + numSpecialTokens
            if let str = String(bytes: [UInt8(byteValue)], encoding: .utf8) {
                reverseVocabulary[byteValue + numSpecialTokens] = str
            } else {
                // Non-UTF8 byte: store a placeholder that won't be emitted as text
                reverseVocabulary[byteValue + numSpecialTokens] = ""
            }
        }
    }
    
    /**
     * Encode text using BPE (équivalent tiktoken.Encoding.encode + Tekkenizer offset)
     * Python: tokens = self._model.encode(s); tokens = [t + self.num_special_tokens for t in tokens]
     */
    func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }
        
        // 1. Découper le texte selon regex pattern (comme tiktoken)
        let chunks = splitByRegexPattern(text)
        
        // 2. Appliquer BPE sur chaque chunk (retourne ranks originaux du vocab)
        var rawTokens: [Int] = []
        
        for chunk in chunks {
            let chunkTokens = encodeBPEChunk(chunk)
            rawTokens.append(contentsOf: chunkTokens)
        }
        
        // 3. Appliquer offset Tekkenizer (comme Python: +1000)
        let finalTokens = rawTokens.map { $0 + numSpecialTokens }
        
        return finalTokens
    }
    
    /**
     * Découpe le texte selon la regex pattern (équivalent tiktoken pat_str matching)
     */
    private func splitByRegexPattern(_ text: String) -> [String] {
        guard let regex = compiledRegex else {
            // Fallback: découpe par mots si regex failed
            return text.components(separatedBy: CharacterSet.whitespacesAndNewlines).filter { !$0.isEmpty }
        }
        
        let range = NSRange(location: 0, length: text.utf16.count)
        let matches = regex.matches(in: text, options: [], range: range)
        
        return matches.compactMap { match in
            guard let swiftRange = Range(match.range, in: text) else { return nil }
            return String(text[swiftRange])
        }
    }
    
    /**
     * Encode un chunk de texte avec BPE (équivalent tiktoken merge algorithm)
     * Algorithme BPE: merge itératif des paires les plus fréquentes
     */
    private func encodeBPEChunk(_ chunk: String) -> [Int] {
        guard !chunk.isEmpty else { return [] }
        guard let chunkData = chunk.data(using: .utf8) else { return [] }
        
        // Si le chunk entier existe dans mergeable_ranks, utiliser directement
        if let directRank = mergeableRanks[chunkData] {
            return [directRank]  // Rank déjà correct (pas d'offset à ajouter)
        }
        
        // Sinon, BPE avec algorithm de merge (EXACT tiktoken)
        let bytes = Array(chunkData)
        
        // Si un seul byte, lookup direct
        if bytes.count == 1 {
            let byteData = Data([bytes[0]])
            if let rank = mergeableRanks[byteData] {
                return [rank]  // Rank déjà correct
            } else {
                return [unkTokenId]
            }
        }
        
        // Initialiser word comme array de bytes individuels
        var word: [Data] = bytes.map { Data([$0]) }
        
        // BPE merge algorithm (EXACT tiktoken)
        while word.count >= 2 {
            // Trouver toutes les paires adjacentes possibles
            var pairs: [(Data, Data, Int)] = []  // (first, second, position)
            
            for i in 0..<(word.count - 1) {
                let pair = word[i] + word[i + 1]  // Concaténer les bytes
                if mergeableRanks[pair] != nil {
                    // Seuls les tokens dans mergeableRanks peuvent être utilisés
                    pairs.append((word[i], word[i + 1], i))
                }
            }
            
            // Si aucune paire mergeable, arrêter
            if pairs.isEmpty { break }
            
            // Trouver la paire avec le rank le plus faible (priorité haute)
            let bestPair = pairs.min { pair1, pair2 in
                let rank1 = mergeableRanks[pair1.0 + pair1.1] ?? Int.max
                let rank2 = mergeableRanks[pair2.0 + pair2.1] ?? Int.max
                return rank1 < rank2
            }!
            
            // Merger la meilleure paire
            let newData = bestPair.0 + bestPair.1
            let position = bestPair.2
            
            var newWord: [Data] = []
            var i = 0
            while i < word.count {
                if i == position {
                    // Remplacer les deux éléments par le merge
                    newWord.append(newData)
                    i += 2  // Skip le prochain élément aussi
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            
            word = newWord
        }
        
        // Convertir word final en token IDs
        let tokens = word.compactMap { data -> Int? in
            if let rank = mergeableRanks[data] {
                return rank  // Rank déjà correct
            }
            return nil
        }
        
        return tokens.isEmpty ? [unkTokenId] : tokens
    }
    
    /**
     * Decode tokens back to text (équivalent tiktoken.Encoding.decode)
     * Python: return self._model.decode([t - self.num_special_tokens for t in tokens])
     *
     * Accumulates raw bytes from the vocabulary and converts the full byte buffer to
     * UTF-8 at the end. This correctly handles multi-byte UTF-8 sequences (accented
     * characters, CJK, emoji) that span multiple BPE tokens.
     */
    func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        var rawBytes = Data()

        for tokenId in tokens {
            // Every id below the special-token count is a control token (e.g. Realtime [STREAMING_PAD] 32 and
            // [STREAMING_WORD] 33), not text: it would map to rank 0, the byte 0x00 (K-13)
            if skipSpecialTokens
                && (tokenId < numSpecialTokens || tokenId == bosTokenId || tokenId == eosTokenId || tokenId == padTokenId) {
                continue
            }

            // Convert tokenId back to raw rank (remove special token offset)
            let rawTokenId = max(0, tokenId - numSpecialTokens)

            // First try the string stored in reverseVocabulary (fast O(1) path, handles
            // real multi-token strings like "hello" correctly).
            if let tokenString = reverseVocabulary[tokenId], !tokenString.isEmpty,
               let stringBytes = tokenString.data(using: .utf8) {
                rawBytes.append(stringBytes)
            } else {
                // Fallback: locate the raw bytes via mergeableRanks (O(n) scan).
                // Required for byte-level demo tokenizer entries that are non-UTF8
                // individual bytes (stored as "" in reverseVocabulary).
                if let bytes = mergeableRanks.first(where: { $0.value == rawTokenId })?.key {
                    rawBytes.append(bytes)
                }
            }
            // Unknown tokens are silently skipped to keep round-trips clean
        }

        // Decode the full byte buffer as UTF-8. Replace invalid sequences with the
        // Unicode replacement character so we always produce a valid Swift String.
        return String(decoding: rawBytes, as: UTF8.self)
    }
    
    /**
     * Encode transcription request (équivalent encode_transcription dans Python)
     * Cette méthode sera utilisée pour les requêtes audio/transcription
     */
    
    var vocabSize: Int { mergeableRanks.count + numSpecialTokens }
    var bosToken: Int { bosTokenId }
    var eosToken: Int { eosTokenId }
    
    // Compatibility methods for VoxtralProcessor interface
    func getControlToken(_ token: String) -> Int {
        // LOGIC PYTHON EXACTE : retourner les mêmes valeurs que Python Tekkenizer.get_control_token()
        switch token {
        case "<s>":
            return bosTokenId  // 1
        case "</s>":
            return eosTokenId  // 2
        case "[INST]":
            return 3  // Valeur Python exacte
        case "[/INST]":
            return 4  // Valeur Python exacte
        case "[AUDIO]":
            return audioTokenIdInternal  // 24
        case "[BEGIN_AUDIO]":
            return 25  // Valeur Python exacte
        case "[TRANSCRIBE]":
            return 34  // Valeur Python exacte
        default:
            // Return -1 for unknown control tokens (Python Tekkenizer behaviour)
            return -1
        }
    }
    
    var audioTokenId: Int { return audioTokenIdInternal }
    var padTokenIdValue: Int { return padTokenId }
    var eosTokenIdValue: Int { return eosTokenId }
    var bosTokenIdValue: Int { return bosTokenId }
    var hasGetControlToken: Bool = true  // getControlToken corrigé, retourne les bonnes valeurs
    var hasAudioTokenId: Bool = true
    var hasVocab: Bool { return !mergeableRanks.isEmpty }
    var hasPadTokenId: Bool = true
    var hasDecodeMethod: Bool = true
    var hasEncodeMethod: Bool = true
    var hasCallMethod: Bool = true
    
    // For compatibility with VoxtralProcessor that expects callAsFunction
    func callAsFunction(
        text: String,
        returnTensors: String = "mlx",
        padding: Bool = true
    ) throws -> [String: MLXArray] {
        let tokenIds = encode(text)
        var result: [String: MLXArray] = [
            "input_ids": MLXArray(tokenIds, [1, tokenIds.count])
        ]
        
        if padding {
            result["attention_mask"] = MLXArray.ones(like: result["input_ids"]!)
        }
        
        return result
    }
    
    static func fromPretrained(
        _ modelPath: String,
        progress: TokenizerProgressCallback? = nil
    ) throws -> TekkenTokenizer {
        return try load(modelPath: modelPath, progress: progress)
    }
    
    func batchDecode(_ tokenIdsList: [[Int]], skipSpecialTokens: Bool = true) -> [String] {
        return tokenIdsList.map { decode($0, skipSpecialTokens: skipSpecialTokens) }
    }
}

// MARK: - 2. AUDIO ENCODER (VALIDÉ)

/**
 * AudioEncoder - Audio tokenisation 75fps exacte
 */

// MARK: - 3. CHAT TEMPLATE PROCESSOR (VALIDÉ)

/**
 * ChatTemplateProcessor - Application template Voxtral exacte
 */