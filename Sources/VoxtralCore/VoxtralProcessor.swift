/**
 * VoxtralProcessor - Swift equivalent of mlx.voxtral/processing_voxtral.py
 * 
 * EXACT conversion of Python VoxtralProcessor class functionality.
 * Direct line-by-line translation following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

import Foundation
import MLX
import MLXNN

/**
 * Direct Python equivalent: class VoxtralProcessor:
 */
public class VoxtralProcessor {
    
    // Python: self.feature_extractor = feature_extractor or VoxtralFeatureExtractor()
    let featureExtractor: VoxtralFeatureExtractor
    
    // Python: self.tokenizer = tokenizer  
    let tokenizer: TekkenTokenizer?
    
    // Python: self._special_token_ids = self._get_special_token_ids()
    public let specialTokenIds: [String: Int]?
    
    /**
     * Direct Python equivalent: def __init__(self, feature_extractor=None, tokenizer=None):
     */
    public init(
        featureExtractor: VoxtralFeatureExtractor? = nil,
        tokenizer: TekkenTokenizer? = nil
    ) {
        self.featureExtractor = featureExtractor ?? VoxtralFeatureExtractor()
        self.tokenizer = tokenizer
        
        // Initialize specialTokenIds after all stored properties
        if let tokenizer = tokenizer {
            var specialTokens: [String: Int] = [:]
            
            // LOGIQUE PYTHON EXACTE : Suivre la même approche de fallback à 3 niveaux
            // Tier 1: Essayer get_control_token (si disponible)
            if tokenizer.hasGetControlToken {
                specialTokens["bos"] = tokenizer.getControlToken("<s>")
                specialTokens["eos"] = tokenizer.getControlToken("</s>")
                specialTokens["inst"] = tokenizer.getControlToken("[INST]")
                specialTokens["inst_end"] = tokenizer.getControlToken("[/INST]")
                specialTokens["audio"] = tokenizer.getControlToken("[AUDIO]")
                specialTokens["begin_audio"] = tokenizer.getControlToken("[BEGIN_AUDIO]")
                specialTokens["transcribe"] = tokenizer.getControlToken("[TRANSCRIBE]")
            }
            // Tier 2: Utiliser audio_token_id + vocab (si disponible)
            else if tokenizer.hasAudioTokenId {
                specialTokens["audio"] = tokenizer.audioTokenId
                if tokenizer.hasVocab {
                    let vocab = tokenizer.vocab
                    specialTokens["bos"] = vocab["<s>"] ?? 1
                    specialTokens["eos"] = vocab["</s>"] ?? 2
                    specialTokens["inst"] = vocab["[INST]"] ?? 3
                    specialTokens["inst_end"] = vocab["[/INST]"] ?? 4
                    specialTokens["begin_audio"] = vocab["[BEGIN_AUDIO]"] ?? 25
                    specialTokens["transcribe"] = vocab["[TRANSCRIBE]"] ?? 34
                } else {
                    // Fallback si pas de vocab
                    specialTokens["bos"] = tokenizer.bosTokenIdValue
                    specialTokens["eos"] = tokenizer.eosTokenIdValue
                    specialTokens["inst"] = 3
                    specialTokens["inst_end"] = 4
                    specialTokens["begin_audio"] = 25
                    specialTokens["transcribe"] = 34
                }
            }
            // Tier 3: Hardcoded defaults (VALEURS PYTHON EXACTES)
            else {
                specialTokens["bos"] = 1
                specialTokens["eos"] = 2
                specialTokens["inst"] = 3
                specialTokens["inst_end"] = 4
                specialTokens["audio"] = 24
                specialTokens["begin_audio"] = 25
                specialTokens["transcribe"] = 34
            }
            
            if tokenizer.hasPadTokenId {
                specialTokens["pad"] = tokenizer.padTokenIdValue
            } else {
                specialTokens["pad"] = 0
            }
            
            self.specialTokenIds = specialTokens
        } else {
            self.specialTokenIds = nil
        }
    }
    
    /**
     * Direct Python equivalent: def _get_special_token_ids(self) -> Dict[str, int]:
     */
    private func _getSpecialTokenIds() -> [String: Int] {
        var specialTokens: [String: Int] = [:]
        
        guard let tokenizer = self.tokenizer else {
            return [:]
        }
        
        // Python: if hasattr(self.tokenizer, 'get_control_token'):
        if tokenizer.hasGetControlToken {
            // Python: special_tokens['bos'] = self.tokenizer.get_control_token('<s>')
            specialTokens["bos"] = tokenizer.getControlToken("<s>")
            specialTokens["eos"] = tokenizer.getControlToken("</s>")
            specialTokens["inst"] = tokenizer.getControlToken("[INST]")
            specialTokens["inst_end"] = tokenizer.getControlToken("[/INST]")
            specialTokens["audio"] = tokenizer.getControlToken("[AUDIO]")
            specialTokens["begin_audio"] = tokenizer.getControlToken("[BEGIN_AUDIO]")
            specialTokens["transcribe"] = tokenizer.getControlToken("[TRANSCRIBE]")
        } else if tokenizer.hasAudioTokenId {
            // Python: elif hasattr(self.tokenizer, 'audio_token_id'):
            specialTokens["audio"] = tokenizer.audioTokenId
            if tokenizer.hasVocab {
                let vocab = tokenizer.vocab
                specialTokens["bos"] = vocab["<s>"] ?? 1
                specialTokens["eos"] = vocab["</s>"] ?? 2
                specialTokens["inst"] = vocab["[INST]"] ?? 3
                specialTokens["inst_end"] = vocab["[/INST]"] ?? 4
                specialTokens["begin_audio"] = vocab["[BEGIN_AUDIO]"] ?? 25
                specialTokens["transcribe"] = vocab["[TRANSCRIBE]"] ?? 34
            }
        } else {
            // Python: else: special_tokens = { 'bos': 1, ... }
            specialTokens = [
                "bos": 1,
                "eos": 2,
                "inst": 3,
                "inst_end": 4,
                "audio": 24,
                "begin_audio": 25,
                "transcribe": 34
            ]
        }
        
        // Python: if hasattr(self.tokenizer, 'pad_token_id'):
        if tokenizer.hasPadTokenId {
            specialTokens["pad"] = tokenizer.padTokenIdValue
        } else {
            specialTokens["pad"] = 0
        }
        
        return specialTokens
    }
    
    /**
     * Direct Python equivalent: def __call__(self, text: Optional[str] = None, audio: Optional[Union[np.ndarray, List[float], str]] = None, ...):
     */
    public func callAsFunction(
        text: String? = nil,
        audio: Any? = nil, // Union[MLXArray, [Float], String]
        samplingRate: Int = 16000,
        padding: Bool = true
    ) throws -> [String: MLXArray] {
        
        var encoding: [String: MLXArray] = [:]
        
        // Python: if audio is not None:
        if let audio = audio {
            // Python: audio_features = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="mlx")
            let audioFeatures = try featureExtractor.callAsFunction(
                rawSpeech: audio,
                samplingRate: samplingRate,
                returnTensors: "mlx"
            )
            // Python: encoding["input_features"] = audio_features["input_features"]
            encoding["input_features"] = (audioFeatures["input_features"] as! MLXArray)
        }
        
        // Python: if text is not None and self.tokenizer is not None:
        if let text = text, let tokenizer = self.tokenizer {
            
            // Python: if hasattr(self.tokenizer, "__call__"):
            if tokenizer.hasCallMethod {
                // Python: text_encoding = self.tokenizer(text, return_tensors="mlx", padding=padding, **kwargs)
                let textEncoding = try tokenizer.callAsFunction(
                    text: text,
                    returnTensors: "mlx",
                    padding: padding
                )
                encoding["input_ids"] = textEncoding["input_ids"]!
                if let attentionMask = textEncoding["attention_mask"] {
                    encoding["attention_mask"] = attentionMask
                }
            } else {
                // Python: token_ids = self.tokenizer.encode(text)
                var tokenIds = tokenizer.encode(text)
                
                // Python: if len(token_ids) > 0 and token_ids[-1] == 2:  # 2 is EOS
                if tokenIds.count > 0 && tokenIds.last == 2 {
                    tokenIds = Array(tokenIds.dropLast())
                }
                
                // Python: if len(token_ids) > 1 and token_ids[0] == 1 and token_ids[1] == 1:
                if tokenIds.count > 1 && tokenIds[0] == 1 && tokenIds[1] == 1 {
                    tokenIds = Array(tokenIds.dropFirst())
                }
                
                // Python: encoding["input_ids"] = mx.array([token_ids])
                encoding["input_ids"] = MLXArray(tokenIds, [1, tokenIds.count])
                
                if padding {
                    // Python: encoding["attention_mask"] = mx.ones_like(encoding["input_ids"])
                    encoding["attention_mask"] = MLXArray.ones(like: encoding["input_ids"]!)
                }
            }
            
            // Python: if audio is not None and self._special_token_ids is not None:
            if audio != nil, let specialTokenIds = self.specialTokenIds {
                // Python: audio_placeholder_sequences = [ [8175, 9383, 1062], [1534, 57401, 1062], ]
                let audioPlaceholderSequences: [[Int]] = [
                    [8175, 9383, 1062],  // "<audio>"
                    [1534, 57401, 1062]  // " <audio>"
                ]
                
                var newInputIds: [MLXArray] = []
                
                // Python: for batch_idx in range(encoding["input_ids"].shape[0]):
                for batchIdx in 0..<encoding["input_ids"]!.shape[0] {
                    // Python: batch_ids = encoding["input_ids"][batch_idx]
                    let batchIds = encoding["input_ids"]![batchIdx]
                    // Python: ids_list = batch_ids.tolist()
                    var idsList = batchIds.asArray(Int.self)
                    
                    var found = false
                    
                    // Python: for audio_placeholder_sequence in audio_placeholder_sequences:
                    for audioPlaceholderSequence in audioPlaceholderSequences {
                        // Python: for i in range(len(ids_list) - len(audio_placeholder_sequence) + 1):
                        for i in 0..<(idsList.count - audioPlaceholderSequence.count + 1) {
                            // Python: if ids_list[i : i + len(audio_placeholder_sequence)] == audio_placeholder_sequence:
                            let slice = Array(idsList[i..<(i + audioPlaceholderSequence.count)])
                            if slice == audioPlaceholderSequence {
                                // Python: for _ in range(len(audio_placeholder_sequence)):
                                for _ in 0..<audioPlaceholderSequence.count {
                                    idsList.remove(at: i)
                                }
                                
                                // Python: audio_token_id = self._special_token_ids.get('audio', 24)
                                let audioTokenId = specialTokenIds["audio"] ?? 24
                                
                                // Python: for _ in range(375):
                                for j in 0..<375 {
                                    idsList.insert(audioTokenId, at: i + j)
                                }
                                found = true
                                break
                            }
                        }
                        if found { break }
                    }
                    
                    // Python: new_input_ids.append(mx.array(ids_list))
                    newInputIds.append(MLXArray(idsList.map { Int32($0) }))
                    
                    // Python: if found and "attention_mask" in encoding:
                    if found && encoding["attention_mask"] != nil {
                        // Python: encoding["attention_mask"] = mx.ones((encoding["input_ids"].shape[0], len(ids_list)))
                        encoding["attention_mask"] = ones([encoding["input_ids"]!.shape[0], idsList.count])
                    }
                }
                
                // Python: if new_input_ids:
                if !newInputIds.isEmpty {
                    // Python: max_len = max(len(ids) for ids in new_input_ids)
                    let maxLen = newInputIds.map { $0.shape[0] }.max() ?? 0
                    
                    var paddedIds: [MLXArray] = []
                    
                    // Python: for ids in new_input_ids:
                    for ids in newInputIds {
                        // Python: if len(ids) < max_len:
                        if ids.shape[0] < maxLen {
                            // Python: pad_token_id = self._special_token_ids.get('pad', 0)
                            let padTokenId = specialTokenIds["pad"] ?? 0
                            // Python: padding = [pad_token_id] * (max_len - len(ids))
                            let paddingArray = Array(repeating: Int32(padTokenId), count: maxLen - ids.shape[0])
                            // Python: padded_ids.append(mx.concatenate([ids, mx.array(padding)]))
                            paddedIds.append(concatenated([ids, MLXArray(paddingArray)]))
                        } else {
                            paddedIds.append(ids)
                        }
                    }
                    
                    // Python: encoding["input_ids"] = mx.stack(padded_ids)
                    encoding["input_ids"] = stacked(paddedIds)
                }
            }
        }
        
        return encoding
    }
    
    /**
     * Direct Python equivalent: def batch_decode(self, token_ids, **kwargs):
     */
    public func batchDecode(_ tokenIds: Any, skipSpecialTokens: Bool = true) throws -> [String] {
        guard let tokenizer = self.tokenizer else {
            throw VoxtralError.tokenizerNotAvailable
        }
        
        // Python: if isinstance(token_ids, mx.array):
        let processedTokenIds: [[Int]]
        if let mlxTokenIds = tokenIds as? MLXArray {
            // Convert MLXArray to [[Int]]
            let batchSize = mlxTokenIds.shape[0]
            processedTokenIds = (0..<batchSize).map { batchIdx in
                mlxTokenIds[batchIdx].asArray(Int.self)
            }
        } else if let arrayTokenIds = tokenIds as? [[Int]] {
            processedTokenIds = arrayTokenIds
        } else {
            throw VoxtralError.invalidTokenFormat
        }
        
        return tokenizer.batchDecode(processedTokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    
    /**
     * Direct Python equivalent: def decode(self, token_ids, **kwargs):
     */
    public func decode(_ tokenIds: Any, skipSpecialTokens: Bool = true) throws -> String {
        guard let tokenizer = self.tokenizer else {
            throw VoxtralError.tokenizerNotAvailable
        }
        
        // Python: if isinstance(token_ids, mx.array):
        let processedTokenIds: [Int]
        if let mlxTokenIds = tokenIds as? MLXArray {
            processedTokenIds = mlxTokenIds.asArray(Int.self)
        } else if let arrayTokenIds = tokenIds as? [Int] {
            processedTokenIds = arrayTokenIds
        } else {
            throw VoxtralError.invalidTokenFormat
        }
        
        // Python: if hasattr(self.tokenizer, "decode") and callable(self.tokenizer.decode):
        if tokenizer.hasDecodeMethod {
            return tokenizer.decode(processedTokenIds, skipSpecialTokens: skipSpecialTokens)
        } else {
            return tokenizer.decode(processedTokenIds)
        }
    }
    
    /**
     * Direct Python equivalent: def apply_transcrition_request(self, audio: Union[str, np.ndarray, List[float]], language: Optional[str] = None, sampling_rate: Optional[int] = None):
     */
    public func applyTranscritionRequest(
        audio: Any, // Union[String, MLXArray, [Float]]
        language: String? = nil,
        samplingRate: Int? = nil
    ) throws -> ProcessedInputs {
        // Python: audio_features = self.feature_extractor(audio, sampling_rate=sampling_rate if sampling_rate else 16000, return_tensors="mlx")
        let audioFeatures = try featureExtractor.callAsFunction(
            rawSpeech: audio,
            samplingRate: samplingRate ?? 16000,
            returnTensors: "mlx"
        )
        
        // Python: if self._special_token_ids is None:
        guard let specialTokenIds = self.specialTokenIds else {
            throw VoxtralError.tokenizerRequired("Tokenizer is required for applyTranscritionRequest")
        }
        
        // Python: tokens = [self._special_token_ids['bos']]
        var tokens = [specialTokenIds["bos"]!]
        
        // Python: tokens.append(self._special_token_ids['inst'])
        tokens.append(specialTokenIds["inst"]!)
        
        // Python: tokens.append(self._special_token_ids['begin_audio'])
        tokens.append(specialTokenIds["begin_audio"]!)
        
        // Python: if isinstance(audio_features, dict):
        let inputFeatures: MLXArray
        if let featuresDict = audioFeatures as? [String: MLXArray] {
            inputFeatures = featuresDict["input_features"]!
        } else {
            // Handle case where audioFeatures has input_features property
            inputFeatures = audioFeatures["input_features"] as! MLXArray
        }
        
        // Python: num_chunks = input_features.shape[0]
        let numChunks = inputFeatures.shape[0]
        // Python: num_audio_tokens = num_chunks * 375
        let numAudioTokens = numChunks * 375
        
        // Python: tokens.extend([self._special_token_ids['audio']] * num_audio_tokens)
        let audioTokenId = specialTokenIds["audio"]!
        for _ in 0..<numAudioTokens {
            tokens.append(audioTokenId)
        }
        
        // Python: tokens.append(self._special_token_ids['inst_end'])
        tokens.append(specialTokenIds["inst_end"]!)
        
        // Python: if language is not None:
        if let language = language {
            // Python: lang_str = f"lang:{language}"
            let langStr = "lang:\(language)"
            
            // Python: if hasattr(self.tokenizer, 'encode'):
            if let tokenizer = self.tokenizer, tokenizer.hasEncodeMethod {
                // Python: lang_tokens = self.tokenizer.encode(lang_str, add_special_tokens=False)
                let langTokens = tokenizer.encode(langStr, addSpecialTokens: false)
                // Python: tokens.extend(lang_tokens)
                tokens.append(contentsOf: langTokens)
            } else {
                // Python: if language == "en":
                if language == "en" {
                    // Python: tokens.extend([9909, 1058, 1262])  # "lang:en"
                    tokens.append(contentsOf: [9909, 1058, 1262])
                } else {
                    throw VoxtralError.languageNotSupported("Language \(language) not yet supported")
                }
            }
        }
        
        // Python: tokens.append(self._special_token_ids['transcribe'])
        tokens.append(specialTokenIds["transcribe"]!)
        
        // Python: input_ids = mx.array([tokens], dtype=mx.int32)
        let inputIds = MLXArray(tokens.map { Int32($0) }, [1, tokens.count])
        
        // Python: return TranscriptionInputs(input_ids, input_features)
        return ProcessedInputs(inputIds: inputIds, inputFeatures: inputFeatures)
    }
    
    /// Progress callback type for processor loading
    public typealias ProcessorProgressCallback = @Sendable (Double, String) -> Void

    /**
     * Direct Python equivalent: @classmethod def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
     */
    public static func fromPretrained(
        _ pretrainedModelNameOrPath: String,
        progress: ProcessorProgressCallback? = nil
    ) throws -> VoxtralProcessor {
        // Python: tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        progress?(0.0, "Loading tokenizer...")
        let tokenizer = try TekkenTokenizer.fromPretrained(pretrainedModelNameOrPath) { tokenizerProgress, status in
            // Map tokenizer progress (0-1) to processor progress (0-0.9)
            progress?(tokenizerProgress * 0.9, status)
        }

        // Python: feature_extractor = VoxtralFeatureExtractor()
        progress?(0.9, "Initializing feature extractor...")
        let featureExtractor = VoxtralFeatureExtractor()

        progress?(1.0, "Processor ready")

        // Python: return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
        return VoxtralProcessor(featureExtractor: featureExtractor, tokenizer: tokenizer)
    }
    
    /**
     * Direct Python equivalent: def _parse_message_content(self, content: Union[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Union[str, np.ndarray]]]
     */
    private func _parseMessageContent(_ content: Any) throws -> (contentItems: [[String: Any]], audioData: [Any]) {
        // Python: if isinstance(content, str):
        //         return [{"type": "text", "text": content}], []
        if let textContent = content as? String {
            return ([["type": "text", "text": textContent]], [])
        }
        
        // Python: content_items = []
        //         audio_data = []
        var contentItems: [[String: Any]] = []
        var audioData: [Any] = []
        
        guard let contentArray = content as? [[String: Any]] else {
            throw VoxtralError.invalidInput("Content must be string or array of dictionaries")
        }
        
        // Python: for item in content:
        for item in contentArray {
            guard let itemType = item["type"] as? String else {
                throw VoxtralError.invalidInput("Content item must have 'type' field")
            }
            
            // Python: if item["type"] == "text":
            //         content_items.append(item)
            if itemType == "text" {
                contentItems.append(item)
            } else if itemType == "audio" {
                // Python: elif item["type"] == "audio":
                //         content_items.append(item)
                contentItems.append(item)
                
                // Python: if "audio" in item: audio_data.append(item["audio"])
                if let audio = item["audio"] {
                    audioData.append(audio)
                } else if let path = item["path"] as? String {
                    // Python: elif "path" in item: audio_data.append(item["path"])
                    audioData.append(path)
                } else if let url = item["url"] as? String {
                    // Python: elif "url" in item: audio_data.append(item["url"])
                    audioData.append(url)
                } else if let base64 = item["base64"] as? String {
                    // Python: elif "base64" in item: audio_data.append(item["base64"])
                    audioData.append(base64)
                } else {
                    throw VoxtralError.invalidInput("Audio content must have 'audio', 'path', 'url', or 'base64' field")
                }
            } else {
                // Python: else: raise ValueError(f"Unknown content type: {item['type']}")
                throw VoxtralError.invalidInput("Unknown content type: \(itemType)")
            }
        }
        
        return (contentItems, audioData)
    }
    
    /**
     * Direct Python equivalent: def apply_chat_template(self, conversation: Union[List[Dict[str, Any]], Dict[str, Any]], tokenize: bool = True, add_generation_prompt: bool = False, return_tensors: Optional[str] = None, **kwargs) -> Union[str, Dict[str, mx.array]]
     */
    public func applyChatTemplate(
        conversation: Any,
        tokenize: Bool = true,
        addGenerationPrompt: Bool = false,
        returnTensors: String? = nil
    ) throws -> Any {
        // Python: if self._special_token_ids is None:
        //         raise ValueError("Tokenizer is required for apply_chat_template")
        guard let specialTokenIds = self.specialTokenIds else {
            throw VoxtralError.tokenizerRequired("Tokenizer is required for apply_chat_template")
        }
        
        // Python: if isinstance(conversation, dict):
        //         conversation = [conversation]
        let conversationArray: [[String: Any]]
        if let singleMessage = conversation as? [String: Any] {
            conversationArray = [singleMessage]
        } else if let messageArray = conversation as? [[String: Any]] {
            conversationArray = messageArray
        } else {
            throw VoxtralError.invalidInput("Conversation must be a dictionary or array of dictionaries")
        }
        
        // Python: all_tokens = []
        //         all_audio_features = []
        var allTokens: [Int] = []
        var allAudioFeatures: [MLXArray] = []
        
        // Python: all_tokens.append(self._special_token_ids['bos'])
        allTokens.append(specialTokenIds["bos"]!)
        
        // Python: for i, message in enumerate(conversation):
        for (i, message) in conversationArray.enumerated() {
            guard let role = message["role"] as? String,
                  let content = message["content"] else {
                throw VoxtralError.invalidInput("Message must have 'role' and 'content' fields")
            }
            
            // Python: content_items, audio_data = self._parse_message_content(content)
            let (contentItems, _) = try _parseMessageContent(content)
            
            if role == "system" {
                // Python: # System message format: system\n{content}</s>
                //         all_tokens.extend(self.tokenizer.encode("system\n", add_special_tokens=False))
                if let tokenizer = self.tokenizer {
                    let systemTokens = tokenizer.encode("system\n", addSpecialTokens: false)
                    allTokens.append(contentsOf: systemTokens)
                    
                    // Python: for item in content_items:
                    //         if item["type"] == "text":
                    //             text_tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
                    //             all_tokens.extend(text_tokens)
                    for item in contentItems {
                        if let itemType = item["type"] as? String, itemType == "text",
                           let text = item["text"] as? String {
                            let textTokens = tokenizer.encode(text, addSpecialTokens: false)
                            allTokens.append(contentsOf: textTokens)
                        }
                    }
                    
                    // Python: all_tokens.append(self._special_token_ids['eos'])
                    allTokens.append(specialTokenIds["eos"]!)
                }
                
            } else if role == "user" {
                // Python: # User message format: [INST]{content}[/INST]
                //         all_tokens.append(self._special_token_ids['inst'])
                allTokens.append(specialTokenIds["inst"]!)
                
                // Python: for item in content_items:
                for item in contentItems {
                    if let itemType = item["type"] as? String {
                        if itemType == "text" {
                            // Python: if item["type"] == "text":
                            //         text_tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
                            //         all_tokens.extend(text_tokens)
                            if let text = item["text"] as? String,
                               let tokenizer = self.tokenizer {
                                let textTokens = tokenizer.encode(text, addSpecialTokens: false)
                                allTokens.append(contentsOf: textTokens)
                            }
                        } else if itemType == "audio" {
                            // Python: elif item["type"] == "audio":
                            let audioInput: Any
                            if let audio = item["audio"] {
                                audioInput = audio
                            } else if let path = item["path"] {
                                audioInput = path
                            } else if let url = item["url"] {
                                audioInput = url
                            } else {
                                throw VoxtralError.invalidInput("Audio item must have audio, path, or url")
                            }
                            
                            // Python: sampling_rate = item.get("sampling_rate", None)
                            let samplingRate = item["sampling_rate"] as? Int
                            
                            // Python: audio_features = self.feature_extractor(audio_input, sampling_rate=sampling_rate, return_tensors="mlx")
                            let audioFeatures = try featureExtractor.callAsFunction(
                                rawSpeech: audioInput,
                                samplingRate: samplingRate,
                                returnTensors: "mlx"
                            )
                            
                            // Python: if isinstance(audio_features, dict):
                            //         features = audio_features["input_features"]
                            let features: MLXArray
                            if let featuresDict = audioFeatures as? [String: MLXArray] {
                                features = featuresDict["input_features"]!
                            } else {
                                features = audioFeatures["input_features"] as! MLXArray
                            }
                            
                            allAudioFeatures.append(features)
                            
                            // Python: all_tokens.append(self._special_token_ids['begin_audio'])
                            allTokens.append(specialTokenIds["begin_audio"]!)
                            // Python: num_chunks = features.shape[0]
                            //         num_audio_tokens = num_chunks * 375
                            //         all_tokens.extend([self._special_token_ids['audio']] * num_audio_tokens)
                            let numChunks = features.shape[0]
                            let numAudioTokens = numChunks * 375
                            for _ in 0..<numAudioTokens {
                                allTokens.append(specialTokenIds["audio"]!)
                            }
                        }
                    }
                }
                
                // Python: all_tokens.append(self._special_token_ids['inst_end'])
                allTokens.append(specialTokenIds["inst_end"]!)
                
                // Python: if i == len(conversation) - 1 and len(content_items) == 1 and content_items[0]["type"] == "audio":
                if i == conversationArray.count - 1 && contentItems.count == 1 &&
                   (contentItems[0]["type"] as? String) == "audio" {
                    // Python: lang_str = "lang:en"  # Default to English
                    //         lang_tokens = self.tokenizer.encode(lang_str, add_special_tokens=False)
                    //         all_tokens.extend(lang_tokens)
                    //         all_tokens.append(self._special_token_ids['transcribe'])
                    let langStr = "lang:en"
                    if let tokenizer = self.tokenizer {
                        let langTokens = tokenizer.encode(langStr, addSpecialTokens: false)
                        allTokens.append(contentsOf: langTokens)
                    }
                    allTokens.append(specialTokenIds["transcribe"]!)
                }
                
            } else if role == "assistant" {
                // Python: elif role == "assistant":
                //         for item in content_items:
                //             if item["type"] == "text":
                //                 text_tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
                //                 all_tokens.extend(text_tokens)
                for item in contentItems {
                    if let itemType = item["type"] as? String, itemType == "text",
                       let text = item["text"] as? String,
                       let tokenizer = self.tokenizer {
                        let textTokens = tokenizer.encode(text, addSpecialTokens: false)
                        allTokens.append(contentsOf: textTokens)
                    }
                }
                
                // Python: if not (i == len(conversation) - 1 and add_generation_prompt):
                //         all_tokens.append(self._special_token_ids['eos'])
                if !(i == conversationArray.count - 1 && addGenerationPrompt) {
                    allTokens.append(specialTokenIds["eos"]!)
                }
            } else {
                // Python: else: raise ValueError(f"Unknown role: {role}")
                throw VoxtralError.invalidInput("Unknown role: \(role)")
            }
        }
        
        // Python: if not tokenize:
        //         if hasattr(self.tokenizer, 'decode'):
        //             return self.tokenizer.decode(all_tokens)
        //         else:
        //             return "<formatted string not available>"
        if !tokenize {
            if let tokenizer = self.tokenizer {
                return tokenizer.decode(allTokens, skipSpecialTokens: false)
            } else {
                return "<formatted string not available>"
            }
        }
        
        // Python: output = {}
        var output: [String: Any] = [:]
        
        // Python: if return_tensors == "mlx":
        //         output["input_ids"] = mx.array([all_tokens], dtype=mx.int32)
        if returnTensors == "mlx" {
            output["input_ids"] = MLXArray(allTokens.map { Int32($0) }, [1, allTokens.count])
            
            // Python: if all_audio_features:
            //         output["input_features"] = mx.concatenate(all_audio_features, axis=0)
            if !allAudioFeatures.isEmpty {
                output["input_features"] = concatenated(allAudioFeatures, axis: 0)
            }
            
            // Python: output["attention_mask"] = mx.ones_like(output["input_ids"])
            output["attention_mask"] = ones(like: output["input_ids"] as! MLXArray)
        } else {
            // Python: else:
            //         output["input_ids"] = [all_tokens]
            output["input_ids"] = [allTokens]
            
            // Python: if all_audio_features:
            //         output["input_features"] = all_audio_features
            if !allAudioFeatures.isEmpty {
                output["input_features"] = allAudioFeatures
            }
            
            // Python: output["attention_mask"] = [[1] * len(all_tokens)]
            output["attention_mask"] = [Array(repeating: 1, count: allTokens.count)]
        }
        
        return output
    }
}