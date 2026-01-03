/**
 * VoxtralConfiguration - Swift equivalent of mlx.voxtral/configuration_voxtral.py
 * 
 * Exact conversion of Python VoxtralEncoderConfig, VoxtralTextConfig, and VoxtralConfig classes.
 * Direct line-by-line translation following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

import Foundation

/**
 * Direct Python equivalent: class VoxtralEncoderConfig
 * Python reference: lines 5-96 in configuration_voxtral.py
 */
public struct VoxtralEncoderConfig: Codable {
    // Python: model_type = "voxtral_encoder" (line 12)
    public static let modelType = "voxtral_encoder"
    
    // Python: attribute_map (lines 14-20) - compatibility mapping
    public static let attributeMap: [String: String] = [
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers", 
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop"
    ]
    
    // Python: def __init__ parameters with exact defaults (lines 22-42)
    public let vocab_size: Int          // Python: vocab_size: int = 51866
    public let hidden_size: Int         // Python: hidden_size: int = 1280
    public let intermediate_size: Int   // Python: intermediate_size: int = 5120
    public let num_hidden_layers: Int   // Python: num_hidden_layers: int = 32
    public let num_attention_heads: Int // Python: num_attention_heads: int = 20
    public let scale_embedding: Bool    // Python: scale_embedding: bool = False
    public let activation_function: String // Python: activation_function: str = "gelu"
    public let num_mel_bins: Int        // Python: num_mel_bins: int = 128
    public let max_source_positions: Int // Python: max_source_positions: int = 1500
    public let initializer_range: Double // Python: initializer_range: float = 0.02
    public let attention_dropout: Double // Python: attention_dropout: float = 0.0
    public let dropout: Double          // Python: dropout: float = 0.0
    public let layerdrop: Double        // Python: layerdrop: float = 0.0
    public let activation_dropout: Double // Python: activation_dropout: float = 0.0
    public let pad_token_id: Int        // Python: pad_token_id: int = 0
    public let head_dim: Int            // Python: head_dim: int = 64
    public let num_key_value_heads: Int // Python: num_key_value_heads: int = 20
    
    // Python: def __init__ with exact default values (lines 22-62)
    public init(
        vocab_size: Int = 51866,
        hidden_size: Int = 1280,
        intermediate_size: Int = 5120,
        num_hidden_layers: Int = 32,
        num_attention_heads: Int = 20,
        scale_embedding: Bool = false,
        activation_function: String = "gelu",
        num_mel_bins: Int = 128,
        max_source_positions: Int = 1500,
        initializer_range: Double = 0.02,
        attention_dropout: Double = 0.0,
        dropout: Double = 0.0,
        layerdrop: Double = 0.0,
        activation_dropout: Double = 0.0,
        pad_token_id: Int = 0,
        head_dim: Int = 64,
        num_key_value_heads: Int = 20
    ) {
        // Python: lines 43-59 - direct assignment
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        self.pad_token_id = pad_token_id
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        
        // Python: for key, value in kwargs.items(): setattr(self, key, value) (lines 61-62)
        // Note: Swift doesn't support dynamic properties like Python, this is handled in from_dict
    }
    
    // Python: @property methods (lines 64-82) - computed properties for compatibility
    public var d_model: Int { return hidden_size }
    public var encoder_layers: Int { return num_hidden_layers }
    public var encoder_attention_heads: Int { return num_attention_heads }
    public var encoder_ffn_dim: Int { return intermediate_size }
    public var encoder_layerdrop: Double { return layerdrop }
    
    /**
     * Direct Python equivalent: def to_dict(self) -> Dict (lines 84-90)
     * Python: Convert config to dictionary
     */
    public func to_dict() -> [String: Any] {
        // Python: output = {}
        var output: [String: Any] = [:]
        
        // Python: for key, value in self.__dict__.items():
        //         if not key.startswith("_"): output[key] = value
        output["vocab_size"] = vocab_size
        output["hidden_size"] = hidden_size
        output["intermediate_size"] = intermediate_size
        output["num_hidden_layers"] = num_hidden_layers
        output["num_attention_heads"] = num_attention_heads
        output["scale_embedding"] = scale_embedding
        output["activation_function"] = activation_function
        output["num_mel_bins"] = num_mel_bins
        output["max_source_positions"] = max_source_positions
        output["initializer_range"] = initializer_range
        output["attention_dropout"] = attention_dropout
        output["dropout"] = dropout
        output["layerdrop"] = layerdrop
        output["activation_dropout"] = activation_dropout
        output["pad_token_id"] = pad_token_id
        output["head_dim"] = head_dim
        output["num_key_value_heads"] = num_key_value_heads
        
        return output
    }
    
    /**
     * Direct Python equivalent: @classmethod def from_dict(cls, config_dict: Dict) (lines 92-95)
     * Python: Create config from dictionary
     */
    public static func from_dict(_ config_dict: [String: Any]) -> VoxtralEncoderConfig {
        // Python: return cls(**config_dict)
        return VoxtralEncoderConfig(
            vocab_size: config_dict["vocab_size"] as? Int ?? 51866,
            hidden_size: config_dict["hidden_size"] as? Int ?? 1280,
            intermediate_size: config_dict["intermediate_size"] as? Int ?? 5120,
            num_hidden_layers: config_dict["num_hidden_layers"] as? Int ?? 32,
            num_attention_heads: config_dict["num_attention_heads"] as? Int ?? 20,
            scale_embedding: config_dict["scale_embedding"] as? Bool ?? false,
            activation_function: config_dict["activation_function"] as? String ?? "gelu",
            num_mel_bins: config_dict["num_mel_bins"] as? Int ?? 128,
            max_source_positions: config_dict["max_source_positions"] as? Int ?? 1500,
            initializer_range: config_dict["initializer_range"] as? Double ?? 0.02,
            attention_dropout: config_dict["attention_dropout"] as? Double ?? 0.0,
            dropout: config_dict["dropout"] as? Double ?? 0.0,
            layerdrop: config_dict["layerdrop"] as? Double ?? 0.0,
            activation_dropout: config_dict["activation_dropout"] as? Double ?? 0.0,
            pad_token_id: config_dict["pad_token_id"] as? Int ?? 0,
            head_dim: config_dict["head_dim"] as? Int ?? 64,
            num_key_value_heads: config_dict["num_key_value_heads"] as? Int ?? 20
        )
    }
    
    // Compatibility method for existing code
    public static func fromDictionary(_ config_dict: [String: Any]) -> VoxtralEncoderConfig {
        return from_dict(config_dict)
    }
}

/**
 * Direct Python equivalent: @dataclass class VoxtralTextConfig (lines 98-159)
 * Python reference: Configuration for Mistral/Llama text decoder
 */
public struct VoxtralTextConfig: Codable {
    // Python: @dataclass fields with exact default values (lines 102-122)
    public let vocab_size: Int                    // Python: vocab_size: int = 131072
    public let hidden_size: Int                   // Python: hidden_size: int = 3072  
    public let intermediate_size: Int             // Python: intermediate_size: int = 8192
    public let num_hidden_layers: Int             // Python: num_hidden_layers: int = 30
    public let num_attention_heads: Int           // Python: num_attention_heads: int = 32
    public let num_key_value_heads: Int           // Python: num_key_value_heads: int = 8
    public let max_position_embeddings: Int       // Python: max_position_embeddings: int = 131072
    public let rms_norm_eps: Double               // Python: rms_norm_eps: float = 1e-05
    public let rope_theta: Double                 // Python: rope_theta: float = 100000000.0
    public let rope_scaling: [String: Double]?    // Python: rope_scaling: Optional[Dict] = None
    public let tie_word_embeddings: Bool          // Python: tie_word_embeddings: bool = False
    public let use_cache: Bool                    // Python: use_cache: bool = True
    public let hidden_act: String                 // Python: hidden_act: str = "silu"
    public let initializer_range: Double          // Python: initializer_range: float = 0.02
    public let attention_bias: Bool               // Python: attention_bias: bool = False
    public let attention_dropout: Double          // Python: attention_dropout: float = 0.0
    public let mlp_bias: Bool                     // Python: mlp_bias: bool = False
    public let head_dim: Int                      // Python: head_dim: int = 128
    public let model_type: String                 // Python: model_type: str = "llama"
    public let pretraining_tp: Int                // Python: pretraining_tp: int = 1
    public let sliding_window: Int?               // Python: sliding_window: Optional[int] = None
    
    public init(
        vocab_size: Int = 131072,
        hidden_size: Int = 3072,
        intermediate_size: Int = 8192,
        num_hidden_layers: Int = 30,
        num_attention_heads: Int = 32,
        num_key_value_heads: Int = 8,
        max_position_embeddings: Int = 131072,
        rms_norm_eps: Double = 1e-05,
        rope_theta: Double = 100000000.0,
        rope_scaling: [String: Double]? = nil,
        tie_word_embeddings: Bool = false,
        use_cache: Bool = true,
        hidden_act: String = "silu",
        initializer_range: Double = 0.02,
        attention_bias: Bool = false,
        attention_dropout: Double = 0.0,
        mlp_bias: Bool = false,
        head_dim: Int = 128,
        model_type: String = "llama",
        pretraining_tp: Int = 1,
        sliding_window: Int? = nil
    ) {
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim
        self.model_type = model_type
        self.pretraining_tp = pretraining_tp
        self.sliding_window = sliding_window
    }
    
    /**
     * Direct Python equivalent: def to_dict(self) -> Dict (lines 124-130)
     * Python: Convert config to dictionary
     */
    public func to_dict() -> [String: Any] {
        // Python: output = {}
        var output: [String: Any] = [:]
        
        // Python: for key, value in self.__dict__.items():
        //         if not key.startswith("_") and value is not None: output[key] = value
        output["vocab_size"] = vocab_size
        output["hidden_size"] = hidden_size
        output["intermediate_size"] = intermediate_size
        output["num_hidden_layers"] = num_hidden_layers
        output["num_attention_heads"] = num_attention_heads
        output["num_key_value_heads"] = num_key_value_heads
        output["max_position_embeddings"] = max_position_embeddings
        output["rms_norm_eps"] = rms_norm_eps
        output["rope_theta"] = rope_theta
        output["tie_word_embeddings"] = tie_word_embeddings
        output["use_cache"] = use_cache
        output["hidden_act"] = hidden_act
        output["initializer_range"] = initializer_range
        output["attention_bias"] = attention_bias
        output["attention_dropout"] = attention_dropout
        output["mlp_bias"] = mlp_bias
        output["head_dim"] = head_dim
        output["model_type"] = model_type
        output["pretraining_tp"] = pretraining_tp
        
        // Python: conditional inclusion of optional values
        if let rope_scaling = rope_scaling {
            output["rope_scaling"] = rope_scaling
        }
        
        if let sliding_window = sliding_window {
            output["sliding_window"] = sliding_window
        }
        
        return output
    }
    
    /**
     * Direct Python equivalent: @classmethod def from_dict(cls, config_dict: Dict) (lines 132-159)
     * Python: Create config from dictionary
     */
    public static func from_dict(_ config_dict: [String: Any]) -> VoxtralTextConfig {
        // Python: known_fields = {...} (lines 135-157)
        // Python: filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        // Python: return cls(**filtered_dict)
        
        return VoxtralTextConfig(
            vocab_size: config_dict["vocab_size"] as? Int ?? 131072,
            hidden_size: config_dict["hidden_size"] as? Int ?? 3072,
            intermediate_size: config_dict["intermediate_size"] as? Int ?? 8192,
            num_hidden_layers: config_dict["num_hidden_layers"] as? Int ?? 30,
            num_attention_heads: config_dict["num_attention_heads"] as? Int ?? 32,
            num_key_value_heads: config_dict["num_key_value_heads"] as? Int ?? 8,
            max_position_embeddings: config_dict["max_position_embeddings"] as? Int ?? 131072,
            rms_norm_eps: config_dict["rms_norm_eps"] as? Double ?? 1e-05,
            rope_theta: config_dict["rope_theta"] as? Double ?? 100000000.0,
            rope_scaling: config_dict["rope_scaling"] as? [String: Double],
            tie_word_embeddings: config_dict["tie_word_embeddings"] as? Bool ?? false,
            use_cache: config_dict["use_cache"] as? Bool ?? true,
            hidden_act: config_dict["hidden_act"] as? String ?? "silu",
            initializer_range: config_dict["initializer_range"] as? Double ?? 0.02,
            attention_bias: config_dict["attention_bias"] as? Bool ?? false,
            attention_dropout: config_dict["attention_dropout"] as? Double ?? 0.0,
            mlp_bias: config_dict["mlp_bias"] as? Bool ?? false,
            head_dim: config_dict["head_dim"] as? Int ?? 128,
            model_type: config_dict["model_type"] as? String ?? "llama",
            pretraining_tp: config_dict["pretraining_tp"] as? Int ?? 1,
            sliding_window: config_dict["sliding_window"] as? Int
        )
    }
    
    // Compatibility method for existing code
    public static func fromDictionary(_ config_dict: [String: Any]) -> VoxtralTextConfig {
        return from_dict(config_dict)
    }
}

/**
 * Direct Python equivalent: class VoxtralConfig (lines 162-231)
 * Python reference: Configuration class for full Voxtral model, compatible with transformers VoxtralConfig
 * 
 * This is the exact Python-compatible configuration class. For Swift MLX usage, see VoxtralConfig struct below.
 */
public class PythonVoxtralConfig: Codable {
    // Python: model_type = "voxtral" (line 169)
    public static let model_type = "voxtral"
    
    // Python: def __init__ parameters (lines 171-180)
    public let audio_config: VoxtralEncoderConfig     // Python: audio_config: Optional[Union[VoxtralEncoderConfig, Dict]] = None
    public let text_config: VoxtralTextConfig         // Python: text_config: Optional[Union[VoxtralTextConfig, Dict]] = None  
    public let audio_token_id: Int                    // Python: audio_token_id: Optional[int] = 24
    public let projector_hidden_act: String          // Python: projector_hidden_act: str = "gelu"
    public let pad_token_id: Int                      // Python: pad_token_id: int = 11
    public let bos_token_id: Int                      // Python: bos_token_id: int = 1
    public let eos_token_id: Int                      // Python: eos_token_id: int = 2
    
    // Python: computed properties (lines 202-203)
    public let vocab_size: Int                        // Python: self.vocab_size = self.text_config.vocab_size
    public let hidden_size: Int                       // Python: self.hidden_size = self.text_config.hidden_size
    
    public init(
        audio_config: VoxtralEncoderConfig? = nil,
        text_config: VoxtralTextConfig? = nil,
        audio_token_id: Int = 24,
        projector_hidden_act: String = "gelu",
        pad_token_id: Int = 11,
        bos_token_id: Int = 1,
        eos_token_id: Int = 2
    ) {
        // Python: if isinstance(audio_config, dict): self.audio_config = VoxtralEncoderConfig(**audio_config)
        //         elif audio_config is None: self.audio_config = VoxtralEncoderConfig()
        //         else: self.audio_config = audio_config (lines 182-187)
        self.audio_config = audio_config ?? VoxtralEncoderConfig()
        
        // Python: if isinstance(text_config, dict): self.text_config = VoxtralTextConfig(**text_config)
        //         elif text_config is None: self.text_config = VoxtralTextConfig()
        //         else: self.text_config = text_config (lines 189-194)
        self.text_config = text_config ?? VoxtralTextConfig()
        
        // Python: direct assignment (lines 196-200)
        self.audio_token_id = audio_token_id
        self.projector_hidden_act = projector_hidden_act
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        // Python: self.vocab_size = self.text_config.vocab_size (line 202)
        self.vocab_size = self.text_config.vocab_size
        // Python: self.hidden_size = self.text_config.hidden_size (line 203)
        self.hidden_size = self.text_config.hidden_size
        
        // Python: for key, value in kwargs.items(): setattr(self, key, value) (lines 205-206)
        // Note: Swift doesn't support dynamic properties like Python, this is handled in from_dict
    }
    
    /**
     * Direct Python equivalent: def to_dict(self) -> Dict (lines 208-226)
     * Python: Convert config to dictionary
     */
    public func to_dict() -> [String: Any] {
        // Python: output = { "model_type": self.model_type, ... } (lines 210-220)
        var output: [String: Any] = [
            "model_type": Self.model_type,
            "audio_config": audio_config.to_dict(),
            "text_config": text_config.to_dict(),
            "audio_token_id": audio_token_id,
            "projector_hidden_act": projector_hidden_act
        ]
        
        // Python: for key, value in self.__dict__.items():
        //         if key not in output and not key.startswith("_"): output[key] = value (lines 222-224)
        output["pad_token_id"] = pad_token_id
        output["bos_token_id"] = bos_token_id
        output["eos_token_id"] = eos_token_id
        output["vocab_size"] = vocab_size
        output["hidden_size"] = hidden_size
        
        return output
    }
    
    // Compatibility method for existing code
    public func toDictionary() -> [String: Any] {
        return to_dict()
    }
    
    /**
     * Direct Python equivalent: @classmethod def from_dict(cls, config_dict: Dict) (lines 228-231)
     * Python: Create config from dictionary
     */
    public static func from_dict(_ config_dict: [String: Any]) -> PythonVoxtralConfig {
        // Python: return cls(**config_dict)
        
        // Handle nested audio_config (similar to Python isinstance checks in __init__)
        let audio_config: VoxtralEncoderConfig
        if let audio_config_dict = config_dict["audio_config"] as? [String: Any] {
            audio_config = VoxtralEncoderConfig.from_dict(audio_config_dict)
        } else {
            audio_config = VoxtralEncoderConfig()
        }
        
        // Handle nested text_config (similar to Python isinstance checks in __init__)
        let text_config: VoxtralTextConfig
        if let text_config_dict = config_dict["text_config"] as? [String: Any] {
            text_config = VoxtralTextConfig.from_dict(text_config_dict)
        } else {
            text_config = VoxtralTextConfig()
        }
        
        return PythonVoxtralConfig(
            audio_config: audio_config,
            text_config: text_config,
            audio_token_id: config_dict["audio_token_id"] as? Int ?? 24,
            projector_hidden_act: config_dict["projector_hidden_act"] as? String ?? "gelu",
            pad_token_id: config_dict["pad_token_id"] as? Int ?? 11,
            bos_token_id: config_dict["bos_token_id"] as? Int ?? 1,
            eos_token_id: config_dict["eos_token_id"] as? Int ?? 2
        )
    }
    
    /**
     * Swift-compatible alias for from_dict
     */
    public static func fromDictionary(_ configDict: [String: Any]) throws -> PythonVoxtralConfig {
        return from_dict(configDict)
    }
}

/**
 * Swift VoxtralConfig with nested types for MLX Swift compatibility
 * 
 * This struct provides the simplified config structure expected by Swift MLX implementations.
 * For Python-compatible full configuration, use PythonVoxtralConfig instead.
 * 
 * This addresses the type mismatch between VoxtralEncoderConfig and VoxtralConfig.AudioConfig
 * expected by the existing Swift codebase.
 */
public struct VoxtralConfig: Codable {
    
    /**
     * Nested AudioConfig type expected by VoxtralAudioEncoder
     */
    public struct AudioConfig: Codable {
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let numLayers: Int
        public let intermediate_size: Int
        
        public init(hiddenSize: Int = 1024, numAttentionHeads: Int = 16, numLayers: Int = 24, intermediate_size: Int = 4096) {
            self.hiddenSize = hiddenSize
            self.numAttentionHeads = numAttentionHeads
            self.numLayers = numLayers
            self.intermediate_size = intermediate_size
        }
    }
    
    /**
     * Nested TextConfig type expected by the codebase
     */
    public struct TextConfig: Codable {
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numberOfHiddenLayers: Int
        public let numberOfAttentionHeads: Int
        public let numberOfKeyValueHeads: Int
        public let headDimension: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Double
        public let rmsNormEpsilon: Double
        
        public init(
            vocabularySize: Int = 32000,
            hiddenSize: Int = 4096,
            intermediateSize: Int = 11008,
            numberOfHiddenLayers: Int = 32,
            numberOfAttentionHeads: Int = 32,
            numberOfKeyValueHeads: Int = 8,
            headDimension: Int = 128,
            maxPositionEmbeddings: Int = 4096,
            ropeTheta: Double = 10000.0,
            rmsNormEpsilon: Double = 1e-5
        ) {
            self.vocabularySize = vocabularySize
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numberOfHiddenLayers = numberOfHiddenLayers
            self.numberOfAttentionHeads = numberOfAttentionHeads
            self.numberOfKeyValueHeads = numberOfKeyValueHeads
            self.headDimension = headDimension
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.ropeTheta = ropeTheta
            self.rmsNormEpsilon = rmsNormEpsilon
        }
    }
    
    public let audioConfig: AudioConfig
    public let audio_config: AudioConfig  // Python property name for compatibility
    public let textConfig: TextConfig
    public let text_config: TextConfig    // Python property name for compatibility
    public let audioTokenId: Int
    public let audio_token_id: Int       // Python property name for compatibility
    
    public init(
        audioConfig: AudioConfig? = nil,
        textConfig: TextConfig? = nil,
        audioTokenId: Int = 24,
    ) {
        self.audioConfig = audioConfig ?? AudioConfig()
        self.audio_config = self.audioConfig  // Same instance for compatibility
        self.textConfig = textConfig ?? TextConfig()
        self.text_config = self.textConfig    // Same instance for compatibility
        self.audioTokenId = audioTokenId
        self.audio_token_id = audioTokenId
    }
    
    /**
     * Convenience initializer from Python-compatible configuration
     * Converts PythonVoxtralConfig to Swift VoxtralConfig
     */
    public init(from pythonConfig: PythonVoxtralConfig) {
        self.audioConfig = AudioConfig(
            hiddenSize: pythonConfig.audio_config.hidden_size,
            numAttentionHeads: pythonConfig.audio_config.num_attention_heads,
            numLayers: pythonConfig.audio_config.num_hidden_layers,
            intermediate_size: pythonConfig.audio_config.intermediate_size
        )
        self.audio_config = self.audioConfig  // Same instance for compatibility
        
        self.textConfig = TextConfig(
            vocabularySize: pythonConfig.text_config.vocab_size,
            hiddenSize: pythonConfig.text_config.hidden_size,
            intermediateSize: pythonConfig.text_config.intermediate_size,
            numberOfHiddenLayers: pythonConfig.text_config.num_hidden_layers,
            numberOfAttentionHeads: pythonConfig.text_config.num_attention_heads,
            numberOfKeyValueHeads: pythonConfig.text_config.num_key_value_heads,
            headDimension: pythonConfig.text_config.head_dim,
            maxPositionEmbeddings: pythonConfig.text_config.max_position_embeddings,
            ropeTheta: pythonConfig.text_config.rope_theta,
            rmsNormEpsilon: pythonConfig.text_config.rms_norm_eps
        )
        self.text_config = self.textConfig    // Same instance for compatibility
        
        self.audioTokenId = pythonConfig.audio_token_id
        self.audio_token_id = self.audioTokenId
    }
    
    /**
     * Convert to Python-compatible configuration
     * Creates PythonVoxtralConfig from Swift VoxtralConfig
     */
    public func toPythonConfig() -> PythonVoxtralConfig {
        let audioConfig = VoxtralEncoderConfig(
            vocab_size: 51866,  // Default from Python
            hidden_size: self.audioConfig.hiddenSize,
            intermediate_size: self.audioConfig.hiddenSize * 4,  // Reasonable default
            num_hidden_layers: self.audioConfig.numLayers,
            num_attention_heads: self.audioConfig.numAttentionHeads,
            scale_embedding: false,
            activation_function: "gelu",
            num_mel_bins: 128,
            max_source_positions: 1500,
            initializer_range: 0.02,
            attention_dropout: 0.0,
            dropout: 0.0,
            layerdrop: 0.0,
            activation_dropout: 0.0,
            pad_token_id: 0,
            head_dim: self.audioConfig.hiddenSize / self.audioConfig.numAttentionHeads,
            num_key_value_heads: self.audioConfig.numAttentionHeads
        )
        
        let textConfig = VoxtralTextConfig(
            vocab_size: self.textConfig.vocabularySize,
            hidden_size: self.textConfig.hiddenSize,
            intermediate_size: self.textConfig.intermediateSize,
            num_hidden_layers: self.textConfig.numberOfHiddenLayers,
            num_attention_heads: self.textConfig.numberOfAttentionHeads,
            num_key_value_heads: self.textConfig.numberOfKeyValueHeads,
            max_position_embeddings: self.textConfig.maxPositionEmbeddings,
            rms_norm_eps: self.textConfig.rmsNormEpsilon,
            rope_theta: self.textConfig.ropeTheta,
            rope_scaling: nil,
            tie_word_embeddings: false,
            use_cache: true,
            hidden_act: "silu",
            initializer_range: 0.02,
            attention_bias: false,
            attention_dropout: 0.0,
            mlp_bias: false,
            head_dim: self.textConfig.headDimension,
            model_type: "llama",
            pretraining_tp: 1,
            sliding_window: nil
        )
        
        return PythonVoxtralConfig(
            audio_config: audioConfig,
            text_config: textConfig,
            audio_token_id: self.audioTokenId,
            projector_hidden_act: "gelu",
            pad_token_id: 11,
            bos_token_id: 1,
            eos_token_id: 2
        )
    }
}