/**
 * VoxtralCore - Swift equivalent of mlx.voxtral/__init__.py
 * 
 * Main module exports for Voxtral Swift MLX implementation.
 * Direct equivalent of Python __init__.py files that define public API.
 * Following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

// MARK: - Core Configuration Classes
// Python: from .configuration_voxtral import VoxtralConfig, VoxtralEncoderConfig, VoxtralTextConfig
// Use PythonVoxtralConfig directly to avoid confusion with VoxtralConfig in VoxtralChat.swift

// MARK: - Main Model Classes  
// Python: from .modeling_voxtral import VoxtralForConditionalGeneration, VoxtralAttention, etc.
// These are already publicly defined in their respective files

// MARK: - Processing Classes
// Python: from .processing_voxtral import VoxtralProcessor
// Python: from .audio_processing import VoxtralFeatureExtractor
// These are already publicly defined in their respective files

// MARK: - Tokenizer
// Python: from .tokenization_voxtral import VoxtralTokenizer
// Using TekkenTokenizer as real tokenizer implementation

// MARK: - Quantization Utilities
// Python: from .quantization import *
// Already publicly defined in VoxtralQuantization.swift

// MARK: - Model Loading Utilities  
// Python: from .utils.model_loading import load_voxtral_model, download_model, load_config, load_weights
// Already publicly defined in Utils/VoxtralModelLoading.swift

// MARK: - Llama Model Components
// Python: from .models.llama import LlamaModel, LlamaAttention, LlamaMLP, LlamaDecoderLayer
// Already publicly defined in Models/VoxtralLlama.swift

// MARK: - CLI Scripts
// Python: from .scripts.generate import main as generate_main
// Python: from .scripts.quantize_voxtral import main as quantize_main  
// These are defined as @main structs in their respective Script files

// MARK: - Version Information
// Python: __version__ = "0.1.0"
public let VoxtralCoreVersion = "0.1.0"

// MARK: - Public API Summary
/**
 * Main classes available in VoxtralCore module:
 *
 * Configuration:
 * - PythonVoxtralConfig (main config)
 * - VoxtralEncoderConfig (audio encoder config)  
 * - VoxtralTextConfig (text/language model config)
 *
 * Models:
 * - VoxtralForConditionalGeneration (main model)
 * - VoxtralAttention, VoxtralMLP, VoxtralDecoderLayer (components)
 * - LlamaModel, LlamaAttention, LlamaMLP, LlamaDecoderLayer (Llama components)
 * 
 * Processing:
 * - VoxtralProcessor (main processor)
 * - VoxtralFeatureExtractor (audio processing)
 * - TekkenTokenizer (text tokenization)
 *
 * Utilities:
 * - loadVoxtralModel() (model loading)
 * - downloadModel() (Hugging Face download)
 * - loadConfig(), loadWeights() (file utilities)
 * - Quantization functions (mixed precision, etc.)
 *
 * CLI Scripts:
 * - VoxtralGenerate (generation script)
 * - VoxtralQuantize (quantization script)
 */