/**
 * Utils - Swift equivalent of mlx.voxtral/utils/__init__.py
 * 
 * Utility functions module exports.
 * Direct equivalent of Python utils/__init__.py that defines utility exports.
 * Following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

// MARK: - Model Loading Utilities
// Python: from .model_loading import load_voxtral_model, download_model, load_config, load_weights

/**
 * Utility function exports - equivalent to Python utils/__init__.py
 *
 * Available functions:
 * - loadVoxtralModel(): Main model loading function
 *   Direct equivalent of Python load_voxtral_model()
 *   Loads model from path or Hugging Face Hub
 *
 * - downloadModel(): Hugging Face Hub download
 *   Direct equivalent of Python download_model()  
 *   Downloads model files from HF Hub
 *
 * - loadConfig(): Configuration file loader
 *   Direct equivalent of Python load_config()
 *   Loads config.json files
 *
 * - loadWeights(): Weight file loader
 *   Direct equivalent of Python load_weights()
 *   Loads safetensors weight files
 *
 * These utilities handle model downloading, configuration loading,
 * and weight file management for Voxtral models.
 */

// All functions are already publicly defined in VoxtralModelLoading.swift
// This file serves as documentation of the utils module structure