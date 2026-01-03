/**
 * Models - Swift equivalent of mlx.voxtral/models/__init__.py
 * 
 * Model components module exports.
 * Direct equivalent of Python models/__init__.py that defines model exports.
 * Following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

// MARK: - Llama Model Components
// Python: from .llama import LlamaModel, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModelArgs

/**
 * Main Llama model exports - equivalent to Python models/__init__.py
 *
 * Available classes:
 * - LlamaModel: Main Llama language model
 * - LlamaAttention: Multi-head attention implementation  
 * - LlamaMLP: Feed-forward network with SiLU activation
 * - LlamaDecoderLayer: Transformer decoder layer
 * - LlamaModelArgs: Configuration structure for Llama models
 *
 * These classes provide the core Llama architecture components used by Voxtral.
 */

// All classes are already publicly defined in VoxtralLlama.swift
// This file serves as documentation of the models module structure