/**
 * Scripts - Swift equivalent of mlx.voxtral/scripts/__init__.py
 * 
 * CLI scripts module exports.
 * Direct equivalent of Python scripts/__init__.py that defines script exports.
 * Following the rule: "si ça existe en python mlx ça doit exister en mlx swift"
 */

// MARK: - CLI Scripts
// Python: from .generate import main as generate_main
// Python: from .quantize_voxtral import main as quantize_main (removed - using standard loader now)

/**
 * CLI script exports - equivalent to Python scripts/__init__.py
 *
 * Available scripts:
 * - VoxtralGenerate: Audio transcription generation script
 *   Direct equivalent of generate.py main() function
 *   Usage: swift run VoxtralGenerate --audio path/to/audio.wav
 *
 * These scripts provide command-line interfaces for common Voxtral operations.
 * They use Swift ArgumentParser for CLI argument handling, equivalent to Python's argparse.
 */

// All scripts are defined as @main structs in their respective files:
// - VoxtralGenerate.swift
// This file serves as documentation of the scripts module structure