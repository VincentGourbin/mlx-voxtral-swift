#!/usr/bin/env python3
"""
Convert MLX Voxtral weights to PyTorch format.

This script:
1. Loads weights from safetensors files (MLX format)
2. Maps weight keys from MLX naming to PyTorch naming
3. Handles any necessary tensor transformations
4. Saves the weights as a PyTorch state_dict

Usage:
    python convert_weights.py --model-path /path/to/voxtral/model --output encoder_weights.pt

    # Or use a Hugging Face model ID (will download)
    python convert_weights.py --model-id mlx-community/voxtral-mini-3b-4bit --output encoder_weights.pt
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Import our PyTorch model
from voxtral_encoder_pytorch import (
    VoxtralEncoderConfig,
    VoxtralProjectorConfig,
    VoxtralEncoderWithProjector,
    create_default_model,
    create_model_for_variant,
    VOXTRAL_VARIANTS
)


def load_safetensors_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """
    Load weights from all safetensors files in the model directory.

    Args:
        model_path: Path to directory containing safetensors files

    Returns:
        Dictionary mapping weight names to numpy arrays
    """
    weights = {}

    # Find all safetensors files (exclude consolidated.*)
    safetensor_files = sorted([
        f for f in model_path.glob("*.safetensors")
        if not f.name.startswith("consolidated")
        and not f.name.startswith("._")
    ])

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    print(f"Found {len(safetensor_files)} safetensors files:")
    for f in safetensor_files:
        print(f"  - {f.name}")

    # Load weights from each file
    # Use PyTorch framework to handle bfloat16, then convert to numpy
    for sf_path in safetensor_files:
        try:
            # Try numpy first (faster for float32/float16)
            with safe_open(sf_path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        except TypeError as e:
            if "bfloat16" in str(e):
                # BF16 not supported by numpy, use PyTorch
                print(f"  Using PyTorch for BF16 weights in {sf_path.name}")
                with safe_open(sf_path, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Convert BF16 to FP32 numpy
                        weights[key] = tensor.float().numpy()
            else:
                raise

    print(f"\nLoaded {len(weights)} weight tensors total")
    return weights


def filter_audio_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Filter weights to only include audio encoder and projector components.

    MLX weight keys we need:
    - audio_tower.conv1.weight
    - audio_tower.conv2.weight
    - audio_tower.embed_positions.weight
    - audio_tower.layers.X.self_attn.{q,k,v}_proj.weight
    - audio_tower.layers.X.self_attn.out_proj.weight
    - audio_tower.layers.X.self_attn_layer_norm.{weight,bias}
    - audio_tower.layers.X.fc1.{weight,bias}
    - audio_tower.layers.X.fc2.{weight,bias}
    - audio_tower.layers.X.final_layer_norm.{weight,bias}
    - audio_tower.layer_norm.{weight,bias}
    - multi_modal_projector.linear_1.weight
    - multi_modal_projector.linear_2.weight
    """
    audio_weights = {}

    for key, value in weights.items():
        if key.startswith("audio_tower.") or key.startswith("multi_modal_projector."):
            audio_weights[key] = value

    print(f"\nFiltered to {len(audio_weights)} audio/projector weights")
    return audio_weights


def map_mlx_to_pytorch_key(mlx_key: str) -> str:
    """
    Map MLX weight key to PyTorch model key.

    MLX uses snake_case, PyTorch model uses the same naming.
    Main difference: MLX has flat naming, PyTorch has nested modules.

    Examples:
        audio_tower.conv1.weight -> encoder.conv1.weight
        audio_tower.layers.0.self_attn.q_proj.weight -> encoder.layers.0.self_attn.q_proj.weight
        multi_modal_projector.linear_1.weight -> projector.linear_1.weight
    """
    # Replace audio_tower with encoder
    pytorch_key = mlx_key.replace("audio_tower.", "encoder.")

    # Replace multi_modal_projector with projector
    pytorch_key = pytorch_key.replace("multi_modal_projector.", "projector.")

    return pytorch_key


def check_quantization(weights: Dict[str, np.ndarray]) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Check if weights are quantized and return quantization parameters.

    Returns:
        Tuple of (is_quantized, bits, group_size)
    """
    # Look for quantization indicators (scales, biases in weight names)
    has_scales = any(".scales" in k for k in weights.keys())
    has_biases = any(".biases" in k for k in weights.keys())

    if has_scales:
        # Try to infer quantization parameters from weight shapes
        # Quantized weights have packed format
        sample_key = [k for k in weights.keys() if k.endswith(".weight") and "audio_tower" in k][0]
        sample_weight = weights[sample_key]

        print(f"\nQuantization detected:")
        print(f"  Has scales: {has_scales}")
        print(f"  Has biases: {has_biases}")
        print(f"  Sample weight dtype: {sample_weight.dtype}")
        print(f"  Sample weight shape: {sample_weight.shape}")

        # Infer bits from dtype (uint32 = 4-bit or 8-bit packed)
        if sample_weight.dtype == np.uint32:
            # 4-bit: 8 values packed per uint32
            return True, 4, 64  # Default group_size
        elif sample_weight.dtype == np.uint8:
            return True, 8, 64

        return True, None, None

    return False, None, None


def dequantize_weight(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: Optional[np.ndarray],
    bits: int,
    group_size: int
) -> np.ndarray:
    """
    Dequantize a weight tensor from packed format to float32.

    This reverses the MLX quantization process.
    """
    # For 4-bit quantization
    if bits == 4:
        # Unpack 4-bit values from uint32
        # Each uint32 contains 8 4-bit values
        packed = weight.astype(np.uint32)

        # Create output shape
        out_features = weight.shape[0]
        packed_in_features = weight.shape[1]
        in_features = packed_in_features * 8  # 8 values per uint32

        # Unpack
        unpacked = np.zeros((out_features, in_features), dtype=np.float32)
        for i in range(8):
            shift = i * 4
            mask = 0xF << shift
            values = ((packed & mask) >> shift).astype(np.float32)
            unpacked[:, i::8] = values

        # Apply scales and biases
        num_groups = in_features // group_size
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            scale = scales[:, g:g+1]
            if biases is not None:
                bias = biases[:, g:g+1]
                unpacked[:, start:end] = unpacked[:, start:end] * scale + bias
            else:
                unpacked[:, start:end] = unpacked[:, start:end] * scale

        return unpacked

    elif bits == 8:
        # 8-bit quantization
        unpacked = weight.astype(np.float32)
        # Apply scales
        num_groups = weight.shape[1] // group_size
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            scale = scales[:, g:g+1]
            if biases is not None:
                bias = biases[:, g:g+1]
                unpacked[:, start:end] = unpacked[:, start:end] * scale + bias
            else:
                unpacked[:, start:end] = unpacked[:, start:end] * scale

        return unpacked

    else:
        raise ValueError(f"Unsupported quantization bits: {bits}")


def convert_weights_to_pytorch(
    mlx_weights: Dict[str, np.ndarray],
    dequantize: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert MLX weights to PyTorch format.

    Args:
        mlx_weights: Dictionary of MLX weights (numpy arrays)
        dequantize: Whether to dequantize quantized weights

    Returns:
        Dictionary of PyTorch weights (torch.Tensor)
    """
    is_quantized, bits, group_size = check_quantization(mlx_weights)

    pytorch_weights = {}
    processed_keys = set()

    for mlx_key, weight in mlx_weights.items():
        # Skip scales and biases (handled with weights)
        if ".scales" in mlx_key or ".biases" in mlx_key:
            continue

        # Skip already processed
        if mlx_key in processed_keys:
            continue

        pytorch_key = map_mlx_to_pytorch_key(mlx_key)

        # Handle quantized weights
        if is_quantized and dequantize and mlx_key.endswith(".weight"):
            scales_key = mlx_key.replace(".weight", ".scales")
            biases_key = mlx_key.replace(".weight", ".biases")

            if scales_key in mlx_weights:
                scales = mlx_weights[scales_key]
                biases = mlx_weights.get(biases_key)

                print(f"  Dequantizing {mlx_key}...")
                weight = dequantize_weight(weight, scales, biases, bits, group_size)
                processed_keys.add(scales_key)
                if biases_key in mlx_weights:
                    processed_keys.add(biases_key)

        # Convert to torch tensor
        # Handle dtype conversion
        if weight.dtype in [np.float16, np.float32, np.float64]:
            tensor = torch.from_numpy(weight.astype(np.float32))
        elif weight.dtype in [np.int32, np.int64]:
            tensor = torch.from_numpy(weight.astype(np.int64))
        else:
            # For quantized weights that weren't dequantized
            tensor = torch.from_numpy(weight.astype(np.float32))

        # Handle Conv1d weight shape difference
        # MLX Conv1d: [out_channels, in_channels, kernel_size] (same as PyTorch)
        # No transpose needed for Conv1d

        pytorch_weights[pytorch_key] = tensor
        processed_keys.add(mlx_key)

    return pytorch_weights


def validate_weights(
    pytorch_weights: Dict[str, torch.Tensor],
    model: VoxtralEncoderWithProjector
) -> bool:
    """
    Validate that converted weights match the PyTorch model structure.
    """
    model_state_dict = model.state_dict()

    print("\n" + "="*60)
    print("Weight Validation")
    print("="*60)

    # Check for missing keys
    model_keys = set(model_state_dict.keys())
    weight_keys = set(pytorch_weights.keys())

    missing_in_weights = model_keys - weight_keys
    extra_in_weights = weight_keys - model_keys

    if missing_in_weights:
        print(f"\nWARNING: {len(missing_in_weights)} keys missing from converted weights:")
        for k in sorted(missing_in_weights)[:10]:
            print(f"  - {k}")
        if len(missing_in_weights) > 10:
            print(f"  ... and {len(missing_in_weights) - 10} more")

    if extra_in_weights:
        print(f"\nWARNING: {len(extra_in_weights)} extra keys in converted weights:")
        for k in sorted(extra_in_weights)[:10]:
            print(f"  - {k}")
        if len(extra_in_weights) > 10:
            print(f"  ... and {len(extra_in_weights) - 10} more")

    # Check shape compatibility
    shape_mismatches = []
    for key in model_keys & weight_keys:
        model_shape = model_state_dict[key].shape
        weight_shape = pytorch_weights[key].shape
        if model_shape != weight_shape:
            shape_mismatches.append((key, model_shape, weight_shape))

    if shape_mismatches:
        print(f"\nERROR: {len(shape_mismatches)} shape mismatches:")
        for key, model_shape, weight_shape in shape_mismatches[:10]:
            print(f"  - {key}: model={model_shape}, weight={weight_shape}")
        return False

    print(f"\nValidation passed:")
    print(f"  - {len(model_keys & weight_keys)} matching keys")
    print(f"  - All shapes compatible")

    return True


def load_weights_into_model(
    model: VoxtralEncoderWithProjector,
    pytorch_weights: Dict[str, torch.Tensor],
    strict: bool = False
) -> VoxtralEncoderWithProjector:
    """
    Load converted weights into PyTorch model.
    """
    # Load with strict=False to allow missing keys (we only have encoder weights)
    missing, unexpected = model.load_state_dict(pytorch_weights, strict=strict)

    if missing:
        print(f"\nMissing keys ({len(missing)}):")
        for k in missing[:5]:
            print(f"  - {k}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    if unexpected:
        print(f"\nUnexpected keys ({len(unexpected)}):")
        for k in unexpected[:5]:
            print(f"  - {k}")
        if len(unexpected) > 5:
            print(f"  ... and {len(unexpected) - 5} more")

    return model


def test_model_output(model: VoxtralEncoderWithProjector, variant: str = "mini") -> None:
    """
    Test the model with dummy input to verify it works.
    """
    print("\n" + "="*60)
    print("Testing Model Output")
    print("="*60)

    model.eval()

    with torch.no_grad():
        # Create dummy mel spectrogram input
        dummy_input = torch.randn(1, 128, 3000)

        # Forward pass
        output = model(dummy_input)

        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

        # Expected output shape depends on variant
        text_hidden_size = VOXTRAL_VARIANTS[variant]["text_hidden_size"]
        expected_shape = (1, 375, text_hidden_size)
        if output.shape == expected_shape:
            print(f"\n✅ Output shape matches expected {expected_shape} for {variant}")
        else:
            print(f"\n❌ WARNING: Output shape {output.shape} != expected {expected_shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLX Voxtral weights to PyTorch format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to directory containing MLX model weights"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mistralai/Voxtral-Mini-3B-2507",
        help="Hugging Face model ID (default: official Mistral full-precision model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="voxtral_encoder.pt",
        help="Output file path for PyTorch weights"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["mini", "small"],
        default="mini",
        help="Model variant: 'mini' (3B, output 3072) or 'small' (24B, output 5120)"
    )
    parser.add_argument(
        "--no-dequantize",
        action="store_true",
        help="Don't dequantize quantized weights (keep in packed format)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model with dummy input after loading"
    )
    parser.add_argument(
        "--use-quantized",
        action="store_true",
        help="Use quantized MLX model instead of official full-precision (not recommended)"
    )

    args = parser.parse_args()

    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.model_id:
        # Try to find in Hugging Face cache
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(
            repo_id=args.model_id,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["consolidated.*"]
        ))
    else:
        parser.error("Either --model-path or --model-id is required")

    print("="*60)
    print("MLX to PyTorch Weight Conversion")
    print("="*60)
    print(f"\nModel path: {model_path}")
    print(f"Output: {args.output}")

    # Step 1: Load MLX weights
    print("\n" + "-"*40)
    print("Step 1: Loading MLX weights")
    print("-"*40)
    all_weights = load_safetensors_weights(model_path)

    # Step 2: Filter to audio weights
    print("\n" + "-"*40)
    print("Step 2: Filtering audio/projector weights")
    print("-"*40)
    audio_weights = filter_audio_weights(all_weights)

    # Print weight keys for debugging
    print("\nAudio weight keys:")
    for key in sorted(audio_weights.keys())[:20]:
        shape = audio_weights[key].shape
        dtype = audio_weights[key].dtype
        print(f"  {key}: {shape} ({dtype})")
    if len(audio_weights) > 20:
        print(f"  ... and {len(audio_weights) - 20} more")

    # Step 3: Convert to PyTorch format
    print("\n" + "-"*40)
    print("Step 3: Converting to PyTorch format")
    print("-"*40)
    pytorch_weights = convert_weights_to_pytorch(
        audio_weights,
        dequantize=not args.no_dequantize
    )

    print(f"\nConverted {len(pytorch_weights)} weights")

    # Step 4: Create model and validate
    print("\n" + "-"*40)
    print(f"Step 4: Creating {args.variant.upper()} model and validating weights")
    print("-"*40)
    model = create_model_for_variant(args.variant)

    if validate_weights(pytorch_weights, model):
        # Step 5: Load weights into model
        print("\n" + "-"*40)
        print("Step 5: Loading weights into model")
        print("-"*40)
        model = load_weights_into_model(model, pytorch_weights, strict=False)

        # Step 6: Test model (optional)
        if args.test:
            test_model_output(model, args.variant)

        # Step 7: Save weights
        print("\n" + "-"*40)
        print("Step 6: Saving PyTorch weights")
        print("-"*40)

        # Save as PyTorch state dict
        torch.save(model.state_dict(), args.output)
        print(f"\nSaved to: {args.output}")

        # Also save as safetensors for compatibility
        safetensors_output = args.output.replace(".pt", ".safetensors")
        if not safetensors_output.endswith(".safetensors"):
            safetensors_output = args.output + ".safetensors"
        save_file(model.state_dict(), safetensors_output)
        print(f"Also saved as: {safetensors_output}")

    else:
        print("\nWeight validation failed, not saving")
        return 1

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    exit(main())
