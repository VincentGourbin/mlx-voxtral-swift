#!/usr/bin/env python3
"""
Convert ANE-optimized Voxtral Encoder to Core ML

This script:
1. Loads the standard PyTorch weights
2. Converts them to ANE-optimized format
3. Converts to Core ML with ANE optimizations

Usage:
    python convert_to_coreml_ane.py --weights voxtral_encoder.pt --output VoxtralEncoderANE.mlpackage
"""

import argparse
import torch
import numpy as np
from pathlib import Path

# Import our models
from voxtral_encoder_pytorch import (
    VoxtralEncoderWithProjector,
    create_default_model as create_standard_model
)
from voxtral_encoder_ane import (
    ANEVoxtralEncoderWithProjector,
    create_ane_model,
    convert_linear_to_conv2d_weight
)


def load_standard_weights(weights_path: Path) -> dict:
    """Load weights from standard PyTorch model."""
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    print(f"  Loaded {len(state_dict)} weight tensors")
    return state_dict


def convert_weights_standard_to_ane(standard_state_dict: dict) -> dict:
    """
    Convert weights from standard format to ANE-optimized format.

    Key changes:
    - Linear weights (2D) -> Conv2d weights (4D): add 2 dimensions
    - Key remapping for structural differences (fc1/fc2 -> ffn.fc1/ffn.fc2)
    - LayerNorm wrapper path adjustment
    """
    ane_state_dict = {}

    for key, value in standard_state_dict.items():
        new_key = key
        new_value = value

        # === Step 1: Handle FFN key remapping first ===
        # Standard: encoder.layers.X.fc1 -> ANE: encoder.layers.X.ffn.fc1
        if '.layers.' in key and ('.fc1.' in key or '.fc2.' in key):
            new_key = key.replace('.fc1.', '.ffn.fc1.').replace('.fc2.', '.ffn.fc2.')

        # === Step 2: Handle LayerNorm wrapper ===
        # ANE uses ANELayerNorm which wraps standard LayerNorm
        # Standard: encoder.layers.X.self_attn_layer_norm.weight
        # ANE: encoder.layers.X.self_attn_layer_norm.layer_norm.weight
        if 'layer_norm' in new_key:
            # For encoder.layer_norm (final)
            if 'encoder.layer_norm.' in new_key and '.layers.' not in new_key:
                new_key = new_key.replace('encoder.layer_norm.', 'encoder.layer_norm.layer_norm.')
            # For layer-level norms
            elif '.self_attn_layer_norm.' in new_key:
                new_key = new_key.replace('.self_attn_layer_norm.', '.self_attn_layer_norm.layer_norm.')
            elif '.final_layer_norm.' in new_key:
                new_key = new_key.replace('.final_layer_norm.', '.final_layer_norm.layer_norm.')

        # === Step 3: Convert Linear weights to Conv2d format ===
        # Linear: (out_features, in_features)
        # Conv2d: (out_channels, in_channels, 1, 1)
        if 'weight' in new_key and value.dim() == 2:
            # Check if this is a layer that should be Conv2d in ANE model
            is_attention = any(proj in new_key for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj'])
            is_ffn = any(fc in new_key for fc in ['.ffn.fc1', '.ffn.fc2'])
            is_projector = any(lin in new_key for lin in ['projector.linear_1', 'projector.linear_2'])

            if is_attention or is_ffn or is_projector:
                new_value = value.unsqueeze(-1).unsqueeze(-1)
                print(f"  Conv2d: {key} {value.shape} -> {new_value.shape}")

        ane_state_dict[new_key] = new_value

    return ane_state_dict


def validate_weights(ane_model: ANEVoxtralEncoderWithProjector, ane_state_dict: dict):
    """Validate that all weights are properly converted."""
    model_keys = set(ane_model.state_dict().keys())
    converted_keys = set(ane_state_dict.keys())

    missing = model_keys - converted_keys
    extra = converted_keys - model_keys

    if missing:
        print(f"\n⚠️  Missing keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if extra:
        print(f"\n⚠️  Extra keys ({len(extra)}):")
        for k in sorted(extra)[:10]:
            print(f"    {k}")
        if len(extra) > 10:
            print(f"    ... and {len(extra) - 10} more")

    # Check shapes
    print("\nValidating shapes...")
    shape_errors = []
    for key in model_keys & converted_keys:
        model_shape = ane_model.state_dict()[key].shape
        converted_shape = ane_state_dict[key].shape
        if model_shape != converted_shape:
            shape_errors.append((key, model_shape, converted_shape))

    if shape_errors:
        print(f"\n⚠️  Shape mismatches ({len(shape_errors)}):")
        for key, expected, got in shape_errors[:10]:
            print(f"    {key}: expected {expected}, got {got}")
    else:
        print("  ✓ All shapes match!")

    return len(missing) == 0 and len(extra) == 0 and len(shape_errors) == 0


def test_model_output(
    standard_model: VoxtralEncoderWithProjector,
    ane_model: ANEVoxtralEncoderWithProjector,
    test_input: torch.Tensor
) -> float:
    """Compare outputs between standard and ANE models."""
    print("\nComparing model outputs...")

    with torch.no_grad():
        standard_output = standard_model(test_input)
        ane_output = ane_model(test_input)

    # Calculate difference
    diff = (standard_output - ane_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Standard output shape: {standard_output.shape}")
    print(f"  ANE output shape:      {ane_output.shape}")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    return max_diff


def convert_to_coreml(
    model: ANEVoxtralEncoderWithProjector,
    output_path: Path,
    precision: str = "float16"
):
    """Convert ANE model to Core ML."""
    import coremltools as ct

    print(f"\nConverting to Core ML...")
    print(f"  Precision: {precision}")
    print(f"  Output: {output_path}")

    model.eval()

    # Create example input
    example_input = torch.randn(1, 128, 3000)

    # Trace the model
    print("  Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML
    print("  Converting with coremltools...")

    # Use compute_precision based on precision argument
    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="mel_spectrogram",
                shape=(1, 128, 3000),
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="audio_embeddings", dtype=np.float32)
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.ALL  # Allow ANE + GPU + CPU
    )

    # Add metadata
    mlmodel.author = "Voxtral ANE Conversion"
    mlmodel.short_description = "Voxtral Audio Encoder optimized for Apple Neural Engine"
    mlmodel.version = "2.0.0"

    # Save
    print(f"  Saving to {output_path}...")
    mlmodel.save(str(output_path))

    # Print model info
    spec = mlmodel.get_spec()
    print(f"\n  Model saved successfully!")
    print(f"  Input:  mel_spectrogram [1, 128, 3000]")
    print(f"  Output: audio_embeddings [1, 375, 3072]")

    return mlmodel


def main():
    parser = argparse.ArgumentParser(description="Convert Voxtral to Core ML with ANE optimizations")
    parser.add_argument("--weights", type=str, required=True, help="Path to standard PyTorch weights")
    parser.add_argument("--output", type=str, default="VoxtralEncoderANE.mlpackage", help="Output path")
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip-validation", action="store_true", help="Skip output validation")
    parser.add_argument("--test", action="store_true", help="Run comparison test")
    args = parser.parse_args()

    print("=" * 60)
    print("VOXTRAL ANE CORE ML CONVERSION")
    print("=" * 60)

    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        return 1

    # Step 1: Load standard weights
    print("\n[1/4] Loading standard weights...")
    standard_state_dict = load_standard_weights(weights_path)

    # Step 2: Create models
    print("\n[2/4] Creating models...")
    standard_model = create_standard_model()
    ane_model = create_ane_model()

    # Load weights into standard model for comparison
    standard_model.load_state_dict(standard_state_dict)
    standard_model.eval()

    print(f"  Standard model parameters: {sum(p.numel() for p in standard_model.parameters()):,}")
    print(f"  ANE model parameters:      {sum(p.numel() for p in ane_model.parameters()):,}")

    # Step 3: Convert weights
    print("\n[3/4] Converting weights to ANE format...")
    ane_state_dict = convert_weights_standard_to_ane(standard_state_dict)

    # Validate
    if not args.skip_validation:
        is_valid = validate_weights(ane_model, ane_state_dict)
        if not is_valid:
            print("\n❌ Weight validation failed!")
            return 1

    # Load converted weights
    ane_model.load_state_dict(ane_state_dict)
    ane_model.eval()

    # Test output equivalence
    if args.test:
        test_input = torch.randn(1, 128, 3000)
        max_diff = test_model_output(standard_model, ane_model, test_input)
        if max_diff > 1.0:
            print(f"\n⚠️  Warning: Large output difference ({max_diff:.4f})")

    # Step 4: Convert to Core ML
    print("\n[4/4] Converting to Core ML...")
    mlmodel = convert_to_coreml(ane_model, output_path, args.precision)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")

    # Get size
    import subprocess
    result = subprocess.run(["du", "-sh", str(output_path)], capture_output=True, text=True)
    if result.returncode == 0:
        size = result.stdout.split()[0]
        print(f"Size:   {size}")

    print("\nNext steps:")
    print("  1. Compile: xcrun coremlcompiler compile", output_path, "Resources/")
    print("  2. Test:    ./.build/release/VoxtralCLI benchmark-coreml --model-path Resources/VoxtralEncoderANE.mlmodelc")

    return 0


if __name__ == "__main__":
    exit(main())
