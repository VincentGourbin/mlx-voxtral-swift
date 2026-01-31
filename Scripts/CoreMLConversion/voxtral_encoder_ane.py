#!/usr/bin/env python3
"""
VoxtralEncoder PyTorch Implementation OPTIMIZED FOR Apple Neural Engine (ANE)

Based on Apple's ml-ane-transformers reference implementation.
https://github.com/apple/ml-ane-transformers

Key ANE optimizations applied:
1. 4D Channels-First Format: (B, S, C) -> (B, C, 1, S)
2. Conv2d instead of Linear: All nn.Linear replaced with nn.Conv2d kernel 1x1
3. Split Attention Heads: Process each head separately instead of batched
4. Optimized Einsum: Use 'bchq,bkhc->bkhq' for attention computation

Expected performance improvement: Up to 10x faster on ANE vs baseline
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class VoxtralEncoderConfig:
    """Configuration for VoxtralEncoder"""
    vocab_size: int = 51866
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    head_dim: int = 64  # hidden_size / num_attention_heads
    num_key_value_heads: int = 20
    pad_token_id: int = 0
    scale_embedding: bool = False
    activation_function: str = "gelu"


@dataclass
class VoxtralProjectorConfig:
    """Configuration for VoxtralMultiModalProjector"""
    audio_intermediate_size: int = 5120
    text_hidden_size: int = 3072  # 3072 for Mini, 5120 for Small


# Variant configurations
VOXTRAL_VARIANTS = {
    "mini": {
        "text_hidden_size": 3072,
        "description": "Voxtral Mini 3B - output [1, 375, 3072]"
    },
    "small": {
        "text_hidden_size": 5120,
        "description": "Voxtral Small 24B - output [1, 375, 5120]"
    }
}


# =============================================================================
# ANE-Optimized Attention
# =============================================================================

class ANEMultiHeadAttention(nn.Module):
    """
    ANE-optimized multi-head self-attention.

    Key optimizations:
    - Uses Conv2d instead of Linear for projections
    - 4D channels-first format: (B, C, 1, S)
    - Splits attention heads explicitly
    - Uses optimized einsum for attention computation
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # ANE: Replace Linear with Conv2d (kernel_size=1)
        # Input: (B, C, 1, S) -> Output: (B, C, 1, S)
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)  # k has no bias
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ANE-optimized attention.

        Args:
            hidden_states: (B, C, 1, S) where C=embed_dim, S=seq_len
            attention_mask: Optional mask

        Returns:
            Output tensor (B, C, 1, S)
        """
        # Input shape: (B, C, 1, S)
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(3)

        # Project Q, K, V using Conv2d
        q = self.q_proj(hidden_states)  # (B, C, 1, S)
        k = self.k_proj(hidden_states)  # (B, C, 1, S)
        v = self.v_proj(hidden_states)  # (B, C, 1, S)

        # Split heads: (B, C, 1, S) -> list of (B, head_dim, 1, S)
        # Split along channel dimension
        mh_q = q.split(self.head_dim, dim=1)  # List of num_heads tensors
        mh_v = v.split(self.head_dim, dim=1)

        # For K, we need to transpose for the einsum pattern
        # K: (B, C, 1, S) -> (B, S, 1, C) for einsum compatibility
        k_transposed = k.permute(0, 3, 2, 1)  # (B, S, 1, C)
        mh_k = k_transposed.split(self.head_dim, dim=3)  # List of (B, S, 1, head_dim)

        # Compute attention for each head separately (ANE optimization)
        attn_outputs = []
        for head_idx in range(self.num_heads):
            qi = mh_q[head_idx]  # (B, head_dim, 1, S) = (B, C, H, Q)
            ki = mh_k[head_idx]  # (B, S, 1, head_dim) = (B, K, H, C)
            vi = mh_v[head_idx]  # (B, head_dim, 1, S) = (B, C, H, K)

            # Attention scores using ANE-friendly einsum
            # qi: (B, head_dim, 1, S) as (b, c, h, q)
            # ki: (B, S, 1, head_dim) as (b, k, h, c)
            # Result: (B, S, 1, S) as (b, k, h, q)
            attn_weights = torch.einsum('bchq,bkhc->bkhq', qi, ki) * self.scaling

            # Apply mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Softmax over key dimension (dim=1)
            attn_weights = F.softmax(attn_weights, dim=1)

            # Apply attention to values
            # attn_weights: (B, K, 1, Q) = (b, k, h, q)
            # vi: (B, head_dim, 1, K) = (b, c, h, k)
            # Result: (B, head_dim, 1, Q) = (b, c, h, q)
            attn_output = torch.einsum('bkhq,bchk->bchq', attn_weights, vi)
            attn_outputs.append(attn_output)

        # Concatenate heads: list of (B, head_dim, 1, S) -> (B, C, 1, S)
        concatenated = torch.cat(attn_outputs, dim=1)

        # Output projection
        output = self.out_proj(concatenated)

        return output


# =============================================================================
# ANE-Optimized Feed-Forward Network
# =============================================================================

class ANEFFN(nn.Module):
    """
    ANE-optimized feed-forward network.

    Uses Conv2d instead of Linear layers for ANE efficiency.
    """

    def __init__(self, embed_dim: int, intermediate_size: int):
        super().__init__()

        # ANE: Replace Linear with Conv2d
        self.fc1 = nn.Conv2d(embed_dim, intermediate_size, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(intermediate_size, embed_dim, kernel_size=1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, 1, S)
        Returns:
            Output: (B, C, 1, S)
        """
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# =============================================================================
# ANE-Optimized Layer Normalization
# =============================================================================

class ANELayerNorm(nn.Module):
    """
    ANE-compatible layer normalization for 4D tensors.

    Standard LayerNorm expects (B, S, C), but ANE uses (B, C, 1, S).
    This wrapper handles the format conversion.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, 1, S) where C = normalized_shape
        Returns:
            Normalized tensor (B, C, 1, S)
        """
        # Convert to (B, S, C) for LayerNorm
        # (B, C, 1, S) -> (B, S, C)
        x = x.squeeze(2).permute(0, 2, 1)

        # Apply LayerNorm
        x = self.layer_norm(x)

        # Convert back to (B, C, 1, S)
        # (B, S, C) -> (B, C, 1, S)
        x = x.permute(0, 2, 1).unsqueeze(2)

        return x


# =============================================================================
# ANE-Optimized Encoder Layer
# =============================================================================

class ANEEncoderLayer(nn.Module):
    """
    ANE-optimized transformer encoder layer.
    Pre-LN architecture with all operations in 4D channels-first format.
    """

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        # Layer norms
        self.self_attn_layer_norm = ANELayerNorm(self.embed_dim, eps=1e-5)
        self.final_layer_norm = ANELayerNorm(self.embed_dim, eps=1e-5)

        # Self-attention
        self.self_attn = ANEMultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=True
        )

        # Feed-forward network
        self.ffn = ANEFFN(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, 1, S)
            attention_mask: Optional mask
        Returns:
            Output: (B, C, 1, S)
        """
        # Pre-LN: Norm -> Attention -> Residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-LN: Norm -> FFN -> Residual
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# ANE-Optimized Full Encoder
# =============================================================================

class ANEVoxtralEncoder(nn.Module):
    """
    ANE-optimized Voxtral audio encoder.

    All operations use 4D channels-first format (B, C, 1, S) for ANE compatibility.

    Input: Mel spectrogram (B, 128, 3000)
    Output: Hidden states (B, 1280, 1, 1500)
    """

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        # Conv layers (these stay as Conv1d for the initial audio processing)
        self.conv1 = nn.Conv1d(
            in_channels=self.num_mel_bins,
            out_channels=self.embed_dim,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Positional embeddings
        self.embed_positions = nn.Embedding(
            num_embeddings=self.max_source_positions,
            embedding_dim=self.embed_dim
        )

        # Final layer norm
        self.layer_norm = ANELayerNorm(self.embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            ANEEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (B, 128, 3000) mel spectrogram
        Returns:
            Hidden states (B, 1280, 1, 1500) in ANE format
        """
        # Initial conv processing
        hidden_states = F.gelu(self.conv1(input_features))  # (B, 1280, 3000)
        hidden_states = F.gelu(self.conv2(hidden_states))    # (B, 1280, 1500)

        # Add positional embeddings
        # hidden_states: (B, C, S) -> need to add (S, C) embeddings
        seq_len = 1500  # Static for Core ML
        positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
        embed_pos = self.embed_positions(positions)  # (1500, 1280)

        # Add positions: need to align shapes
        # hidden_states: (B, 1280, 1500), embed_pos: (1500, 1280)
        # Transpose embed_pos to (1280, 1500) and add
        hidden_states = hidden_states + embed_pos.transpose(0, 1).unsqueeze(0)

        # Convert to ANE 4D format: (B, C, S) -> (B, C, 1, S)
        hidden_states = hidden_states.unsqueeze(2)  # (B, 1280, 1, 1500)

        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states  # (B, 1280, 1, 1500)


# =============================================================================
# ANE-Optimized Projector
# =============================================================================

class ANEMultiModalProjector(nn.Module):
    """
    ANE-optimized multi-modal projector.

    Uses Conv2d for ANE efficiency.
    """

    def __init__(self, config: VoxtralProjectorConfig):
        super().__init__()

        # ANE: Use Conv2d
        self.linear_1 = nn.Conv2d(
            config.audio_intermediate_size,
            config.text_hidden_size,
            kernel_size=1,
            bias=False
        )
        self.linear_2 = nn.Conv2d(
            config.text_hidden_size,
            config.text_hidden_size,
            kernel_size=1,
            bias=False
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (B, 5120, 1, 375) in ANE format
        Returns:
            Projected features (B, 3072, 1, 375)
        """
        hidden_states = F.gelu(self.linear_1(audio_features))
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# =============================================================================
# Full ANE-Optimized Pipeline
# =============================================================================

class ANEVoxtralEncoderWithProjector(nn.Module):
    """
    Complete ANE-optimized audio processing pipeline.

    Input: Mel spectrogram (1, 128, 3000)
    Output: Audio embeddings (1, 375, text_hidden_size)
            - Mini: (1, 375, 3072)
            - Small: (1, 375, 5120)
    """

    def __init__(
        self,
        encoder_config: Optional[VoxtralEncoderConfig] = None,
        projector_config: Optional[VoxtralProjectorConfig] = None
    ):
        super().__init__()

        self.encoder_config = encoder_config or VoxtralEncoderConfig()
        self.projector_config = projector_config or VoxtralProjectorConfig()
        self.text_hidden_size = self.projector_config.text_hidden_size

        self.encoder = ANEVoxtralEncoder(self.encoder_config)
        self.projector = ANEMultiModalProjector(self.projector_config)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Full audio processing pipeline.

        Args:
            mel_spectrogram: (1, 128, 3000)
        Returns:
            Audio embeddings (1, 375, 3072)
        """
        # Encode
        encoder_output = self.encoder(mel_spectrogram)  # (1, 1280, 1, 1500)

        # Reshape for projector
        # (1, 1280, 1, 1500) -> (1, 5120, 1, 375)
        # Combine 4 consecutive positions: 1500/4 = 375, 1280*4 = 5120
        batch_size = 1

        # Remove singleton dimension: (1, 1280, 1, 1500) -> (1, 1280, 1500)
        encoder_output = encoder_output.squeeze(2)

        # Reshape: (1, 1280, 1500) -> (1, 1280, 375, 4) -> (1, 5120, 375)
        encoder_output = encoder_output.view(batch_size, 1280, 375, 4)
        encoder_output = encoder_output.permute(0, 1, 3, 2)  # (1, 1280, 4, 375)
        encoder_output = encoder_output.reshape(batch_size, 5120, 375)  # (1, 5120, 375)

        # Add back singleton for ANE format: (1, 5120, 375) -> (1, 5120, 1, 375)
        encoder_output = encoder_output.unsqueeze(2)

        # Project
        projected = self.projector(encoder_output)  # (1, 3072, 1, 375)

        # Convert back to standard format for output: (1, 3072, 1, 375) -> (1, 375, 3072)
        output = projected.squeeze(2).permute(0, 2, 1)

        return output


# =============================================================================
# Weight Conversion Utilities
# =============================================================================

def convert_linear_to_conv2d_weight(linear_weight: torch.Tensor) -> torch.Tensor:
    """
    Convert Linear weight to Conv2d weight format.

    Linear weight: (out_features, in_features)
    Conv2d weight: (out_channels, in_channels, 1, 1)
    """
    return linear_weight.unsqueeze(-1).unsqueeze(-1)


def convert_standard_to_ane_weights(standard_state_dict: dict) -> dict:
    """
    Convert weights from standard encoder to ANE-optimized format.

    Key transformations:
    - Linear weights -> Conv2d weights (add 2 dimensions)
    - Keep conv1d, embedding, and layernorm weights as-is
    """
    ane_state_dict = {}

    for key, value in standard_state_dict.items():
        new_key = key
        new_value = value

        # Handle attention projections (q_proj, k_proj, v_proj, out_proj)
        if any(proj in key for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
            if 'weight' in key and value.dim() == 2:
                new_value = convert_linear_to_conv2d_weight(value)

        # Handle FFN layers (fc1, fc2)
        elif any(fc in key for fc in ['fc1', 'fc2', 'linear_1', 'linear_2']):
            if 'weight' in key and value.dim() == 2:
                new_value = convert_linear_to_conv2d_weight(value)

        # Handle layer norm - ANELayerNorm wraps standard LayerNorm
        elif 'layer_norm' in key:
            # Adjust key path for ANELayerNorm wrapper
            new_key = key.replace('self_attn_layer_norm.', 'self_attn_layer_norm.layer_norm.')
            new_key = new_key.replace('final_layer_norm.', 'final_layer_norm.layer_norm.')
            if 'encoder.layer_norm.' in new_key and 'layer_norm.layer_norm' not in new_key:
                new_key = new_key.replace('encoder.layer_norm.', 'encoder.layer_norm.layer_norm.')

        # Handle FFN key remapping (fc1/fc2 -> ffn.fc1/ffn.fc2)
        if '.fc1.' in key or '.fc2.' in key:
            # Already handled above for weight conversion
            # Remap key structure
            new_key = key.replace('.fc1.', '.ffn.fc1.')
            new_key = new_key.replace('.fc2.', '.ffn.fc2.')

        ane_state_dict[new_key] = new_value

    return ane_state_dict


def create_ane_model(variant: str = "mini") -> ANEVoxtralEncoderWithProjector:
    """Create ANE-optimized model with Voxtral configuration for the specified variant.

    Args:
        variant: "mini" (3072 output) or "small" (5120 output)

    Returns:
        ANEVoxtralEncoderWithProjector configured for the variant
    """
    if variant not in VOXTRAL_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Must be one of: {list(VOXTRAL_VARIANTS.keys())}")

    variant_config = VOXTRAL_VARIANTS[variant]
    text_hidden_size = variant_config["text_hidden_size"]

    encoder_config = VoxtralEncoderConfig(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_mel_bins=128,
        max_source_positions=1500
    )

    projector_config = VoxtralProjectorConfig(
        audio_intermediate_size=5120,
        text_hidden_size=text_hidden_size
    )

    print(f"  Creating ANE model for variant '{variant}'")
    print(f"    Projector output: {text_hidden_size}")
    print(f"    Description: {variant_config['description']}")

    return ANEVoxtralEncoderWithProjector(encoder_config, projector_config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ANE-Optimized VoxtralEncoder Test")
    print("=" * 60)

    # Parse variant from command line
    variant = "mini"
    if len(sys.argv) > 1 and sys.argv[1] in VOXTRAL_VARIANTS:
        variant = sys.argv[1]

    print(f"\nTesting variant: {variant}")
    variant_config = VOXTRAL_VARIANTS[variant]
    expected_output_size = variant_config["text_hidden_size"]

    # Create model
    print("\nCreating ANE-optimized model...")
    model = create_ane_model(variant=variant)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 128, 3000)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     [1, 375, {expected_output_size}]")

    # Verify output shape
    expected_shape = (1, 375, expected_output_size)
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
    print("\n✓ Forward pass successful!")

    # Test intermediate shapes
    print("\nIntermediate shapes (ANE format):")
    with torch.no_grad():
        encoder_out = model.encoder(dummy_input)
        print(f"  Encoder output: {encoder_out.shape} (B, C, 1, S)")

    print("\n" + "=" * 60)
