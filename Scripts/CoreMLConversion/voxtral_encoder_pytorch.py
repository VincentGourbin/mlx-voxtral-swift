#!/usr/bin/env python3
"""
VoxtralEncoder PyTorch Implementation for Core ML Conversion

This is an exact PyTorch port of the Swift VoxtralModeling.swift encoder components.
Used as an intermediate step for converting MLX weights to Core ML.

Architecture:
- VoxtralAttention: Multi-head self-attention
- VoxtralEncoderLayer: Transformer encoder layer (Pre-LN architecture)
- VoxtralEncoder: Full audio encoder with conv1d + positional embeddings + transformer layers
- VoxtralMultiModalProjector: Projects encoder output to LLM hidden size

Configuration (identical for Mini 3B and Small 24B):
- hidden_size: 1280
- intermediate_size: 5120
- num_hidden_layers: 32
- num_attention_heads: 20
- num_mel_bins: 128
- max_source_positions: 1500
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VoxtralEncoderConfig:
    """Configuration for VoxtralEncoder - matches Swift VoxtralEncoderConfig"""
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
    audio_intermediate_size: int = 5120  # From audio encoder
    text_hidden_size: int = 3072  # LLM hidden size (3072 for Mini, 5120 for Small)


# Model variant configurations
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


class VoxtralAttention(nn.Module):
    """
    Multi-head self-attention for Voxtral encoder.
    Exact port of Swift VoxtralAttention class.

    Swift reference: VoxtralModeling.swift lines 59-145
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Swift: self.q_proj = Linear(embedDim, embedDim, bias: bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Swift: self.k_proj = Linear(embedDim, embedDim, bias: false)  # NO BIAS for k_proj!
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Swift: self.v_proj = Linear(embedDim, embedDim, bias: bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Swift: self.out_proj = Linear(embedDim, embedDim, bias: bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching Swift callAsFunction.

        Args:
            hidden_states: [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Swift: let query = q_proj(hiddenStates)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape and transpose for multi-head attention
        # Swift: reshaped(query, [batchSize, seqLen, numHeads, headDim])
        # Swift: transposed(queryReshaped, axes: [0, 2, 1, 3])
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Swift: scaledDotProductAttention(queries, keys, values, scale: scaling, mask: attentionMask)
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            scale=self.scaling
        )

        # Transpose back and reshape
        # Swift: transposed(attnOutput, axes: [0, 2, 1, 3]).reshaped([batchSize, seqLen, embedDim])
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        # Swift: out_proj(outputReshaped)
        output = self.out_proj(attn_output)

        return output, None  # Swift: return (finalOutput, nil)


class VoxtralEncoderLayer(nn.Module):
    """
    Transformer encoder layer with Pre-LN architecture.
    Exact port of Swift VoxtralEncoderLayer class.

    Swift reference: VoxtralModeling.swift lines 150-236
    """

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        # Swift: self.self_attn_layer_norm = LayerNorm(dimensions: embedDim, eps: 1e-5)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

        # Swift: self.final_layer_norm = LayerNorm(dimensions: embedDim, eps: 1e-5)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

        # Swift: self.self_attn = VoxtralAttention(embedDim, numHeads, bias: true)
        self.self_attn = VoxtralAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=True
        )

        # Swift: self.fc1 = Linear(embedDim, config.intermediate_size, bias: true)
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size, bias=True)

        # Swift: self.fc2 = Linear(config.intermediate_size, embedDim, bias: true)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim, bias=True)

        # Activation function
        if config.activation_function == "gelu":
            self.activation_fn = F.gelu
        elif config.activation_function == "relu":
            self.activation_fn = F.relu
        elif config.activation_function == "silu":
            self.activation_fn = F.silu
        else:
            self.activation_fn = F.gelu

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching Swift callAsFunction.
        Pre-LN architecture: LayerNorm before attention/FFN.
        """
        # Swift: let residual = hiddenStates
        residual = hidden_states

        # Swift: let normalizedStates = self_attn_layer_norm(hiddenStates)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Swift: let (attnOutput, attnWeights) = self_attn(normalizedStates, ...)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # Swift: let afterAttnResidual = residual + attnOutput
        hidden_states = residual + hidden_states

        # Swift: let residual2 = afterAttnResidual
        residual = hidden_states

        # Swift: let normalizedStates2 = final_layer_norm(afterAttnResidual)
        hidden_states = self.final_layer_norm(hidden_states)

        # Swift: activation(fc1(normalizedStates2))
        hidden_states = self.activation_fn(self.fc1(hidden_states))

        # Swift: fc2(activatedStates)
        hidden_states = self.fc2(hidden_states)

        # Swift: let finalOutput = residual2 + fc2Output
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class VoxtralEncoder(nn.Module):
    """
    Full Voxtral audio encoder with conv1d + positional embeddings + transformer layers.
    Exact port of Swift VoxtralEncoder class.

    Swift reference: VoxtralModeling.swift lines 241-397

    Input: [batch, n_mels, n_frames] = [batch, 128, 3000]
    Output: [batch, seq_len, hidden_size] = [batch, 1500, 1280]
    """

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        # Swift: self.conv1 = Conv1d(inputChannels: numMelBins, outputChannels: embedDim, kernelSize: 3, padding: 1)
        self.conv1 = nn.Conv1d(
            in_channels=self.num_mel_bins,
            out_channels=self.embed_dim,
            kernel_size=3,
            padding=1
        )

        # Swift: self.conv2 = Conv1d(inputChannels: embedDim, outputChannels: embedDim, kernelSize: 3, stride: 2, padding: 1)
        self.conv2 = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Swift: self.embed_positions = Embedding(embeddingCount: maxSourcePositions, dimensions: embedDim)
        self.embed_positions = nn.Embedding(
            num_embeddings=self.max_source_positions,
            embedding_dim=self.embed_dim
        )

        # Swift: self.layer_norm = LayerNorm(dimensions: embedDim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # Swift: self.layers = [VoxtralEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            VoxtralEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[torch.Tensor, ...]]]:
        """
        Forward pass matching Swift callAsFunction.

        Args:
            input_features: Mel spectrogram [batch, n_mels, n_frames] = [batch, 128, 3000]
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of (last_hidden_state, all_hidden_states, all_attentions)
        """
        # Swift: var hiddenStates = transposed(inputFeatures, axes: [0, 2, 1])
        # Note: In Swift, input is [batch, n_mels, n_frames] and we transpose to [batch, n_frames, n_mels]
        # But Conv1d expects [batch, channels, length], so we DON'T transpose here
        # Instead, Swift transposes before conv (which is unusual), let's match that behavior

        # Swift transposes to [batch, n_frames, n_mels] = [batch, 3000, 128]
        # Then applies conv1d which in MLX works on axis=1
        # In PyTorch, Conv1d expects [batch, channels, length]
        # So we need: [batch, 128, 3000] for PyTorch (which is the input format)

        hidden_states = input_features  # Already [batch, 128, 3000]

        # Swift: hiddenStates = gelu(conv1(hiddenStates))
        hidden_states = F.gelu(self.conv1(hidden_states))  # [batch, 1280, 3000]

        # Swift: hiddenStates = gelu(conv2(hiddenStates))
        hidden_states = F.gelu(self.conv2(hidden_states))  # [batch, 1280, 1500] due to stride=2

        # Transpose for transformer: [batch, 1280, 1500] -> [batch, 1500, 1280]
        hidden_states = hidden_states.transpose(1, 2)

        # Swift: let seqLen = hiddenStates.shape[1]
        # STATIC: After conv2 with stride=2, 3000 -> 1500
        # For Core ML compatibility, use fixed value
        seq_len = 1500

        # Swift: let embedPos = self.embed_positions.weight[:seq_len]
        # Use register_buffer for static positions to avoid dynamic arange
        positions = torch.arange(seq_len, dtype=torch.long)
        embed_pos = self.embed_positions(positions)  # [1500, embed_dim]

        # Swift: hiddenStates = hiddenStates + embedPos
        hidden_states = hidden_states + embed_pos

        # Prepare attention mask if provided
        prepared_mask = None
        if attention_mask is not None:
            prepared_mask = self._prepare_attention_mask(attention_mask)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Process through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=prepared_mask,
                output_attentions=output_attentions
            )

            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

        # Swift: hiddenStates = self.layer_norm(hiddenStates)
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions

    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prepare attention mask for transformer layers."""
        batch_size, seq_len = attention_mask.shape

        # Expand mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
        expanded_mask = attention_mask[:, None, None, :]

        # Invert: 1 -> 0, 0 -> -10000
        inverted_mask = (1.0 - expanded_mask) * -1e4

        # Broadcast to [batch, 1, seq_len, seq_len]
        return inverted_mask.expand(batch_size, 1, seq_len, seq_len)


class VoxtralMultiModalProjector(nn.Module):
    """
    Projects audio encoder output to LLM hidden size.
    Exact port of Swift VoxtralMultiModalProjector class.

    Swift reference: VoxtralModeling.swift lines 402-437

    Input: [batch, seq_len, 5120] (intermediate_size)
    Output: [batch, seq_len, 3072] (text_hidden_size)
    """

    def __init__(self, config: VoxtralProjectorConfig):
        super().__init__()

        # Swift: self.linear_1 = Linear(config.audio_config.intermediate_size, config.text_config.hidden_size, bias: false)
        self.linear_1 = nn.Linear(
            config.audio_intermediate_size,
            config.text_hidden_size,
            bias=False
        )

        # Swift: self.act = gelu
        self.act = F.gelu

        # Swift: self.linear_2 = Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias: false)
        self.linear_2 = nn.Linear(
            config.text_hidden_size,
            config.text_hidden_size,
            bias=False
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching Swift callAsFunction.

        Args:
            audio_features: [batch, seq_len, audio_intermediate_size]

        Returns:
            Projected features [batch, seq_len, text_hidden_size]
        """
        # Swift: let hiddenStates = linear_1(audioFeatures)
        hidden_states = self.linear_1(audio_features)

        # Swift: let activatedStates = act(hiddenStates)
        hidden_states = self.act(hidden_states)

        # Swift: return linear_2(activatedStates)
        return self.linear_2(hidden_states)


class VoxtralEncoderWithProjector(nn.Module):
    """
    Combined encoder + projector for Core ML conversion.
    This is the full audio processing pipeline.

    Input: Mel spectrogram [batch, 128, 3000]
    Output: Audio embeddings [batch, num_frames, text_hidden_size]
            - Mini:  [1, 375, 3072]
            - Small: [1, 375, 5120]

    Processing:
    1. VoxtralEncoder: [1, 128, 3000] -> [1, 1500, 1280]
    2. Reshape: [1, 1500, 1280] -> [375, 5120] (4 frames combined)
    3. VoxtralMultiModalProjector: [375, 5120] -> [375, text_hidden_size]
    4. Add batch dim: [375, text_hidden_size] -> [1, 375, text_hidden_size]
    """

    def __init__(
        self,
        encoder_config: Optional[VoxtralEncoderConfig] = None,
        projector_config: Optional[VoxtralProjectorConfig] = None,
        variant: str = "mini"  # "mini" or "small"
    ):
        super().__init__()

        self.encoder_config = encoder_config or VoxtralEncoderConfig()
        self.projector_config = projector_config or VoxtralProjectorConfig()
        self.variant = variant

        self.encoder = VoxtralEncoder(self.encoder_config)
        self.projector = VoxtralMultiModalProjector(self.projector_config)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Full audio processing pipeline.

        Args:
            mel_spectrogram: [1, 128, 3000] mel spectrogram (batch=1 fixed for Core ML)

        Returns:
            Audio embeddings [1, 375, text_hidden_size]
            - Mini:  [1, 375, 3072]
            - Small: [1, 375, 5120]
        """
        # Step 1: Encode
        # Swift: let (hiddenStates, _, _) = audio_tower(inputFeatures, ...)
        encoder_output, _, _ = self.encoder(mel_spectrogram)  # [1, 1500, 1280]

        # Step 2: Reshape for projector
        # IMPORTANT: Use STATIC shapes for Core ML compatibility
        # Core ML doesn't support dynamic reshape with -1 or .shape operations
        #
        # Math: 1500 * 1280 = 1,920,000 / 5120 = 375
        # We combine 4 consecutive frames of 1280 into 5120
        # 1500 / 4 = 375, 1280 * 4 = 5120
        #
        # [1, 1500, 1280] -> [1, 375, 4, 1280] -> [1, 375, 5120]

        # Use torch.reshape with literal integers (no .shape access)
        reshaped = encoder_output.reshape(1, 375, 4, 1280)
        reshaped = reshaped.reshape(1, 375, 5120)  # [1, 375, 5120]

        # Step 3: Project
        # Swift: audioEmbeds = multi_modal_projector(audioHiddenStatesProcessed)
        projected = self.projector(reshaped)  # [1, 375, text_hidden_size]

        return projected


def create_model_for_variant(variant: str = "mini") -> VoxtralEncoderWithProjector:
    """
    Create model for a specific Voxtral variant.

    Args:
        variant: "mini" for Mini 3B (output 3072) or "small" for Small 24B (output 5120)

    Returns:
        VoxtralEncoderWithProjector configured for the variant
    """
    if variant not in VOXTRAL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(VOXTRAL_VARIANTS.keys())}")

    variant_config = VOXTRAL_VARIANTS[variant]

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
        text_hidden_size=variant_config["text_hidden_size"]
    )

    print(f"Creating model for {variant_config['description']}")
    return VoxtralEncoderWithProjector(encoder_config, projector_config, variant=variant)


def create_default_model() -> VoxtralEncoderWithProjector:
    """Create model with default Voxtral Mini configuration (for backward compatibility)."""
    return create_model_for_variant("mini")


if __name__ == "__main__":
    # Test the model
    print("Creating VoxtralEncoderWithProjector...")
    model = create_default_model()

    # Print model structure
    print(f"\nModel structure:")
    print(f"  Encoder layers: {len(model.encoder.layers)}")
    print(f"  Encoder hidden_size: {model.encoder.embed_dim}")
    print(f"  Projector output: {model.projector_config.text_hidden_size}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(1, 128, 3000)  # [batch, n_mels, n_frames]

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [1, 375, 3072]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
