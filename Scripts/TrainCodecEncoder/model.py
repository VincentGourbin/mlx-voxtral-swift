"""Voxtral Codec Encoder + frozen Decoder + Quantizer.

The encoder architecture mirrors the decoder (paper Section 2.1, Table 1):
- Input projection: CausalConv(patch_size→dim, k=7)
- 4 blocks: Transformer(ALiBi, window halving) → CausalConv(downsampling)
- Output: 292-dim latent (256 semantic + 36 acoustic)

The decoder and quantizer codebook are loaded from the TTS checkpoint and frozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from config import CodecConfig


# ==============================================================================
# Shared Components (used by both encoder and decoder)
# ==============================================================================


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 use_weight_norm: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, bias=False)
        if use_weight_norm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution for upsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 use_weight_norm: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, bias=False)
        if use_weight_norm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        out = self.conv(x)
        # Trim for causality
        return out[:, :, :T * self.stride]


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for n heads."""
    def _slopes_pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _slopes_pow2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = _slopes_pow2(closest)
        extra = _slopes_pow2(2 * closest)
        slopes += extra[0::2][:n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)


class Attention(nn.Module):
    """Multi-head attention with ALiBi bias, QK-norm, and sliding window."""

    def __init__(self, cfg: CodecConfig, window_size: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        self.window_size = window_size

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        if cfg.qk_norm:
            self.q_norm = nn.RMSNorm(cfg.n_heads * cfg.head_dim, eps=cfg.qk_norm_eps)
            self.k_norm = nn.RMSNorm(cfg.n_kv_heads * cfg.head_dim, eps=cfg.qk_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.register_buffer('alibi_slopes', get_alibi_slopes(cfg.n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # ALiBi bias
        pos = torch.arange(T, device=x.device)
        dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # [T, T], dist[i,j] = j - i
        alibi = self.alibi_slopes[:, None, None] * dist[None, :, :]
        scores = scores + alibi

        # Causal + sliding window mask
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        if self.window_size > 0:
            window_mask = torch.where(dist < -self.window_size, float('-inf'), 0.0).to(x.device)
            causal_mask = causal_mask + window_mask

        scores = scores + causal_mask
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerLayer(nn.Module):
    """Single transformer layer with LayerScale."""

    def __init__(self, cfg: CodecConfig, window_size: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attention = Attention(cfg, window_size)
        self.feed_forward = FeedForward(cfg)

        if cfg.layer_scale:
            self.attention_scale = nn.Parameter(torch.full((cfg.dim,), cfg.layer_scale_init))
            self.ffn_scale = nn.Parameter(torch.full((cfg.dim,), cfg.layer_scale_init))
        else:
            self.attention_scale = None
            self.ffn_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attention(self.attention_norm(x))
        if self.attention_scale is not None:
            h = h * self.attention_scale
        x = x + h

        h = self.feed_forward(self.ffn_norm(x))
        if self.ffn_scale is not None:
            h = h * self.ffn_scale
        x = x + h
        return x


class TransformerBlock(nn.Module):
    """Block of N transformer layers."""

    def __init__(self, n_layers: int, cfg: CodecConfig, window_size: int):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(cfg, window_size) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ==============================================================================
# Quantization
# ==============================================================================


class SemanticVQ(nn.Module):
    """Vector Quantizer for semantic tokens.

    Uses EMA-updated codebook from the frozen TTS checkpoint.
    During training, VQ is applied with 50% probability.
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        # Codebook stored as embedding_sum / cluster_usage (EMA format from checkpoint)
        self.register_buffer('embedding_sum', torch.zeros(codebook_size, dim))
        self.register_buffer('cluster_usage', torch.ones(codebook_size))

    @property
    def codebook(self) -> torch.Tensor:
        return self.embedding_sum.float() / self.cluster_usage.float().clamp(min=1e-8).unsqueeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → indices: [B, T]"""
        flat = x.float().reshape(-1, self.dim)
        cb = self.codebook
        dists = torch.cdist(flat, cb, p=2)
        indices = dists.argmin(dim=-1)
        return indices.reshape(x.shape[0], x.shape[1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, T] → embeddings: [B, T, D]"""
        return self.codebook[indices]

    def forward(self, x: torch.Tensor, vq_prob: float = 0.5):
        """x: [B, T, D] → quantized, indices, commitment_loss"""
        indices = self.encode(x)
        quantized = self.decode(indices)

        # Straight-through estimator
        if self.training:
            # Apply VQ with probability vq_prob
            if torch.rand(1).item() < vq_prob:
                out = x + (quantized - x).detach()
            else:
                out = x
                indices = self.encode(x)  # Still compute indices for monitoring
        else:
            out = quantized

        commitment_loss = F.mse_loss(x, quantized.detach())
        return out, indices, commitment_loss


class AcousticFSQ(nn.Module):
    """Finite Scalar Quantization for acoustic tokens.

    During training: 50% quantized, 25% dithered, 25% pass-through.
    """

    def __init__(self, n_levels: int = 21, dim: int = 36):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to n_levels uniform levels."""
        x = torch.tanh(x)
        scaled = (x + 1) / 2 * (self.n_levels - 1)
        return torch.round(scaled)

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes [0, n_levels-1] → [-1, 1]"""
        return codes.float() / (self.n_levels - 1) * 2 - 1

    def forward(self, x: torch.Tensor, quant_prob: float = 0.5, dither_prob: float = 0.25):
        """x: [B, T, D] → quantized, codes"""
        x_tanh = torch.tanh(x)

        if self.training:
            r = torch.rand(1).item()
            if r < quant_prob:
                # Quantize with straight-through estimator
                codes = self.quantize(x)
                quantized = self.dequantize(codes)
                out = x_tanh + (quantized - x_tanh).detach()
            elif r < quant_prob + dither_prob:
                # Dither: add uniform noise of magnitude 1/L
                noise = torch.rand_like(x_tanh) * (2.0 / self.n_levels) - (1.0 / self.n_levels)
                out = x_tanh + noise
                codes = self.quantize(x)  # For monitoring
            else:
                # Pass through
                out = x_tanh
                codes = self.quantize(x)  # For monitoring
        else:
            codes = self.quantize(x)
            out = self.dequantize(codes)

        return out, codes.long()


# ==============================================================================
# Encoder
# ==============================================================================


class VoxtralCodecEncoder(nn.Module):
    """Voxtral Codec Encoder (paper Section 2.1, Table 1).

    Compresses 24kHz audio to 12.5Hz latent (292-dim = 256 semantic + 36 acoustic).
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection: patch_size → dim
        self.input_proj = CausalConv1d(cfg.patch_size * cfg.channels, cfg.dim,
                                        kernel_size=7, use_weight_norm=cfg.use_weight_norm)

        # 4 encoder blocks: [transformer, conv_downsample]
        self.blocks = nn.ModuleList()
        window_size = cfg.encoder_sliding_windows[0]  # Start at 16
        for i in range(len(cfg.encoder_transformer_lengths)):
            window_size = cfg.encoder_sliding_windows[i]
            # Transformer
            self.blocks.append(TransformerBlock(
                cfg.encoder_transformer_lengths[i], cfg, window_size
            ))
            # Conv (downsampling)
            is_last = (i == len(cfg.encoder_transformer_lengths) - 1)
            out_ch = cfg.latent_dim if is_last else cfg.dim
            self.blocks.append(CausalConv1d(
                cfg.dim, out_ch,
                kernel_size=cfg.encoder_convs_kernels[i],
                stride=cfg.encoder_convs_strides[i],
                use_weight_norm=cfg.use_weight_norm,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → latent: [B, latent_dim, T'] where T'=T/(240*8)"""
        B, C, T = x.shape

        # Patchify
        n_patches = T // self.cfg.patch_size
        x = x[:, :, :n_patches * self.cfg.patch_size]
        x = x.reshape(B, C * self.cfg.patch_size, n_patches)  # [B, 240, n_patches]

        # Input projection
        x = self.input_proj(x)  # [B, 1024, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, 1024] for transformer

        # Encoder blocks: [transformer, conv] × 4
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                x = block(x)
            elif isinstance(block, CausalConv1d):
                x = x.transpose(1, 2)  # [B, D, T] for conv
                x = block(x)
                x = x.transpose(1, 2)  # [B, T, D] for transformer

        # x: [B, T', latent_dim=292]
        return x.transpose(1, 2)  # [B, 292, T']


# ==============================================================================
# Decoder (Frozen — loaded from TTS checkpoint)
# ==============================================================================


class VoxtralCodecDecoder(nn.Module):
    """Voxtral Codec Decoder. Loaded from TTS checkpoint and frozen."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg

        # Build decoder blocks: [conv, transformer] × 4
        self.blocks = nn.ModuleList()
        strides = cfg.decoder_convs_strides
        kernels = cfg.decoder_convs_kernels
        window_sizes = [2, 4, 8, 16]  # Doubling (reverse of encoder halving)

        for i in range(len(cfg.decoder_transformer_lengths)):
            # Conv (first block takes latent_dim, rest take dim)
            in_ch = cfg.latent_dim if i == 0 else cfg.dim
            if strides[i] > 1:
                self.blocks.append(CausalConvTranspose1d(
                    in_ch, cfg.dim, kernel_size=kernels[i], stride=strides[i],
                    use_weight_norm=cfg.use_weight_norm
                ))
            else:
                self.blocks.append(CausalConv1d(
                    in_ch, cfg.dim, kernel_size=kernels[i], stride=strides[i],
                    use_weight_norm=cfg.use_weight_norm
                ))

            # Transformer
            self.blocks.append(TransformerBlock(
                cfg.decoder_transformer_lengths[i], cfg, window_sizes[i]
            ))

        # Output projection
        self.output_proj = CausalConv1d(cfg.dim, cfg.patch_size, kernel_size=7,
                                         use_weight_norm=cfg.use_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, latent_dim, T'] → waveform: [B, 1, T]"""
        strides = self.cfg.decoder_convs_strides

        for i in range(0, len(self.blocks), 2):
            conv_block = self.blocks[i]
            xformer_block = self.blocks[i + 1]

            # Conv
            x = conv_block(x)

            # Transformer
            x = x.transpose(1, 2)
            x = xformer_block(x)
            x = x.transpose(1, 2)

        # Output projection
        x = self.output_proj(x)  # [B, 240, T_up]

        # Reshape patches to waveform
        B = x.shape[0]
        return x.reshape(B, 1, -1)


# ==============================================================================
# Full Codec Model
# ==============================================================================


class VoxtralCodec(nn.Module):
    """Full Voxtral Codec: Encoder → Quantize → Decoder.

    The decoder and quantizer codebook are frozen (loaded from TTS checkpoint).
    Only the encoder is trainable.
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VoxtralCodecEncoder(cfg)
        self.decoder = VoxtralCodecDecoder(cfg)
        self.semantic_vq = SemanticVQ(cfg.semantic_codebook_size, cfg.semantic_dim)
        self.acoustic_fsq = AcousticFSQ(cfg.acoustic_codebook_size, cfg.acoustic_dim)

        # ASR distillation projection (semantic_dim → whisper_dim)
        self.asr_projection = nn.Linear(cfg.semantic_dim, 512)  # Will be resized for actual Whisper dim

    def freeze_decoder(self):
        """Freeze decoder and quantizer codebook — only encoder is trainable."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.semantic_vq.embedding_sum.requires_grad = False
        self.semantic_vq.cluster_usage.requires_grad = False

    def load_decoder_weights(self, checkpoint_path: str):
        """Load decoder + quantizer weights from TTS checkpoint."""
        weights = {}
        with safe_open(checkpoint_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("audio_tokenizer."):
                    weights[key[len("audio_tokenizer."):]] = f.get_tensor(key)

        # Map decoder_blocks
        decoder_state = {}
        for k, v in weights.items():
            if k.startswith("decoder_blocks."):
                # Map alternating conv/transformer blocks
                parts = k.split(".")
                block_idx = int(parts[1])
                rest = ".".join(parts[2:])

                # In checkpoint: even=conv, odd=transformer
                # In our model: same structure
                decoder_state[f"blocks.{block_idx}.{rest}"] = v

            elif k.startswith("output_proj."):
                rest = k[len("output_proj."):]
                decoder_state[f"output_proj.{rest}"] = v

            elif k.startswith("quantizer.semantic_codebook."):
                rest = k[len("quantizer.semantic_codebook."):]
                if rest == "embedding_sum":
                    self.semantic_vq.embedding_sum.data.copy_(v.float())
                elif rest == "cluster_usage":
                    self.semantic_vq.cluster_usage.data.copy_(v.float())

        # Load decoder weights (handling weight_norm parametrizations)
        missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
        print(f"Decoder loaded: {len(decoder_state)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"  Missing: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}...")

    def forward(self, x: torch.Tensor, vq_prob: float = 0.5):
        """Full forward: encode → quantize → decode.

        Args:
            x: [B, 1, T] audio waveform (24kHz, normalized to [-1, 1])

        Returns:
            reconstructed: [B, 1, T'] reconstructed waveform
            semantic_codes: [B, T_frames] semantic token indices
            acoustic_codes: [B, T_frames, 36] acoustic FSQ indices
            commit_loss: VQ commitment loss
            latent_semantic: [B, T_frames, 256] post-VQ semantic embeddings
        """
        # Encode
        latent = self.encoder(x)  # [B, 292, T']
        latent = latent.transpose(1, 2)  # [B, T', 292]

        # Split semantic/acoustic
        semantic = latent[:, :, :self.cfg.semantic_dim]   # [B, T', 256]
        acoustic = latent[:, :, self.cfg.semantic_dim:]   # [B, T', 36]

        # Quantize
        sem_quantized, sem_codes, commit_loss = self.semantic_vq(semantic, vq_prob=vq_prob)
        ac_quantized, ac_codes = self.acoustic_fsq(acoustic)

        # Reconstruct latent
        quantized_latent = torch.cat([sem_quantized, ac_quantized], dim=-1)  # [B, T', 292]
        quantized_latent = quantized_latent.transpose(1, 2)  # [B, 292, T']

        # Decode (frozen)
        with torch.no_grad():
            reconstructed = self.decoder(quantized_latent)

        return reconstructed, sem_codes, ac_codes, commit_loss, semantic
