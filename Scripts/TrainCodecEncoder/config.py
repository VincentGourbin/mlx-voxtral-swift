"""Training configuration for Voxtral Codec Encoder.

Hyperparameters from paper Section 2.1, Table 1, and training Section (Eq. 3).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodecConfig:
    """Voxtral Codec architecture config (from paper Table 1)."""
    # Audio
    sampling_rate: int = 24000
    patch_size: int = 240
    channels: int = 1

    # Encoder/Decoder shared
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    norm_eps: float = 0.01
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    causal: bool = True
    use_weight_norm: bool = True

    # Encoder
    encoder_transformer_lengths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    encoder_convs_kernels: List[int] = field(default_factory=lambda: [4, 4, 4, 3])
    encoder_convs_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 1])
    encoder_sliding_windows: List[int] = field(default_factory=lambda: [16, 8, 4, 2])

    # Decoder (mirror of encoder)
    decoder_transformer_lengths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_convs_kernels: List[int] = field(default_factory=lambda: [3, 4, 4, 4])
    decoder_convs_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    # Decoder windows double: [2, 4, 8, 16] (reverse of encoder halving)

    # Quantization
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21  # FSQ levels
    acoustic_dim: int = 36

    @property
    def latent_dim(self) -> int:
        return self.semantic_dim + self.acoustic_dim  # 292

    @property
    def frame_rate(self) -> float:
        scale = 1
        for s in self.encoder_convs_strides:
            scale *= s
        return self.sampling_rate / (self.patch_size * scale)  # 12.5 Hz


@dataclass
class DiscriminatorConfig:
    """Multi-resolution STFT discriminator config (from paper Section 2.1)."""
    fft_sizes: List[int] = field(default_factory=lambda: [2296, 1418, 876, 542, 334, 206, 126, 76])
    channels: int = 256


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Dataset
    dataset: str = "librispeech"  # or path to audio directory
    librispeech_split: str = "train-clean-100"  # start small
    max_audio_seconds: float = 10.0
    min_audio_seconds: float = 1.0

    # Loss weights (paper Eq. 3)
    alpha: float = 1.0       # feature matching (adversarial)
    beta: float = 1.0        # ASR distillation
    gamma_base: float = 0.9999  # reconstruction decay base
    delta: float = 0.1       # VQ commitment

    # VQ training tricks
    vq_prob: float = 0.5     # probability of applying VQ during training
    fsq_quant_prob: float = 0.5   # 50% quantized
    fsq_dither_prob: float = 0.25  # 25% dithered
    # remaining 25% pass through unquantized

    # Whisper distillation
    whisper_model: str = "base"  # or "small", "medium", "large-v3"

    # Optimization
    batch_size: int = 8
    learning_rate: float = 3e-4
    lr_discriminator: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100_000
    warmup_steps: int = 1000
    log_interval: int = 100
    eval_interval: int = 5000
    save_interval: int = 10000
    gradient_clip: float = 1.0

    # Model checkpoint
    tts_model_path: str = ""  # path to Voxtral-4B-TTS-2603 checkpoint
    output_dir: str = "checkpoints/codec_encoder"

    # Hardware
    device: str = "mps"  # Apple Silicon
    dtype: str = "float32"  # bf16 has issues on MPS
    num_workers: int = 4

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "voxtral-codec-encoder"
