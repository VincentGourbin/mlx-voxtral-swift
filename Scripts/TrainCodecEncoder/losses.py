"""Loss functions for Voxtral Codec Encoder training.

From paper Eq. 3:
  Loss = α·L_feature + β·L_ASR + γ·L_L1 + γ·L_STFT + δ·L_commit

Where:
  α=1.0 (adversarial feature matching)
  β=1.0 (ASR distillation)
  γ=0.9999^t (exponentially decaying reconstruction)
  δ=0.1 (VQ commitment)
"""

import torch
import torch.nn.functional as F
from typing import List


def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 waveform reconstruction loss."""
    min_len = min(x.shape[-1], y.shape[-1])
    return F.l1_loss(x[..., :min_len], y[..., :min_len])


def stft_loss(x: torch.Tensor, y: torch.Tensor,
              fft_size: int = 1024, hop_length: int = 256, win_length: int = 1024) -> torch.Tensor:
    """L1 loss on STFT magnitudes."""
    min_len = min(x.shape[-1], y.shape[-1])
    x, y = x[..., :min_len], y[..., :min_len]

    window = torch.hann_window(win_length, device=x.device)
    x_stft = torch.stft(x.squeeze(1), fft_size, hop_length, win_length, window=window, return_complex=True)
    y_stft = torch.stft(y.squeeze(1), fft_size, hop_length, win_length, window=window, return_complex=True)

    return F.l1_loss(x_stft.abs(), y_stft.abs())


def multi_scale_stft_loss(x: torch.Tensor, y: torch.Tensor,
                          fft_sizes: List[int] = [512, 1024, 2048]) -> torch.Tensor:
    """Multi-scale STFT loss averaged over multiple FFT sizes."""
    loss = 0.0
    for fft_size in fft_sizes:
        loss += stft_loss(x, y, fft_size=fft_size, hop_length=fft_size // 4, win_length=fft_size)
    return loss / len(fft_sizes)


def feature_matching_loss(real_features: List[List[torch.Tensor]],
                          fake_features: List[List[torch.Tensor]]) -> torch.Tensor:
    """L1 feature matching loss across all discriminators and layers.

    Paper Eq. 2: L_feature = (1/MN) Σ_m Σ_n ||D_n^m(x) - D_n^m(x̂)||_1
    """
    loss = 0.0
    count = 0
    for real_disc_feats, fake_disc_feats in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_disc_feats, fake_disc_feats):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            count += 1
    return loss / max(count, 1)


def discriminator_hinge_loss(real_logits: List[torch.Tensor],
                             fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for discriminator training.

    D_loss = Σ_n [max(0, 1 - D_n(x)) + max(0, 1 + D_n(x̂))]
    """
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss += torch.mean(F.relu(1.0 - real)) + torch.mean(F.relu(1.0 + fake))
    return loss / len(real_logits)


def generator_hinge_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Generator loss (minimize -D(x̂))."""
    loss = 0.0
    for fake in fake_logits:
        loss += -torch.mean(fake)
    return loss / len(fake_logits)


def vq_commitment_loss(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """VQ commitment loss: ||z_e - sg(z_q)||^2

    Paper: L_commit = ||z_e - sg(z_q)||^2
    """
    return F.mse_loss(z_e, z_q.detach())
