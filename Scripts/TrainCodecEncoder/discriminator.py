"""Multi-resolution STFT Discriminator for Voxtral Codec training.

8 discriminators with FFT sizes: [2296, 1418, 876, 542, 334, 206, 126, 76].
Each uses STFT → Conv layers → hinge loss.
Feature matching loss: L1 on intermediate activations.

Reference: paper Section 2.1 (Adversarial Training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from config import DiscriminatorConfig


class STFTDiscriminator(nn.Module):
    """Single STFT-based discriminator for one resolution."""

    def __init__(self, fft_size: int, channels: int = 256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = fft_size // 4
        self.win_length = fft_size

        # Input: STFT magnitude [B, freq_bins, time_frames]
        freq_bins = fft_size // 2 + 1

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.LeakyReLU(0.2),
            ),
            nn.Conv2d(channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ])

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude. x: [B, T] → [B, 1, F, T']"""
        window = torch.hann_window(self.win_length, device=x.device)
        spec = torch.stft(x, self.fft_size, self.hop_length, self.win_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # [B, F, T']
        return mag.unsqueeze(1)  # [B, 1, F, T']

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x: [B, T] → (logits, features_list)"""
        h = self.stft(x)
        features = []
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        return features[-1], features[:-1]


class MultiResolutionSTFTDiscriminator(nn.Module):
    """8 STFT discriminators at different resolutions.

    FFT sizes from paper: [2296, 1418, 876, 542, 334, 206, 126, 76]
    """

    def __init__(self, cfg: DiscriminatorConfig = DiscriminatorConfig()):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(fft_size, cfg.channels) for fft_size in cfg.fft_sizes
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """x: [B, T] → (list of logits, list of feature lists)"""
        all_logits = []
        all_features = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features
