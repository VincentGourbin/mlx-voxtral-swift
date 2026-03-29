"""Whisper ASR Distillation Loss for Voxtral Codec training.

From paper Section 2.1 (Semantic Token Learning):
L_ASR = 1 - (1/L) Σ_l cos_sim(z̃_l, h_l)

Simplified: we align the codec's semantic embeddings with Whisper's encoder
output using interpolation + cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper


class WhisperDistillationLoss(nn.Module):
    """ASR distillation loss using a frozen Whisper encoder."""

    def __init__(self, whisper_model: str = "base", semantic_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.whisper_sr = 16000

        # Load Whisper on target device
        self.whisper = whisper.load_model(whisper_model, device=device)
        for param in self.whisper.parameters():
            param.requires_grad = False
        self.whisper.eval()
        self._device = device

        # Projection: semantic_dim → whisper encoder dim
        whisper_dim = self.whisper.dims.n_audio_state  # 512 for base
        self.projection = nn.Linear(semantic_dim, whisper_dim)

    @torch.no_grad()
    def _get_whisper_features(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Get Whisper encoder features from 24kHz audio.

        Args:
            audio_24k: [B, T] audio at 24kHz on any device

        Returns:
            [B, T_whisper, D_whisper] encoder features on same device as audio_24k
        """
        device = audio_24k.device
        B = audio_24k.shape[0]

        all_features = []
        for i in range(B):
            # Resample 24kHz → 16kHz
            audio_sample = audio_24k[i].float()
            audio_16k = F.interpolate(
                audio_sample.unsqueeze(0).unsqueeze(0),
                scale_factor=self.whisper_sr / 24000,
                mode='linear', align_corners=False
            ).squeeze()

            # Pad/trim to 30s (Whisper requirement) — must be on CPU for whisper utils
            audio_padded = whisper.pad_or_trim(audio_16k.cpu())

            # Mel spectrogram (CPU) then move to Whisper's device
            mel = whisper.log_mel_spectrogram(audio_padded).to(self._device).unsqueeze(0)
            features = self.whisper.encoder(mel)  # [1, 1500, D]
            all_features.append(features)

        # Stack and move to original device
        return torch.cat(all_features, dim=0).to(device)  # [B, 1500, D]

    def forward(self, semantic_embeddings: torch.Tensor, audio: torch.Tensor,
                codec_frame_rate: float = 12.5) -> torch.Tensor:
        """Compute ASR distillation loss.

        Args:
            semantic_embeddings: [B, T_codec, semantic_dim] post-VQ semantic embeddings
            audio: [B, 1, T_audio] original audio at 24kHz

        Returns:
            loss: scalar cosine alignment loss
        """
        audio_flat = audio.squeeze(1)  # [B, T_audio]

        # Get Whisper encoder features
        whisper_features = self._get_whisper_features(audio_flat)  # [B, 1500, D_whisper]
        T_whisper = whisper_features.shape[1]
        T_codec = semantic_embeddings.shape[1]

        # Project semantic embeddings to Whisper dimension
        projected = self.projection(semantic_embeddings)  # [B, T_codec, D_whisper]

        # Align lengths via interpolation (codec ~12.5Hz, Whisper 1500 frames for 30s = 50Hz)
        # Only use the portion of Whisper features that corresponds to actual audio
        audio_duration = audio_flat.shape[1] / 24000.0
        whisper_frames_used = min(T_whisper, int(audio_duration * 50))  # 50 frames/sec
        whisper_features = whisper_features[:, :whisper_frames_used, :]

        # Interpolate codec frames to match Whisper frame count
        if T_codec > 1 and whisper_frames_used > 1:
            projected = F.interpolate(
                projected.transpose(1, 2),
                size=whisper_frames_used,
                mode='linear', align_corners=False
            ).transpose(1, 2)

            # Cosine similarity
            cos_sim = F.cosine_similarity(
                projected.float(), whisper_features.float(), dim=-1
            )
            loss = 1.0 - cos_sim.mean()
        else:
            loss = torch.tensor(0.0, device=audio.device, requires_grad=True)

        return loss
