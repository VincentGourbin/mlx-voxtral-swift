"""Dataset loading for Voxtral Codec Encoder training.

Supports LibriSpeech and custom audio directories.
All audio is resampled to 24kHz mono and normalized to [-1, 1].
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class SpeechDataset(Dataset):
    """Speech audio dataset for codec training.

    Loads audio files, resamples to 24kHz, and returns fixed-length segments.
    """

    def __init__(
        self,
        root: str,
        sample_rate: int = 24000,
        max_seconds: float = 10.0,
        min_seconds: float = 1.0,
        split: str = "train-clean-100",
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.min_samples = int(min_seconds * sample_rate)

        # Collect audio files
        self.files = []
        self._torchaudio_ds = None
        if root == "librispeech" or not os.path.isdir(root):
            # Download LibriSpeech
            data_dir = os.path.expanduser("~/.cache/voxtral_training/librispeech")
            os.makedirs(data_dir, exist_ok=True)
            try:
                ds = torchaudio.datasets.LIBRISPEECH(data_dir, url=split, download=True)
                self._torchaudio_ds = ds
                self.files = list(range(len(ds)))
                print(f"LibriSpeech {split}: {len(self.files)} samples")
            except Exception as e:
                raise ValueError(f"Cannot download LibriSpeech: {e}")
        elif os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                        self.files.append(os.path.join(dirpath, f))
        else:
            raise ValueError(f"Cannot load dataset from {root}")

        print(f"Dataset: {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._torchaudio_ds is not None:
            waveform, sr, *_ = self._torchaudio_ds[idx]
        else:
            waveform, sr = torchaudio.load(self.files[idx])

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Normalize to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Trim/pad to max_samples
        T = waveform.shape[1]
        if T > self.max_samples:
            # Random crop
            start = torch.randint(0, T - self.max_samples, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif T < self.min_samples:
            # Pad with zeros
            waveform = torch.nn.functional.pad(waveform, (0, self.min_samples - T))

        # Ensure length is multiple of patch_size (240)
        patch_size = 240
        trim = waveform.shape[1] % patch_size
        if trim > 0:
            waveform = waveform[:, :-trim]

        return waveform  # [1, T]


def _collate_fn(batch):
    """Pad batch to same length (must be top-level for multiprocessing pickle)."""
    max_len = max(x.shape[1] for x in batch)
    frame_size = 240 * 8
    max_len = ((max_len + frame_size - 1) // frame_size) * frame_size

    padded = []
    for x in batch:
        if x.shape[1] < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - x.shape[1]))
        padded.append(x)
    return torch.stack(padded)


def create_dataloader(
    root: str,
    split: str = "train-clean-100",
    batch_size: int = 8,
    sample_rate: int = 24000,
    max_seconds: float = 10.0,
    min_seconds: float = 1.0,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for speech audio."""

    dataset = SpeechDataset(
        root=root,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
        min_seconds=min_seconds,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
