"""Dataset loading for Voxtral Codec Encoder training.

Supports:
- LibriSpeech (English only)
- Mozilla Common Voice (multilingual — 9 Voxtral languages)
- Custom audio directory

All audio is resampled to 24kHz mono and normalized to [-1, 1].
"""

import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, List


# The 9 languages supported by Voxtral TTS
VOXTRAL_LANGUAGES = ["en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"]


class SpeechDataset(Dataset):
    """Speech audio dataset for codec training."""

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

        self.files = []
        self._torchaudio_ds = None
        if root == "librispeech" or (not os.path.isdir(root) and "librispeech" in root.lower()):
            data_dir = os.path.expanduser("~/.cache/voxtral_training/librispeech")
            os.makedirs(data_dir, exist_ok=True)
            ds = torchaudio.datasets.LIBRISPEECH(data_dir, url=split, download=True)
            self._torchaudio_ds = ds
            self.files = list(range(len(ds)))
            print(f"  LibriSpeech {split}: {len(self.files)} samples")
        elif os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.endswith(('.wav', '.flac', '.mp3', '.ogg', '.opus')):
                        self.files.append(os.path.join(dirpath, f))
            print(f"  Audio directory: {len(self.files)} files")
        else:
            raise ValueError(f"Cannot load dataset from {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._torchaudio_ds is not None:
            waveform, sr, *_ = self._torchaudio_ds[idx]
        else:
            waveform, sr = torchaudio.load(self.files[idx])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        T = waveform.shape[1]
        if T > self.max_samples:
            start = torch.randint(0, T - self.max_samples, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif T < self.min_samples:
            waveform = F.pad(waveform, (0, self.min_samples - T))

        patch_size = 240
        trim = waveform.shape[1] % patch_size
        if trim > 0:
            waveform = waveform[:, :-trim]

        return waveform


class CommonVoiceDataset(Dataset):
    """Mozilla Common Voice dataset for a single language.

    Downloads via HuggingFace datasets library.
    """

    def __init__(
        self,
        language: str,
        sample_rate: int = 24000,
        max_seconds: float = 10.0,
        min_seconds: float = 1.0,
        split: str = "train",
        max_samples_per_lang: int = 50000,
        cache_dir: str = "~/.cache/voxtral_training/commonvoice",
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.min_samples = int(min_seconds * sample_rate)

        try:
            from datasets import load_dataset, Audio
            cache_dir = os.path.expanduser(cache_dir)
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                language,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
            if len(ds) > max_samples_per_lang:
                ds = ds.shuffle(seed=42).select(range(max_samples_per_lang))
            self._ds = ds
            print(f"  Common Voice {language}: {len(ds)} samples")
        except Exception as e:
            print(f"  Warning: Could not load Common Voice {language}: {e}")
            self._ds = None

    def __len__(self) -> int:
        return len(self._ds) if self._ds else 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self._ds[idx]
        audio = item["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        T = waveform.shape[1]
        if T > self.max_samples:
            start = torch.randint(0, T - self.max_samples, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif T < self.min_samples:
            waveform = F.pad(waveform, (0, self.min_samples - T))

        patch_size = 240
        trim = waveform.shape[1] % patch_size
        if trim > 0:
            waveform = waveform[:, :-trim]

        return waveform


def _collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(x.shape[1] for x in batch)
    frame_size = 240 * 8
    max_len = ((max_len + frame_size - 1) // frame_size) * frame_size

    padded = []
    for x in batch:
        if x.shape[1] < max_len:
            x = F.pad(x, (0, max_len - x.shape[1]))
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
    languages: Optional[List[str]] = None,
    max_samples_per_lang: int = 50000,
) -> DataLoader:
    """Create a DataLoader for speech audio.

    Args:
        root: "librispeech", "commonvoice", or path to audio directory
        languages: For commonvoice, list of language codes (default: all 9 Voxtral languages)
    """
    use_pin = torch.cuda.is_available()

    if root == "commonvoice":
        langs = languages or VOXTRAL_LANGUAGES
        print(f"Loading Common Voice for {len(langs)} languages...")
        datasets = []
        for lang in langs:
            ds = CommonVoiceDataset(
                language=lang, sample_rate=sample_rate,
                max_seconds=max_seconds, min_seconds=min_seconds,
                split="train", max_samples_per_lang=max_samples_per_lang,
            )
            if len(ds) > 0:
                datasets.append(ds)

        if not datasets:
            raise ValueError("No Common Voice data loaded. Install: pip install datasets")
        dataset = ConcatDataset(datasets)
        print(f"Total: {len(dataset)} samples across {len(datasets)} languages")
    else:
        dataset = SpeechDataset(
            root=root, sample_rate=sample_rate,
            max_seconds=max_seconds, min_seconds=min_seconds, split=split,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=use_pin,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
