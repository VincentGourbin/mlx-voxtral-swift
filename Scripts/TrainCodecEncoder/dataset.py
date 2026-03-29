"""Dataset loading for Voxtral Codec Encoder training.

Supports:
- LibriSpeech (English only, no auth needed)
- Multilingual LibriSpeech / MLS (8 languages, no auth needed)
- FLEURS (102 languages including ar/hi, no auth needed)
- Custom audio directory

All audio is resampled to 24kHz mono and normalized to [-1, 1].
"""

import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, List

# Voxtral TTS languages → dataset mapping
# MLS covers: en, fr, es, de, it, pt, nl, pl (missing ar, hi)
# FLEURS covers all including ar, hi
MLS_LANGUAGES = {
    "en": "english", "fr": "french", "es": "spanish", "de": "german",
    "it": "italian", "pt": "portuguese", "nl": "dutch",
}
FLEURS_LANGUAGES = {"ar": "ar_eg", "hi": "hi_in"}  # Arabic (Egypt), Hindi (India)


class HFAudioDataset(Dataset):
    """Generic HuggingFace audio dataset wrapper."""

    def __init__(self, hf_dataset, sample_rate=24000, max_seconds=10.0, min_seconds=1.0,
                 audio_column="audio"):
        self.ds = hf_dataset
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.min_samples = int(min_seconds * sample_rate)
        self.audio_column = audio_column

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        audio = item[self.audio_column]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)

        # Resample if needed
        if audio.get("sampling_rate", self.sample_rate) != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, audio["sampling_rate"], self.sample_rate
            )

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Trim/pad
        T = waveform.shape[1]
        if T > self.max_samples:
            start = torch.randint(0, T - self.max_samples, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif T < self.min_samples:
            waveform = F.pad(waveform, (0, self.min_samples - T))

        # Align to patch size
        trim = waveform.shape[1] % 240
        if trim > 0:
            waveform = waveform[:, :-trim]

        return waveform


class SpeechDataset(Dataset):
    """Local audio files or LibriSpeech dataset."""

    def __init__(self, root, sample_rate=24000, max_seconds=10.0, min_seconds=1.0,
                 split="train-clean-100"):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.min_samples = int(min_seconds * sample_rate)
        self.files = []
        self._torchaudio_ds = None

        if root == "librispeech":
            data_dir = os.path.expanduser("~/.cache/voxtral_training/librispeech")
            os.makedirs(data_dir, exist_ok=True)
            ds = torchaudio.datasets.LIBRISPEECH(data_dir, url=split, download=True)
            self._torchaudio_ds = ds
            self.files = list(range(len(ds)))
            print(f"  LibriSpeech {split}: {len(self.files)} samples")
        elif os.path.isdir(root):
            for dp, _, fns in os.walk(root):
                for f in fns:
                    if f.endswith(('.wav', '.flac', '.mp3', '.ogg', '.opus')):
                        self.files.append(os.path.join(dp, f))
            print(f"  Audio dir: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
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
        trim = waveform.shape[1] % 240
        if trim > 0:
            waveform = waveform[:, :-trim]
        return waveform


def _collate_fn(batch):
    max_len = max(x.shape[1] for x in batch)
    frame_size = 240 * 8
    max_len = ((max_len + frame_size - 1) // frame_size) * frame_size
    padded = []
    for x in batch:
        if x.shape[1] < max_len:
            x = F.pad(x, (0, max_len - x.shape[1]))
        padded.append(x)
    return torch.stack(padded)


def _load_multilingual(sample_rate, max_seconds, min_seconds, max_per_lang):
    """Load MLS (8 langs) + FLEURS (ar, hi) for full 9-language coverage."""
    from datasets import load_dataset, Audio

    datasets_list = []

    # MLS: en, fr, es, de, it, pt, nl
    for code, name in MLS_LANGUAGES.items():
        try:
            print(f"  Loading MLS {code} ({name})...")
            ds = load_dataset("facebook/multilingual_librispeech", name,
                              split="train", streaming=False)
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
            if len(ds) > max_per_lang:
                ds = ds.shuffle(seed=42).select(range(max_per_lang))
            datasets_list.append(HFAudioDataset(ds, sample_rate, max_seconds, min_seconds))
            print(f"    → {len(ds)} samples")
        except Exception as e:
            print(f"    Warning: MLS {code} failed: {e}")

    # FLEURS: ar, hi (smaller but covers missing languages)
    for code, fleurs_code in FLEURS_LANGUAGES.items():
        try:
            print(f"  Loading FLEURS {code} ({fleurs_code})...")
            ds = load_dataset("google/fleurs", fleurs_code, split="train")
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
            if len(ds) > max_per_lang:
                ds = ds.shuffle(seed=42).select(range(max_per_lang))
            datasets_list.append(HFAudioDataset(ds, sample_rate, max_seconds, min_seconds))
            print(f"    → {len(ds)} samples")
        except Exception as e:
            print(f"    Warning: FLEURS {code} failed: {e}")

    return datasets_list


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
    """Create a DataLoader.

    root options:
        "librispeech" → English only (no auth)
        "commonvoice" or "multilingual" → MLS + FLEURS, 9 languages (no auth)
        "/path/to/audio" → custom directory
    """
    use_pin = torch.cuda.is_available()

    if root.startswith("mls_"):
        # Single-language MLS: mls_french, mls_german, mls_spanish, etc.
        lang_name = root[4:]  # "french", "german", etc.
        print(f"Loading MLS {lang_name}...")
        from datasets import load_dataset, Audio
        ds = load_dataset("facebook/multilingual_librispeech", lang_name, split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        if max_samples_per_lang and len(ds) > max_samples_per_lang:
            ds = ds.shuffle(seed=42).select(range(max_samples_per_lang))
        dataset = HFAudioDataset(ds, sample_rate, max_seconds, min_seconds)
        print(f"  → {len(dataset)} samples")
    elif root in ("commonvoice", "multilingual"):
        print(f"Loading multilingual dataset (MLS + FLEURS)...")
        ds_list = _load_multilingual(sample_rate, max_seconds, min_seconds, max_samples_per_lang)
        if not ds_list:
            raise ValueError("No multilingual data loaded. pip install datasets")
        dataset = ConcatDataset(ds_list)
        print(f"Total: {len(dataset)} samples across {len(ds_list)} languages")
    else:
        dataset = SpeechDataset(root, sample_rate, max_seconds, min_seconds, split)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_collate_fn,
        pin_memory=use_pin, drop_last=True,
        persistent_workers=num_workers > 0,
    )
