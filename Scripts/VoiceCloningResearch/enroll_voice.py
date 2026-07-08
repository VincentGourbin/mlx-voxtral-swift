#!/usr/bin/env python3
"""Enroll a voice for Voxtral TTS from a reference recording.

End-to-end offline pipeline (the codec encoder was never published by
Mistral, so codes are recovered by gradient descent through the frozen
decoder — see README.md):

  1. Prepare the reference: mono, 24 kHz, cut at the last natural pause
     before the target duration (fade + trailing silence). A reference
     that ends mid-speech destabilizes the start of later syntheses.
  2. Optimize discrete codes [T, 37] against the reference
     (upstream/training_script.py, with our MPS gradient fixes applied).
  3. Convert codes to a voice embedding [T+1, 3072] WITH the END_AUDIO
     terminator frame (all official presets end with it; without it the
     model starts generating as if mid-stream).
  4. Export as .safetensors, directly usable by VoxtralCLI:

     VoxtralCLI tts "Any text" -o out.wav --model tts-4b \
         --voice-embedding voices/<name>.safetensors

Usage:
    .venv/bin/python enroll_voice.py --reference my_voice.wav --name my_voice
    .venv/bin/python enroll_voice.py --reference clip.mp3 --name demo --epochs 15000

Cost: ~30 min for 5000 epochs (default) on an M-series Mac, ~1h30 for 15000.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
WEIGHTS = HERE / "voxtral-tts-weights"

SAMPLING_RATE = 24_000
FRAME_RATE = 12.5  # codec frames per second


def check_workspace() -> None:
    """Fail early with actionable messages if the workspace is incomplete."""
    if not (UPSTREAM / "training_script.py").exists():
        sys.exit(
            "upstream/ missing. Run:\n"
            "  git clone https://github.com/MarvinRomson/voxtral-tts-codes-for-audio.git upstream\n"
            "  git -C upstream apply ../patches/upstream_fixes.patch"
        )
    if "MPS workaround" not in (UPSTREAM / "training_script.py").read_text():
        sys.exit(
            "upstream/training_script.py lacks the MPS gradient fixes "
            "(torch.stft backward is silently wrong on MPS beyond ~2.5 s of signal).\n"
            "Run:  git -C upstream apply ../patches/upstream_fixes.patch"
        )
    if not (WEIGHTS / "consolidated.safetensors").exists():
        sys.exit(
            "voxtral-tts-weights/ missing. Run:\n"
            "  .venv/bin/hf download mistralai/Voxtral-4B-TTS-2603 --local-dir voxtral-tts-weights"
        )


def prepare_reference(src: Path, dst: Path, duration: float) -> None:
    """Mono 24 kHz WAV of exactly `duration` seconds, ending on a pause."""
    x, sr = sf.read(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != SAMPLING_RATE:
        import torchaudio

        x = torchaudio.functional.resample(
            torch.from_numpy(x), sr, SAMPLING_RATE
        ).numpy()
        sr = SAMPLING_RATE
    if len(x) < duration * sr:
        sys.exit(f"Reference too short: {len(x)/sr:.2f}s < {duration}s required")
    x = x[: int(duration * sr)]

    # Cut at the quietest 50 ms window in the last 1.5 s, so the learned
    # codes end on a natural pause instead of interrupted speech.
    win = sr // 20
    search_start = int((duration - 1.5) * sr)
    best, best_rms = None, np.inf
    for s in range(search_start, len(x) - win, sr // 100):
        rms = float(np.sqrt((x[s : s + win] ** 2).mean()))
        if rms < best_rms:
            best_rms, best = rms, s
    cut = best + win // 2
    if best_rms > 0.02:
        print(
            f"  warning: no clear pause found near the end (min rms {best_rms:.4f}); "
            "consider a reference that ends on silence"
        )

    y = np.zeros(int(duration * sr), dtype=np.float32)
    y[:cut] = x[:cut]
    fade = sr // 50  # 20 ms
    y[cut - fade : cut] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    sf.write(dst, y, sr)
    print(f"  reference prepared: speech 0-{cut/sr:.2f}s + silence to {duration}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reference", required=True, type=Path, help="Reference recording (wav/mp3/flac/...)")
    p.add_argument("--name", required=True, help="Voice name (output file basename)")
    p.add_argument("--epochs", type=int, default=5000, help="Optimization epochs (5000 ok, 15000 better)")
    p.add_argument("--duration", type=float, default=8.0, help="Reference duration in seconds")
    p.add_argument("--device", default="mps", help="Torch device for the decoder")
    p.add_argument("--output-dir", type=Path, default=HERE / "voices")
    args = p.parse_args()

    check_workspace()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = HERE / "checkpoints" / f"enroll_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    num_frames = int(args.duration * FRAME_RATE)

    print(f"[1/4] Preparing reference ({args.duration}s @ 24 kHz mono)...")
    prepared = run_dir / "reference_prepared.wav"
    prepare_reference(args.reference, prepared, args.duration)

    print(f"[2/4] Optimizing codes ({args.epochs} epochs on {args.device}, "
          f"~{args.epochs * 0.35 / 60:.0f} min)...")
    subprocess.run(
        [
            sys.executable, "training_script.py",
            "--reference-audio", str(prepared),
            "--model-path", str(WEIGHTS),
            "--num-epochs", str(args.epochs),
            "--num-frames", str(num_frames),
            "--device", args.device,
            "--learning-rate", "0.1",
            "--reconstruction-weight", "0.5",
            "--speaker-weight", "0.5",
            "--checkpoint-dir", str(run_dir),
            "--log-every", "500",
        ],
        cwd=UPSTREAM,
        check=True,
    )

    print("[3/4] Converting codes to voice embedding (with END_AUDIO frame)...")
    embedding_pt = run_dir / f"{args.name}.pt"
    subprocess.run(
        [
            sys.executable, "codes_to_embeddings.py",
            "--codes", str(run_dir / "final_codes.pt"),
            "--embedding-weight", str(WEIGHTS / "consolidated.safetensors"),
            "--output", str(embedding_pt),
            "--add-end-token",
        ],
        cwd=UPSTREAM,
        check=True,
    )

    print("[4/4] Exporting safetensors...")
    from safetensors.torch import save_file

    emb = torch.load(embedding_pt, map_location="cpu", weights_only=False)
    if isinstance(emb, dict):
        emb = next(iter(emb.values()))
    out = args.output_dir / f"{args.name}.safetensors"
    save_file({"embedding": emb.float().contiguous()}, str(out))

    print(f"\nVoice enrolled: {out}  [{emb.shape[0]} frames, {emb.shape[1]} dims]")
    print("Try it:")
    print(f'  VoxtralCLI tts "Hello, this is my cloned voice." -o test.wav \\')
    print(f"      --model tts-4b --voice-embedding {out}")


if __name__ == "__main__":
    main()
