"""Training script for Voxtral Codec Encoder.

Trains the encoder while keeping the decoder and VQ codebook frozen.
Uses all losses from the paper (Eq. 3):
  Loss = α·L_feature + β·L_ASR + γ·L_L1 + γ·L_STFT + δ·L_commit

Usage (RunPod / CUDA):
    python train.py --tts_model_path /path/to/Voxtral-4B-TTS-2603 --device cuda --batch_size 16 --dtype bfloat16

Usage (Apple Silicon / MPS):
    python train.py --tts_model_path /path/to/Voxtral-4B-TTS-2603 --device mps --batch_size 1 --dtype float32

Usage (CPU):
    python train.py --tts_model_path /path/to/Voxtral-4B-TTS-2603 --device cpu --batch_size 4
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

from config import CodecConfig, DiscriminatorConfig, TrainConfig
from model import VoxtralCodec
from discriminator import MultiResolutionSTFTDiscriminator
from losses import (l1_loss, multi_scale_stft_loss, feature_matching_loss,
                    discriminator_hinge_loss)
from whisper_distillation import WhisperDistillationLoss
from dataset import create_dataloader


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    dtype = get_dtype(cfg.dtype)
    use_amp = cfg.dtype != "float32" and cfg.device == "cuda"

    print(f"Device: {device}, dtype: {dtype}, AMP: {use_amp}")

    # ---- Model ----
    codec_cfg = CodecConfig()
    codec = VoxtralCodec(codec_cfg).to(device)

    # Load frozen decoder + quantizer from TTS checkpoint
    assert cfg.tts_model_path, "Must provide --tts_model_path"
    ckpt_path = os.path.join(cfg.tts_model_path, "consolidated.safetensors")
    if not os.path.exists(ckpt_path):
        ckpt_files = sorted(Path(cfg.tts_model_path).glob("model-*.safetensors"))
        if ckpt_files:
            from safetensors import safe_open
            merged = {}
            for f in ckpt_files:
                with safe_open(str(f), framework="pt") as sf:
                    for k in sf.keys():
                        if k.startswith("audio_tokenizer."):
                            merged[k] = sf.get_tensor(k)
            ckpt_path = "/tmp/_merged_decoder.safetensors"
            save_file(merged, ckpt_path)
            print(f"Merged {len(merged)} audio_tokenizer weights from {len(ckpt_files)} shards")

    codec.load_decoder_weights(ckpt_path)
    codec.freeze_decoder()

    # Optional: torch.compile for CUDA speedup
    if cfg.compile and cfg.device == "cuda":
        print("Compiling encoder with torch.compile...")
        codec.encoder = torch.compile(codec.encoder)

    encoder_params = sum(p.numel() for p in codec.encoder.parameters())
    total_params = sum(p.numel() for p in codec.parameters() if p.requires_grad)
    print(f"Encoder params: {encoder_params / 1e6:.1f}M, Trainable: {total_params / 1e6:.1f}M")

    # ---- Discriminator ----
    disc_cfg = DiscriminatorConfig()
    discriminator = MultiResolutionSTFTDiscriminator(disc_cfg).to(device)
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.1f}M")

    # ---- Whisper Distillation ----
    whisper_loss_fn = WhisperDistillationLoss(
        whisper_model=cfg.whisper_model,
        semantic_dim=codec_cfg.semantic_dim,
        device=str(device),
    ).to(device)

    # ---- Optimizers ----
    optimizer_g = torch.optim.AdamW(
        list(codec.encoder.parameters()) + list(codec.asr_projection.parameters()) +
        list(whisper_loss_fn.projection.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.lr_discriminator, weight_decay=cfg.weight_decay,
    )

    # AMP scaler for CUDA mixed precision
    scaler_g = torch.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.amp.GradScaler(enabled=use_amp)
    amp_ctx = lambda: torch.amp.autocast(device_type=cfg.device, dtype=dtype) if use_amp else nullcontext()

    # ---- Dataset ----
    dataloader = create_dataloader(
        root=cfg.dataset,
        split=cfg.librispeech_split,
        batch_size=cfg.batch_size,
        sample_rate=codec_cfg.sampling_rate,
        max_seconds=cfg.max_audio_seconds,
        min_seconds=cfg.min_audio_seconds,
        num_workers=cfg.num_workers,
    )

    # ---- Wandb ----
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # ---- Resume ----
    global_step = 0
    resume_path = os.path.join(cfg.output_dir, "latest_checkpoint.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        codec.encoder.load_state_dict(ckpt["encoder"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        global_step = ckpt["global_step"]
        print(f"Resumed from step {global_step}")

    # ---- Training Loop ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    codec.train()
    discriminator.train()

    print(f"\n{'='*60}")
    print(f"Training for {cfg.max_steps} steps (starting from {global_step})")
    print(f"Batch size: {cfg.batch_size}, Audio: {cfg.max_audio_seconds}s")
    print(f"{'='*60}\n")

    while global_step < cfg.max_steps:
        for batch in dataloader:
            if global_step >= cfg.max_steps:
                break

            audio = batch.to(device, non_blocking=True)
            t0 = time.time()
            gamma = cfg.gamma_base ** global_step

            # ---- Forward (with optional AMP) ----
            with amp_ctx():
                reconstructed, sem_codes, ac_codes, commit_loss, semantic_embs = codec(
                    audio, vq_prob=cfg.vq_prob
                )
                audio_flat = audio.squeeze(1)
                recon_flat = reconstructed.squeeze(1)
                min_len = min(audio_flat.shape[-1], recon_flat.shape[-1])
                audio_flat = audio_flat[..., :min_len]
                recon_flat = recon_flat[..., :min_len]

            # ---- Discriminator step ----
            with amp_ctx():
                recon_detached = recon_flat.detach()
                real_logits, real_features = discriminator(audio_flat)
                fake_logits, fake_features = discriminator(recon_detached)
                d_loss = discriminator_hinge_loss(real_logits, fake_logits)

            optimizer_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.gradient_clip)
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # ---- Generator step ----
            with amp_ctx():
                fake_logits_g, fake_features_g = discriminator(recon_flat)
                _, real_features_g = discriminator(audio_flat.detach())
                l_feature = feature_matching_loss(real_features_g, fake_features_g)
                l_asr = whisper_loss_fn(semantic_embs, audio, codec_cfg.frame_rate)
                l_l1 = l1_loss(reconstructed, audio)
                l_stft = multi_scale_stft_loss(reconstructed, audio)
                l_commit = commit_loss

                g_loss = (
                    cfg.alpha * l_feature +
                    cfg.beta * l_asr +
                    gamma * (l_l1 + l_stft) +
                    cfg.delta * l_commit
                )

            optimizer_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(
                list(codec.encoder.parameters()) + list(whisper_loss_fn.projection.parameters()),
                cfg.gradient_clip
            )
            scaler_g.step(optimizer_g)
            scaler_g.update()

            step_time = time.time() - t0
            global_step += 1

            # ---- Logging ----
            if global_step % cfg.log_interval == 0:
                print(f"Step {global_step}: g={g_loss.item():.4f} d={d_loss.item():.4f} "
                      f"feat={l_feature.item():.4f} asr={l_asr.item():.4f} "
                      f"l1={l_l1.item():.4f} stft={l_stft.item():.4f} "
                      f"commit={l_commit.item():.4f} γ={gamma:.6f} "
                      f"time={step_time:.2f}s")

                if cfg.use_wandb:
                    import wandb
                    wandb.log({
                        "loss/total_g": g_loss.item(), "loss/d": d_loss.item(),
                        "loss/feature": l_feature.item(), "loss/asr": l_asr.item(),
                        "loss/l1": l_l1.item(), "loss/stft": l_stft.item(),
                        "loss/commit": l_commit.item(), "gamma": gamma,
                        "step_time": step_time,
                    }, step=global_step)

            # ---- Save checkpoint ----
            if global_step % cfg.save_interval == 0:
                # Encoder weights only
                save_path = os.path.join(cfg.output_dir, f"encoder_step{global_step}.safetensors")
                save_file(dict(codec.encoder.state_dict()), save_path)

                # Full checkpoint for resume
                torch.save({
                    "encoder": codec.encoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "global_step": global_step,
                }, resume_path)

                print(f"Saved checkpoint at step {global_step}")

    # ---- Final save ----
    final_path = os.path.join(cfg.output_dir, "encoder_final.safetensors")
    save_file(dict(codec.encoder.state_dict()), final_path)

    swift_weights = {f"audio_tokenizer.{k}": v for k, v in codec.encoder.state_dict().items()}
    swift_path = os.path.join(cfg.output_dir, "encoder_for_swift.safetensors")
    save_file(swift_weights, swift_path)

    print(f"\nTraining complete!")
    print(f"  Encoder: {final_path}")
    print(f"  Swift-compatible: {swift_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Voxtral Codec Encoder")
    parser.add_argument("--tts_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="librispeech")
    parser.add_argument("--librispeech_split", type=str, default="train-clean-100")
    parser.add_argument("--output_dir", type=str, default="checkpoints/codec_encoder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--max_audio_seconds", type=float, default=10.0)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA only)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = TrainConfig(
        tts_model_path=args.tts_model_path,
        dataset=args.dataset,
        librispeech_split=args.librispeech_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype=args.dtype,
        whisper_model=args.whisper_model,
        use_wandb=args.use_wandb,
        max_audio_seconds=args.max_audio_seconds,
        compile=args.compile,
        num_workers=args.num_workers,
    )

    train(cfg)


if __name__ == "__main__":
    main()
