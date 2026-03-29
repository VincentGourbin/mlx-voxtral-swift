"""Training script for Voxtral Codec Encoder.

Trains the encoder while keeping the decoder and VQ codebook frozen.
Uses all losses from the paper (Eq. 3):
  Loss = α·L_feature + β·L_ASR + γ·L_L1 + γ·L_STFT + δ·L_commit

Usage:
    python train.py --tts_model_path /path/to/Voxtral-4B-TTS-2603 --dataset /path/to/librispeech
    python train.py --tts_model_path /path/to/Voxtral-4B-TTS-2603 --dataset librispeech --librispeech_split train-clean-100
"""

import argparse
import os
import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file
from tqdm import tqdm

from config import CodecConfig, DiscriminatorConfig, TrainConfig
from model import VoxtralCodec
from discriminator import MultiResolutionSTFTDiscriminator
from losses import (l1_loss, multi_scale_stft_loss, feature_matching_loss,
                    discriminator_hinge_loss, generator_hinge_loss)
from whisper_distillation import WhisperDistillationLoss
from dataset import create_dataloader


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # ---- Model ----
    codec_cfg = CodecConfig()
    codec = VoxtralCodec(codec_cfg).to(device)

    # Load frozen decoder + quantizer from TTS checkpoint
    assert cfg.tts_model_path, "Must provide --tts_model_path"
    ckpt_path = os.path.join(cfg.tts_model_path, "consolidated.safetensors")
    if not os.path.exists(ckpt_path):
        # Try mlx-community sharded format
        ckpt_files = sorted(Path(cfg.tts_model_path).glob("model-*.safetensors"))
        if ckpt_files:
            # Merge sharded weights temporarily
            from safetensors import safe_open
            merged = {}
            for f in ckpt_files:
                with safe_open(str(f), framework="pt") as sf:
                    for k in sf.keys():
                        if k.startswith("audio_tokenizer."):
                            merged[k] = sf.get_tensor(k)
            # Save merged for loading
            ckpt_path = "/tmp/_merged_decoder.safetensors"
            save_file(merged, ckpt_path)
            print(f"Merged {len(merged)} audio_tokenizer weights from {len(ckpt_files)} shards")

    codec.load_decoder_weights(ckpt_path)
    codec.freeze_decoder()

    encoder_params = sum(p.numel() for p in codec.encoder.parameters())
    total_params = sum(p.numel() for p in codec.parameters() if p.requires_grad)
    print(f"Encoder params: {encoder_params / 1e6:.1f}M")
    print(f"Trainable params: {total_params / 1e6:.1f}M")

    # ---- Discriminator ----
    disc_cfg = DiscriminatorConfig()
    discriminator = MultiResolutionSTFTDiscriminator(disc_cfg).to(device)
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator params: {disc_params / 1e6:.1f}M")

    # ---- Whisper Distillation ----
    # Everything on same device to avoid cross-device deadlocks
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

    # ---- Training Loop ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    global_step = 0
    codec.train()
    discriminator.train()

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg.max_steps} steps")
    print(f"{'='*60}\n")

    while global_step < cfg.max_steps:
        for batch in dataloader:
            if global_step >= cfg.max_steps:
                break

            audio = batch.to(device)  # [B, 1, T]
            t0 = time.time()

            # ---- Reconstruction decay ----
            gamma = cfg.gamma_base ** global_step

            # ---- Forward ----
            reconstructed, sem_codes, ac_codes, commit_loss, semantic_embs = codec(
                audio, vq_prob=cfg.vq_prob
            )

            audio_flat = audio.squeeze(1)  # [B, T]
            recon_flat = reconstructed.squeeze(1)  # [B, T']

            # Align lengths
            min_len = min(audio_flat.shape[-1], recon_flat.shape[-1])
            audio_flat = audio_flat[..., :min_len]
            recon_flat = recon_flat[..., :min_len]

            # ---- Discriminator step ----
            with torch.no_grad():
                recon_detached = recon_flat.detach()
            real_logits, real_features = discriminator(audio_flat)
            fake_logits, fake_features = discriminator(recon_detached)

            d_loss = discriminator_hinge_loss(real_logits, fake_logits)

            optimizer_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.gradient_clip)
            optimizer_d.step()

            # ---- Generator step ----
            fake_logits_g, fake_features_g = discriminator(recon_flat)
            _, real_features_g = discriminator(audio_flat.detach())

            # Feature matching loss (α=1.0)
            l_feature = feature_matching_loss(real_features_g, fake_features_g)

            # ASR distillation loss (β=1.0)
            l_asr = whisper_loss_fn(semantic_embs, audio, codec_cfg.frame_rate)

            # Reconstruction losses (γ decaying)
            l_l1 = l1_loss(reconstructed, audio)
            l_stft = multi_scale_stft_loss(reconstructed, audio)

            # VQ commitment loss (δ=0.1)
            l_commit = commit_loss

            # Total generator loss
            g_loss = (
                cfg.alpha * l_feature +
                cfg.beta * l_asr +
                gamma * (l_l1 + l_stft) +
                cfg.delta * l_commit
            )

            optimizer_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(codec.encoder.parameters()) + list(whisper_loss_fn.projection.parameters()),
                cfg.gradient_clip
            )
            optimizer_g.step()

            step_time = time.time() - t0
            global_step += 1

            # ---- Logging ----
            if global_step % cfg.log_interval == 0:
                log_dict = {
                    "loss/total_g": g_loss.item(),
                    "loss/d": d_loss.item(),
                    "loss/feature": l_feature.item(),
                    "loss/asr": l_asr.item(),
                    "loss/l1": l_l1.item(),
                    "loss/stft": l_stft.item(),
                    "loss/commit": l_commit.item(),
                    "gamma": gamma,
                    "step_time": step_time,
                }
                print(f"Step {global_step}: g={g_loss.item():.4f} d={d_loss.item():.4f} "
                      f"feat={l_feature.item():.4f} asr={l_asr.item():.4f} "
                      f"l1={l_l1.item():.4f} stft={l_stft.item():.4f} "
                      f"commit={l_commit.item():.4f} γ={gamma:.6f} "
                      f"time={step_time:.2f}s")

                if cfg.use_wandb:
                    import wandb
                    wandb.log(log_dict, step=global_step)

            # ---- Save ----
            if global_step % cfg.save_interval == 0:
                save_path = os.path.join(cfg.output_dir, f"encoder_step{global_step}.safetensors")
                encoder_state = {
                    k: v for k, v in codec.encoder.state_dict().items()
                }
                save_file(encoder_state, save_path)
                print(f"Saved encoder checkpoint: {save_path}")

    # ---- Final save ----
    final_path = os.path.join(cfg.output_dir, "encoder_final.safetensors")
    save_file(dict(codec.encoder.state_dict()), final_path)
    print(f"\nTraining complete! Final encoder saved to: {final_path}")

    # Also save in a format ready for our Swift model
    # The Swift model expects keys like: audio_tokenizer.input_proj.*, audio_tokenizer.encoder_blocks.*
    swift_weights = {}
    for k, v in codec.encoder.state_dict().items():
        swift_weights[f"audio_tokenizer.{k}"] = v
    swift_path = os.path.join(cfg.output_dir, "encoder_for_swift.safetensors")
    save_file(swift_weights, swift_path)
    print(f"Swift-compatible encoder saved to: {swift_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Voxtral Codec Encoder")
    parser.add_argument("--tts_model_path", type=str, required=True,
                        help="Path to Voxtral-4B-TTS model directory")
    parser.add_argument("--dataset", type=str, default="librispeech",
                        help="Path to audio directory or 'librispeech'")
    parser.add_argument("--librispeech_split", type=str, default="train-clean-100")
    parser.add_argument("--output_dir", type=str, default="checkpoints/codec_encoder")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--max_audio_seconds", type=float, default=10.0)
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
        whisper_model=args.whisper_model,
        use_wandb=args.use_wandb,
        max_audio_seconds=args.max_audio_seconds,
    )

    train(cfg)


if __name__ == "__main__":
    main()
