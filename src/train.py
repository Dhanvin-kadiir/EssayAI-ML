"""
train.py - Training loop with checkpointing, LR scheduling, and gradient accumulation.

Key features:
  - Mixed-precision training (fp16) via torch.cuda.amp
  - Gradient accumulation to simulate large batch sizes on low-VRAM GPUs
  - Cosine LR schedule with warmup (standard for transformer fine-tuning)
  - Skips a batch and clears gradients on CUDA OOM errors
  - Checkpoint saving on best validation QWK (the competition metric)
  - JSON log written after every epoch
"""

import os
import json
import math
import random
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from src import config
from src.dataset  import load_dataframe, split_dataframe, get_tokenizer, get_dataloaders
from src.model    import EssayScoringModel
from src.evaluate import evaluate_epoch


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    """Save model weights + metadata to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics"    : metrics,
    }, path)
    print(f"  [SAVED] Checkpoint -> {path}")


def log_metrics(log_path: str, entry: dict):
    """Append one epoch's metrics to the JSON log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    history = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)


def run_training(batch_size: int = config.BATCH_SIZE):
    """
    Main training function.

    Skips an out-of-memory batch after clearing any partial accumulation window.
    """
    print("=" * 60)
    print("  EssayAI ML — Training")
    print("=" * 60)
    print(f"  Device      : {config.DEVICE}")
    print(f"  Model       : {config.MODEL_NAME}")
    print(f"  Epochs      : {config.EPOCHS}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Max length  : {config.MAX_LENGTH}")
    print("=" * 60)

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    df              = load_dataframe(config.TRAIN_CSV)
    train_df, val_df = split_dataframe(df)
    tokenizer       = get_tokenizer()
    train_loader, val_loader = get_dataloaders(train_df, val_df, tokenizer, batch_size)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = EssayScoringModel().to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Total parameters: {total_params:,}")

    # ── 3. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.GRAD_ACCUM)
    total_steps  = updates_per_epoch * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler('cuda', enabled=config.FP16)
    loss_fn      = nn.MSELoss()

    best_qwk     = -1.0
    # ── 4. Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        processed_batches = 0
        t0          = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            try:
                input_ids   = batch["input_ids"].to(config.DEVICE)
                attn_mask   = batch["attention_mask"].to(config.DEVICE)
                labels      = batch["labels"].to(config.DEVICE).unsqueeze(1)  # (B,1)

                with autocast('cuda', enabled=config.FP16):
                    logits = model(input_ids, attn_mask)
                    window_start = (step // config.GRAD_ACCUM) * config.GRAD_ACCUM
                    window_size = min(config.GRAD_ACCUM, len(train_loader) - window_start)
                    batch_loss = loss_fn(logits, labels)
                    loss = batch_loss / window_size

                scaler.scale(loss).backward()
                epoch_loss += batch_loss.item()
                processed_batches += 1

                # Accumulate gradients
                if (step + 1) % config.GRAD_ACCUM == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                          f"| Loss: {batch_loss.item():.4f} "
                          f"| Elapsed: {elapsed:.1f}s")

            except torch.cuda.OutOfMemoryError:
                print("  ⚠ CUDA OOM — clearing cache and skipping step")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

        avg_train_loss = epoch_loss / max(processed_batches, 1)

        # ── 5. Validation ─────────────────────────────────────────────────────
        val_metrics = evaluate_epoch(model, val_loader)

        elapsed = time.time() - t0
        print(f"\n{'-'*55}")
        print(f"  Epoch {epoch}/{config.EPOCHS} Complete  [{elapsed:.1f}s]")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val RMSE   : {val_metrics['rmse']:.4f}")
        print(f"  Val QWK    : {val_metrics['qwk']:.4f}")
        print(f"  Val Acc    : {val_metrics['accuracy']:.2f}%")
        print(f"{'-'*55}\n")

        # ── 6. Checkpoint on improvement ──────────────────────────────────────
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            save_checkpoint(model, optimizer, epoch, val_metrics, config.BEST_MODEL_PATH)

        # ── 7. Log ────────────────────────────────────────────────────────────
        entry = {
            "epoch"      : epoch,
            "train_loss" : round(avg_train_loss, 4),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        }
        log_metrics(config.LOG_PATH, entry)

    print(f"\n✅ Training complete. Best QWK = {best_qwk:.4f}")
    print(f"   Model saved to: {config.BEST_MODEL_PATH}")
    return best_qwk


if __name__ == "__main__":
    run_training(batch_size=config.BATCH_SIZE)
