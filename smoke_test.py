"""
smoke_test.py - Validates the entire pipeline with 10 samples before full training.
Run: python smoke_test.py
The first run downloads the DeBERTa tokenizer/model files; download size and time vary.
This checks the data/tokenizer/model-forward/metric path, not training quality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from src import config
from src.dataset  import load_dataframe, split_dataframe, get_tokenizer, EssayDataset
from src.model    import EssayScoringModel, denormalise_score
from src.evaluate import compute_qwk, compute_rmse, compute_accuracy
import numpy as np

print("=" * 55)
print("  SMOKE TEST — Validating full pipeline")
print("=" * 55)
print(f"  Device : {config.DEVICE}")
print(f"  Model  : {config.MODEL_NAME}")

# 1. Dataset
print("\n[1/5] Loading dataset...")
df = load_dataframe(config.TRAIN_CSV)
train_df, val_df = split_dataframe(df)

# 2. Tokenizer (downloads required model files on first run)
print("\n[2/5] Loading tokenizer (first run downloads model files)...")
tokenizer = get_tokenizer()

# 3. Dataset & DataLoader with tiny subset
print("\n[3/5] Creating DataLoader (10-sample subset)...")
tiny_train = train_df.head(10).reset_index(drop=True)
tiny_val   = val_df.head(10).reset_index(drop=True)
train_ds   = EssayDataset(tiny_train, tokenizer)
val_ds     = EssayDataset(tiny_val,   tokenizer)
train_dl   = DataLoader(train_ds, batch_size=2)
val_dl     = DataLoader(val_ds,   batch_size=2)

batch = next(iter(train_dl))
print(f"  input_ids shape    : {batch['input_ids'].shape}")
print(f"  attention_mask     : {batch['attention_mask'].shape}")
print(f"  labels             : {batch['labels']}")

# 4. Model forward pass
print("\n[4/5] Testing model forward pass...")
model = EssayScoringModel().to(config.DEVICE)
model.eval()
with torch.no_grad():
    ids   = batch["input_ids"].to(config.DEVICE)
    mask  = batch["attention_mask"].to(config.DEVICE)
    out   = model(ids, mask)
    print(f"  Raw logits (norm)  : {out.squeeze().cpu().numpy().round(4)}")
    scores = denormalise_score(out.squeeze().cpu().numpy())
    print(f"  Denorm scores      : {scores.round(2)}")

# 5. Metrics
print("\n[5/5] Checking metrics...")
preds  = np.array([3.2, 4.1, 2.8, 5.0, 3.5])
labels = np.array([3.0, 4.0, 3.0, 5.0, 4.0])
print(f"  RMSE     : {compute_rmse(preds, labels):.4f}")
print(f"  QWK      : {compute_qwk(preds, labels):.4f}")
print(f"  Accuracy : {compute_accuracy(preds, labels):.1f}%")

print("\n" + "=" * 55)
print("  ALL TESTS PASSED - Ready to train!")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e6:.1f} MB" if config.DEVICE == "cuda" else "")
print("=" * 55)
print("\n  Run training with:")
print("    python -m src.train")
