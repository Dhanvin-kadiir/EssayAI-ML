"""
dataset.py - Custom PyTorch Dataset for the AES 2.0 Kaggle competition data.

Key decisions:
  - Handles variable-length essays via tokenizer truncation + padding
  - Returns attention masks so the model ignores padding tokens
  - Normalises scores to [0, 1] for stable regression training,
    then we reverse-normalise during evaluation
  - Rejects essays shorter than MIN_CHARS to avoid noise
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src import config

# ─────────────────────────────────────────────────────────────────────────────
MIN_CHARS = 50   # Discard essays shorter than 50 characters (safety net)
# ─────────────────────────────────────────────────────────────────────────────


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Read CSV, validate columns, clean text, drop short essays."""
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    required = {"full_text", "score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Basic cleaning
    df["full_text"] = df["full_text"].astype(str).str.strip()
    df = df[df["full_text"].str.len() >= MIN_CHARS].reset_index(drop=True)

    # Score range guard
    df = df[df["score"].between(config.SCORE_MIN, config.SCORE_MAX)].reset_index(drop=True)

    print(f"[Dataset] Loaded {len(df)} valid essays | "
          f"Score dist:\n{df['score'].value_counts().sort_index().to_dict()}")
    return df


def split_dataframe(df: pd.DataFrame, val_size: float = config.VAL_SPLIT,
                    seed: int = config.SEED):
    """Stratified train/val split to keep score distribution balanced."""
    train_df, val_df = train_test_split(
        df, test_size=val_size,
        stratify=df["score"],
        random_state=seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"[Dataset] Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df


class EssayDataset(Dataset):
    """
    PyTorch Dataset that tokenizes essays on-the-fly.

    Args:
        df         : DataFrame with 'full_text' and (optionally) 'score'
        tokenizer  : HuggingFace tokenizer (DeBERTa)
        max_length : Maximum token count (default 512)
        is_test    : If True, skip score loading (for inference)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = config.MAX_LENGTH,
                 is_test: bool = False):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        text  = self.df.loc[idx, "full_text"]
        enc   = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )

        item = {
            "input_ids"      : enc["input_ids"].squeeze(0),          # (max_len,)
            "attention_mask" : enc["attention_mask"].squeeze(0),      # (max_len,)
        }

        # Token type IDs are optional (DeBERTa doesn't use them)
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if not self.is_test:
            raw_score = float(self.df.loc[idx, "score"])
            # Normalise to [0, 1] → smoother gradient landscape for regression
            norm_score = (raw_score - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN)
            item["labels"] = torch.tensor(norm_score, dtype=torch.float32)

        return item


def get_tokenizer():
    """Load DeBERTa tokenizer from HuggingFace Hub (cached after first download)."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print(f"[Dataset] Tokenizer loaded: {config.MODEL_NAME}")
    return tokenizer


def get_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame,
                    tokenizer, batch_size: int = config.BATCH_SIZE):
    """Create DataLoader objects for train and validation splits."""
    train_ds = EssayDataset(train_df, tokenizer)
    val_ds   = EssayDataset(val_df,   tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2,   # Larger batch: no gradients needed
        shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader
