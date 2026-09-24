"""
model.py - DeBERTa-v3 based regression model for Automated Essay Scoring.

Architecture:
  1. DeBERTa-v3-base backbone   → contextual embeddings
  2. Multi-Sample Dropout (MSD) → regularisation trick from winning Kaggle solutions
     - Run [CLS] embedding through N different dropout rates
     - Average logits → reduces variance, improves generalisation
  3. Linear regression head     → single score in [0, 1] (denormalised later)

Why MSD?  It implicitly ensembles the model with itself under different noise
levels, similar to MC Dropout at inference time, without inference overhead.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from src import config


class MultiSampleDropout(nn.Module):
    """
    Applies multiple dropout rates and averages the resulting logits.
    Each dropout mask is independently sampled → approximates an ensemble.

    Args:
        in_features  : Size of input embedding
        out_features : Size of output (1 for regression)
        dropout_rates: List of dropout probabilities to average over
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout_rates: list = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = config.DROPOUT_RATES

        self.dropouts   = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.linear     = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) → output: (batch, out_features)"""
        x = x.to(self.linear.weight.dtype)
        # Average logits across all dropout samples
        logits = [self.linear(drop(x)) for drop in self.dropouts]
        return torch.stack(logits, dim=0).mean(dim=0)   # (batch, out_features)


class EssayScoringModel(nn.Module):
    """
    Full DeBERTa-based regression model.

    Forward pass outputs:
      - logits : raw regression score in [0, 1] (after sigmoid)

    Usage:
      model = EssayScoringModel()
      out   = model(input_ids, attention_mask)   # shape: (batch, 1)
    """

    def __init__(self, model_name: str = config.MODEL_NAME,
                 dropout_rates: list = None):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size   # 768 for base models

        # ── Regression Head (Multi-Sample Dropout) ────────────────────────────
        dropout_rates = dropout_rates or config.DROPOUT_RATES
        self.head = MultiSampleDropout(hidden_size, 1, dropout_rates)

        # ── Activation ────────────────────────────────────────────────────────
        # Sigmoid bounds output to (0, 1) matching our normalised score labels
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids      : (batch, seq_len) tokenised essay
            attention_mask : (batch, seq_len) 1=real token, 0=padding
            token_type_ids : Not used by DeBERTa but kept for API compatibility

        Returns:
            logits : (batch, 1) normalised score in (0, 1)
        """
        outputs = self.encoder(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # [CLS] token embedding → sentence-level representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        logits = self.sigmoid(self.head(cls_embedding))     # (batch, 1)
        return logits

    def get_embeddings(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns per-token hidden states for feedback generation.
        Shape: (batch, seq_len, hidden_size)
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )
        return outputs.last_hidden_state


def denormalise_score(norm_score: float) -> float:
    """Convert model output [0,1] → original score range [SCORE_MIN, SCORE_MAX]."""
    return norm_score * (config.SCORE_MAX - config.SCORE_MIN) + config.SCORE_MIN


def normalise_score(raw_score: float) -> float:
    """Convert raw score [SCORE_MIN, SCORE_MAX] → [0,1] for model input."""
    return (raw_score - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN)
