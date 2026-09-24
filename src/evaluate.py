"""
evaluate.py - Evaluation module with RMSE, Accuracy (±0.5), and Quadratic Weighted Kappa.

QWK (Quadratic Weighted Kappa) is the official metric for essay scoring competitions.
It penalises large disagreements more than small ones — perfectly aligned with human
grading rubrics where being 3 points off is far worse than being 0.5 off.
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from src import config
from src.model import denormalise_score


# ─────────────────────────────────────────────────────────────────────────────
def compute_qwk(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Quadratic Weighted Kappa between predictions and ground-truth scores.

    Both arrays must be in the original score range (1–6), ROUNDED to integers,
    because QWK is defined for ordinal (integer) ratings.
    """
    preds_rounded  = np.clip(np.round(preds),  config.SCORE_MIN, config.SCORE_MAX).astype(int)
    labels_rounded = np.clip(np.round(labels), config.SCORE_MIN, config.SCORE_MAX).astype(int)
    return cohen_kappa_score(labels_rounded, preds_rounded, weights="quadratic")


def compute_rmse(preds: np.ndarray, labels: np.ndarray) -> float:
    """Root Mean Squared Error — measures average distance in score units."""
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def compute_accuracy(preds: np.ndarray, labels: np.ndarray,
                     tolerance: float = 0.5) -> float:
    """
    Percentage of predictions within `tolerance` of the true score.
    (±0.5 is a reasonable human-grader-agreement threshold)
    """
    correct = np.abs(preds - labels) <= tolerance
    return float(correct.mean() * 100)


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_epoch(model, dataloader) -> dict:
    """
    Run one full pass over a DataLoader and return all three metrics.

    Returns:
        {
          "rmse"    : float,
          "qwk"     : float,
          "accuracy": float   (% within ±0.5)
        }
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for batch in dataloader:
        input_ids  = batch["input_ids"].to(config.DEVICE)
        attn_mask  = batch["attention_mask"].to(config.DEVICE)
        labels     = batch["labels"].numpy()   # normalised [0,1]

        logits = model(input_ids, attn_mask).cpu().numpy().squeeze()  # (B,)

        # Denormalise both preds and labels → original score range
        preds_raw  = denormalise_score(logits)
        labels_raw = denormalise_score(labels)

        all_preds.append(preds_raw)
        all_labels.append(labels_raw)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "rmse"    : compute_rmse(all_preds, all_labels),
        "qwk"     : compute_qwk(all_preds, all_labels),
        "accuracy": compute_accuracy(all_preds, all_labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_and_report(model, dataloader, save_path: str = None) -> dict:
    """
    Evaluate and optionally save a JSON report to disk.
    """
    metrics = evaluate_epoch(model, dataloader)

    report = {
        "rmse"              : round(metrics["rmse"],     4),
        "qwk"               : round(metrics["qwk"],      4),
        "accuracy_pm0.5"    : round(metrics["accuracy"], 2),
    }

    print("\n" + "=" * 45)
    print("  EVALUATION RESULTS")
    print("=" * 45)
    print(f"  RMSE           : {report['rmse']}")
    print(f"  QWK            : {report['qwk']}")
    print(f"  Accuracy (±0.5): {report['accuracy_pm0.5']}%")
    print("=" * 45 + "\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved → {save_path}")

    return report
