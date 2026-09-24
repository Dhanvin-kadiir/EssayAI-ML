"""
config.py - Central configuration for the Essay Scoring System
All hyperparameters, paths, and settings live here for easy tuning.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
MODELS_DIR      = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR     = os.path.join(OUTPUTS_DIR, "results")

TRAIN_CSV       = os.path.join(DATA_DIR, "train.csv")
TEST_CSV        = os.path.join(DATA_DIR, "test.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pt")
LOG_PATH        = os.path.join(RESULTS_DIR, "training_log.json")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "microsoft/deberta-v3-base"
MAX_LENGTH      = 512                            # Token limit per essay
NUM_CLASSES     = 1                              # Regression: single score output
DROPOUT_RATES   = [0.1, 0.2, 0.3, 0.4, 0.5]    # Multi-sample dropout (model averaging)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
SEED            = 42
EPOCHS          = 3             # Starting point; tune for dataset and hardware
BATCH_SIZE      = 4             # Adjust for available accelerator memory
LEARNING_RATE   = 2e-5          # Standard for transformer fine-tuning
WEIGHT_DECAY    = 0.01          # AdamW regularization
WARMUP_RATIO    = 0.06          # 6% of steps for LR warm-up
VAL_SPLIT       = 0.15          # 15% for validation
GRAD_ACCUM      = 4             # Effective batch = 4 * 4 = 16
MAX_GRAD_NORM   = 1.0           # Gradient clipping

# Expected score range for the configured essay scoring dataset
SCORE_MIN       = 1
SCORE_MAX       = 6

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
FP16            = torch.cuda.is_available()      # Mixed-precision on GPU only

# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────
FEEDBACK_TOP_K_SENTENCES       = 3    # Maximum heuristic suggestions returned

print(f"[Config] Device: {DEVICE} | FP16: {FP16} | Model: {MODEL_NAME}")
