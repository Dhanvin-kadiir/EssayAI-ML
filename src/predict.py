"""
predict.py - Inference + Feedback Generation for the Essay Scoring System.

Feedback is generated with simple text-structure heuristics; it is not model-generated.
"""

import re
import os
import torch
import numpy as np

from src import config
from src.model   import EssayScoringModel, denormalise_score
from src.dataset import get_tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Feedback templates keyed by issue type
FEEDBACK_TEMPLATES = {
    "low_coherence"   : "⚠  Paragraph {n} could be structured more clearly — try using a topic sentence.",
    "weak_evidence"   : "💡 Paragraph {n} makes a claim but lacks specific evidence. Add statistics, examples, or quotes.",
    "vague_language"  : "🔍 Paragraph {n} uses vague language. Replace phrases like 'things' or 'stuff' with precise terms.",
    "good_structure"  : "✅ Paragraph {n} shows strong logical structure — great work!",
    "strong_intro"    : "✅ Your introduction sets up the argument well — compelling hook!",
    "weak_conclusion" : "💡 Your conclusion could be stronger — try restating your thesis with new insight.",
}

# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str = config.BEST_MODEL_PATH):
    """Load a trained EssayScoringModel from a checkpoint file."""
    model = EssayScoringModel()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Please run train.py first or provide a valid path."
        )
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE)
    model.eval()
    print(f"[Predict] Model loaded from epoch {ckpt.get('epoch', '?')}")
    print(f"[Predict] Checkpoint metrics: {ckpt.get('metrics', {})}")
    return model


def split_into_sentences(text: str) -> list:
    """Split essay text into individual sentences (simple regex-based)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def group_into_paragraphs(text: str) -> list:
    """Split essay into paragraphs by blank lines, with a sentence fallback."""
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paras) <= 1:
        # Fall back: group every ~3 sentences into a pseudo-paragraph
        sentences = split_into_sentences(text)
        paras = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return paras


def generate_feedback(essay: str,
                      top_k: int = config.FEEDBACK_TOP_K_SENTENCES) -> list:
    """
    Return heuristic feedback strings based on basic text structure.

    Strategy (heuristic-based, no external knowledge base needed):
      - Score them by vocabulary richness and sentence length (proxy for depth)
      - Identify structural issues: missing intro hook, weak conclusion, vague claims
    """
    paragraphs = group_into_paragraphs(essay)
    feedback   = []

    for i, para in enumerate(paragraphs):
        n = i + 1
        words      = para.lower().split()
        unique_r   = len(set(words)) / max(len(words), 1)   # Lexical diversity
        avg_len    = np.mean([len(s.split()) for s in split_into_sentences(para)] or [0])

        # Heuristic rules
        if i == 0 and avg_len > 12 and unique_r > 0.6:
            feedback.append(FEEDBACK_TEMPLATES["strong_intro"])
        elif i == len(paragraphs) - 1 and avg_len < 10:
            feedback.append(FEEDBACK_TEMPLATES["weak_conclusion"].format(n=n))
        elif unique_r < 0.45:
            feedback.append(FEEDBACK_TEMPLATES["vague_language"].format(n=n))
        elif avg_len < 8:
            feedback.append(FEEDBACK_TEMPLATES["low_coherence"].format(n=n))
        elif avg_len > 18 and unique_r > 0.65:
            feedback.append(FEEDBACK_TEMPLATES["good_structure"].format(n=n))
        else:
            feedback.append(FEEDBACK_TEMPLATES["weak_evidence"].format(n=n))

    # Return up to top_k unique feedback items
    seen = set()
    unique_fb = []
    for fb in feedback:
        if fb not in seen:
            seen.add(fb)
            unique_fb.append(fb)
        if len(unique_fb) == top_k:
            break
    return unique_fb


@torch.no_grad()
def predict_essay(essay: str, model, tokenizer,
                  verbose: bool = True) -> dict:
    """
    End-to-end prediction pipeline for a single essay.

    Returns:
        {
          "score"       : float (1–6),
          "confidence"  : str  ("Uncalibrated"),
          "feedback"    : list[str],
        }
    """
    # ── Validate input ────────────────────────────────────────────────────────
    if not essay or len(essay.strip()) < 50:
        return {
            "score"     : None,
            "confidence": "Uncalibrated",
            "feedback"  : ["⚠ Essay is too short (< 50 characters). Please provide a complete essay."],
            "error"     : "SHORT_ESSAY",
        }

    # ── Tokenise & predict ────────────────────────────────────────────────────
    enc = tokenizer(
        essay,
        max_length     = config.MAX_LENGTH,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )
    input_ids  = enc["input_ids"].to(config.DEVICE)
    attn_mask  = enc["attention_mask"].to(config.DEVICE)

    norm_logit = model(input_ids, attn_mask).item()   # (0, 1)
    raw_score  = denormalise_score(norm_logit)        # (1, 6)

    # ── Feedback ──────────────────────────────────────────────────────────────
    feedback = generate_feedback(essay)

    result = {
        "score"      : round(raw_score, 2),
        "confidence" : "Uncalibrated",
        "feedback"   : feedback,
    }

    if verbose:
        _print_result(essay, result)

    return result


def _print_result(essay: str, result: dict):
    """Pretty-print the prediction result to console."""
    preview = essay[:120].replace('\n', ' ')
    print(f"\nProcessing essay:\n\"{preview}...\"\n")
    print("=" * 50)
    print("  RESULTS")
    print("=" * 50)
    if result.get("error"):
        print(f"  ⚠ Error: {result['feedback'][0]}")
        return

    score = result["score"]
    print(f"  Predicted Score : {score:.1f} / 6")
    print(f"  Confidence      : {result['confidence']}")
    print(f"\n  FEEDBACK:")
    for fb in result["feedback"]:
        print(f"    {fb}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    model     = load_model()
    tokenizer = get_tokenizer()

    if len(sys.argv) > 1:
        # Predict on a file passed as argument
        essay_file = sys.argv[1]
        with open(essay_file, "r", encoding="utf-8") as f:
            essay = f.read()
    else:
        # Interactive mode: read from stdin
        print("\n[Predict] Paste your essay below. Press Enter twice + Ctrl-D to submit.\n")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        essay = "\n".join(lines)

    predict_essay(essay, model, tokenizer)
