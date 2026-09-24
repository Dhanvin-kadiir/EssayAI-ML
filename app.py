"""
app.py - Flask web server for the Essay Scoring System.
Works in two modes:
  1. MODEL mode — a trained PyTorch + DeBERTa checkpoint is loaded
  2. DEMO mode  — no ML deps required, uses heuristic scoring for UI preview

Run: python app.py
"""

import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app = Flask(__name__)

# ── Try importing ML stack (optional for demo) ────────────────────────────────
ML_AVAILABLE = False
_model = _tokenizer = None

try:
    from src import config
    ML_AVAILABLE = True
except ImportError:
    pass

_model_loaded = False
_load_error   = None


def load_model_lazy():
    global _model, _tokenizer, _model_loaded, _load_error
    if _model_loaded:
        return True, None
    if _load_error:
        return False, _load_error
    if not ML_AVAILABLE:
        return False, "PyTorch not installed — running in demo mode"
    best = config.BEST_MODEL_PATH
    if not os.path.exists(best):
        return False, "No trained checkpoint found — running in demo mode"
    try:
        from src.dataset import get_tokenizer
        _tokenizer = get_tokenizer()
        from src.predict import load_model as lm
        _model = lm(best)
        _model_loaded = True
        return True, None
    except Exception as e:
        _load_error = str(e)
        return False, _load_error


# ── Demo scoring (heuristic, no ML needed) ────────────────────────────────────
def demo_score_essay(essay: str) -> dict:
    """
    Heuristic essay scorer used when PyTorch is not installed.
    Analyses: word count, lexical diversity, paragraph structure, sentence length.
    """
    words       = essay.split()
    sentences   = re.split(r'[.!?]+', essay)
    sentences   = [s.strip() for s in sentences if len(s.strip()) > 5]
    paragraphs  = [p.strip() for p in essay.split('\n') if p.strip()]

    word_count  = len(words)
    unique_r    = len(set(w.lower() for w in words)) / max(word_count, 1)
    avg_sent_len= sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    para_count  = len(paragraphs)

    # Score heuristic (1–6)
    score = 1.0
    score += min(word_count / 120, 1.5)         # Length (up to +1.5)
    score += unique_r * 1.2                      # Vocabulary diversity
    score += min(avg_sent_len / 25, 0.8)         # Sentence complexity
    score += min(para_count / 4, 0.5)            # Structure
    score  = min(max(score, 1.0), 6.0)

    # Feedback
    feedback = []
    if para_count == 1:
        feedback.append("⚠  Essay is a single block — break it into clear paragraphs (intro, body, conclusion).")
    elif para_count >= 4:
        feedback.append("✅ Good paragraph structure — clear organisation helps the reader follow your argument.")

    if avg_sent_len < 10:
        feedback.append("💡 Sentences are very short. Aim for 15–20 word sentences to show analytical depth.")
    elif avg_sent_len > 30:
        feedback.append("💡 Some sentences are very long. Break complex ideas into 2–3 focused sentences.")

    if unique_r < 0.45:
        feedback.append("⚠  Vocabulary is repetitive. Use synonyms and precise academic terms to strengthen your writing.")
    elif unique_r > 0.65:
        feedback.append("✅ Strong vocabulary diversity — your word choices demonstrate good command of language.")

    if word_count < 150:
        feedback.append("💡 Essay is quite short. Aim for at least 300 words to fully develop your argument.")
    elif word_count > 400:
        feedback.append("✅ Sufficient essay length — you've given your ideas room to develop.")

    if not feedback:
        feedback.append("💡 Consider adding specific examples, statistics, or quotes to strengthen your claims.")

    return {
        "score"     : round(score, 2),
        "confidence": "Uncalibrated",
        "feedback"  : feedback[:3],
        "demo_mode" : True,
    }


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status"      : "ok",
        "ml_available": ML_AVAILABLE,
        "model_loaded": _model_loaded,
        "checkpoint_available": bool(ML_AVAILABLE and os.path.exists(config.BEST_MODEL_PATH)),
        "demo_mode"   : not _model_loaded,
    })


@app.route("/api/score", methods=["POST"])
def score_essay():
    t0    = time.time()
    data  = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    essay = (data or {}).get("essay", "")
    if not isinstance(essay, str):
        return jsonify({"error": "Essay must be provided as text"}), 400
    essay = essay.strip()

    if not essay:
        return jsonify({"error": "No essay text provided"}), 400
    if len(essay) < 50:
        return jsonify({"error": "Essay too short (minimum 50 characters)",
                        "feedback": ["Please write at least 50 characters."]}), 400
    if len(essay) > 10000:
        return jsonify({"error": "Essay exceeds the 10,000 character limit"}), 400

    # Try full ML mode first
    ok, err = load_model_lazy()
    if ok:
        try:
            from src.predict import predict_essay
            result   = predict_essay(essay, _model, _tokenizer, verbose=False)
            elapsed  = int((time.time() - t0) * 1000)
            return jsonify({**result, "time_ms": elapsed, "demo_mode": False})
        except Exception as e:
            err = f"Model inference failed: {e}"

    # Demo mode (heuristic)
    result  = demo_score_essay(essay)
    elapsed = int((time.time() - t0) * 1000)
    return jsonify({**result, "time_ms": elapsed, "demo_reason": err})


@app.route("/api/demo")
def demo_essay():
    sample = """Social media has fundamentally transformed the way teenagers communicate, learn, and perceive the world around them. While platforms like Instagram and TikTok offer unprecedented opportunities for self-expression and global connection, they also introduce significant challenges to adolescent mental health and academic performance.

On one hand, social media democratizes information access. Students can now follow scientists, historians, and thought leaders directly, exposing themselves to diverse perspectives that traditional education might overlook. A teenager in rural India can learn from MIT professors; a student in Brazil can collaborate on projects with peers in Germany. This global interconnectedness fosters cultural empathy and broadens intellectual horizons in ways previous generations could not have imagined.

However, the algorithm-driven nature of these platforms creates an echo chamber effect. Research by the American Psychological Association found that teenagers who spend more than three hours daily on social media report significantly higher rates of anxiety and depression compared to their peers. The constant comparison to curated, filtered representations of others lives distorts self-perception and sets unrealistic standards.

Furthermore, the dopamine feedback loops engineered into these platforms are designed to maximize engagement, not wellbeing. Features like infinite scroll and notification badges exploit psychological vulnerabilities particularly pronounced during adolescence.

In conclusion, social media impact on teenagers is neither wholly positive nor negative but depends critically on how and how much it is used. Educational institutions and parents must work together to establish healthy digital habits and advocate for platform designs that genuinely prioritize user wellbeing over engagement metrics."""
    return jsonify({"essay": sample.strip()})


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/static/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(WEB_DIR, "static", "css"), filename)


@app.route("/static/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(WEB_DIR, "static", "js"), filename)


@app.route("/static/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(WEB_DIR, "static", "assets"), filename)


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


if __name__ == "__main__":
    print("=" * 55)
    print("  EssayAI ML — Essay Score Prediction")
    print(f"  ML Stack: {'Available' if ML_AVAILABLE else 'Not installed (demo mode)'}")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=False)
