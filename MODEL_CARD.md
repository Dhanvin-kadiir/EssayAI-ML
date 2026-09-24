# EssayAI ML model card

## Overview

EssayAI ML is an educational prototype for experimenting with automated essay score regression. Its intended score range is 1–6. The model code uses DeBERTa-v3-base with a regression head, but this checkout contains no trained checkpoint.

## Intended use

Use the code to learn about text preprocessing, transformer fine-tuning, regression metrics, and serving a local model. The included sample dataset is synthetic and suitable only for exercising the pipeline.

Do not use this project to make admissions, hiring, disciplinary, placement, or other consequential decisions. Do not present predictions as reliable human-equivalent grades.

## Data

The bundled CSV has 2,100 template-generated essays, 350 per integer score from 1 through 6. The synthetic generator deliberately connects writing templates to score labels. A model trained on these examples can learn template cues rather than meaningful writing quality, so its metrics do not transfer to real essays.

The Kaggle AES 2.0 competition dataset is mentioned as an optional replacement, but is not included. Users must obtain it independently and comply with its terms.

## Model behavior and limits

- The encoder input is capped at 512 tokens by default; longer essays are truncated.
- The output is a bounded regression value mapped to 1–6.
- Confidence is reported as uncalibrated; the project does not estimate predictive uncertainty.
- Feedback is based on simple paragraph, sentence-length, and vocabulary-diversity heuristics. It does not verify facts, evaluate evidence quality, or understand a rubric.
- No human evaluation, subgroup analysis, or validated real-data metrics are provided in this repository.
- Without `outputs/models/best_model.pt`, the web service uses a heuristic demo scorer and marks its response as demo mode.

## Evaluation

The code can calculate RMSE, quadratic weighted kappa (QWK), and accuracy within ±0.5 points. No trained evaluation result is supplied. Any future result should disclose the dataset, split strategy, preprocessing, checkpoint, and whether the essays are synthetic or real.

## Ethical considerations

Automated scoring can reproduce dataset and rubric bias, and surface-level signals such as length or vocabulary can be mistaken for quality. Keep a human reviewer in the loop and provide a way to contest results if this prototype is adapted for real learners.
