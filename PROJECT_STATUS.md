# EssayAI ML project status

## Implemented

- Flask endpoints for health, sample essay, and essay scoring.
- A DeBERTa-v3 regression training and inference path.
- RMSE, QWK, and ±0.5 accuracy calculations.
- A browser interface and a clearly identified heuristic fallback.
- A generator for balanced synthetic sample essays.

## Repository state at handoff

- Local sample data: `data/train.csv`, 2,100 synthetic rows (350 per score); regenerate with `python data/generate_sample_data.py`. The CSV is excluded from Git.
- Trained checkpoint: absent.
- Evaluation report: absent.
- Kaggle data: absent.
- EDA notebook: absent; the `notebooks` directory is currently empty.
- License: not specified.
- Git remote: not configured in the local project directory. Create the GitHub repository and add its remote before pushing.

## Before presenting model performance

1. Replace the synthetic sample CSV with a permitted real dataset.
2. Use a split strategy that prevents related or duplicated essays from leaking across train and validation sets.
3. Train and save a checkpoint.
4. Evaluate on a held-out set and publish the data source, split, and actual metrics.
5. Review score calibration, subgroup behavior, long-essay truncation, and feedback quality.
