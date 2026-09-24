# EssayAI ML roadmap

Ideas for future work, in priority order:

1. Replace generated examples with a properly licensed, representative essay dataset and document its source and score rubric.
2. Add data validation, duplicate checks, and a split strategy that limits topic, prompt, or author leakage.
3. Report held-out metrics and error analysis, including score-level and subgroup results where permitted.
4. Improve the model's handling of essays longer than the encoder's 512-token limit.
5. Validate feedback suggestions with educators and provide rubric-specific guidance.
6. Add reproducible environment locking and a lightweight automated build workflow.
7. Consider pooling strategies, calibration, and model alternatives only after establishing a reliable baseline.
