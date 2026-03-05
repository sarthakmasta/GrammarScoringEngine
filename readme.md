# Grammar Scoring Engine

A hybrid acoustic + semantic regression pipeline to score spoken English grammar from audio samples.

## Approach

Predicts Grammar Scores (1–5) from 45–60 second audio files using a three-stage pipeline:

1. **Transcription**: Whisper large-v3 via faster-whisper
2. **Feature Extraction**: Whisper encoder embeddings (1280-dim) + handcrafted signal features (MFCC, ZCR, RMS) + DeBERTa-v3-base text embeddings (768-dim)
3. **Modeling**: XGBoost (Optuna-tuned) + LightGBM/Ridge stacking ensemble with RidgeCV

## Results

| Metric | Score |
|--------|-------|
| Stacker OOF RMSE | 0.5438 |
|XGBoost OOF RMSE | 0.5559 |
| Kaggle RMSE | 0.47 |
| Leaderboard Position | 4 |

## Pipeline
```
Audio Files
    │
    ├── faster-whisper large-v3 ──► Transcripts
    │                                    │
    ├── Whisper Encoder ──► Acoustic     │
    │   Embeddings (1280-dim)            │
    │                                    ▼
    ├── Librosa ──► Handcrafted     DeBERTa-v3-base
    │   Features (30-dim)          Text Embeddings
    │                              (768-dim)
    │
    └── Concatenate All Features (2078-dim)
            │
            ├── XGBoost (Optuna-tuned, 5-fold CV)
            │
            └── LightGBM + Ridge Stacking Ensemble
                        │
                        ▼
                Final Predictions (averaged, clipped 1–5)
```