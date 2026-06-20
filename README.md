# emotion-helper

Emotion assistant for short text messages.

## Project goal
This project aims to build a machine learning model that automatically detects the emotional tone of short text messages, primarily to help autistic users who find emotion recognition difficult.

## Task
- Classify short texts (messenger-style messages) into emotion categories.
- Output a predicted emotion label and probabilities for all supported emotions.

## Input / output format
Input JSON:
```json
{
  "text": "Пу-пу-пу"
}
```
Output JSON:
```json
{
  "label": "joy",
  "probs": {
    "joy": 0.82,
    "sadness": 0.05,
    "anger": 0.03,
    "neutral": 0.10
  }
}
```

## Metrics
Primary metrics:
- Accuracy
- Macro F1-score (preferred for class imbalance)

Target ranges:
- Accuracy: 0.75 - 1.0
- F1-score: 0.70 - 1.0

## Validation
- Stratified train/validation/test split by class
- Experiment logging for reproducibility
- Use a fixed random seed to make results reproducible

## Datasets
Primary dataset: GoEmotions
- ~58k texts
- 27 emotion labels
- Short Reddit comments
- Noisy, multi-label data

Potential additional data: Twitter sentiment datasets from Kaggle.

## Modeling
### Baseline
- Logistic regression with TF-IDF features
- Expected baseline Macro F1: 0.50 - 0.60

### Main model
- Train a model from scratch without pretrained language models
- Architecture:
  - Embedding layer
  - Bidirectional LSTM
  - Fully connected classifier

## Deployment
- Expose the model via a REST API inference service
- Support external applications such as desktop apps, corporate messengers, and Telegram bots

## Current status
- The project structure is present, but training and inference plumbing is still under development.
- `emotion_assistant/train.py` currently loads Hydra config and should be extended to instantiate data, model, and trainer.
- The README will be updated as the implementation matures.
