# Emotion Assistant

Emotion Assistant is a machine learning system for automatic emotion classification of short text messages.

The project is designed as an industrial-style MLOps pipeline and demonstrates the complete lifecycle of an NLP model: data management, training, experiment tracking, packaging, and inference.

---

# Problem Statement

Many people experience difficulties interpreting emotional intent in written communication.

This project aims to classify short text messages into emotion categories and provide probability estimates for all supported emotions.

Example input:

```json
{
  "text": "I feel really excited and hopeful today."
}
```

Example output:

```json
{
  "label": "joy",
  "probs": {
    "joy": 0.82,
    "sadness": 0.04,
    "anger": 0.02,
    "neutral": 0.12
  }
}
```

---

# Dataset

The primary dataset is **GoEmotions**.

Dataset characteristics:

* approximately 58,000 text samples
* 27 emotion classes
* collected from Reddit comments
* contains noisy real-world language
* originally supports multiple labels per sample

Dataset storage is managed through DVC and is not stored directly in Git.

---

# Model Architecture

## Baseline

* TF-IDF
* Logistic Regression

## Main Model

The main model is trained from scratch without pretrained language models.

Architecture:

```text
Text
 ↓
Tokenizer
 ↓
Vocabulary Encoding
 ↓
Embedding Layer
 ↓
Bidirectional LSTM
 ↓
Fully Connected Layer
 ↓
Emotion Probabilities
```

---

# Metrics

The following metrics are used:

* Accuracy
* Macro F1 Score

Expected performance:

| Metric   | Target |
| -------- | ------ |
| Accuracy | 0.75+  |
| Macro F1 | 0.70+  |

Dataset split:

* Train: 80%
* Validation: 10%
* Test: 10%

Stratified splitting is used whenever possible.

---

# Technology Stack

* Python
* PyTorch
* PyTorch Lightning
* Hydra
* DVC
* MLflow
* uv
* Ruff
* pre-commit

---

# Project Structure

```text
.
├── configs/
│   ├── train.yaml
│   ├── data/
│   ├── model/
│   └── logging/
│
├── emotion_assistant/
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── utils/
│   ├── train.py
│   └── infer.py
│
├── scripts/
├── plots/
├── outputs/
├── dvc.yaml
├── pyproject.toml
└── README.md
```

---

# Setup

Create the environment and install dependencies:

```bash
uv sync
```

Activate environment if necessary:

```bash
.venv\Scripts\activate
```

Install pre-commit hooks:

```bash
pre-commit install
```

Verify code quality tools:

```bash
pre-commit run -a
```

---

# Data Preparation

Download the dataset:

```bash
uv run python datasetDownload.py --output-dir data/raw
```

If DVC storage is configured:

```bash
dvc pull
```

---

# Training

Run training with default Hydra configuration:

```bash
uv run python -m emotion_assistant.train
```

Example with Hydra overrides:

```bash
uv run python -m emotion_assistant.train trainer.max_epochs=20 data.batch_size=128
```

Training pipeline:

1. Load dataset
2. Build vocabulary
3. Create train/validation/test splits
4. Train BiLSTM model
5. Evaluate performance
6. Log metrics to MLflow
7. Save checkpoints
8. Export production artifacts

Generated artifacts:

```text
outputs/checkpoints/last.ckpt
outputs/checkpoints/metadata.pkl
```

---

# Experiment Tracking

MLflow server is expected at:

```text
http://127.0.0.1:8080
```

Logged information:

* training loss
* validation loss
* accuracy
* F1 score
* hyperparameters
* git commit hash

Training plots are saved to:

```text
plots/
```

---

# Inference

Run prediction for a text message:

```bash
uv run python -m emotion_assistant.infer "I feel really excited today."
```

Specify checkpoint manually:

```bash
uv run python -m emotion_assistant.infer \
    "I feel really excited today." \
    --checkpoint outputs/checkpoints/last.ckpt \
    --metadata outputs/checkpoints/metadata.pkl
```

---

# ONNX Export

Export trained model:

```bash
uv run python -m emotion_assistant.inference.export_onnx \
    --checkpoint outputs/checkpoints/last.ckpt \
    --metadata outputs/checkpoints/metadata.pkl \
    --output outputs/model.onnx
```

Generated artifact:

```text
outputs/model.onnx
```

---

# TensorRT Conversion

Convert ONNX model into TensorRT engine:

```bash
python scripts/export_tensorrt.py \
    --onnx outputs/model.onnx \
    --output outputs/model.engine
```

Generated artifact:

```text
outputs/model.engine
```

---

# DVC Pipeline

Reproduce the entire pipeline:

```bash
dvc repro
```

Pull artifacts from remote storage:

```bash
dvc pull
```

Push artifacts:

```bash
dvc push
```

---

# Production Preparation

Production deployment includes:

1. Trained checkpoint export
2. ONNX conversion
3. TensorRT optimization
4. Packaging inference code
5. Deployment through inference server

Required artifacts:

```text
model.onnx
metadata.pkl
vocabulary
inference code
configuration files
```

---

# Inference Server

The model can be deployed using:

* MLflow Serving
* Triton Inference Server

Future deployment targets:

* Desktop applications
* Corporate messengers
* Telegram bots
* Communication assistance tools

---

# Development Notes

The repository follows the course requirements:

* Hydra-based configuration management
* PyTorch Lightning training
* DVC data management
* MLflow experiment tracking
* ONNX export
* TensorRT packaging
* Reproducible training pipeline

Large datasets, checkpoints, and generated artifacts must never be committed directly to Git.
