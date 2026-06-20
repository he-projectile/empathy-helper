from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import torch

from emotion_assistant.data.labels import GOEMOTIONS_LABELS
from emotion_assistant.data.preprocessing import clean_text
from emotion_assistant.models.bilstm import BiLSTMClassifier


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> torch.Tensor:
    tokens = clean_text(text).split()
    ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]
    ids = ids[:max_length]
    ids += [vocab.get("<PAD>", 0)] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def load_checkpoint(
    checkpoint_path: Path, metadata_path: Path
) -> Tuple[BiLSTMClassifier, Dict[str, int], Dict[str, object]]:
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    vocab = metadata["vocab"]
    num_classes = metadata["num_classes"]
    model_config = metadata["model_config"]

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        num_classes=num_classes,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if all(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {
            key.replace("model.", "", 1): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab, metadata


def predict_text(
    text: str,
    model: BiLSTMClassifier,
    vocab: Dict[str, int],
    max_length: int,
    top_k: int = 5,
) -> Dict[str, object]:
    input_ids = encode_text(text, vocab, max_length)
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    ranked_indices = sorted(
        range(len(probs)), key=lambda index: probs[index], reverse=True
    )
    top_indices = ranked_indices[:top_k]
    top_labels = [
        GOEMOTIONS_LABELS[index] if index < len(GOEMOTIONS_LABELS) else str(index)
        for index in top_indices
    ]

    return {
        "text": text,
        "predicted_label": {
            "index": top_indices[0],
            "emotion": top_labels[0],
        },
        "probs": {
            top_labels[i]: probs[top_indices[i]] for i in range(len(top_indices))
        },
    }
