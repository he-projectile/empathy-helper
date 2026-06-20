import argparse
import json
import os
import pickle
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from emotion_assistant.data.labels import GOEMOTIONS_LABELS
from emotion_assistant.data.preprocessing import clean_text
from emotion_assistant.models.bilstm import BiLSTMClassifier


def encode_text(text: str, vocab: dict, max_length: int) -> torch.Tensor:
    tokens = clean_text(text).split()
    ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]
    ids = ids[:max_length]
    ids += [vocab.get("<PAD>", 0)] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def load_checkpoint(checkpoint_path: str, metadata_path: str):
    with open(metadata_path, "rb") as handle:
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
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab, metadata


def predict(text: str, model: torch.nn.Module, vocab: dict, max_length: int, top_k: int = 5):
    input_ids = encode_text(text, vocab, max_length)
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    topk = ranked[:top_k]
    topk_labels = [GOEMOTIONS_LABELS[i] if i < len(GOEMOTIONS_LABELS) else str(i) for i in topk]
    return {
        "text": text,
        "predicted_label": {
            "index": topk[0],
            "emotion": topk_labels[0],
        },
        "probs": {topk_labels[i]: probs[topk[i]] for i in range(len(topk))},
    }


def main():
    parser = argparse.ArgumentParser(description="Infer emotion from a single text input.")
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/last.ckpt",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="outputs/checkpoints/metadata.pkl",
        help="Path to saved model metadata",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top K probabilities to return")
    args = parser.parse_args()

    model, vocab, metadata = load_checkpoint(args.checkpoint, args.metadata)
    result = predict(args.text, model, vocab, metadata["max_length"], top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
