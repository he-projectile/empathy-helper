import os
import pickle
import sys
from typing import List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import hydra
import pandas as pd
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from emotion_assistant.data.dataset import EmotionDataset
from emotion_assistant.data.labels import GOEMOTIONS_LABELS
from emotion_assistant.data.preprocessing import build_vocab, clean_text
from emotion_assistant.models.bilstm import BiLSTMClassifier
from emotion_assistant.models.lightning_module import EmotionClassifierModule


def parse_label_list(raw_label: str) -> List[int]:
    text = str(raw_label).strip()
    text = text.strip("[]")
    if not text:
        return []

    values = [
        value.strip() for value in text.replace(",", " ").split() if value.strip()
    ]
    return [int(value) for value in values if value.isdigit()]


def build_multihot_labels(
    label_lists: List[List[int]], num_classes: int
) -> torch.Tensor:
    labels = torch.zeros((len(label_lists), num_classes), dtype=torch.float32)
    for idx, label_list in enumerate(label_lists):
        for label in label_list:
            if 0 <= label < num_classes:
                labels[idx, label] = 1.0
    return labels


def load_data(csv_path: str, max_length: int):
    dataframe = pd.read_csv(csv_path)
    texts = [clean_text(str(text)) for text in dataframe["text"].fillna("")]
    label_lists = [parse_label_list(label) for label in dataframe["labels"].fillna("")]
    return texts, label_lists


def make_dataloader(
    dataset: EmotionDataset, batch_size: int, num_workers: int, shuffle: bool = False
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    data_root = to_absolute_path(cfg.data.data_dir)

    train_texts, train_label_lists = load_data(
        os.path.join(data_root, "train.csv"),
        cfg.data.max_length,
    )
    validation_texts, validation_label_lists = load_data(
        os.path.join(data_root, "validation.csv"),
        cfg.data.max_length,
    )
    test_texts, test_label_lists = load_data(
        os.path.join(data_root, "test.csv"),
        cfg.data.max_length,
    )

    all_label_indices = [label for labels in train_label_lists for label in labels]
    num_classes = max(all_label_indices, default=-1) + 1
    if num_classes <= 0:
        raise ValueError("Unable to infer number of classes from training labels.")

    vocab = build_vocab(train_texts)

    train_dataset = EmotionDataset(
        train_texts,
        build_multihot_labels(train_label_lists, num_classes),
        vocab,
        cfg.data.max_length,
    )
    validation_dataset = EmotionDataset(
        validation_texts,
        build_multihot_labels(validation_label_lists, num_classes),
        vocab,
        cfg.data.max_length,
    )
    test_dataset = EmotionDataset(
        test_texts,
        build_multihot_labels(test_label_lists, num_classes),
        vocab,
        cfg.data.max_length,
    )

    train_loader = make_dataloader(
        train_dataset,
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=True,
    )
    validation_loader = make_dataloader(
        validation_dataset, cfg.data.batch_size, cfg.data.num_workers
    )
    test_loader = make_dataloader(
        test_dataset, cfg.data.batch_size, cfg.data.num_workers
    )

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_classes=num_classes,
    )

    lightning_module = EmotionClassifierModule(
        model=model,
        learning_rate=cfg.model.learning_rate,
    )

    if cfg.trainer.get("accelerator", None) == "gpu" and not torch.cuda.is_available():
        print("CUDA is not available, switching trainer to CPU.")
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

    trainer = hydra.utils.get_class("lightning.Trainer")(**cfg.trainer)
    trainer.fit(lightning_module, train_loader, validation_loader)
    trainer.test(lightning_module, dataloaders=test_loader)

    original_cwd = get_original_cwd()
    output_dir = os.path.join(original_cwd, "outputs", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "last.ckpt")
    trainer.save_checkpoint(checkpoint_path)

    metadata = {
        "vocab": vocab,
        "max_length": cfg.data.max_length,
        "num_classes": num_classes,
        "label_names": GOEMOTIONS_LABELS[:num_classes],
        "model_config": {
            "embedding_dim": cfg.model.embedding_dim,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
            "dropout": cfg.model.dropout,
        },
    }

    metadata_path = os.path.join(output_dir, "metadata.pkl")
    with open(metadata_path, "wb") as handle:
        pickle.dump(metadata, handle)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
