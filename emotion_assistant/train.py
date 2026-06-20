from pathlib import Path
from typing import Dict, List

import hydra
import mlflow
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from emotion_assistant.data.dataset import (
    EmotionDataset,
    build_multihot_labels,
    load_data,
)
from emotion_assistant.data.labels import GOEMOTIONS_LABELS
from emotion_assistant.data.preprocessing import build_vocab
from emotion_assistant.models.bilstm import BiLSTMClassifier
from emotion_assistant.models.lightning_module import EmotionClassifierModule


def make_dataloader(
    dataset: EmotionDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def log_config(cfg: DictConfig) -> None:
    mlflow.log_param("seed", cfg.seed)
    for prefix, section in [
        ("data", cfg.data),
        ("model", cfg.model),
        ("trainer", cfg.trainer),
    ]:
        for key, value in section.items():
            mlflow.log_param(f"{prefix}.{key}", str(value))


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    source_root = Path(to_absolute_path(cfg.data.data_dir))
    train_texts, train_label_lists = load_data(source_root / "train.csv")
    validation_texts, validation_label_lists = load_data(source_root / "validation.csv")
    test_texts, test_label_lists = load_data(source_root / "test.csv")

    label_indices = [label for labels in train_label_lists for label in labels]
    num_classes = max(label_indices, default=-1) + 1
    if num_classes <= 0:
        raise ValueError("Unable to infer the number of classes from training labels.")

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
        validation_dataset,
        cfg.data.batch_size,
        cfg.data.num_workers,
    )
    test_loader = make_dataloader(
        test_dataset,
        cfg.data.batch_size,
        cfg.data.num_workers,
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
        model=model, learning_rate=cfg.model.learning_rate
    )

    if cfg.trainer.get("accelerator") == "gpu" and not torch.cuda.is_available():
        print("CUDA is not available, switching trainer to CPU.")
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

    mlflow.set_experiment("emotion_assistant")
    with mlflow.start_run():
        log_config(cfg)
        trainer = hydra.utils.get_class("lightning.Trainer")(**cfg.trainer)
        trainer.fit(lightning_module, train_loader, validation_loader)
        test_results = trainer.test(lightning_module, dataloaders=test_loader)

        metrics = test_results[0] if test_results else {}
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        original_cwd = Path(get_original_cwd())
        output_dir = original_cwd / "outputs" / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / "last.ckpt"
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
        metadata_path = output_dir / "metadata.pkl"
        with metadata_path.open("wb") as handle:
            import pickle

            pickle.dump(metadata, handle)

        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")

        print(f"Saved checkpoint to {checkpoint_path}")
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
