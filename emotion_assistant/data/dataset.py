from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from emotion_assistant.data.preprocessing import clean_text


def parse_label_list(raw_label: str) -> List[int]:
    text = str(raw_label).strip()
    text = text.strip("[]")
    if not text:
        return []

    values = [
        value.strip() for value in text.replace(",", " ").split() if value.strip()
    ]
    return [int(value) for value in values if value.isdigit()]


def load_data(csv_path: Path) -> Tuple[List[str], List[List[int]]]:
    dataframe = pd.read_csv(csv_path)
    texts = [clean_text(str(text)) for text in dataframe["text"].fillna("")]
    label_lists = [parse_label_list(label) for label in dataframe["labels"].fillna("")]
    return texts, label_lists


def build_multihot_labels(
    label_lists: List[List[int]], num_classes: int
) -> torch.Tensor:
    labels = torch.zeros((len(label_lists), num_classes), dtype=torch.float32)
    for index, label_list in enumerate(label_lists):
        for label in label_list:
            if 0 <= label < num_classes:
                labels[index, label] = 1.0
    return labels


class EmotionDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: torch.Tensor,
        vocab: Dict[str, int],
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def encode(self, text: str) -> torch.Tensor:
        tokens = text.split()
        ids = [self.vocab.get(token, self.vocab.get("<UNK>", 1)) for token in tokens]
        ids = ids[: self.max_length]
        ids += [self.vocab.get("<PAD>", 0)] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encode(self.texts[idx]),
            "label": self.labels[idx],
        }

    def __len__(self) -> int:
        return len(self.texts)
