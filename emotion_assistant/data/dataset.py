import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        vocab,
        max_length,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def encode(self, text):
        tokens = text.split()

        ids = [self.vocab.get(token, 1) for token in tokens]

        ids = ids[: self.max_length]

        ids += [0] * (self.max_length - len(ids))

        return torch.tensor(ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encode(self.texts[idx]),
            "label": torch.tensor(self.labels[idx]),
        }

    def __len__(self):
        return len(self.texts)
