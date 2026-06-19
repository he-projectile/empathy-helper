import torch.nn as nn
import torch


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        _, (hidden, _) = self.lstm(embedded)

        features = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1,
        )

        return self.classifier(features)
