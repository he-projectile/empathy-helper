import torch
from lightning import LightningModule
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score


class EmotionClassifierModule(LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.train_accuracy = MultilabelAccuracy(
            num_labels=model.classifier.out_features
        )
        self.val_accuracy = MultilabelAccuracy(num_labels=model.classifier.out_features)
        self.test_accuracy = MultilabelAccuracy(
            num_labels=model.classifier.out_features
        )
        self.train_f1 = MultilabelF1Score(
            num_labels=model.classifier.out_features, average="macro"
        )
        self.val_f1 = MultilabelF1Score(
            num_labels=model.classifier.out_features, average="macro"
        )
        self.test_f1 = MultilabelF1Score(
            num_labels=model.classifier.out_features, average="macro"
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.sigmoid(logits)
        self.train_accuracy.update(preds, batch["label"])
        self.train_f1.update(preds, batch["label"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.sigmoid(logits)
        self.val_accuracy.update(preds, batch["label"])
        self.val_f1.update(preds, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.sigmoid(logits)
        self.test_accuracy.update(preds, batch["label"])
        self.test_f1.update(preds, batch["label"])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
