import math
import torch

from torch import nn, Tensor
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryRecall, BinaryPrecision, BinaryAccuracy, BinaryROC)
from torchmetrics.regression import (
    MeanSquaredError, MeanAbsoluteError)


class PositionalEncoding(nn.Module):
    """ From
    https://pytorch.org/tutorials/beginner/transformer_tutorial.htm://pytorch.org/tutorials/beginner/transformer_tutorial.html  # noqa E504
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x


class TransformerOracle(LightningModule):

    def __init__(
        self,
        input_size,
        output_size,
        n_head,
        n_layers,
        lr,
        loss=nn.MSELoss
    ):
        super(TransformerOracle, self).__init__()

        self.hparams["name"] = self.__class__.__name__

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.n_head = n_head
        self.n_layers = n_layers

        self.embedding_size = 32

        self.cls_token = nn.Parameter(torch.randn((3)))

        layer = nn.TransformerEncoderLayer(
            self.embedding_size, n_head, batch_first=True)

        self.embedding = nn.Sequential(
            *(nn.Linear(3, self.embedding_size),
              nn.ReLU()))

        self.pos_encoding = PositionalEncoding(
            self.embedding_size, max_len=(input_size//3) + 1)
        self.bert = nn.TransformerEncoder(layer, self.n_layers)
        self.head = nn.Linear(self.embedding_size, output_size)

        self.loss = loss()
        self.sig = nn.Sigmoid()

        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.roc = BinaryROC()

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.trainer.max_epochs
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def forward(self, x):

        N, L, D = x.shape  # Batch size, length of sequence, nb. of dims
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.embedding(x) * math.sqrt(self.embedding_size)

        encoding = self.pos_encoding(x)

        hidden = self.bert(encoding)

        y = self.head(hidden[:, 0])

        if self.loss is not nn.BCEWithLogitsLoss:
            y = self.sig(y)

        return y.squeeze(-1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_hat = self(x)
        pred_loss = self.loss(y_hat, y)

        acc = self.accuracy(y_hat, torch.round(y))
        recall = self.recall(y_hat, torch.round(y))
        precision = self.precision(y_hat, torch.round(y))
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)

        self.log('train_loss', pred_loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_recall', recall, on_step=True, on_epoch=False)
        self.log('train_precision', precision, on_step=True, on_epoch=False)
        self.log('train_mse', mse, on_step=True, on_epoch=False)
        self.log('train_mae', mae, on_step=True, on_epoch=False)

        return pred_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_hat = self(x)
        pred_loss = self.loss(y_hat, y)

        acc = self.accuracy(y_hat, torch.round(y))
        recall = self.recall(y_hat, torch.round(y))
        precision = self.precision(y_hat, torch.round(y))
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)

        self.log('val_loss', pred_loss)
        self.log('val_acc', acc)
        self.log('val_recall', recall)
        self.log('val_precision', precision)
        self.log('val_mse', mse, on_step=True, on_epoch=False)
        self.log('val_mae', mae, on_step=True, on_epoch=False)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_hat = self(x)

        acc = self.accuracy(y_hat, torch.round(y))
        recall = self.recall(y_hat, torch.round(y))
        precision = self.precision(y_hat, torch.round(y))
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        fpr, tpr, thresholds = self.roc(y_hat, y.int())

        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True)
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        for f, p, t in zip(fpr, tpr, thresholds):
            self.log('test_{}_fpr'.format(t), f, on_step=False, on_epoch=True)
            self.log('test_{}_tpr'.format(t), t, on_step=False, on_epoch=True)
        # self.log('test_roc', roc, on_step=False, on_epoch=True)
