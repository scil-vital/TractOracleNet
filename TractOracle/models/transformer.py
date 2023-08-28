import math
import torch

from torch import nn
from torch.nn import functional as F
from lightning.pytorch import LightningModule

from TractOracle.models.utils import calc_accuracy, PositionalEncoding


class TransformerOracle(LightningModule):

    def __init__(self, input_size, output_size, n_head, n_layers, lr):
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

        self.sig = nn.Sigmoid()

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, threshold=0.01, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "pred_train_loss"
        }

    def forward(self, x):

        N, L, D = x.shape  # Batch size, length of sequence, nb. of dims
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.embedding(x) * math.sqrt(self.embedding_size)

        encoding = self.pos_encoding(x)

        hidden = self.bert(encoding)

        y = self.head(hidden[:, 0])

        y = self.sig(y)

        return y.squeeze(-1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_hat = self(x)
        pred_loss = F.mse_loss(y_hat, y)

        acc_05 = calc_accuracy(y, y_hat, threshold=0.5)
        acc_075 = calc_accuracy(y, y_hat, threshold=0.75)

        self.log('pred_train_loss', pred_loss, on_step=False, on_epoch=True)
        self.log('pred_train_acc_0.5', acc_05, on_step=False, on_epoch=True)
        self.log('pred_train_acc_0.75', acc_075, on_step=False, on_epoch=True)

        return pred_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)

        y_hat = self(x)
        pred_loss = F.mse_loss(y_hat, y)

        acc_05 = calc_accuracy(y, y_hat, threshold=0.5)
        acc_075 = calc_accuracy(y, y_hat, threshold=0.75)

        self.log('pred_val_loss', pred_loss, on_step=False, on_epoch=True)
        self.log('pred_val_acc_0.5', acc_05, on_step=False, on_epoch=True)
        self.log('pred_val_acc_0.75', acc_075, on_step=False, on_epoch=True)
