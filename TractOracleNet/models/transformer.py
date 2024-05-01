import math
import torch

from matplotlib import pyplot as plt
from torch import nn, Tensor
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryRecall, BinaryPrecision, BinaryAccuracy, BinaryROC,
    BinarySpecificity, BinaryF1Score)
from torchmetrics.regression import (
    MeanSquaredError, MeanAbsoluteError)


class PositionalEncoding(nn.Module):
    """ Modified from
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
    """ Transformer model for streamline scoring.

    The model consits of an embedding layer, a positional encoding layer,
    a transformer encoder and a linear layer.
    """

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
        # Keep the name of the model
        self.hparams["name"] = self.__class__.__name__

        # Save the hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.n_head = n_head
        self.n_layers = n_layers

        # Embedding size, could be defined by the user ?
        self.embedding_size = 32

        # Class token, initialized randomly
        self.cls_token = nn.Parameter(torch.randn((3)))

        # Embedding layer
        self.embedding = nn.Sequential(
            *(nn.Linear(3, self.embedding_size),
              nn.ReLU()))

        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(
            self.embedding_size, max_len=(input_size//3) + 1)

        # Transformer encoder layer
        layer = nn.TransformerEncoderLayer(
            self.embedding_size, n_head, batch_first=True)

        # Transformer encoder
        self.bert = nn.TransformerEncoder(layer, self.n_layers)
        # Linear layer
        self.head = nn.Linear(self.embedding_size, output_size)
        # Sigmoid layer
        self.sig = nn.Sigmoid()

        # Loss function
        self.loss = loss()

        # Metrics
        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.spec = BinarySpecificity()
        self.precision = BinaryPrecision()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.roc = BinaryROC()
        self.f1 = BinaryF1Score()

        # Save the hyperparameters to the checkpoint
        self.save_hyperparameters()

    def configure_optimizers(self):
        # Define the optimizer
        # Use Cosine Annealing as learning rate scheduler and AdamW as
        # optimizer
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
        """ Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor (N, L, D)
            Input tensor with shape (N, L, D) where N is the batch size,
            L is the length of the sequence and D is the number of
            dimensions.

        Returns
        -------
        y : torch.Tensor (N)
            Output tensor with shape (N) where N is the batch size.

        """
        # Get the shape of the input
        N, L, D = x.shape  # Batch size, length of sequence, nb. of dims
        # Add class token to the input
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        # Concatenate class token with input
        x = torch.cat((cls_tokens, x), dim=1)
        # Apply embedding layer and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding_size)
        encoding = self.pos_encoding(x)

        # Apply transformer
        hidden = self.bert(encoding)
        # Apply linear layer
        y = self.head(hidden[:, 0])

        # No need to apply sigmoid if using BCEWithLogitsLoss
        if self.loss is not nn.BCEWithLogitsLoss:
            y = self.sig(y)
        else:
            y = hidden[:, 0]

        # Return the output
        return y.squeeze(-1)

    def training_step(self, train_batch, batch_idx):
        """ Training step of the model.

        Parameters
        ----------
        train_batch : tuple
            Tuple containing the input and output of the model.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        pred_loss : torch.Tensor
            Prediction loss.

        """

        # Get the input and output
        x, y = train_batch
        x, y = x.squeeze(0), y.squeeze(0)

        # Forward pass
        y_hat = self(x)
        # Compute the loss
        pred_loss = self.loss(y_hat, y)

        # Compute metrics
        acc = self.accuracy(y_hat, torch.round(y))
        recall = self.recall(y_hat, torch.round(y))
        spec = self.spec(y_hat, torch.round(y))
        precision = self.precision(y_hat, torch.round(y))
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)

        # Log the metrics
        self.log('train_loss', pred_loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_recall', recall, on_step=True, on_epoch=False)
        self.log('train_spec', spec, on_step=True, on_epoch=False)
        self.log('train_precision', precision, on_step=True, on_epoch=False)
        self.log('train_mse', mse, on_step=True, on_epoch=False)
        self.log('train_mae', mae, on_step=True, on_epoch=False)

        return pred_loss

    def validation_step(self, val_batch, batch_idx):
        """ Validation step of the model.

        Parameters
        ----------
        val_batch : tuple
            Tuple containing the input and output of the model.
        batch_idx : int
            Index of the batch.

        """

        # Get the input and output
        x, y = val_batch

        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)

        # Forward pass
        y_hat = self(x)
        # Compute the loss
        pred_loss = self.loss(y_hat, y)

        # Compute metrics
        acc = self.accuracy(y_hat, torch.round(y))
        recall = self.recall(y_hat, torch.round(y))
        precision = self.precision(y_hat, torch.round(y))
        spec = self.spec(y_hat, torch.round(y))
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        f1 = self.f1(y_hat, y)

        # Log the metrics
        self.log('val_loss', pred_loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_spec', spec, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        """ Test step of the model.

        Parameters
        ----------
        test_batch : tuple
            Tuple containing the input and output of the model.
        batch_idx : int
            Index of the batch.
        """
        # Get the input and output
        x, y = test_batch
        if len(x.shape) > 3:
            x = x.squeeze(0)
            y = y.squeeze(0)
        # Forward pass
        y_hat = self(x)

        # Compute metrics
        acc = self.accuracy(y_hat, torch.round(y))
        recall = self.recall(y_hat, torch.round(y))
        precision = self.precision(y_hat, torch.round(y))
        spec = self.spec(y_hat, y)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.roc.update(y_hat, y.int())

        # Log the metrics
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True)
        self.log('test_spec', spec, on_step=False, on_epoch=True)
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        """ Plot ROC curve and save it to file. """

        fig, ax_ = self.roc.plot(score=True)

        fig.savefig('roc.png')
        plt.close(fig)
