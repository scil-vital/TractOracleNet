import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch import LightningModule


def format_widths(widths_str):
    return [int(i) for i in widths_str.split('-')]


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU, dropout=0.5,
    last_activation=nn.Identity
):
    layers = [nn.Flatten(), nn.Linear(input_size, widths[0]),
              activation(), nn.Dropout(dropout)]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation(),
             nn.Dropout(dropout)])

    layers.extend(
        [nn.Linear(widths[-1], output_size), last_activation()])
    return nn.Sequential(*layers)


class FeedForwardOracle(LightningModule):

    def __init__(self, input_size, output_size, layers, lr):
        super(FeedForwardOracle, self).__init__()

        self.hparams["name"] = self.__class__.__name__

        self.input_size = input_size
        self.output_size = output_size
        self.layers = format_widths(layers)
        self.lr = lr

        self.network = make_fc_network(
            self.layers, self.input_size, self.output_size)

        self.save_hyperparameters()

    def forward(self, x):
        return self.network(x).squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)