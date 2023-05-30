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


class AutoencoderOracle(LightningModule):

    def __init__(self, input_size, output_size, layers, lr):
        super(AutoencoderOracle, self).__init__()

        self.hparams["name"] = self.__class__.__name__

        self.input_size = input_size
        self.output_size = output_size
        self.layers = format_widths(layers)
        self.lr = lr

        self.network = make_fc_network(
            self.layers, self.input_size, self.output_size)

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        # TODO: Add latent-space predictor
        return self.network(x).squeeze()

    def training_step(self, train_batch, batch_idx):
        # TODO: Add latent-space predictor
        x, y = train_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # TODO: Add latent-space predictor
        x, y = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
