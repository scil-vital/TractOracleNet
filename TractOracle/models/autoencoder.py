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
        self.kernel_size = 3
        self.layers = format_widths(layers)
        self.lr = lr

        # TODO: Make the autoencoder architecture parametrizable ?

        factor = 2

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32 * factor, 64 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64 * factor, 128 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(128 * factor, 256 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(256 * factor, 512 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(512 * factor, 1024 * factor, 3, stride=1, padding=0))

        self.network = make_fc_network(
            self.layers, 1024 * factor, self.output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024 * factor, 512 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(512 * factor, 256 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(256 * factor, 128 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(128 * factor, 64 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(64 * factor, 32 * factor, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(32 * factor, 3, 3, stride=2, padding=0),
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.encoder(x).squeeze()
        return self.network(z).squeeze()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = x.squeeze(0)
        y = y.squeeze(0)

        x = x.permute(0, 2, 1)
        z = self.encoder(x)

        y_hat = self.network(z.squeeze()).squeeze()
        pred_loss = F.mse_loss(y_hat.float(), y.float())

        x_hat = self.decoder(z)
        reconst_loss = F.mse_loss(x_hat, x)
        self.log(
            'reconst_train_loss', reconst_loss, on_step=False, on_epoch=True)
        self.log('pred_train_loss', pred_loss, on_step=False, on_epoch=True)
        return reconst_loss + pred_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x = x.squeeze(0)
        y = y.squeeze(0)

        x = x.permute(0, 2, 1)
        z = self.encoder(x)

        y_hat = self.network(z.squeeze()).squeeze()
        pred_loss = F.mse_loss(y_hat.float(), y.float())

        x_hat = self.decoder(z)
        reconst_loss = F.mse_loss(x_hat, x)
        self.log(
            'reconst_val_loss', reconst_loss, on_step=False, on_epoch=True)
        self.log('pred_val_loss', pred_loss, on_step=False, on_epoch=True)
