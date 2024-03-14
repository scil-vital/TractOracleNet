#!/usr/bin/env python
import argparse
import torch

from argparse import RawTextHelpFormatter
from os.path import join

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from TractOracleNet.models.transformer import TransformerOracle
from TractOracleNet.trainers.data_module import StreamlineDataModule

# Set the default precision to float32 to
# speed up training and reduce memory usage
torch.set_float32_matmul_precision("medium")


class TractOracleNetTransformerTraining():
    """ Train a Transformer model to score streamlines.
    """

    def __init__(
        self,
        train_dto: dict,
    ):
        """ Initialize the training process.
        """
        # Experiment parameters
        self.experiment_path = train_dto['path']
        self.experiment = train_dto['experiment']
        self.id = train_dto['id']

        # Model parameters
        self.lr = train_dto['lr']
        self.max_ep = train_dto['max_ep']
        self.n_head = train_dto['n_head']
        self.n_layers = train_dto['n_layers']
        self.checkpoint = train_dto['checkpoint']

        # Data loading parameters
        self.num_workers = train_dto['num_workers']
        self.batch_size = train_dto['batch_size']

        # Data files
        self.train_dataset_file = train_dto['train_dataset_file']
        self.val_dataset_file = train_dto['val_dataset_file']
        self.test_dataset_file = train_dto['test_dataset_file']

    def train(
        self,
    ):
        """ Train the model.
        """
        # Working directory
        root_dir = join(self.experiment_path, self.experiment, self.id)

        # Get example input to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        if self.checkpoint:
            model = TransformerOracle.load_from_checkpoint(self.checkpoint)
        else:
            model = TransformerOracle(
                self.input_size, self.output_size, self.n_head,
                self.n_layers, self.lr)

        # Instanciate the datamodule
        dm = StreamlineDataModule(
            self.train_dataset_file, self.val_dataset_file,
            self.test_dataset_file,
            self.batch_size, self.num_workers)

        # Training
        comet_logger = CometLogger(
            project_name="tractoracle",
            experiment_name='-'.join((self.experiment, self.id)))

        # Log parameters
        comet_logger.log_hyperparams({
            "model": TransformerOracle.__name__,
            "lr": self.lr,
            "max_ep": self.max_ep,
            "n_layers": self.n_layers,
            "n_head": self.n_head,
            "batch_size": self.batch_size})

        # Log the learning rate during training as it will vary
        # from Cosine Annealing
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Define the trainer
        # Mixed precision is used to speed up training and
        # reduce memory usage
        trainer = Trainer(logger=comet_logger,
                          log_every_n_steps=1,
                          num_sanity_val_steps=0,
                          max_epochs=self.max_ep,
                          enable_checkpointing=True,
                          default_root_dir=root_dir,
                          precision='16-mixed',
                          callbacks=[lr_monitor])
        # Train the model
        trainer.fit(model, dm, ckpt_path=self.checkpoint)
        # Test the model
        trainer.test(model, dm)


def add_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('id', type=str,
                        help='ID of experiment.')
    parser.add_argument('max_ep', type=int,
                        help='Number of epochs.')
    parser.add_argument('train_dataset_file', type=str,
                        help='Training dataset.')
    parser.add_argument('val_dataset_file', type=str,
                        help='Validation dataset.')
    parser.add_argument('test_dataset_file', type=str,
                        help='Testing dataset.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads.')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of encoder layers.')
    parser.add_argument('--batch_size', type=int, default=(2**11+768),
                        help='Batch size, in number of streamlines.')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of workers for dataloader.')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint. If not provided, '
                             'train from scratch.')


def parse_args():
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    return args


def main():
    " Main function."

    args = parse_args()
    # Train the model
    training = TractOracleNetTransformerTraining(vars(args))
    training.train()


if __name__ == "__main__":
    main()
