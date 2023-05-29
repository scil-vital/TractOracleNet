#!/usr/bin/env python
import argparse
import torch

from argparse import RawTextHelpFormatter
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from os.path import join
# from lightning.pytorch.tuner import Tuner

from TractOracle.models.feed_forward import FeedForwardOracle
from TractOracle.trainers.data_module import StreamlineDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TractOracleTraining():
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        train_dto: dict,
    ):
        """
        """
        self.experiment_path = train_dto['path']
        self.experiment = train_dto['experiment']
        self.id = train_dto['id']

        # RL parameters
        self.lr = train_dto['lr']
        self.max_ep = train_dto['max_ep']
        self.layers = train_dto['layers']
        self.checkpoint = train_dto['checkpoint']
        self.render = train_dto['render']

        self.num_workers = train_dto['num_workers']
        self.batch_size = train_dto['batch_size']

        #  Tracking parameters
        self.train_dataset_file = train_dto['train_dataset_file']
        self.test_dataset_file = train_dto['test_dataset_file']

    def train(
        self,
    ):
        """
        """

        # Get example state to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        if self.checkpoint:
            model = FeedForwardOracle.load_from_checkpoint(self.checkpoint)
        else:
            model = FeedForwardOracle(
                self.input_size, self.output_size, self.layers, self.lr)

        dm = StreamlineDataModule(
            self.train_dataset_file, self.test_dataset_file,
            self.batch_size, self.num_workers)

        root_dir = join(self.experiment_path, self.experiment, self.id)

        # Training
        logger = TensorBoardLogger(root_dir, name=self.id)
        trainer = Trainer(logger=logger,
                          num_sanity_val_steps=0,
                          max_epochs=self.max_ep,
                          enable_checkpointing=True,
                          default_root_dir=root_dir,
                          profiler='simple')

        trainer.fit(model, dm, ckpt_path=self.checkpoint)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        # Start training !
        self.train()


def add_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('id', type=str,
                        help='ID of experiment.')
    parser.add_argument('lr', type=float,
                        help='Learning rate.')
    parser.add_argument('max_ep', type=int,
                        help='Number of epochs.')
    parser.add_argument('layers', type=str,
                        help='Layers of network.')
    parser.add_argument('train_dataset_file', type=str,
                        help='Training dataset.')
    parser.add_argument('test_dataset_file', type=str,
                        help='Testing dataset.')
    parser.add_argument('--batch_size', type=int, default=2**12,
                        help='TODO')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='TODO')
    parser.add_argument('--checkpoint', type=str,
                        help='TODO')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu or not')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')
    parser.add_argument('--render', action='store_true',
                        help='Save screenshots of tracking as it goes along.' +
                        'Preferably disabled on non-graphical environments')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_args(parser)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    experiment = TractOracleTraining(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
