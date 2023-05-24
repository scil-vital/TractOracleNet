#!/usr/bin/env python
import argparse
import pytorch_lightning as pl
import torch


from argparse import RawTextHelpFormatter
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from feedforwardmodel import FeedForwardOracle
from StreamlineDataset import StreamlineDataset

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
        self.name = train_dto['name']

        # RL parameters
        self.lr = train_dto['lr']
        self.max_ep = train_dto['max_ep']
        self.layers = train_dto['layers']
        self.render = train_dto['render']

        self.num_workers = 8
        self.batch_size = 8192

        #  Tracking parameters
        self.train_dataset_file = train_dto['train_dataset_file']
        self.test_dataset_file = train_dto['test_dataset_file']

    def train(
        self,
        model,
        train_dataset: StreamlineDataset,
        test_dataset: StreamlineDataset,
    ):
        """
        """
        len_train = len(train_dataset)
        print(len_train)
        train_data, val_data = random_split(train_dataset,
                                            [int(len_train*0.8),
                                             int(len_train*0.2)])

        train_loader = DataLoader(train_data,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size,
                                  prefetch_factor=1,
                                  persistent_workers=True
                                  if self.num_workers > 0 else False,
                                  pin_memory=True)

        valid_loader = DataLoader(val_data,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size,
                                  prefetch_factor=1,
                                  persistent_workers=True
                                  if self.num_workers > 0 else False,
                                  pin_memory=True)

        # training

        logger = TensorBoardLogger("tb_logs", name="my_model")
        trainer = pl.Trainer(log_every_n_steps=1, logger=logger)

        trainer.fit(model, train_loader, valid_loader)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        train_dataset = StreamlineDataset(
            self.train_dataset_file)
        test_dataset = StreamlineDataset(
            self.test_dataset_file)

        # Get example state to define NN input size
        self.input_size = train_dataset.input_size
        self.output_size = 1

        model = FeedForwardOracle(
            self.input_size, self.output_size, self.layers)

        # Start training !
        self.train(model, train_dataset, test_dataset)


def add_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('name', type=str,
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
