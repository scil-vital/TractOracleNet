#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor

from TractOracle.models.transformer import TransformerOracle
from TractOracle.trainers.data_module import StreamlineDataModule


class TractOracleTransformerTraining():
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

        self.checkpoint = train_dto['checkpoint']

        self.num_workers = train_dto['num_workers']
        self.batch_size = train_dto['batch_size']

        #  Tracking parameters
        self.train_dataset_file = train_dto['train_dataset_file']
        self.val_dataset_file = train_dto['val_dataset_file']
        self.test_dataset_file = train_dto['test_dataset_file']

    def test(
        self,
    ):
        """
        """

        # Get example state to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        if self.checkpoint:
            model = TransformerOracle.load_from_checkpoint(self.checkpoint)
        else:
            model = TransformerOracle(
                self.input_size, self.output_size, self.n_head,
                self.n_layers, self.lr)

        dm = StreamlineDataModule(
            self.train_dataset_file, self.val_dataset_file,
            self.test_dataset_file,
            self.batch_size,
            self.num_workers)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(log_every_n_steps=1,
                          num_sanity_val_steps=0,
                          enable_checkpointing=True,
                          precision='16-mixed',
                          callbacks=[lr_monitor])

        trainer.test(model, dm, ckpt_path=self.checkpoint)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        # Start testing!
        self.test()


def add_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('train_dataset_file', type=str,
                        help='Training dataset.')
    parser.add_argument('val_dataset_file', type=str,
                        help='Validation dataset.')
    parser.add_argument('test_dataset_file', type=str,
                        help='Testing dataset.')
    parser.add_argument('--batch_size', type=int, default=(2**13+2**11),
                        help='TODO')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='TODO')
    parser.add_argument('--checkpoint', type=str,
                        help='TODO')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu or not')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')


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

    experiment = TractOracleTransformerTraining(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
