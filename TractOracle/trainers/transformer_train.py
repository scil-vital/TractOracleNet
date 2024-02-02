#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CometLogger
from os.path import join
# from lightning.pytorch.tuner import Tuner
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
        self.experiment = train_dto['experiment']
        self.id = train_dto['id']

        # RL parameters
        self.lr = train_dto['lr']
        self.max_ep = train_dto['max_ep']
        self.n_head = train_dto['n_head']
        self.n_layers = train_dto['n_layers']
        self.checkpoint = train_dto['checkpoint']
        self.render = train_dto['render']

        self.num_workers = train_dto['num_workers']
        self.batch_size = train_dto['batch_size']

        #  Tracking parameters
        self.train_dataset_file = train_dto['train_dataset_file']
        self.val_dataset_file = train_dto['val_dataset_file']
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
            model = TransformerOracle.load_from_checkpoint(self.checkpoint)
        else:
            model = TransformerOracle(
                self.input_size, self.output_size, self.n_head,
                self.n_layers, self.lr)

        dm = StreamlineDataModule(
            self.train_dataset_file, self.val_dataset_file,
            self.test_dataset_file,
            self.batch_size, self.num_workers)

        root_dir = join(self.experiment_path, self.experiment, self.id)

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

        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(logger=comet_logger,
                          log_every_n_steps=1,
                          num_sanity_val_steps=0,
                          max_epochs=self.max_ep,
                          enable_checkpointing=True,
                          default_root_dir=root_dir,
                          precision='16-mixed',
                          callbacks=[lr_monitor])

        trainer.fit(model, dm, ckpt_path=self.checkpoint)
        trainer.test(model, dm)

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
    parser.add_argument('n_head', type=int,
                        help='Number of attention heads.')
    parser.add_argument('n_layers', type=int,
                        help='Number of decoder layers.')
    parser.add_argument('train_dataset_file', type=str,
                        help='Training dataset.')
    parser.add_argument('val_dataset_file', type=str,
                        help='Validation dataset.')
    parser.add_argument('test_dataset_file', type=str,
                        help='Testing dataset.')
    parser.add_argument('--batch_size', type=int, default=(2**11+768),
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

    experiment = TractOracleTransformerTraining(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
