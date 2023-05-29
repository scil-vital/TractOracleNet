#!/usr/bin/env python
import argparse
import numpy as np
import torch

from argparse import RawTextHelpFormatter
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points

from TractOracle.models.feed_forward import FeedForwardOracle

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
        self.checkpoint = train_dto['checkpoint']
        self.tractogram = train_dto['tractogram']
        self.reference = train_dto['reference']

    def train(
        self,
    ):
        """
        """

        # Get example state to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        checkpoint = torch.load(self.checkpoint)
        hyper_parameters = checkpoint["hyper_parameters"]

        models = {
            'FeedForwardOracle': FeedForwardOracle
        }

        model = models[hyper_parameters[
            'name']].load_from_checkpoint(self.checkpoint)
        model.eval()

        sft = load_tractogram(self.tractogram, self.reference)
        sft.to_vox()

        streamlines = set_number_of_points(sft.streamlines, 128)
        dirs = np.diff(streamlines, axis=1)

        data = torch.as_tensor(dirs, dtype=torch.float, device='cuda')

        with torch.no_grad():
            predictions = model(data).cpu().numpy()

        print(predictions)

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
    parser.add_argument('checkpoint', type=str,
                        help='TODO')
    parser.add_argument('tractogram', type=str,
                        help='TODO')
    parser.add_argument('reference', type=str,
                        help='TODO change this for scilpy')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_args(parser)

    args = parser.parse_args()

    return parser, args


def main():

    parser, args = parse_args()

    experiment = TractOracleTraining(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
