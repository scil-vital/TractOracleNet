#!/usr/bin/env python
import argparse
import numpy as np
import torch

from argparse import RawTextHelpFormatter
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import set_number_of_points

from TractOracle.models.autoencoder import AutoencoderOracle
from TractOracle.models.feed_forward import FeedForwardOracle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TractOraclePredictor():
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
        self.threshold = train_dto['threshold']
        self.out = train_dto['out']

    def run(self):
        """
        Main method where the magic happens
        """

        # Get example state to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # TODO: Get this from datamodule
        self.output_size = 1  # TODO: Get this from datamodule.
        # Might quantize score in the future ?

        # Load the model's hyper and actual params from a saved checkpoint
        checkpoint = torch.load(self.checkpoint)
        hyper_parameters = checkpoint["hyper_parameters"]

        # The model's class is saved in hparams
        models = {
            'AutoencoderOracle': AutoencoderOracle,
            'FeedForwardOracle': FeedForwardOracle
        }

        # Load it from the checkpoint
        model = models[hyper_parameters[
            'name']].load_from_checkpoint(self.checkpoint)
        # Put the model in eval mode to fix dropout and other stuff
        model.eval()

        # Load the tractogram using a reference to make sure it can
        # go into proper voxel space.
        sft = load_tractogram(self.tractogram, self.reference)
        sft.to_vox()

        # Resample streamlines to a fixed number of points. This should be
        # set by the model ? TODO?
        resampled_streamlines = set_number_of_points(sft.streamlines, 128)
        # Compute streamline features as the directions between points
        dirs = np.diff(resampled_streamlines, axis=1)

        # Load the features as torch tensors and predict
        with torch.no_grad():
            data = torch.as_tensor(dirs, dtype=torch.float, device='cuda')
            predictions = model(data).cpu().numpy()

        # Fetch the streamlines that passed the gauntlet
        ids = np.argwhere(predictions > self.threshold).squeeze()
        filtered = sft[ids]

        # Save the filtered streamlines
        print('Kept {}/{} streamlines.'.format(len(sft), len(filtered)))
        save_tractogram(filtered, self.out, bbox_valid_check=False)

        # TODO: Save all streamlines and add scores as dps


def add_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('id', type=str,
                        help='ID of experiment.')
    parser.add_argument('checkpoint', type=str,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model.')
    parser.add_argument('tractogram', type=str,
                        help='Tractogram file to score.')
    parser.add_argument('reference', type=str, default='same',
                        help='Reference file for tractogram (.nii.gz).'
                             'For .trk, can be \'same\'.')
    parser.add_argument('out', type=str,
                        help='Output file.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold score for filtering.')


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

    experiment = TractOraclePredictor(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
