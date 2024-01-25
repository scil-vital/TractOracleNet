#!/usr/bin/env python
import argparse
import numpy as np
import torch

from argparse import RawTextHelpFormatter
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import set_number_of_points
from tqdm import tqdm

from scilpy.utils.streamlines import uniformize_bundle_sft
from scilpy.viz.utils import get_colormap

from TractOracle.models.autoencoder import AutoencoderOracle
from TractOracle.models.feed_forward import FeedForwardOracle
from TractOracle.models.transformer import TransformerOracle

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
        self.checkpoint = train_dto['checkpoint']
        self.tractogram = train_dto['tractogram']
        self.reference = train_dto['reference']
        self.threshold = train_dto['threshold']
        self.batch_size = train_dto['batch_size']
        self.out = train_dto['out']
        self.failed = train_dto['failed']
        self.all = train_dto['all']

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
            'FeedForwardOracle': FeedForwardOracle,
            'TransformerOracle': TransformerOracle
        }

        # Load it from the checkpoint
        model = models[hyper_parameters[
            'name']].load_from_checkpoint(self.checkpoint)
        # Put the model in eval mode to fix dropout and other stuff
        model.eval()

        # Load the tractogram using a reference to make sure it can
        # go into proper voxel space.
        sft = load_tractogram(self.tractogram, self.reference,
                              bbox_valid_check=False)
        sft.to_vox()
        sft.to_corner()
        uniformize_bundle_sft(sft)

        lengths = [len(s) for s in sft.streamlines]
        scores_per_point = np.zeros((len(lengths), max(lengths), 1))

        for i, s in enumerate(tqdm(sft.streamlines)):
            length = len(scores_per_point[i])
            streamlines = [s[:l] for l in range(3, length)]

            # Resample streamlines to a fixed number of points. This should be
            # set by the model ? TODO?
            resampled_streamlines = set_number_of_points(streamlines, 128)
            # Compute streamline features as the directions between points
            dirs = np.diff(resampled_streamlines, axis=1)

            with torch.no_grad():
                data = torch.as_tensor(
                    dirs, dtype=torch.float, device='cuda')
                pred_batch = model(data).cpu().numpy()

            scores_per_point[i][3:] = pred_batch[:, None]

        scores = [list(scores_per_point[i, :l])
                  for i, l in enumerate(lengths)]
        sft.data_per_point['score'] = scores

        cmap = get_colormap('jet')
        color = cmap(np.squeeze(sft.data_per_point['score']._data))[:, 0:3] * 255

        sft.data_per_point['color'] = sft.streamlines
        sft.data_per_point['color']._data = color
        save_tractogram(sft, self.out, bbox_valid_check=False)

        # TODO: Save all streamlines and add scores as dps


def add_args(parser):
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
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for predictions.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold score for filtering.')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--all', action='store_true',
                   help='Output a tractogram containing all streamlines '
                   'and scores.')
    g.add_argument('--failed', type=str, default=None,
                   help='Output file for invalid streamlines.')


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
