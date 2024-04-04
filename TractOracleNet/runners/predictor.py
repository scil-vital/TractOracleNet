#!/usr/bin/env python
import argparse
import numpy as np
import torch

from argparse import RawTextHelpFormatter
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import set_number_of_points
from tqdm import tqdm

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg)

from TractOracleNet.utils import get_data, save_filtered_streamlines
from TractOracleNet.models.utils import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cast_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TractOracleNetPredictor():
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
        self.dense = train_dto['dense']
        self.tractogram = train_dto['tractogram']
        self.reference = train_dto['reference']
        self.threshold = train_dto['threshold']
        self.batch_size = train_dto['batch_size']
        self.out = train_dto['out']
        self.rejected = train_dto['rejected']
        self.nofilter = train_dto['nofilter']

    def predict(self, model, sft):
        """ Predict the scores of the streamlines.

        Args:
            model: The model to use for prediction.
            data: The data to predict on.

        Returns:
            The scores of the streamlines.
        """

        # Get the directions between points of the streamlines
        total = len(sft)

        predictions = np.zeros((total))
        for i in tqdm(range(0, total, self.batch_size)):
            j = i + self.batch_size
            # Load the features as torch tensors and predict
            batch_dirs = get_data(sft[i:j], device)

            with torch.autocast(cast_device, enabled=cast_device == 'cuda'):
                with torch.no_grad():
                    batch = torch.as_tensor(
                        batch_dirs, dtype=torch.float, device=device)
                    pred_batch = model(batch)
                    predictions[i:j] = pred_batch.cpu().numpy()

        return predictions

    def dense_predict(self, model, sft):
        """ Predict the scores of the streamlines point by point. This will
        be slower than predict, but is useful for visualizing the scores.

        Args:
            model: The model to use for prediction.
            data: The data to predict on.

        Returns:
            scores: The scores of the streamlines.

        """

        sft.to_vox()
        sft.to_corner()

        lengths = [len(s) for s in sft.streamlines]
        scores_per_point = np.zeros((len(lengths), max(lengths), 1))

        # Predict the scores of the streamlines point by point, one streamlne
        # at a time.
        for i, s in enumerate(tqdm(sft.streamlines)):
            length = len(scores_per_point[i])
            streamlines = [s[:le] for le in range(3, length)]

            # Resample streamlines to a fixed number of points. This should be
            # set by the model ? TODO?
            resampled_streamlines = set_number_of_points(streamlines, 128)
            # Compute streamline features as the directions between points
            dirs = np.diff(resampled_streamlines, axis=1)

            with torch.autocast(cast_device, enabled=cast_device == 'cuda'):
                with torch.no_grad():
                    data = torch.as_tensor(
                        dirs, dtype=torch.float, device=device)
                    pred_batch = model(data).cpu().numpy()

            scores_per_point[i][3:] = pred_batch[:, None]

        scores = [list(scores_per_point[i, :l])
                  for i, l in enumerate(lengths)]

        return scores

    def run(self):
        """
        Main method where the magic happens
        """

        model = get_model(self.checkpoint)

        # Load the tractogram using a reference to make sure it can
        # go into proper voxel space.
        sft = load_tractogram(self.tractogram, self.reference,
                              bbox_valid_check=False, trk_header_check=False)

        if self.dense:
            # Predict the scores of the streamlines point by point
            predictions = self.dense_predict(model, sft)
        else:
            # Predict the scores of the streamlines
            predictions = self.predict(model, sft)

        # Save the filtered streamlines
        if not self.dense:
            # Fetch the streamlines that passed the gauntlet
            if self.nofilter:
                ids = np.arange(0, len(predictions))
            else:
                # Save the filtered streamlines
                print('Kept {}/{} streamlines ({}%).'.format(len(ids),
                      len(sft), (len(ids) / len(sft) * 100)))

                ids = np.argwhere(
                    predictions > self.threshold).squeeze()

            new_sft = StatefulTractogram.from_sft(sft[ids].streamlines, sft)

            # Save the streamlines
            save_filtered_streamlines(new_sft, predictions[ids], self.out)

            # Save the streamlines that rejected
            if self.rejected:
                # Fetch the streamlines that rejected
                rejected_ids = np.setdiff1d(np.arange(predictions.shape[0]),
                                            ids)

                new_sft = StatefulTractogram.from_sft(
                    sft[rejected_ids].streamlines, sft)

                # Save the streamlines
                save_filtered_streamlines(
                    new_sft, predictions[rejected_ids], self.rejected)
        else:
            # Save all streamlines
            sft.data_per_point['score'] = predictions

            save_filtered_streamlines(
                sft, predictions, self.out, dense=self.dense)


def _build_arg_parser(parser):
    parser.add_argument('tractogram', type=str,
                        help='Tractogram file to score.')
    parser.add_argument('out', type=str,
                        help='Output file.')
    parser.add_argument('--reference', type=str, default='same',
                        help='Reference file for tractogram (.nii.gz).'
                             'For .trk, can be \'same\'. Default is '
                             '[%(default)s].')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for predictions. Default is '
                             '[%(default)s].')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold score for filtering. Default is '
                             '[%(default)s].')
    parser.add_argument('--checkpoint', type=str,
                        default='model/tractoracle.ckpt',
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s].')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--nofilter', action='store_true',
                   help='Output a tractogram containing all streamlines '
                   'instead of only plausible ones.')
    g.add_argument('--rejected', type=str, default=None,
                   help='Output file for invalid streamlines.')
    g.add_argument('--dense', action='store_true',
                   help='Predict the scores of the streamlines point by point.'
                        ' Streamlines\' endpoints should be uniformized for'
                        ' best visualization.')

    add_overwrite_arg(parser)

def parse_args():
    """ Filter a tractogram. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    _build_arg_parser(parser)
    args = parser.parse_args()

    assert_inputs_exist(parser, args.tractogram)
    assert_outputs_exist(parser, args, args.out, optional=args.rejected)

    return parser, args


def main():

    parser, args = parse_args()

    experiment = TractOracleNetPredictor(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
