#!/usr/bin/env python
import argparse
import numpy as np
import torch

from argparse import RawTextHelpFormatter
from dipy.io.streamline import load_tractogram
from tqdm import tqdm

from TractOracle.utils import get_data, save_filtered_streamlines
from TractOracle.models.utils import get_model

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

    def predict(self, model, data):
        """ Predict the scores of the streamlines.

        Args:
            model: The model to use for prediction.
            data: The data to predict on.

        Returns:
            The scores of the streamlines.
        """

        predictions = []
        for i in tqdm(range(0, data.shape[0], self.batch_size)):
            j = i + self.batch_size
            # Load the features as torch tensors and predict
            batch_dirs = data[i:j, :, :]
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    batch = torch.as_tensor(
                        batch_dirs, dtype=torch.float, device='cuda')
                    pred_batch = model(batch)
                    predictions.extend(pred_batch.cpu().numpy().tolist())

        predictions = np.asarray(predictions)

        return predictions

    def run(self):
        """
        Main method where the magic happens
        """

        model = get_model(self.checkpoint)

        # Load the tractogram using a reference to make sure it can
        # go into proper voxel space.
        sft = load_tractogram(self.tractogram, self.reference,
                              bbox_valid_check=False, trk_header_check=False)
        # Get the directions between points of the streamlines
        dirs = get_data(sft)

        # Predict the scores of the streamlines
        predictions = self.predict(model, dirs)

        # Save the filtered streamlines
        if not self.all:
            # Fetch the streamlines that passed the gauntlet
            ids = np.argwhere(
                predictions > self.threshold).squeeze()

            # Save the filtered streamlines
            print('Kept {}/{} streamlines ({}%).'.format(len(ids),
                  len(sft), (len(ids) / len(sft) * 100)))

            # Save the streamlines
            save_filtered_streamlines(sft, predictions, ids, self.out)

            # Save the streamlines that failed
            if self.failed:
                # Fetch the streamlines that failed
                failed_ids = np.setdiff1d(np.arange(predictions.shape[0]), ids)
                # Save the streamlines
                save_filtered_streamlines(
                    sft, predictions, failed_ids, self.failed)
        else:
            # Save all the streamlines
            save_filtered_streamlines(
                sft, predictions, np.arange(0, predictions.shape[0]), self.out)


def add_args(parser):
    parser.add_argument('checkpoint', type=str,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model.')
    parser.add_argument('tractogram', type=str,
                        help='Tractogram file to score.')
    parser.add_argument('out', type=str,
                        help='Output file.')
    parser.add_argument('--reference', type=str, default='same',
                        help='Reference file for tractogram (.nii.gz).'
                             'For .trk, can be \'same\'.')
    parser.add_argument('--batch_size', type=int, default=512,
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
