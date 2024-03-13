#!/usr/bin/env python
import argparse

import h5py
import json
import numpy as np

from argparse import RawTextHelpFormatter
from os.path import expanduser

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'  # much faster on cpu #TODO?

"""
Script to process multiple subjects into a single .hdf5 file.
See example configuration file.

Streamlines will be "pre-reversed" to prevent having to do it on the fly.

Heavly inspired by https://github.com/scil-vital/dwi_ml/blob/master/dwi_ml/data/hdf5/hdf5_creation.py # noqa E405
"""


def parse_args():

    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('config_file', type=str,
                        help="Configuration file to load subjects and their"
                        " volumes.")
    parser.add_argument('output', type=str,
                        help="Output filename including path")
    parser.add_argument('--nb_points', type=int, default=128,
                        help='Number of points to resample streamlines to.')

    arguments = parser.parse_args()

    return arguments


def main():
    """ Parse args, generate dataset and save it on disk """
    args = parse_args()

    generate_dataset(config_file=args.config_file,
                     output=args.output,
                     nb_points=args.nb_points)


def generate_dataset(
    config_file: str,
    output: str,
    nb_points: int,
) -> None:
    """ Generate a dataset

    Args:
        config_file:
        output:
        nb_points:

    """

    dataset_name = output

    with open(config_file, "r") as conf:
        config = json.load(conf)

        add_subjects_to_hdf5(
            config, dataset_name, nb_points)


def add_subjects_to_hdf5(

    config, dataset_name, nb_points
):
    """

    Args:
        config:
        hdf_file:
        nb_points:

    """
    for subject_id in config:
        "Processing subject {}".format(subject_id),

        subject_config = config[subject_id]

        reference_file = subject_config['reference']
        streamlines_file = subject_config['streamlines']

        process_subjects(streamlines_file, reference_file, dataset_name,
                         nb_points)


def process_subjects(
    streamlines_file, reference_file, dataset_name, nb_points,
):
    """ Process bundles.

    Args:
        bundles:
        reference:
        nb_points:

    Returns:
        sfts, scores

    """
    print('Computing size of dataset.')
    nb = get_number_of_streamlines(
            expanduser(streamlines_file), expanduser(reference_file))
    print('Datasets will have {} streamlines'.format(nb))
    print('Writing streamlines to datasets.')

    idices = np.arange(nb)
    np.random.shuffle(idices)

    train_sep = int(nb * 0.70)
    valid_sep = int((nb - train_sep) * 0.66)
    train_idices, rem_idices = idices[:train_sep], idices[train_sep:]
    valid_idices, test_idices = rem_idices[:valid_sep], rem_idices[valid_sep:]

    # Clean existing processed files
    train_dataset_file = "{}_train.hdf5".format(dataset_name)

    # Initialize database
    with h5py.File(train_dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2
        hdf_file.attrs['nb_points'] = nb_points

        ps = load_streamlines(streamlines_file, 'same')
        print('Mean score', np.mean(
            ps[train_idices].data_per_streamline['score']))
        process_bundle(
            ps[train_idices], hdf_file,
            'same', nb_points)

    print("Saved dataset : {}".format(train_dataset_file))

    valid_dataset_file = "{}_valid.hdf5".format(dataset_name)
    # Initialize database
    with h5py.File(valid_dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2
        hdf_file.attrs['nb_points'] = nb_points

        ps = load_streamlines(streamlines_file, 'same')
        print('Mean score', np.mean(
            ps[valid_idices].data_per_streamline['score']))
        process_bundle(
            ps[valid_idices], hdf_file,
            'same', nb_points)

    print("Saved dataset : {}".format(valid_dataset_file))

    test_dataset_file = "{}_test.hdf5".format(dataset_name)
    # Initialize database
    with h5py.File(test_dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2
        hdf_file.attrs['nb_points'] = nb_points

        ps = load_streamlines(streamlines_file, 'same')
        print('Mean score', np.mean(
            ps[test_idices].data_per_streamline['score']))
        process_bundle(
            ps[test_idices], hdf_file,
            'same', nb_points)

    print("Saved dataset : {}".format(test_dataset_file))


def get_number_of_streamlines(
    f, reference
):
    ps = load_streamlines(f, reference)

    return len(ps)


def process_bundle(
    ps, hdf_subject, reference, nb_points
):
    add_streamlines_to_hdf5(hdf_subject, ps, nb_points)


def load_streamlines(
    streamlines_file: str,
    reference,
    nb_points: int = None,
):
    """

    Args:
        streamlines_file:
        reference:
        nb_points:

    Returns:
        sft:

    """

    sft = load_tractogram(expanduser(streamlines_file), 'same', bbox_valid_check=False)
    sft.to_corner()
    sft.to_vox()

    return sft


def add_streamlines_to_hdf5(hdf_subject, sft, nb_points):
    """
    Add streamlines to HDF5

    Copied from: https://github.com/scil-vital/dwi_ml/blob/df0af9296408a7337a892577945e9e4455ce3c67/dwi_ml/data/hdf5/hdf5_creation.py#L490  # noqa E405

    Args:
        hdf_subject:
        sft:
        scores:

    """

    total = len(sft)

    if 'streamlines' not in hdf_subject:

        streamlines_group = hdf_subject.create_group('streamlines')
        streamlines = set_number_of_points(sft.streamlines, nb_points)
        streamlines = np.asarray(streamlines)

        # The hdf5 can only store numpy arrays (it is actually the
        # reason why it can fetch only precise streamlines from
        # their ID). We need to deconstruct the sft and store all
        # its data separately to allow reconstructing it later.
        # (a, d, vs, vo) = sft.space_attributes
        # streamlines_group.attrs['space'] = str(sft.space)
        # streamlines_group.attrs['affine'] = a
        # streamlines_group.attrs['origin'] = str(sft.origin)
        # streamlines_group.attrs['dimensions'] = d
        # streamlines_group.attrs['voxel_sizes'] = vs
        # streamlines_group.attrs['voxel_order'] = vo

        # Accessing private Dipy values, but necessary.
        # We need to deconstruct the streamlines into arrays with
        # types recognizable by the hdf5.
        streamlines_group.create_dataset(
            'data', shape=(total, nb_points, streamlines.shape[-1]))
        # streamlines_group.create_dataset('offsets', maxshape=(None,),
        #                                  data=streamlines._offsets)
        # streamlines_group.create_dataset('lengths', maxshape=(None,),
        #                                  data=streamlines._lengths)
        streamlines_group.create_dataset(
            'scores', shape=(total))

    append_streamlines_to_hdf5(hdf_subject, sft, nb_points)


def append_streamlines_to_hdf5(hdf_subject, sft, nb_points):
    streamlines_group = hdf_subject['streamlines']
    data_group = streamlines_group['data']
    # offsets_group = streamlines_group['offsets']
    # lengths_group = streamlines_group['lengths']
    scores_group = streamlines_group['scores']

    streamlines = set_number_of_points(sft.streamlines, nb_points)
    streamlines = np.asarray(streamlines)
    scores = np.asarray(sft.data_per_streamline['score']).squeeze(-1)

    assert streamlines.shape[0] == scores.shape[0], \
        (streamlines.shape, scores.shape)

    data_group[:] = streamlines
    scores_group[:] = scores


if __name__ == "__main__":
    main()
