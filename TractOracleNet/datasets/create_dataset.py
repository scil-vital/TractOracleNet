#!/usr/bin/env python
import argparse

import h5py
import json
import numpy as np

from argparse import RawTextHelpFormatter
from glob import glob
from os.path import expanduser
from tqdm import tqdm

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points
from nibabel.streamlines import load

"""
Script to process multiple subjects into a single .hdf5 file.
"""


def generate_dataset(
    config_file: str,
    dataset_file: str,
    nb_points: int = 128,
    max_streamline_subject: int = -1
) -> None:
    """ Generate a dataset from a configuration file and save it to disk.

    Parameters:
    -----------
    config_file: str
        Path to the configuration file containing the subjects and their
        streamlines.
    dataset_file: str
        Path to the output file where the dataset will be saved.
    nb_points: int
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """
    # Initialize database
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 1
        hdf_file.attrs['nb_points'] = nb_points

        with open(config_file, "r") as conf:
            config = json.load(conf)

            add_subjects_to_hdf5(
                config, hdf_file, nb_points, max_streamline_subject)

    print("Saved dataset : {}".format(dataset_file))


def add_subjects_to_hdf5(
    config, hdf_file, nb_points=128, max_streamline_subject=-1
):
    """ Process the subjects and add them to the hdf5 file.

    Parameters
    ----------
    config: dict
        Dictionary containing the subjects and their streamlines.
    hdf_file: h5py.File
        HDF5 file to save the dataset to.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """
    sub_files = []
    for subject_id in config:
        "Processing subject {}".format(subject_id),

        subject_config = config[subject_id]

        reference_anat = subject_config['reference']
        streamlines_files_list = subject_config['streamlines']

        sub_files.append((reference_anat, streamlines_files_list))

    process_subjects(sub_files, hdf_file, nb_points, max_streamline_subject)


def process_subjects(
    sub_files, hdf_subject, nb_points, max_streamline_subject
):
    """ Process the subjects and add them to the hdf5 file. First,
    the size of the dataset is computed, then the streamlines are
    loaded and assigned a random index so that they are spread
    across the dataset. Then, the streamlines are added to the
    dataset.

    Parameters
    ----------
    sub_files: list
        List of tuples containing the reference anatomy and the
        streamlines files for each subject.
    hdf_subject: h5py.File
        HDF5 file to save the dataset to.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """

    total = 0
    idx = 0
    max_strml = max_streamline_subject

    print('Computing size of dataset.')
    for anat, strm_files in tqdm(sub_files):
        streamlines_files = glob(expanduser(strm_files[0]))
        for bundle in streamlines_files:
            len_p = load(expanduser(bundle), expanduser(anat), lazy_load=True)
            total += min(max_strml, len(len_p.streamlines))

    print('Dataset will have {} streamlines'.format(total))
    print('Writing streamlines to dataset.')

    # Randomize the order of the streamlines
    idices = np.arange(total)
    np.random.shuffle(idices)

    # Add the streamlines to the dataset
    for anat, strm_files in tqdm(sub_files):
        streamlines_files = glob(expanduser(strm_files[0]))
        for bundle in streamlines_files:
            # Load the streamlines
            ps = load_streamlines(bundle, expanduser(anat))
            # Randomize the order of the streamlines
            ps_idices = np.random.choice(len(ps), min(max_strml, len(ps)),
                                         replace=False)
            print('Processing {}'.format(bundle))
            # Get the indices to use
            idx = idices[:len(ps_idices)]
            # Add the streamlines to the dataset
            add_streamlines_to_hdf5(
                hdf_subject, ps[ps_idices], nb_points, total, idx)
            # Remove the indices that have been used
            idices = idices[len(ps_idices):]


def load_streamlines(
    streamlines_file: str,
    reference,
):
    """ Load the streamlines from a file and make sure they are in
    voxel space and aligned with the corner of the voxels.

    Parameters
    ----------
    streamlines_file: str
        Path to the file containing the streamlines.
    reference: str
        Path to the reference anatomy file.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    """

    sft = load_tractogram(streamlines_file, reference, bbox_valid_check=False)
    sft.to_corner()
    sft.to_vox()

    return sft


def add_streamlines_to_hdf5(hdf_subject, sft, nb_points, total, idx):
    """ Add the streamlines to the hdf5 file.

    Parameters
    ----------
    hdf_subject: h5py.File
        HDF5 file to save the dataset to.
    sft: nib.streamlines.tractogram.Tractogram
        Streamlines to add to the dataset.
    nb_points: int, optional
        Number of points to resample the streamlines to
    total: int
        Total number of streamlines in the dataset
    idx: list
        List of positions to store the streamlines
    """

    # Get the scores and the streamlines
    scores = np.asarray(sft.data_per_streamline['score']).squeeze(-1)
    # Resample the streamlines
    streamlines = set_number_of_points(sft.streamlines, nb_points)
    streamlines = np.asarray(streamlines)

    # Create the dataset if it does not exist
    if 'streamlines' not in hdf_subject:
        # Create the group
        streamlines_group = hdf_subject.create_group('streamlines')
        # Set the number of points
        streamlines = np.asarray(streamlines)
        # 'data' will contain the streamlines
        streamlines_group.create_dataset(
            'data', shape=(total, nb_points, streamlines.shape[-1]))
        # 'scores' will contain the scores
        streamlines_group.create_dataset('scores', shape=(total))

    streamlines_group = hdf_subject['streamlines']
    data_group = streamlines_group['data']
    scores_group = streamlines_group['scores']

    for i, st, sc in zip(idx, streamlines, scores):
        data_group[i] = st
        scores_group[i] = sc


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
    parser.add_argument('--max_streamline_subject', type=int, default=-1,
                        help='Maximum number of streamlines per subject. '
                             'Default is -1, meaning all streamlines are '
                             'used.')

    arguments = parser.parse_args()

    return arguments


def main():
    """ Parse args, generate dataset and save it on disk """
    args = parse_args()

    generate_dataset(config_file=args.config_file,
                     output=args.output,
                     nb_points=args.nb_points,
                     max_streamline_subject=args.max_streamline_subject)


if __name__ == "__main__":
    main()
