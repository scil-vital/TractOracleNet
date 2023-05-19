#!/usr/bin/env python
import argparse

import h5py
import json
import nibabel as nib
import numpy as np

from argparse import RawTextHelpFormatter
from os.path import join

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import set_number_of_points

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'  # much faster on cpu #TODO?

"""
Script to process multiple subjects into a single .hdf5 file.
See example configuration file.

Streamlines will be "pre-reversed" to prevent having to do it on the fly.
TODO?: Pre-compute states ?

Heavly inspired by https://github.com/scil-vital/dwi_ml/blob/master/dwi_ml/data/hdf5/hdf5_creation.py # noqa E405
But modified to suit my needs
"""


def parse_args():

    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('path', type=str,
                        help='Location of the dataset files.')
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

    generate_dataset(path=args.path,
                     config_file=args.config_file,
                     output=args.output,
                     nb_points=args.nb_points,
                     normalize=args.normalize)


def generate_dataset(
    path: str,
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

    # Clean existing processed files
    dataset_file = "{}.hdf5".format(dataset_name)

    # Initialize database
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2
        hdf_file.attrs['nb_points'] = nb_points

        with open(join(path, config_file), "r") as conf:
            config = json.load(conf)

            add_subjects_to_hdf5(
                path, config, hdf_file, "training", nb_points)

            add_subjects_to_hdf5(
                path, config, hdf_file, "validation", nb_points)

            add_subjects_to_hdf5(
                path, config, hdf_file, "testing", nb_points)

    print("Saved dataset : {}".format(dataset_file))


def add_subjects_to_hdf5(

    path, config, hdf_file, dataset_split, nb_points
):
    """

    Args:
        config:
        hdf_file:
        dataset_split:
        nb_points:

    """

    hdf_split = hdf_file.create_group(dataset_split)
    for subject_id in config[dataset_split]:
        "Processing subject {}".format(subject_id),

        subject_config = config[dataset_split][subject_id]
        hdf_subject = hdf_split.create_group(subject_id)
        add_subject_to_hdf5(path, subject_config, hdf_subject, nb_points)


def add_subject_to_hdf5(
    path, config, hdf_subject, nb_points,
):
    """

    Args:
        config:
        hdf_subject:
        nb_points:

    """

    reference_anat = config['reference']
    bundle_files = config['bundles']

    # Process subject's data
    process_subject(
        hdf_subject, path, reference_anat, bundle_files, nb_points)


def process_subject(
    hdf_subject,
    path: str,
    reference: str,
    bundles: str,
    nb_points: float,
):
    """

    Args:
        hdf_subject:
        path:
        reference:
        bundles:
        nb_points:

    """

    ref_volume = nib.load(join(path, reference))

    process_all_streamlines(
        hdf_subject, path, bundles, ref_volume, nb_points)


def process_all_streamlines(
    hdf_subject, path, bundles, reference, nb_points,
):
    """ Process bundles.

    Args:
        bundles:
        reference:
        nb_points:

    Returns:
        sfts, scores

    """
    print('Processing streamlines')

    for bundle in bundles:
        process_bundle(path, bundle, hdf_subject, reference, nb_points)


def process_bundle(
    path, f, hdf_subject, reference, score, nb_points,
):

    ps = load_streamlines(join(path, f), reference, nb_points)
    ps.to_vox()

    add_streamlines_to_hdf5(hdf_subject, ps)


def load_streamlines(
    streamlines_file: str,
    reference,
    nb_points: int,
):
    """

    Args:
        streamlines_file:
        reference:
        nb_points:

    Returns:
        sft:

    """

    sft = load_tractogram(streamlines_file, reference, bbox_valid_check=False)
    sft.to_center()
    lengths = np.asarray([len(s) for s in sft.streamlines])

    sft = StatefulTractogram.from_sft(
        set_number_of_points(sft.streamlines[lengths > 0], nb_points), sft)

    return sft


def add_volume_to_hdf5(hdf_subject, volume_img, volume_name):
    """

    Args:
        hdf_subject:
        volume_img:
        volume_name:

    """

    hdf_input_volume = hdf_subject.create_group(volume_name)
    hdf_input_volume.attrs['vox2rasmm'] = volume_img.affine
    hdf_input_volume.create_dataset('data', data=volume_img.get_fdata())


def add_streamlines_to_hdf5(hdf_subject, sft, scores):
    """
    Add streamlines to HDF5

    Copied from: https://github.com/scil-vital/dwi_ml/blob/df0af9296408a7337a892577945e9e4455ce3c67/dwi_ml/data/hdf5/hdf5_creation.py#L490  # noqa E405

    Args:
        hdf_subject:
        sft:
        scores:

    """

    if 'streamlines' not in hdf_subject:

        streamlines_group = hdf_subject.create_group('streamlines')
        streamlines = sft.streamlines
        scores = sft.data_per_streamline['scores']

        # The hdf5 can only store numpy arrays (it is actually the
        # reason why it can fetch only precise streamlines from
        # their ID). We need to deconstruct the sft and store all
        # its data separately to allow reconstructing it later.
        (a, d, vs, vo) = sft.space_attributes
        streamlines_group.attrs['space'] = str(sft.space)
        streamlines_group.attrs['affine'] = a
        streamlines_group.attrs['dimensions'] = d
        streamlines_group.attrs['voxel_sizes'] = vs
        streamlines_group.attrs['voxel_order'] = vo

        # Accessing private Dipy values, but necessary.
        # We need to deconstruct the streamlines into arrays with
        # types recognizable by the hdf5.
        streamlines_group.create_dataset(
            'data', maxshape=(None, streamlines._data.shape[-1]),
            data=streamlines._data)
        streamlines_group.create_dataset('offsets', maxshape=(None,),
                                         data=streamlines._offsets)
        streamlines_group.create_dataset('lengths', maxshape=(None,),
                                         data=streamlines._lengths)
        streamlines_group.create_dataset('scores', maxshape=(None,),
                                         data=scores)
    else:
        append_streamlines_to_hdf5(hdf_subject, sft, scores)


def append_streamlines_to_hdf5(hdf_subject, sft, scores):
    streamlines_group = hdf_subject['streamlines']
    data_group = streamlines_group['data']
    offsets_group = streamlines_group['offsets']
    lengths_group = streamlines_group['lengths']
    scores_group = streamlines_group['scores']

    streamlines = sft.streamlines
    prev_data_shape = data_group.shape
    data_group.resize(
        prev_data_shape[0] + streamlines._data.shape[0], axis=0)
    data_group[prev_data_shape[0]:] = streamlines._data

    prev_offsets_shape = offsets_group.shape
    new_offsets = streamlines._offsets + offsets_group[-1] + lengths_group[-1]
    offsets_group.resize(
        prev_offsets_shape[0] + streamlines._offsets.shape[0], axis=0)
    offsets_group[prev_offsets_shape[0]:] = new_offsets

    prev_lengths_shape = lengths_group.shape
    lengths_group.resize(
        prev_lengths_shape[0] + streamlines._lengths.shape[0], axis=0)
    lengths_group[prev_lengths_shape[0]:] = streamlines._lengths

    prev_scores_shape = scores_group.shape
    scores_group.resize(
        prev_scores_shape[0] + scores.shape[0], axis=0)
    scores_group[prev_scores_shape[0]:] = scores


if __name__ == "__main__":
    main()
