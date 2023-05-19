import h5py
import numpy as np

from collections import defaultdict

from nibabel.streamlines import Tractogram
from torch.utils.data import Dataset

import SubjectData


class StreamlineDataset(Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(
        self, file_path: str, dataset_split: str, noise=0.0, device=None
    ):
        """
        Args:
        """
        self.file_path = file_path
        self.split = dataset_split
        self.noise = noise
        with h5py.File(self.file_path, 'r') as f:
            self.subject_list = list(f[dataset_split].keys())
            self.indexes, self.rev_indexes, self.lengths = \
                self._build_indexes(f, dataset_split)
            self.state_size = self._compute_state_size(f)

    def _build_indexes(self, dataset_file, split):
        """
        """
        print('Building indexes')
        set_list = list()
        lengths = []
        rev_index = defaultdict(list)

        split_set = dataset_file[split]
        for subject in list(split_set.keys()):
            if subject != 'transitions':
                streamlines = SubjectData.from_hdf_subject(
                    split_set, subject).sft.streamlines
                for i in range(len(streamlines)):
                    k = (subject, i)
                    rev_index[subject].append((len(set_list), i))

                    set_list.append(k)
                lengths.extend(streamlines._lengths)

        print('Done')
        return set_list, rev_index, lengths

    @property
    def archives(self):
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')
        return self.f

    def get_one_input(self):

        state_0, *_ = self[0]
        self.f.close()
        del self.f
        return state_0[0]

    def __getitem__(self, index):
        """This method loads, transforms and returns slice corresponding to the
        corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)
        """
        # return index

        # Map streamline total index -> subject.streamline_id
        subject, strml_idx = self.indexes[index]
        f = self.archives[self.split]
        subject_data = SubjectData.from_hdf_subject(f, subject)
        sft = subject_data.sft.as_sft(strml_idx)
        score = sft.data_per_streamline['scores'][0]
        sft.to_vox()
        streamline = sft.streamlines[0]

        if self.noise > 0.0:
            streamline = streamline + np.random.normal(
                loc=0.0, scale=self.noise, size=streamline.shape)

        return streamline, score

    def __len__(self):
        """
        return the length of the dataset
        """
        return int(len(self.indexes))

    def render(
        self,
        streamline
    ):
        """ Debug function

        Parameters:
        -----------
        tractogram: Tractogram, optional
            Object containing the streamlines and seeds
        path: str, optional
            If set, save the image at the specified location instead
            of displaying directly
        """
        from fury import window, actor
        # Might be rendering from outside the environment
        tractogram = Tractogram(
            streamlines=streamline,
            data_per_streamline={
                'seeds': streamline[:, 0, :]
            })

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines)
        scene.add(stream_actor)
        scene.reset_camera_tight(0.95)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
