import h5py
import numpy as np

from nibabel.streamlines import Tractogram
from torch.utils.data import Dataset

from subject_data import SubjectData


class StreamlineDataset(Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(
        self,
        file_path: str,
        noise: float = 0.1,
        flip_p: float = 0.5,
        device=None
    ):
        """
        Args:
        """
        self.file_path = file_path
        self.noise = noise
        self.flip_p = flip_p
        with h5py.File(self.file_path, 'r') as f:
            self.subject_list = list(f.keys())
            self.indexes = \
                self._build_indexes(f)
            self.input_size = self._compute_input_size()

    def _build_indexes(self, dataset_file):
        """
        """
        print('Building indexes')
        set_list = list()

        split_set = dataset_file
        for subject in list(split_set.keys()):

            streamlines = SubjectData.from_numpy_array(
                split_set, subject).streamlines
            for i in range(len(streamlines)):
                k = (subject, i)

                set_list.append(k)

        return set_list

    def _compute_input_size(self):
        L, P = self.get_one_input().shape
        return L * P

    @property
    def archives(self):
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')
        return self.f

    def get_one_input(self):

        state_0, *_ = self[0]
        self.f.close()
        del self.f
        return state_0

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
        f = self.archives
        # subject_data = SubjectData.from_hdf_subject(f, subject)

        subject_data = SubjectData.from_numpy_array(
            f, subject)

        streamline = subject_data.streamlines[strml_idx]
        score = subject_data.scores[strml_idx]

        if self.noise > 0.0:
            dtype = streamline.dtype
            streamline = streamline + np.random.normal(
                loc=0.0, scale=self.noise, size=streamline.shape).astype(dtype)
            # print(streamline.shape)
        if np.random.random() < self.flip_p:
            streamline = np.flip(streamline, axis=0).copy()
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
