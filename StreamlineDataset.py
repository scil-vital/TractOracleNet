import h5py
import numpy as np

from nibabel.streamlines import Tractogram
from torch.utils.data import Dataset

from subject_data import SubjectData


class StreamlineDataset(Dataset):
    """
    TODO
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
        self.n_f = 0
        with h5py.File(self.file_path, 'r') as f:
            self.subject_list = list(f.keys())
            self.indexes = \
                self._build_indexes(f)
            self.input_size = self._compute_input_size()

    def _build_indexes(self, dataset_file):
        """ TODO
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
        L, P = self._get_one_input().shape
        return L * P

    @property
    def archives(self):
        """ TODO
        """
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')
        return self.f

    def __del__(self):
        self.f.close()
        print('Destructor called, File closed.')

    def _get_one_input(self):
        """ TODO
        """

        state_0, *_ = self[0]
        self.f.close()
        del self.f
        return state_0

    def __getitem__(self, index):
        """This method loads the hdf5, gets the subject and streamline id
        corresonding to the data index and returns the associated data and
        target.

        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)
        """

        # Map streamline total index -> subject.streamline_id
        subject, strml_idx = self.indexes[index]

        f = self.archives

        hdf_subject = f[subject]['streamlines']

        streamline = hdf_subject['data'][strml_idx]
        score = hdf_subject['scores'][strml_idx]

        # Add noise to streamline points for robustness
        if self.noise > 0.0:
            dtype = streamline.dtype
            streamline = streamline + np.random.normal(
                loc=0.0, scale=self.noise, size=streamline.shape).astype(dtype)

        # Flip streamline for robustness
        if np.random.random() < self.flip_p:
            streamline = np.flip(streamline, axis=0).copy()

        # Convert the streamline points to directions
        # Works really well
        dirs = np.diff(streamline, axis=0)

        return dirs, score

    def __len__(self):
        """
        Return the length of the dataset, i.e. the number
        of streamlines in the dataset.
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
