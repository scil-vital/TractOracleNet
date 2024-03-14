import h5py
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from nibabel.streamlines.array_sequence import ArraySequence
from torch.utils.data import Dataset


class StreamlineBatchDataset(Dataset):
    """ Dataset for loading streamlines from hdf5 files. The streamlines
    are loaded in batches and can be augmented with noise and flipping.

    Streamlines from the dataset are presumed to be already shuffled
    and resampled to 128 points. This is done because HDF5 access is
    slow and it takes the same time to access a single streamline or
    a slice of streamlines.
    """

    def __init__(
        self,
        file_path: str,
        noise: float = 0.1,
        flip_p: float = 0.5,
        dense: bool = True,
        partial: bool = False
    ):
        """
        Parameters:
        -----------
        file_path: str
            Path to the hdf5 file containing the streamlines
        noise: float, optional
            Standard deviation of the Gaussian noise to add to the
            streamline points
        flip_p: float, optional
            Probability of flipping the streamline
        dense: bool, optional
            If set, streamlines will be randomly cut to allow the
            model to learn how to score partial streamlines.
        partial: bool, optional
            If set, the scores will be adjusted to account for the
            partial streamlines. i.e. the score of a valid streamline
            cut in half will be 0.5.
        """
        self.file_path = file_path
        self.noise = noise
        self.flip_p = flip_p
        self.dense = dense
        self.partial = partial
        self.input_size = self._compute_input_size()

        f = self.archives
        streamlines = f['streamlines']['data']
        self.length = len(streamlines)

    def _compute_input_size(self):
        """ Compute the size of the input data
        """
        batch = self._get_one_input()
        L, P = batch.shape
        return L * P

    @property
    def archives(self):
        """ Open the hdf5 file and return the file object.
        Keep the file open until the object is deleted.
        """
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')
        return self.f

    def __del__(self):
        """ Destructor to close the hdf5 file.
        """
        if hasattr(self, 'f'):
            self.f.close()

    def _get_one_input(self):
        """ Get one input from the dataset.
        """

        state_0, *_ = self[[0, 1]]
        self.f.close()
        del self.f
        return state_0[0]

    def __getitem__(self, indices):
        """ Get a batch of streamlines and their scores. The streamlines
        are augmented with noise and flipping. The scores are adjusted
        if the streamlines are cut (if partial == True).

        Parameters:
        -----------
        indices: list
            List of indices to select the streamlines from the dataset.

        Returns:
        --------
        dirs: np.ndarray
            Array of shape (N, L, 3) containing the N sequences of
            3D directions of the streamlines. In practice, L=127.
        score: np.ndarray
            Array of shape (N,) containing the scores of the streamlines.
        """

        # Get or open the hdf5 file
        f = self.archives

        # Get the streamlines and their scores
        hdf_subject = f['streamlines']
        data = hdf_subject['data']
        scores_data = hdf_subject['scores']

        # Start and end indices. Presume that the indices are sorted
        # and sequential.
        start, end = indices[0], indices[-1] + 1

        # Handle rollover indices
        if start > end:
            # Concatenate the two parts
            batch_end = max(indices)
            batch_start = min(indices)
            streamlines = np.concatenate(
                (data[start:batch_end], data[batch_start:end]), axis=0)
            score = np.concatenate(
                (scores_data[start:batch_end], scores_data[batch_start:end]),
                axis=0)
        # Slice as usual
        else:
            streamlines = data[start:end]
            score = scores_data[start:end]

        # Flip streamline for robustness
        # Ideally, a proportion p of the streamlines should be flipped
        # not all of them.
        if np.random.random() < self.flip_p:
            streamlines = np.flip(streamlines, axis=1).copy()

        # Randomly cut the streamlines to allow the model to learn
        # how to score partial streamlines
        if self.dense:
            new_lengths = np.random.randint(3, streamlines.shape[1],
                                            size=streamlines.shape[0])
            # Adjust the score to account for the partial streamlines
            if self.partial:
                old_length = streamlines.shape[1]
                score *= new_lengths / old_length

            # Need to convert the list of arrays to an ArraySequence
            # to use set_number_of_points
            array_seq = ArraySequence([streamlines[i, :new_lengths[i]]
                                       for i in range(len(new_lengths))])
            streamlines = set_number_of_points(array_seq, 128)
            streamlines = np.asarray(streamlines)

        # Add noise to streamline points for robustness
        if self.noise > 0.0:
            dtype = streamlines.dtype
            streamlines = streamlines + np.random.normal(
                loc=0.0, scale=self.noise, size=streamlines.shape
            ).astype(dtype)

        # Compute the directions
        dirs = np.diff(streamlines, axis=1)

        return dirs, score
