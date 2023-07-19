import numpy as np

from TractOracle.datasets.StreamlineDataset import StreamlineDataset


class StreamlineBatchDataset(StreamlineDataset):

    def _get_one_input(self):
        """ TODO
        """

        state_0, *_ = self[[0, 1]]
        self.f.close()
        del self.f
        return state_0[0]

    def __getitem__(self, indices):
        """ TODO

        :arg
            indices: TODO
        :return
            A tuple (input, target)
        """

        f = self.archives

        hdf_subject = f['streamlines']
        data = hdf_subject['data']
        scores_data = hdf_subject['scores']

        start, end = indices[0], indices[-1] + 1

        # Handle rollover indices
        if start > end:
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

        streamlines = np.asarray([data[i] for i in indices])

        score = np.asarray([scores_data[i] for i in indices])

        # Flip streamline for robustness
        if np.random.random() < self.flip_p:
            streamlines = np.flip(streamlines, axis=1).copy()

        # Add noise to streamline points for robustness
        if self.noise > 0.0:
            dtype = streamlines.dtype
            streamlines = streamlines + np.random.normal(
                loc=0.0, scale=self.noise, size=streamlines.shape
            ).astype(dtype)

        # Convert the streamline points to directions
        # Works really well
        dirs = np.diff(streamlines, axis=1)

        return dirs, score
