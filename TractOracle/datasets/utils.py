import numpy as np

from torch.utils.data import Sampler


class WeakShuffleSampler(Sampler):
    """ Weak shuffling inspired by https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc  # noqa E501

    Weakly shuffles by return batched ids in a random way, so that
    batches are not encountered in the same order every epoch. Adds
    randomness by adding a "starting index" which shifts the indices,
    so that every batch gets different data each epoch. "Neighboring"
    data may be put in the same batch still.

    Presumes that the dataset is already shuffled on disk.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_length = len(dataset)

        self.n_batches = self.dataset_length // self.batch_size

    def __len__(self):
        return self.dataset_length

    def __iter__(self):

        # Get the shifting index
        starting_idx = np.random.randint(len(self))

        # Shuffle the batches
        batch_ids = np.random.permutation(int(self.n_batches))
        for id in batch_ids:
            # Batch slice beginning
            beg = (starting_idx + (id * self.batch_size)) % len(self)
            # Batch slice end
            end = (starting_idx + ((id + 1) * self.batch_size)) % len(self)
            # Indices are rolling over
            if beg > end:
                # Concatenate indices at the end of the dataset and the
                # beginning in the same batch
                idx = np.concatenate(
                    (np.arange(beg, len(self)), np.arange(end)))
            else:
                idx = range(beg, end)

            # Weirdly enough, precomputing the indices seems slower.
            for index in idx:
                yield int(index)
