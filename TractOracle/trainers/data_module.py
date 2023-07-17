import numpy as np
import lightning.pytorch as pl
from torch.utils.data import (
    DataLoader, BatchSampler, Sampler, SequentialSampler, Subset)

from TractOracle.datasets.StreamlineDataset import StreamlineDataset


class WeakShuffleSampler(Sampler):
    """ Weak shuffling inspired by https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc  # noqa E501

    Weakly shuffles by return batched ids in a random way, so that
    batches are not encountered in the same order every epoch. Adds
    randomness by adding a "starting index" which shifts the indices,
    so that every batch gets different data each epoch. "Neighboring"
    data may be put in the same batch still.

    Ideally, the dataset would also be pre-shuffled on disk.

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


class StreamlineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        test_file: str,
        batch_size: int = 1024,
        num_workers: int = 8,
        valid_pct=0.2,
        total_pct=1.,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_pct = valid_pct
        self.total_pct = total_pct

        self.data_loader_kwargs = {
            'num_workers': self.num_workers,
            'persistent_workers': False,
            'pin_memory': True,
        }

    def prepare_data(self):
        # pass ?
        pass

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            streamline_train_full = StreamlineDataset(
                self.train_file)

            len_train = len(streamline_train_full)

            train_split, valid_split = (int(len_train*(1-self.valid_pct)),
                                        int(len_train*self.valid_pct))
            if train_split + valid_split != len_train:
                train_split += 1

            self.streamline_train, self.streamline_val = \
                Subset(streamline_train_full, range(train_split)), \
                Subset(streamline_train_full, range(
                    train_split, train_split + valid_split))

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.streamline_test = StreamlineDataset(
                self.test_file)

        if stage == "predict":
            self.streamline_test = StreamlineDataset(
                self.test_file)

    def train_dataloader(self):
        sampler = BatchSampler(WeakShuffleSampler(
            self.streamline_train, self.batch_size), self.batch_size,
            drop_last=True)
        return DataLoader(
            self.streamline_train,
            sampler=sampler,
            **self.data_loader_kwargs)

    def val_dataloader(self):
        # WARNING: Using the custom sampler makes lightning skip
        # the validation run. I don't know why.
        sampler = BatchSampler(SequentialSampler(
            self.streamline_val), self.batch_size,
            drop_last=True)
        return DataLoader(
            self.streamline_val,
            sampler=sampler,
            **self.data_loader_kwargs)

    def test_dataloader(self):
        sampler = BatchSampler(SequentialSampler(
            self.streamline_test), self.batch_size,
            drop_last=False)
        return DataLoader(
            self.streamline_test,
            batch_size=None,
            sampler=sampler,
            **self.data_loader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.streamline_test,
                          **self.val_data_loader_kwargs)
