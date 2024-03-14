import lightning.pytorch as pl

from torch.utils.data import (
    BatchSampler, DataLoader, SequentialSampler)

from TractOracle.datasets.StreamlineBatchDataset import StreamlineBatchDataset
from TractOracle.datasets.utils import WeakShuffleSampler


class StreamlineDataModule(pl.LightningDataModule):
    """ Data module for the streamline dataset. This module is used to
    load the data and create the dataloaders for the training, validation
    and test sets.

    A custom sampler is used to shuffle the data in the training set
    while keeping the batches consistent.
    """

    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int = 1024,
        num_workers: int = 20,
    ):
        """ Initialize the data module with the paths to the training,
        validation and test files. The batch size and number of workers
        for the dataloaders can also be set.

        Parameters:
        -----------
        train_file: str
            Path to the hdf5 file containing the training set
        val_file: str
            Path to the hdf5 file containing the validation set
        test_file: str
            Path to the hdf5 file containing the test set
        batch_size: int, optional
            Size of the batches to use for the dataloaders
        num_workers: int, optional
            Number of workers to use for the dataloaders
        """

        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_loader_kwargs = {
            'num_workers': self.num_workers,
            'prefetch_factor': 8,
            'persistent_workers': False,
            'pin_memory': True,
        }

    def prepare_data(self):
        # pass ?
        pass

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            self.streamline_train = StreamlineBatchDataset(
                self.train_file)

            self.streamline_val = StreamlineBatchDataset(
                self.val_file)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.streamline_test = StreamlineBatchDataset(
                self.test_file, noise=0.0, flip_p=0.0)

    def train_dataloader(self):
        """ Create the dataloader for the training set
        """
        sampler = BatchSampler(WeakShuffleSampler(
            self.streamline_train, self.batch_size), self.batch_size,
            drop_last=True)

        return DataLoader(
            self.streamline_train,
            sampler=sampler,
            **self.data_loader_kwargs)

    def val_dataloader(self):
        """ Create the dataloader for the validation set
        """
        sampler = BatchSampler(SequentialSampler(
            self.streamline_val), self.batch_size,
            drop_last=True)
        return DataLoader(
            self.streamline_val,
            sampler=sampler,
            **self.data_loader_kwargs)

    def test_dataloader(self):
        """ Create the dataloader for the test set
        """
        sampler = BatchSampler(SequentialSampler(
            self.streamline_test), self.batch_size,
            drop_last=False)
        return DataLoader(
            self.streamline_test,
            batch_size=None,
            sampler=sampler,
            **self.data_loader_kwargs)

    def predict_dataloader(self):
        pass
