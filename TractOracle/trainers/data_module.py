import lightning.pytorch as pl
import numpy as np

from torch.utils.data import (
    BatchSampler, DataLoader, SequentialSampler, Subset, random_split)


from TractOracle.datasets.StreamlineDataset import StreamlineDataset
from TractOracle.datasets.StreamlineBatchDataset import StreamlineBatchDataset
from TractOracle.datasets.utils import WeakShuffleSampler


class StreamlineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        test_file: str,
        batch_size: int = 1024,
        num_workers: int = 30,
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

        self.batch_dataset = True

        self.data_loader_kwargs = {
            'num_workers': self.num_workers,
            'prefetch_factor': 8,
            'persistent_workers': False,
            'pin_memory': True,
        }
        if not self.batch_dataset:
            self.data_loader_kwargs.update({
                'batch_size': self.batch_size
            })

    def prepare_data(self):
        # pass ?
        pass

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            if self.batch_dataset:

                streamline_train_full = StreamlineBatchDataset(
                    self.train_file)

                all_indices = np.arange(len(streamline_train_full))
                train_size = int(
                    len(streamline_train_full) * (1 - self.valid_pct))
                train_indices = all_indices[:train_size]
                val_indices = all_indices[train_size:]
                self.streamline_train, self.streamline_val = \
                    (Subset(streamline_train_full, train_indices),
                     Subset(streamline_train_full, val_indices))
            else:

                streamline_train_full = StreamlineDataset(
                    self.train_file)

                self.streamline_train, self.streamline_val = random_split(
                    streamline_train_full, [1-self.valid_pct, self.valid_pct])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.streamline_test = StreamlineDataset(
                self.test_file)

        if stage == "predict":
            self.streamline_test = StreamlineDataset(
                self.test_file)

    def train_dataloader(self):
        sampler = None
        if self.batch_dataset:
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
        sampler = None
        if self.batch_dataset:
            sampler = BatchSampler(SequentialSampler(
                self.streamline_val), self.batch_size,
                drop_last=True)
        return DataLoader(
            self.streamline_val,
            sampler=sampler,
            **self.data_loader_kwargs)

    def test_dataloader(self):
        sampler = None
        if self.batch_dataset:
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
