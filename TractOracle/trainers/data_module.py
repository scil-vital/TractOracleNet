import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

from TractOracle.datasets.StreamlineDataset import StreamlineDataset


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

        self.train_data_loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'persistent_workers': False,
            'pin_memory': False,
            'shuffle': True,
        }

        self.val_data_loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'persistent_workers': False,
            'pin_memory': False,
            'shuffle': False,
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

            self.streamline_train, self.streamline_val = random_split(
                streamline_train_full, [train_split, valid_split])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.streamline_test = StreamlineDataset(
                self.test_file)

        if stage == "predict":
            self.streamline_test = StreamlineDataset(
                self.test_file)

    def train_dataloader(self):
        return DataLoader(
            self.streamline_train,
            **self.train_data_loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.streamline_val,
                          **self.val_data_loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.streamline_test,
                          **self.val_data_loader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.streamline_test,
                          **self.val_data_loader_kwargs)
