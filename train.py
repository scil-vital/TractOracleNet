#!/usr/bin/env python
import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from feedforwardmodel import FeedForwardOracle
from StreamlineDataset import StreamlineDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TractOracleTraining():
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        train_dataset_file: str,
        valid_dataset_file: str,
        test_dataset_file: str,
        lr: float,
        layers: str,
        max_ep: int,
        render: bool,
    ):
        """
        """
        self.experiment_path = path
        self.experiment = experiment
        self.name = name

        # RL parameters
        self.lr = lr
        self.max_exp = max_ep

        #  Tracking parameters
        self.train_dataset_file = train_dataset_file
        self.valid_dataset_file = valid_dataset_file
        self.test_dataset_file = test_dataset_file

    def train(
        self,
        train_dataset: StreamlineDataset,
        valid_dataset: StreamlineDataset,
        test_dataset: StreamlineDataset,
    ):
        """
        """

        # The RL training algorithm
        alg = self.get_alg()

        back_env, env = self.get_envs()
        back_test_env, test_env = self.get_test_envs()

        train_loader = DataLoader(train_dataset,
                                  num_workers=self.num_workers,
                                  # batch_size=self.batch_size,
                                  prefetch_factor=2,
                                  persistent_workers=True
                                  if self.num_workers > 0 else False,
                                  pin_memory=True)

        valid_loader = DataLoader(valid_dataset,
                                  num_workers=self.num_workers,
                                  # batch_size=self.batch_size,
                                  prefetch_factor=2,
                                  persistent_workers=True
                                  if self.num_workers > 0 else False,
                                  pin_memory=True)

        # test_loader = DataLoader(test_dataset,
        #                          num_workers=self.num_workers,
        #                          # batch_size=self.batch_size,
        #                          prefetch_factor=2,
        #                          persistent_workers=True
        #                          if self.num_workers > 0 else False,
        #                          pin_memory=True)

        def add_to_means(means, dic):
            return {k: means[k].append(dic[k]) for k in dic.keys()}

        def mean_losses(dic):
            return {k: torch.mean(torch.stack(dic[k])).cpu().numpy()
                    for k in dic.keys()}

        for epoch in range(self.max_ep):
            print("Epoch: {} of {}".format(epoch + 1, self.max_ep))
            means = defaultdict(list)

            for i, (streamlines, scores) in enumerate(tqdm(train_loader), 0):
                # transfer tensors to selected device
                streamlines, scores = \
                    streamlines.to(device, non_blocking=True), \
                    scores.to(device, non_blocking=True)

                losses = alg.update(streamlines, scores)

                add_to_means(means, losses)

            losses = mean_losses(means)
            print(losses)

        for i, (streamlines, scores) in enumerate(tqdm(valid_loader), 0):
            # transfer tensors to selected device
            streamlines, scores = \
                streamlines.to(device, non_blocking=True), \
                scores.to(device, non_blocking=True)

            pred_scores = alg.predict(streamlines, scores)

            print(np.mean(pred_scores - scores))

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        train_dataset = StreamlineDataset(
            self.train_dataset_file, 'training')
        valid_dataset = StreamlineDataset(
            self.valid_dataset_file, 'validation')
        test_dataset = StreamlineDataset(
            self.test_dataset_file, 'testing')

        # Get example state to define NN input size
        self.input_size = 128 * 3
        self.output_size = 1

        model = FeedForwardOracle(
            self.input_size, self.output_size, self.layers)

        # Start training !
        self.train(model, train_dataset, valid_dataset, test_dataset)


def main():

    experiment = TractOracleTraining()
    experiment.run()


if __name__ == "__main__":
    main()
