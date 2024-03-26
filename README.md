# TractOracleNet


> Oracle [ awr-uh-kuhl, or- ] a person who delivers authoritative, wise, or highly regarded and influential pronouncements.


TractOracle-Net is half of [TractOracle](https://preprintcoming), a reinforcement learning system for tractography. **TractOracle-Net** is a streamline classification network which can be used to reward plausible streamlines from [TractOracle-RL](https://github.com/scil-vital/TrackToLearn) or filter streamlines in general.


## Installation

TractOracle-Net should be installed in a [virtual environment](https://virtualenv.pypa.io/en/latest/user_guide.html).

You can install the library after cloning the repo by running `install.sh`.

## Prediction

TractOracle-Net can filter tractograms using `predictor.py`:

```
usage: predictor.py [-h] [--reference REFERENCE] [--batch_size BATCH_SIZE]
                    [--threshold THRESHOLD] [--checkpoint CHECKPOINT]
                    [--nofilter | --rejected REJECTED | --dense]
                    tractogram out

 Filter a tractogram. 

positional arguments:
  tractogram            Tractogram file to score.
  out                   Output file.

options:
  -h, --help            show this help message and exit
  --reference REFERENCE
                        Reference file for tractogram (.nii.gz).For .trk, can be 'same'. Default is [same].
  --batch_size BATCH_SIZE
                        Batch size for predictions. Default is [512].
  --threshold THRESHOLD
                        Threshold score for filtering. Default is [0.5].
  --checkpoint CHECKPOINT
                        Checkpoint (.ckpt) containing hyperparameters and weights of model. Default is [model/tractoracle.ckpt].
  --nofilter            Output a tractogram containing all streamlines and scores instead of only plausible ones.
  --rejected REJECTED   Output file for invalid streamlines.
  --dense               Predict the scores of the streamlines point by point. Streamlines' endpoints should be uniformized for best visualization.
```

Streamlines will be colored according to their predicted scores (if saving a `.trk`). A pretrained model is included in `model/` and will be automatically used. If you want to use your own model, use the `--checkpoint` argument.

## Docker

TractOracle-Net is available through Docker Hub. You can pull the image by running

```
docker pull TODO
```

You can then score a tractogram by running

```
sudo docker run -v .:/workspace/${TRACTOGRAM_LOCATION} tractoracle-net predictor.py /workspace/${TRACTOGRAM_FILE} ${OUT} [...]
```

See [Docker volumes](https://docs.docker.com/storage/volumes/) for an explanation of the `-v` flag. **To use CUDA capabilities with Docker, you will need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**. You will then be able to use the `--gpus` flag. For example:

```
sudo docker run --gpus all -v .:/workspace/${TRACTOGRAM_LOCATION} tractoracle-net predictor.py /workspace/${TRACTOGRAM_FILE} ${OUT} [...]
```

## Training

You will first need to create a dataset. See `example_config` for example configuration files for your datasets. You can then run `python TractOracleNet/datasets/create_dataset.py` to create a dataset in the form of an `.hdf5` file.

```
python TractOracleNet/datasets/create_dataset.py

usage: create_dataset.py [-h] [--nb_points NB_POINTS] [--max_streamline_subject MAX_STREAMLINE_SUBJECT] config_file output

positional arguments:
  config_file           Configuration file to load subjects and their volumes.
  output                Output filename including path

options:
  -h, --help            show this help message and exit
  --nb_points NB_POINTS
                        Number of points to resample streamlines to. Default is [128].
  --max_streamline_subject MAX_STREAMLINE_SUBJECT
                        Maximum number of streamlines per subject. Default is -1, meaning all streamlines are used.
```

With your new dataset, you can then train a model using `python TractOracleNet/trainers/transformer_train.py`.

```
usage: transformer_train.py [-h] [--lr LR] [--n_head N_HEAD] [--n_layers N_LAYERS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--checkpoint CHECKPOINT]
                            path experiment id max_ep train_dataset_file val_dataset_file test_dataset_file

 Parse the arguments.
    

positional arguments:
  path                  Path to experiment
  experiment            Name of experiment.
  id                    ID of experiment.
  max_ep                Number of epochs.
  train_dataset_file    Training dataset.
  val_dataset_file      Validation dataset.
  test_dataset_file     Testing dataset.

options:
  -h, --help            show this help message and exit
  --lr LR               Learning rate.
  --n_head N_HEAD       Number of attention heads.
  --n_layers N_LAYERS   Number of encoder layers.
  --batch_size BATCH_SIZE
                        Batch size, in number of streamlines.
  --num_workers NUM_WORKERS
                        Number of workers for dataloader.
  --checkpoint CHECKPOINT
                        Path to checkpoint. If not provided, train from scratch.
```

## References

See preprint: COMING SOON

See conference paper: (hopefully) COMING SOON
