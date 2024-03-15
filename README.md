# TractOracleNet


> Oracle [ awr-uh-kuhl, or- ] a person who delivers authoritative, wise, or highly regarded and influential pronouncements.


TractOracle-Net is half of [TractOracle](https://preprintcoming), a reinforcement learning system for tractography. **TractOracle-Net** is a streamline classification network which can be used to reward plausible streamlines from __TractOracle-RL__ or filter streamlines in general.


## Installation

TractOracle-Net should be installed in a [virtual environment](https://virtualenv.pypa.io/en/latest/user_guide.html).

You can install the library after cloning the repo by running `install.sh`.

A docker container may come soon-ish maybe, maybe not.

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

## Training

Instructions coming soon.
