import torch

from TractOracle.models.autoencoder import AutoencoderOracle
from TractOracle.models.feed_forward import FeedForwardOracle
from TractOracle.models.transformer import TransformerOracle


def get_model(checkpoint_file):
    """ Get the model from a checkpoint. """

    # Load the model's hyper and actual params from a saved checkpoint
    checkpoint = torch.load(checkpoint_file)

    # The model's class is saved in hparams
    models = {
        'AutoencoderOracle': AutoencoderOracle,
        'FeedForwardOracle': FeedForwardOracle,
        'TransformerOracle': TransformerOracle
    }

    hyper_parameters = checkpoint["hyper_parameters"]
    # Load it from the checkpoint
    model = models[hyper_parameters[
        'name']].load_from_checkpoint(checkpoint_file)
    # Put the model in eval mode to fix dropout and other stuff
    model.eval()

    return model
