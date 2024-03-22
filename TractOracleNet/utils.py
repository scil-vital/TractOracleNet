import numpy as np
import torch

from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import set_number_of_points
from scilpy.viz.utils import get_colormap


def get_data(sft, device):
    sft.to_vox()
    sft.to_corner()

    resampled_streamlines = set_number_of_points(sft.streamlines, 128)
    # Compute streamline features as the directions between points
    dirs = np.diff(resampled_streamlines, axis=1)

    with torch.no_grad():
        data = torch.as_tensor(
            dirs, dtype=torch.float, device=device)

    return data


def save_filtered_streamlines(sft, scores, out_tractogram, dense=False):
    """ Save the filtered streamlines with the scores as colors.

    Parameters
    ----------
    sft : StatefulTractogram
        The input tractogram.
    scores : array
        The scores for each streamline.
    out_tractogram : str
        The output tractogram.
    dense : bool, optional
        If True, the scores are dense and will be used as is.
        If False, the scores are per streamline and will be repeated for each
        point of the streamline.

    """

    cmap = get_colormap('jet')

    if dense:
        tmp = [np.squeeze(scores[s]) for s in
               range(len(sft))]
        data = np.hstack(tmp)
    else:
        data = np.squeeze(scores)

    color = cmap(data)[:, 0:3] * 255
    if not dense:
        tmp = [np.tile([color[i][0], color[i][1], color[i][2]],
                       (len(sft.streamlines[i]), 1))
               for i in range(len(sft.streamlines))]
        sft.data_per_point['color'] = tmp
    else:
        sft.data_per_point['color'] = sft.streamlines
        sft.data_per_point['color']._data = color

    save_tractogram(sft, out_tractogram)
