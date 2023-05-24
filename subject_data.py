import numpy as np

from dwi_ml.data.dataset.streamline_containers import LazySFTData


class SubjectData(object):
    """
    Tractography-related data (input information, tracking mask, peaks and
    streamlines)
    """

    def __init__(
        self,
        subject_id: str,
        sft=None,
        streamlines=None,
        scores=None,
    ):
        self.subject_id = subject_id
        self.sft = sft
        self.streamlines = streamlines
        self.scores = scores

    @classmethod
    def from_numpy_array(cls, hdf_file, subject_id):
        """ Create a SubjectData object from an HDF group object """
        hdf_subject = hdf_file[subject_id]

        streamlines = hdf_subject['streamlines']['data']
        scores = np.array(hdf_subject['streamlines']['scores'])

        return cls(subject_id, streamlines=streamlines, scores=scores)

    @classmethod
    def from_hdf_subject(cls, hdf_file, subject_id):
        """ Create a SubjectData object from an HDF group object """
        hdf_subject = hdf_file[subject_id]

        sft = LazySFTData.init_from_hdf_info(
            hdf_subject['streamlines'])
        scores = np.array(hdf_subject['streamlines']['scores'])

        return cls(subject_id, sft=sft, scores=scores)
