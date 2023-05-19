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
        input_dv=None,
        peaks=None,
        wm=None,
        gm=None,
        csf=None,
        include=None,
        exclude=None,
        interface=None,
        sft=None,
        rewards=None,
        states=None
    ):
        self.subject_id = subject_id
        self.input_dv = input_dv
        self.peaks = peaks
        self.wm = wm
        self.gm = gm
        self.csf = csf
        self.include = include
        self.exclude = exclude
        self.interface = interface
        self.rewards = rewards
        self.states = states
        self.sft = sft

    @classmethod
    def from_hdf_subject(cls, hdf_file, subject_id):
        """ Create a SubjectData object from an HDF group object """
        hdf_subject = hdf_file[subject_id]
        input_dv = MRIDataVolume.from_hdf_group(hdf_subject, 'input_volume')

        peaks = MRIDataVolume.from_hdf_group(hdf_subject, 'peaks_volume')
        wm = MRIDataVolume.from_hdf_group(hdf_subject, 'wm_volume')
        gm = MRIDataVolume.from_hdf_group(hdf_subject, 'gm_volume')
        csf = MRIDataVolume.from_hdf_group(
            hdf_subject, 'csf_volume', 'wm_volume')
        include = MRIDataVolume.from_hdf_group(
            hdf_subject, 'include_volume', 'wm_volume')
        exclude = MRIDataVolume.from_hdf_group(
            hdf_subject, 'exclude_volume', 'wm_volume')
        interface = MRIDataVolume.from_hdf_group(
            hdf_subject, 'interface_volume', 'wm_volume')

        states = None
        sft = None
        rewards = None
        if 'streamlines' in hdf_subject:
            sft = LazySFTData.init_from_hdf_info(
                hdf_subject['streamlines'])
            rewards = np.array(hdf_subject['streamlines']['rewards'])

        return cls(
            subject_id, input_dv=input_dv, wm=wm, gm=gm, csf=csf,
            include=include, exclude=exclude, interface=interface,
            peaks=peaks, sft=sft, rewards=rewards, states=states)
