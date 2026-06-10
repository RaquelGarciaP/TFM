import h5py
import numpy as np


class Snapshot:
    def __init__(self, snapshot_file_name):

        # Open snapshot file
        self.f = h5py.File(snapshot_file_name, "r")

        # Read minimum dark matter mass in [M_sun]
        self.min_dm = (
            np.min(self.f["PartType1/Masses"]) * self.f["Units"].attrs["MsolUnit"]
        )

        # Read minimum time at which stars are formed
        # (min sft) in [years]
        # (31556926.0 sec = 1 year)
        self.min_t = (
            np.min(self.f["PartType4/StellarFormationTime"])
            * self.f["Units"].attrs["SecUnit"]
            / 31556926.0
        )

        # Time of the snapshot (= max time of the simulation) in [years]
        self.snap_time = (
            self.f["Header"].attrs["Time"]
            * self.f["Units"].attrs["SecUnit"]
            / 31556926.0
        )
