import os

import numpy as np

from nif.data.point_wise_data import PointWiseData


class TravelingWaveHighFreq(PointWiseData):
    """
    A class for loading and normalizing the traveling wave high frequency dataset.

    Attributes:
        n_p (int): The number of parameters.
        n_x (int): The number of input features.
        n_o (int): The number of output targets.

    Methods:
        __init__(): Initializes the class and loads the dataset.
        standard_normalize(raw_data, area_weighted=False): Normalizes the given data
            using standard normalization.
        minmax_normalize(raw_data, n_para, n_x, n_target, area_weighted=False):
            Normalizes the given data using min-max normalization.
    """

    def __init__(self):
        """
        Initializes the class and loads the traveling wave high frequency dataset.
        Calls the super class PointWiseData to store the parameter, x, and u data as
        data_raw. Uses the minmax_normalize method to normalize the data and store
        it as data, mean, and std.
        """
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        data = np.load(dir_path + "/dataset/traveling_wave_high_freq.npz")["data"]
        parameter_data = data[:, [0]]
        x_data = data[:, [1]]
        u_data = data[:, [2]]
        super(TravelingWaveHighFreq, self).__init__(parameter_data, x_data, u_data)
        self.data, self.mean, self.std = self.minmax_normalize(
            self.data_raw, n_para=self.n_p, n_x=self.n_x, n_target=1
        )


if __name__ == "__main__":
    tw = TravelingWaveHighFreq()
    print(tw.data.mean(axis=0))
    print(tw.data.std(axis=0))
    print(tw.data.max(axis=0))
    print(tw.parameter.shape)
    print(tw.x.shape)
    print(tw.u.shape)
