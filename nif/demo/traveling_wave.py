import os

import numpy as np

from nif.data.point_wise_data import PointWiseData


class TravelingWave(PointWiseData):
    """
    A class for loading and processing the traveling wave dataset.

    Inherits from the PointWiseData class, which is a base class for point-wise data processing.

    Attributes:
        mean (ndarray): Mean values of the normalized data.
        std (ndarray): Standard deviation values of the normalized data.
    """

    def __init__(self):
        """
        Loads the traveling wave dataset and initializes the object.

        The dataset is loaded from the file 'traveling_wave.npz' located in the 'dataset' directory
        relative to the current file's directory. The data is then processed by normalizing it using
        the `standard_normalize` function from the base class, PointWiseData. The normalized data is
        stored in the `data` attribute, while the mean and standard deviation values are stored in
        the `mean` and `std` attributes, respectively.
        """
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        data = np.load(dir_path + "/dataset/traveling_wave.npz")["data"]
        parameter_data = data[:, [0]]
        x_data = data[:, [1]]
        u_data = data[:, [2]]
        super(TravelingWave, self).__init__(parameter_data, x_data, u_data)
        self.data, self.mean, self.std = self.standard_normalize(self.data_raw)


if __name__ == "__main__":
    tw = TravelingWave()
    print(tw.data.mean(axis=0))
    print(tw.data.std(axis=0))
    print(tw.parameter.shape)
    print(tw.x.shape)
    print(tw.u.shape)
