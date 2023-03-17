import os

import numpy as np

from nif.data.point_wise_data import PointWiseData


class CylinderFlow(PointWiseData):
    """
    A class representing the cylinder flow dataset.

    Inherits from the PointWiseData class.

    Attributes:
    - data (numpy.ndarray): A numpy array containing the normalized data.
    - mean (numpy.ndarray): A numpy array containing the mean values used for normalization.
    - std (numpy.ndarray): A numpy array containing the standard deviation values used for normalization.
    - sample_weight (numpy.ndarray): A numpy array containing the weights assigned to each data sample.
    - n_p (int): The number of parameters in the data.
    - n_x (int): The number of inputs (features) in the data.
    - n_o (int): The number of outputs (targets) in the data.

    """

    def __init__(self):
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        data = np.load(dir_path + "/dataset/cylinderflow.npz")["data"]
        parameter_data = data[:, [0]]
        x_data = data[:, [1, 2]]
        u_data = data[:, [3, 4]]
        sample_weight = data[:, -1:]
        super(CylinderFlow, self).__init__(
            parameter_data, x_data, u_data, sample_weight
        )
        self.data, self.mean, self.std, self.sample_weight = self.minmax_normalize(
            self.data_raw, n_para=self.n_p, n_x=self.n_x, n_target=2, area_weighted=True
        )


if __name__ == "__main__":
    tw = CylinderFlow()
    print(tw.mean)
    print(tw.std)
    print("")
    print("normalized")
    print(tw.data.mean(axis=0))
    print(tw.data.std(axis=0))
    print("")
    print(tw.parameter.shape)
    print(tw.x.shape)
    print(tw.u.shape)
    print(tw.sample_weight.shape)
