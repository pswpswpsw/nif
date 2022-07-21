import numpy as np


class PointWiseData(object):
    def __init__(self, parameter_data, x_data, u_data, sample_weight=None):
        if sample_weight is not None:
            self.data_raw = np.hstack([parameter_data, x_data, u_data, sample_weight])
        else:
            self.data_raw = np.hstack([parameter_data, x_data, u_data])
        self.data = None
        self.sample_weight = None
        self.n_p = parameter_data.shape[-1]
        self.n_x = x_data.shape[-1]
        self.n_o = u_data.shape[-1]

    @property
    def parameter(self):
        return self.data[:, : self.n_p]

    @property
    def x(self):
        return self.data[:, self.n_p : self.n_p + self.n_x]

    @property
    def u(self):
        return self.data[:, self.n_p + self.n_x : self.n_p + self.n_x + self.n_o]

    @staticmethod
    def standard_normalize(raw_data, area_weighted=False):
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        if area_weighted:
            mean[-1] = 0.0
            std[-1] = np.mean(raw_data[:, -1])
            normalized_data = (raw_data - mean) / std
            return normalized_data[:, :-1], mean, std, normalized_data[:, -1]
        else:
            normalized_data = (raw_data - mean) / std
            return normalized_data, mean, std

    @staticmethod
    def minmax_normalize(raw_data, n_para, n_x, n_target, area_weighted=False):
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        for i in range(n_para + n_x):
            mean[i] = 0.5 * (np.min(raw_data[:, i]) + np.max(raw_data[:, i]))
            std[i] = 0.5 * (-np.min(raw_data[:, i]) + np.max(raw_data[:, i]))

        # also we normalize the output target to make sure the maximal is most 1
        for j in range(n_para + n_x, n_para + n_x + n_target):
            std[j] = np.max(np.abs(raw_data[:, j]))

        if area_weighted:
            # for area, simply take the mean as std for normalize
            mean[-1] = 0.0
            std[-1] = np.mean(raw_data[:, -1])
            normalized_data = (raw_data - mean) / std
            return normalized_data[:, :-1], mean, std, normalized_data[:, -1]
        else:
            normalized_data = (raw_data - mean) / std
            return normalized_data, mean, std
