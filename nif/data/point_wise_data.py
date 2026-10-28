import numpy as np


class PointWiseData(object):
    """Represents point-wise data.

    Attributes:
        data_raw (numpy.ndarray): Raw data.
        data (numpy.ndarray): Normalized data.
        sample_weight (numpy.ndarray): Sample weights.
        n_p (int): Number of parameter features.
        n_x (int): Number of state features.
        n_o (int): Number of output features.
    """

    def __init__(self, parameter_data, x_data, u_data, sample_weight=None):
        """Initializes a new instance of the PointWiseData class.

        Args:
            parameter_data (numpy.ndarray): Parameter data.
            x_data (numpy.ndarray): State data.
            u_data (numpy.ndarray): Output data.
            sample_weight (numpy.ndarray): Sample weights. Defaults to None.
        """
        arrays = [np.asarray(a) for a in (parameter_data, x_data, u_data)]
        if any(a.ndim != 2 for a in arrays) or len({a.shape[0] for a in arrays}) != 1:
            raise ValueError(
                "parameter_data, x_data and u_data must be matrices with matching rows"
            )
        parameter_data, x_data, u_data = arrays
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).reshape(-1, 1)
            if sample_weight.shape[0] != parameter_data.shape[0]:
                raise ValueError("sample weights must have one entry per row")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
                raise ValueError("sample weights must be finite and nonnegative")
        if any(not np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("data must be finite")
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
        """Returns the parameter data."""
        return self.data[:, : self.n_p]

    @property
    def x(self):
        """Returns the state data."""
        return self.data[:, self.n_p : self.n_p + self.n_x]

    @property
    def u(self):
        """Returns the output data."""
        return self.data[:, self.n_p + self.n_x : self.n_p + self.n_x + self.n_o]

    @staticmethod
    def standard_normalize(raw_data, area_weighted=False):
        """Performs standard normalization on raw data.

        Args:
            raw_data (numpy.ndarray): Raw data.
            area_weighted (bool): Whether to perform area weighting. Defaults to False.

        Returns:
            numpy.ndarray: Normalized data.
            numpy.ndarray: Mean of raw data.
            numpy.ndarray: Standard deviation of raw data.
            numpy.ndarray: Normalized sample weights.
        """
        raw_data = PointWiseData._validate_data(raw_data, area_weighted)
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        if area_weighted:
            mean[-1] = 0.0
            std[-1] = np.mean(raw_data[:, -1])
            PointWiseData._validate_scale(std)
            normalized_data = (raw_data - mean) / std
            return (
                normalized_data[:, :-1],
                mean,
                std,
                normalized_data[:, -1],
            )
        else:
            PointWiseData._validate_scale(std)
            normalized_data = (raw_data - mean) / std
            return normalized_data, mean, std

    @staticmethod
    def minmax_normalize(raw_data, n_para, n_x, n_target, area_weighted=False):
        """Performs min-max normalization on raw data.

        Args:
            raw_data (numpy.ndarray): Raw data.
            n_para (int): Number of parameter features.
            n_x (int): Number of state features.
            n_target (int): Number of output features.
            area_weighted (bool): Whether to perform area weighting. Defaults to False.

        Returns:
            numpy.ndarray: Normalized data.
            numpy.ndarray: Mean of raw data.
            numpy.ndarray: Standard deviation of raw data.
        """
        raw_data = PointWiseData._validate_data(raw_data, area_weighted)
        if any(
            not isinstance(n, (int, np.integer)) or n < 0
            for n in (n_para, n_x, n_target)
        ):
            raise ValueError("feature counts must be nonnegative integers")
        if n_para + n_x + n_target + int(area_weighted) != raw_data.shape[1]:
            raise ValueError("feature counts do not match the data columns")
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
            PointWiseData._validate_scale(std)
            normalized_data = (raw_data - mean) / std
            return normalized_data[:, :-1], mean, std, normalized_data[:, -1]
        else:
            PointWiseData._validate_scale(std)
            normalized_data = (raw_data - mean) / std
            return normalized_data, mean, std

    @staticmethod
    def _validate_data(raw_data, area_weighted):
        data = np.asarray(raw_data)
        if data.ndim != 2 or not data.shape[0] or not data.shape[1]:
            raise ValueError("data must be a nonempty two-dimensional array")
        if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
            raise ValueError("data must contain finite numeric values")
        if area_weighted and (np.any(data[:, -1] < 0) or data[:, -1].mean() <= 0):
            raise ValueError("area weights must be nonnegative with positive mean")
        return data

    @staticmethod
    def _validate_scale(scale):
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("normalization has zero range or constant columns")

    @staticmethod
    def inverse_normalize(values, mean, scale):
        """Restore physical units: normalized values * scale + mean.

        Pass the output-column slices of mean and scale when restoring model
        predictions. Torch tensors preserve device, dtype and autograd.
        """
        PointWiseData._validate_scale(np.asarray(scale))
        if hasattr(values, "new_tensor"):
            return values * values.new_tensor(scale) + values.new_tensor(mean)
        return np.asarray(values) * np.asarray(scale) + np.asarray(mean)
