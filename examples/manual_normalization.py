import numpy as np

from nif.data.point_wise_data import PointWiseData

data = np.random.rand(100, 3)
normalized_data, mean, std = PointWiseData.minmax_normalize(
    raw_data=data, n_para=1, n_x=1, n_target=1
)

print(normalized_data.min(axis=0))
print(normalized_data.max(axis=0))
