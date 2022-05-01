import numpy as np
import os
from nif.data.point_wise_data import PointWiseData

class TravelingWaveHighFreq(PointWiseData):
    def __init__(self):
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        data = np.load(dir_path+'/dataset/traveling_wave_high_freq.npz')['data']
        parameter_data = data[:,[0]]
        x_data = data[:,[1]]
        u_data = data[:,[2]]
        super(TravelingWaveHighFreq, self).__init__(parameter_data, x_data, u_data)
        self.data, self.mean, self.std = self.minmax_normalize(self.data_raw, n_para=self.n_p, n_x=self.n_x, n_target=1)

if __name__=='__main__':
    tw = TravelingWaveHighFreq()
    print(tw.data.mean(axis=0))
    print(tw.data.std(axis=0))
    print(tw.data.max(axis=0))
    print(tw.parameter.shape)
    print(tw.x.shape)
    print(tw.u.shape)


