import numpy as np
from .point_wise_data import PointWiseData

class CylinderFlow(PointWiseData):
    def __init__(self):
        data = np.load('./data/cylinderflow.npz')['data']
        parameter_data = data[:,[0]]
        x_data = data[:,[1,2]]
        u_data = data[:,[3,4]]
        sample_weight_data = data[:,[-1]]
        super(CylinderFlow, self).__init__(parameter_data, x_data, u_data, sample_weight_data=sample_weight_data)
        self.data, self.mean, self.std = self.minmax_normalize(self.data_raw, n_para=self.n_p, n_x=self.n_x)

if __name__=='__main__':
    tw = CylinderFlow()
    print(tw.data.mean(axis=0))
    print(tw.data.std(axis=0))
    print(tw.parameter.shape)
    print(tw.x.shape)
    print(tw.u.shape)
    print(tw.sample_weight_data.shape)