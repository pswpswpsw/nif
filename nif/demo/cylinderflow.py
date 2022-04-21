import numpy as np
import os
from .point_wise_data import PointWiseData

class CylinderFlow(PointWiseData):
    def __init__(self):
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        data = np.load(dir_path+'/data/cylinderflow.npz')['data']
        parameter_data = data[:,[0]]
        x_data = data[:,[1,2]]
        u_data = data[:,[3,4]]
        super(CylinderFlow, self).__init__(parameter_data, x_data, u_data)
        self.data, self.mean, self.std, self.sample_weight = self.minmax_normalize(self.data_raw,
                                                                                   n_para=self.n_p,
                                                                                   n_x=self.n_x,
                                                                                   n_target=2,
                                                                                   area_weighted=True)



if __name__=='__main__':
    tw = CylinderFlow()
    print(tw.data.mean(axis=0))
    print(tw.data.std(axis=0))
    print(tw.parameter.shape)
    print(tw.x.shape)
    print(tw.u.shape)