import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class Dataset:
    def __init__(self, Gaussian_setting):
        self.category_count = len(Gaussian_setting)
        self.Gaussian_setting = Gaussian_setting
        self.data = []
        # generate data
        for i in range(self.category_count):
            one_class_data = np.random.multivariate_normal(**self.Gaussian_setting[i])
            self.data.append(one_class_data)

    def get_test_data(self, size_list):
        # generate data with mixed classes
        test_data = []
        for ind, g_setting in enumerate(self.Gaussian_setting):
            g_setting['size'] = size_list[ind]
            one_class_data = np.random.multivariate_normal(**self.Gaussian_setting[ind])
            test_data.append(one_class_data)
        return np.array(test_data)
