import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

class Dataset:
    def __init__(self, Gaussian_setting):
        """
        :param Gaussian_setting:
        eg:
        Gaussian_setting_config = [
            {
                'mean': [0, 0],
                'cov': [[1, 0], [0, 15]],
                'size': 1000
            },
            {
                'mean': [5, -5],
                'cov': [[15, 0], [0, 1]],
                'size': 1000
            }
        ]
        """
        self.category_count = len(Gaussian_setting)
        self.Gaussian_setting = Gaussian_setting
        self.data = []
        self.label = []
        # generate data
        for i in range(self.category_count):
            one_class_data = np.random.multivariate_normal(**self.Gaussian_setting[i])
            one_class_label = [0] * self.category_count
            one_class_label[i] = 1

            self.data.append(one_class_data)
            self.label.append([one_class_label] * len(one_class_data))

    def get_more_data(self, size_list):
        # generate data with mixed classes
        more_data = []
        for ind, g_setting in enumerate(self.Gaussian_setting):
            g_setting['size'] = size_list[ind]
            one_class_data = np.random.multivariate_normal(**self.Gaussian_setting[ind])
            more_data.append(one_class_data)
        return np.array(more_data)

    def shuffle_data(self, data, label, shuff=True):
        all_data = []
        all_label = []
        for c_data in data:
            all_data.extend(c_data)
        for c_label in label:
            all_label.extend(c_label)
        all_data_label_pair = zip(all_data, all_label)
        if shuff:
            random.shuffle(all_data_label_pair)
        shuffled_all_data, shuffled_all_label = zip(*all_data_label_pair)
        return shuffled_all_data, shuffled_all_label
