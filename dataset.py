from base.dataset import GenericDataArranger, GenericDataset
from torch.utils.data import Dataset as PytorchDataset

import numpy as np
from PIL import Image


class Dataset(GenericDataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=None,
                 time_delay=0, feature_extraction=0):
        super().__init__(data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=mean_std,
                 time_delay=time_delay, feature_extraction=feature_extraction)


class DataArranger(GenericDataArranger):
    def __init__(self, dataset_info, dataset_path, debug):
        super().__init__(dataset_info, dataset_path, debug)

    @staticmethod
    def get_feature_list():
        feature_list = ['vggish', 'bert', 'egemaps']
        return feature_list

    def partition_range_fn(self):
        partition_range = {
            'train': [np.arange(0, 71), np.arange(71, 142), np.arange(142, 213), np.arange(213, 284), np.arange(284, 356)],
            'validate': [np.arange(356, 432)],
            'test': [],
            'extra': [np.arange(432, 594)]}

        # partition_range = {
        #     'train': [np.arange(0, 62), np.arange(61, 124), np.arange(124, 186), np.arange(186, 248)],
        #     'validate': [np.arange(248, 319)],
        #     'test': [],
        #     'extra': [np.arange(319, 412)]}


        if self.debug == 1:
            partition_range = {
                'train': [np.arange(0, 1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), ],
                'validate': [np.arange(5, 6)],
                'test': [np.arange(6, 7)],
                'extra': [np.arange(7, 8)]}

            # partition_range = {
            #     'train': [np.arange(0, 1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4)],
            #     'validate': [np.arange(5, 6)],
            #     'test': [np.arange(6, 7)],
            #     'extra': [np.arange(7, 8)]}
        return partition_range

    @staticmethod
    def assign_fold_to_partition():
        fold_to_partition = {'train': 5, 'validate': 1, 'test': 0, 'extra': 1}
        return fold_to_partition


