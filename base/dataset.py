import os
from operator import itemgetter

from collections import OrderedDict

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from base.transforms3D import *
from base.utils import load_npy


class GenericDataArranger(object):
    def __init__(self, dataset_info, dataset_path, debug):
        self.dataset_info = dataset_info
        self.debug = debug
        self.trial_list = self.generate_raw_trial_list(dataset_path)
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()

    def generate_iterator(self):
        iterator = self.dataset_info['partition']
        return iterator

    def generate_partitioned_trial_list(self, window_length, hop_length, fold, windowing=True):

        train_validate_range = self.partition_range['train'] + self.partition_range['validate']
        assert  len(train_validate_range) == self.fold_to_partition['train'] + self.fold_to_partition['validate']

        partition_range = list(np.roll(train_validate_range, fold))
        partition_range += self.partition_range['test'] + self.partition_range['extra']
        partitioned_trial = {}

        for partition, num_fold in self.fold_to_partition.items():
            partitioned_trial[partition] = []

            for i in range(num_fold):
                index = partition_range.pop(0)
                trial_of_this_fold = list(itemgetter(*index)(self.trial_list))

                if len(index) == 1:
                    trial_of_this_fold = [trial_of_this_fold]

                for path, trial, length in trial_of_this_fold:
                    if not windowing:
                        window_length = length

                    windowed_indices = self.windowing(np.arange(length), window_length=window_length,
                                                      hop_length=hop_length)

                    for index in windowed_indices:
                        partitioned_trial[partition].append([path, trial, length, index])

        return partitioned_trial

    def calculate_mean_std(self, partitioned_trial):
        feature_list = self.get_feature_list()
        mean_std_dict = {partition: {feature: {'mean': None, 'std': None} for feature in feature_list} for partition in partitioned_trial.keys()}

        # Calculate the mean
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                sums = 0
                for path, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    sums += data.sum()
                mean_std_dict[partition][feature]['mean'] = sums / (lengths + 1e-10)

        # Then calculate the standard deviation.
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                x_minus_mean_square = 0
                mean = mean_std_dict[partition][feature]['mean']
                for path, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    x_minus_mean_square += np.sum((data - mean) ** 2)
                x_minus_mean_square_divide_N_minus_1 = x_minus_mean_square / (lengths - 1)
                mean_std_dict[partition][feature]['std'] = np.sqrt(x_minus_mean_square_divide_N_minus_1)

        return mean_std_dict

    @staticmethod
    def partition_range_fn():
        raise NotImplementedError

    @staticmethod
    def assign_fold_to_partition():
        raise NotImplementedError

    @staticmethod
    def get_feature_list():
        feature_list = ['landmark', 'action_unit', 'mfcc', 'egemaps', 'vggish', 'bert']
        return feature_list

    def generate_raw_trial_list(self, dataset_path):
        trial_path = os.path.join(dataset_path, self.dataset_info['data_folder'])

        trial_dict = OrderedDict({'train': [], 'validate': [], 'extra': [], 'test': []})
        for idx, partition in enumerate(self.generate_iterator()):

            if partition == "unused":
                continue

            trial = self.dataset_info['trial'][idx]
            path = os.path.join(trial_path, trial)
            length = self.dataset_info['length'][idx]

            trial_dict[partition].append([path, trial, length])

        trial_list = []
        for partition, trials in trial_dict.items():
            trial_list.extend(trials)

        return trial_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        length = len(x)

        if length >= window_length:
            steps = (length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < length - 1:
                sampled_x.append(x[-window_length:])
        else:
            sampled_x = [x]

        return sampled_x


class GenericDataset(Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=None,
                 time_delay=0, feature_extraction=0):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

    def get_index_given_emotion(self):
        raise NotImplementedError

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if "video" in self.modality:
            if self.mode == 'train':
                self.transform_dict['video'] = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupRandomCrop(48, 40),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])
            else:
                self.transform_dict['video'] = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupCenterCrop(40),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def get_feature_transform(self, feature):
        if  "logmel" in feature:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        return transform

    def __getitem__(self, index):
        path, trial, length, index = self.data_list[index]

        examples = {}

        for feature in self.modality:
            examples[feature] = self.get_example(path, length, index, feature)

        if len(index) < self.window_length:
            index = np.arange(self.window_length)

        return examples, trial, length, index

    def __len__(self):
        return len(self.data_list)

    def get_example(self, path, length, index, feature):


        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        # Probably, a trial may be shorter than the window, so the zero padding is employed.
        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        # Sometimes we may want to shift the label, so that
        # the ith label point  corresponds to the (i - time_delay)-th data point.
        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")

        # For the test set, labels of zeros are generated as dummies.
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label



