import os
import shutil
from pathlib import Path
import pickle

import numpy as np
import math



def load_npy(path, feature):
    filename = os.path.join(path, feature + ".npy")
    data = np.load(filename, mmap_mode='c')
    return data

def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def save_to_pickle(path, data, replace=False):
    if replace:
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
    else:
        if not os.path.isfile(path):
            with open(path, 'wb') as handle:
                pickle.dump(data, handle)


def copy_file(input_filename, output_filename):
    if not os.path.isfile(output_filename):
        shutil.copy(input_filename, output_filename)


def expand_index_by_multiplier(index, multiplier):
    expanded_index = []
    for value in index:
        expanded_value = [i for i in np.arange(value * multiplier, (value + 1) * multiplier)]
        expanded_index.extend(expanded_value)

    return expanded_index


def get_filename_from_a_folder_given_extension(folder, extension, string=""):
    file_list = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(extension):
            if string in file:
                file_list.append(os.path.join(folder, file))

    return file_list

def get_all_files_recursively_by_ext(root, ext, string=""):
    found = []
    for path in Path(root).rglob('*.{}'.format(ext)):
        if str(path).split(os.sep)[-1].startswith(string) and str(path).split(os.sep)[-1].endswith(string+"."+ext):
            found.append(str(path))
    return sorted(found)

def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
