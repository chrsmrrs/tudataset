import os.path as osp

import numpy as np
from torch_geometric.datasets import TUDataset


# Return classes as a numpy array.
def read_classes(ds_name):
    # Classes
    with open("datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_multi_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [[float(j) for j in i.split(",")] for i in list(f)]
    f.closed

    return np.array(classes)


# Download dataset, regression problem=False, multi-target regression=False.
def get_dataset(dataset, regression=False, multi_target_regression=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    TUDataset(path, name=dataset)

    if multi_target_regression:
        return read_multi_targets(dataset)
    if not regression:
        return read_classes(dataset)
    else:
        return read_targets(dataset)
