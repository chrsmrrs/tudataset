import numpy as np
import networkx as nx
import os.path as path
from torch_geometric.datasets import TUDataset
import os.path as osp



# Return classes as numpy array.
def read_classes(ds_name):
    # Classes
    with open("datasets/" + ds_name  + "/" +  ds_name + "/raw/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)

def read_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name  + "/" + ds_name  + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_multi_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name  + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [[float(j) for j in i.split(",")] for i in list(f)]
    f.closed

    return np.array(classes)


def get_dataset(dataset, regression=False, multigregression=False, one_hot=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    TUDataset(path, name=dataset)

    if multigregression:
        return read_multi_targets(dataset)
    if not regression:
        return read_classes(dataset)
    else:
        return read_targets(dataset)