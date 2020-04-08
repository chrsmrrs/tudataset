from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation, sgd_regressor_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler
import numpy as np

def main():
    print("XXXX")

    targets = dp.get_dataset("ZINC_test", regression=True)
    gm = kb.compute_lwl_2_sparse("ZINC_test", 2, True, True)
    gm = aux.normalize_feature_vector(gm)
    print("XXXX")

    p = sgd_regressor_evaluation([gm], targets, list(range(0, 4000)), list(range(4000, 4500)),
                                      list(range(4500, 5000)), num_repetitions=1)
    print(p)

    exit()

    dp.get_dataset("ZINC_train", regression=True)
    dp.get_dataset("ZINC_val", regression=True)
    dp.get_dataset("ZINC_test", regression=True)

    indices_train = []
    indices_val = []
    indices_test = []

    infile = open("../../../localwl_dev/kgnn/datasets/test.index.txt", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    infile = open("../../../localwl_dev/kgnn/datasets/val.index.txt", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("../../../localwl_dev/kgnn/datasets/train.index.txt", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    print("###")


    targets = kb.read_targets("ZINC_train", indices_train)
    targets.extend(kb.read_targets("ZINC_val", indices_val))
    targets.extend(kb.read_targets("ZINC_test", indices_test))
    targets = np.array(targets)
    print(len(targets))

    print("###")
    all_matrices = []
    for i in range(4,5):
        all_matrices.append(kb.compute_wl_1_sparse_ZINC(i, True, True, indices_train, indices_val, indices_test))

    print("###")
    indices_train = list(range(10000))
    indices_val = list(range(1000))
    indices_test = list(range(1000))
    p = eval.sgd_regressor_evaluation(all_matrices, targets,  indices_train, indices_val, indices_test)
    print(p)








if __name__ == "__main__":
    main()
