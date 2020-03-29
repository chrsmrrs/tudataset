from __future__ import division
import numpy as np

import pandas as pd
from auxiliarymethods import dataset_parsers as dp
from auxiliarymethods import evaluation as val

from scipy import sparse as sp
from gnn_baselines.gin import GIN0
from auxiliarymethods.evaluation import gnn_evaluation

import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.dataset_parsers as dp


def main():
    dataset = "ENZYMES"


    # all_matrices = []
    classes = dp.read_classes(dataset)
    # for i in range(0,6):
    #     gm = kb.compute_wl_1_sparse(dataset, i, True, False)
    #     gm = aux.normalize_feature_vector(gm)
    #     all_matrices.append(gm)
    # print(val.linear_svm_evaluation(all_matrices, classes, num_repetitions=10))
    #
    all_matrices = []
    for i in range(0,6):
        gm = kb.compute_wl_1_dense(dataset, i, True, False)
        gm = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm)
    print(val.kernel_svm_evaluation(all_matrices, classes,num_repetitions=10))


    print(gnn_evaluation(GIN0, dataset, [2], [64], num_repetitions=10, all_std=True))

if __name__ == "__main__":
    main()
