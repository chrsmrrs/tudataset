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
    datataset = [["MCF-7", True, True],
                 ["MOLT-4", True, True],
                 ["TRIANGLES", False, False],
                 ["github_stargazers", False, False],
                 ["reddit_threads", False, False]]
    for d, use_labels, use_edge_labels in datataset:
        dataset = d
        classes = dp.get_dataset(dataset)

        print("WL1")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)
        print("###")
        print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=True))

        print("GR")
        all_matrices = []
        gm = kb.compute_graphlet_sparse(dataset, use_labels, use_edge_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
        print("###")
        print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=True))

        print("SP")
        all_matrices = []
        gm = kb.compute_shortestpath_sparse(dataset, use_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
        print("###")
        print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=True))



    # datataset = [["ENZYMES", True],
    #     #              ["IMDB-BINARY", False], ["IMDB-MULTI", False],
    #     #              ["NCI1", True],
    #     #              ["PROTEINS", True],
    #     #              ["REDDIT-BINARY", False],
    #     #              ["deezer_ego_nets", False]]
    # for d, use_labels in datataset:
    #
    #     dataset = d
    #     classes = dp.get_dataset(dataset)
    #
    #     print("WL1")
    #     all_matrices = []
    #     for i in range(1, 6):
    #         gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
    #         gm_n = aux.normalize_gram_matrix(gm)
    #         all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
    #
    #     print("WLOA")
    #     all_matrices = []
    #     for i in range(1, 6):
    #         gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
    #         gm_n = aux.normalize_gram_matrix(gm)
    #         all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
    #
    #     print("GR")
    #     all_matrices = []
    #     gm = kb.compute_graphlet_dense(dataset, use_labels, False)
    #     gm_n = aux.normalize_gram_matrix(gm)
    #     all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
    #
    #     print("SP")
    #     all_matrices = []
    #     gm = kb.compute_shortestpath_dense(dataset, use_labels)
    #     gm_n = aux.normalize_gram_matrix(gm)
    #     all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))


if __name__ == "__main__":
    main()
