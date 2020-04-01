from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0

import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp


def main():
    datataset = [["ENZYMES", True],
                 ["IMDB-BINARY", False], ["IMDB-MULTI", False],
                 ["NCI1", True],
                 ["PROTEINS", True],
                 ["REDDIT-BINARY", False],
                 ["deezer_ego_nets", False]]
    for d, use_labels in datataset:

        dataset = d
        classes = dp.get_dataset(dataset)

        print("WL1")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("WLOA")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("GR")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_graphlet_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("SP")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_shortestpath_dense(dataset, i, use_labels)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWL2")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_lwl_2_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWLP2")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_lwlp_2_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWL2OA")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_lwl_2_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWLP2OA")
        all_matrices = []
        for i in range(0, 6):
            gm = kb.compute_lwlp_2_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm)
            all_matrices.append(gm_n)
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))


if __name__ == "__main__":
    main()
