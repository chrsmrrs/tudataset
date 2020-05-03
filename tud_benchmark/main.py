from __future__ import division

import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation


def main():
    # Smaller datasets using LIBSVM.
    dataset = [["ENZYMES", True], ["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
                 ["REDDIT-BINARY", False]]

    dataset = [["ENZYMES", True], ["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
                 ["REDDIT-BINARY", False]]

    for d, use_labels in dataset:
        dataset = d
        classes = dp.get_dataset(dataset)

        # print(d + " " + "WL1")
        # all_matrices = []
        # for i in range(1, 6):
        #     gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # print("###")
        # print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
        #
        # print(d + " " + "WLOA")
        # all_matrices = []
        # for i in range(1, 6):
        #     gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # print("###")
        # print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print(d + " " + "GR")
        all_matrices = []
        gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=1, all_std=True))

        # print(d + " " + "SP")
        # all_matrices = []
        # gm = kb.compute_shortestpath_dense(dataset, use_labels)
        # gm_n = aux.normalize_gram_matrix(gm)
        # all_matrices.append(gm_n)
        # print("###")
        # print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))


    ####################################################################################################################

    exit()

    # Larger datasets using LIBLINEAR.
    datataset = [["MCF-7", True, True], ["MOLT-4", True, True], ["TRIANGLES", False, False],
                 ["github_stargazers", False, False],
                 ["reddit_threads", False, False]]
    for d, use_labels, use_edge_labels in datataset:
        print(d)
        dataset = d
        classes = dp.get_dataset(dataset)

        print(d + " " + "WL1")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)
        print("###")
        print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1))

        print(d + " " + "GR")
        all_matrices = []
        gm = kb.compute_graphlet_sparse(dataset, use_labels, use_edge_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
        print("###")
        print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1))

        print(d + " " + "SP")
        all_matrices = []
        gm = kb.compute_shortestpath_sparse(dataset, use_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
        print("###")
        print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1))


if __name__ == "__main__":
    main()
