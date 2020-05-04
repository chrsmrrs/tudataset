import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation


def main():
    dataset = "ENZYMES"
    classes = dp.get_dataset(dataset)

    all_matrices = []
    for i in range(1, 6):
        gm = kb.compute_wl_1_sparse(dataset, i, True, False)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)

    acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True,
                                          max_iterations=-1)
    print(dataset + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))


    exit()

    # Smaller datasets using LIBSVM.
    dataset = [["ENZYMES", True], ["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
                 ["REDDIT-BINARY", False]]

    results = []

    for d, use_labels in dataset:
        dataset = d
        classes = dp.get_dataset(dataset)

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True)
        print(d + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwl_2_dense(dataset, i, use_labels, False, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True)
        print(d + " " + "LWL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "LWL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwlp_2_dense(dataset, i, use_labels, False, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True)
        print(d + " " + "LWLP2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "LWLP2 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True)
        print(d + " " + "WLOA " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "WLOA " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True)
        print(d + " " + "GR " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GR " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        gm = kb.compute_shortestpath_dense(dataset, use_labels)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True)
        print(d + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

    ####################################################################################################################
    # Larger datasets using LIBLINEAR.

    results = []
    dataset = [["MCF-7", True, True], ["MOLT-4", True, True], ["TRIANGLES", False, False],
                 ["github_stargazers", False, False],
                 ["reddit_threads", False, False]]

    for d, use_labels, use_edge_labels in dataset:
        print(d)
        dataset = d
        classes = dp.get_dataset(dataset)

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1)
        print(d + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwl_2_sparse(dataset, i, use_labels, use_edge_labels, False)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1)
        print(d + " " + "LWL2SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "LWL2SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwlp_2_sparse(dataset, i, use_labels, use_edge_labels, False)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1)
        print(d + " " + "LWLP2SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "LWLP2SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        gm = kb.compute_graphlet_sparse(dataset, use_labels, use_edge_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1)
        print(d + " " + "GRSP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GRSP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        gm = kb.compute_shortestpath_sparse(dataset, use_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=False,
                                    max_iterations=-1)
        print(d + " " + "SPSP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "SPSP " + str(acc) + " " + str(s_1) + " " + str(s_2))


    print("DONE! :*")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
