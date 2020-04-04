from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler


def main():
    # dataset = "deezer_ego_nets"
    # classes = dp.get_dataset(dataset)
    # print(classes.sum()/len(classes))
    #
    # print("WL")
    # all_matrices = []
    # for i in range(1, 4):
    #     print(i)
    #     gm = kb.compute_wl_1_sparse(dataset, i, False, False)
    #     gm_n = aux.normalize_feature_vector(gm)
    #     all_matrices.append(gm)
    #     all_matrices.append(gm_n)
    # print("###")
    # print(linear_svm_evaluation(all_matrices, classes, num_repetitions=1, all_std=True, primal=True))

    # print("WL")
    # all_matrices = []
    # for i in range(3, 4):
    #     print(i)
    #     gm = kb.compute_lwlp_2_dense(dataset, i, False, False)
    #     gm_n = aux.normalize_gram_matrix(gm)
    #     #all_matrices.append(gm)
    #     all_matrices.append(gm_n)
    # print("###")
    # print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=1, all_std=True))
    #
    #
    # exit()
    #
    targets = dp.get_dataset("ZINC_val", regression=True)
    f = kb.compute_lwlp_2_sparse("ZINC_val", 4, True, True)
    #f= aux.normalize_feature_vector(f)
    train, test =  train_test_split(list(range(targets.shape[0])), train_size=0.9, shuffle=True)
    print("###")

    X_train = f[train]
    X_test = f[test]

    reg = SGDRegressor(max_iter=5000).fit(X_train, targets[train])
    p = reg.predict(X_test)
    print(mse(targets[test], p))
    exit()

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
        for i in range(1, 6):
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("WLOA")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("GR")
        all_matrices = []
        gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("SP")
        all_matrices = []
        gm = kb.compute_shortestpath_dense(dataset, use_labels)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWL2")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwl_2_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWLP2")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwlp_2_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWL2OA")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwl_2_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

        print("LWLP2OA")
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_lwlp_2_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        print("###")
        print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))


if __name__ == "__main__":
    main()
